import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
import torch.nn.functional as F


from ActionAssociativeRetrieval.AAR_src.utils.aar_env import ActionAssociativeRetrieval


def generate_rajectory(required_act, stay_number, seed):
    env = ActionAssociativeRetrieval(stay_number=stay_number, seed=seed)

    DESIRED_REWARD = 1
    rtg = DESIRED_REWARD

    states, actions, rtgs, dones = [], [], [rtg], []
    state = env.reset()

    for t in range(env.episode_timeout):
        states.append(state)

        # * Optimal Policy
        # ** act = 0 | 1 -> transfer, act = 0 -> stay
        if t == 0 or t == env.episode_timeout - 1:
            act = required_act # S0 -> S1 & S1 -> S0
        else:
            act = 2 # S1 -> S1

        state, reward, done, info = env.step(act)

        actions.append(act)
        rtg = rtg - reward
        rtgs.append(rtg)
        dones.append(done)

        if done:
            states.append(state)
            break
    states, actions, rtgs, dones = np.array(states), np.array(actions), np.array(rtgs), np.array(dones)

    return states, actions, rtgs, dones



def generate_dict_with_trajectories(segments: int, multiplier: int, sampling_action: int = None, context_length: int = 30):
    """
    sampling_action: 
        None -> required_act in [0, 1]
        0 -> required_act in [0]
        1 -> required_act in [1]
    """

    if sampling_action is None:
        REQUIRED_ACTS = [0, 1]
    elif sampling_action == 0:
        REQUIRED_ACTS = [0]
    elif sampling_action == 1:
        REQUIRED_ACTS = [1]

    stay_number = context_length * segments - 3

    data = {}
    iteration = 0
    for required_act in REQUIRED_ACTS:
        if sampling_action is None:
            if required_act == 0:
                RANGE = range(multiplier)
            elif required_act == 1:
                RANGE = range(multiplier+1, 2*multiplier+1)
        else:
            RANGE = range(multiplier)
        for seed_noise in tqdm(RANGE):
            states, actions, rtgs, dones = generate_rajectory(required_act=required_act, stay_number=stay_number, seed=seed_noise)
            data[iteration] = {'obs': states, 'action': actions, 'rtg': rtgs, 'done': dones}
            iteration += 1
    return data

# data = generate_dict_with_trajectories(segments=3, multiplier=10, sampling_action=None)

class AAR_dataset(Dataset):
    def __init__(self, data, gamma, max_length):
        self.data = data
        self.gamma = gamma
        self.max_length = max_length
        
    def discount_cumsum(self, x):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0]-1)):
            discount_cumsum[t] = x[t] + self.gamma * discount_cumsum[t+1]
        return discount_cumsum

    def __getitem__(self, index):
        channels = self.data[index]['obs'].shape[1]
            
        traj = self.data[index]
        length = traj['rtg'].shape[0]
        s = traj['obs']
        s = torch.from_numpy(s).float()
        s = s.reshape(1, length, channels)
        a = torch.from_numpy(traj['action'])
        a = a.reshape(1, length-1, 1)
        d = torch.from_numpy(traj['done'].reshape(1, length-1, 1)).to(dtype=torch.long)
        timesteps = torch.from_numpy(np.arange(0, length).reshape(1, -1, 1))
        rtg = torch.from_numpy(traj['rtg']).reshape(1, length, 1)
        mask = torch.ones_like(a)

        #Pad with zeros if length is less than max_length
        # print(length)
        if length <= self.max_length:
            pad_length = self.max_length - length# + 1
            #print(s.shape)
            state_to_pad = s[:, -1, :].unsqueeze(1)# * 0
            while s.shape[1] < self.max_length:
                s = torch.cat((s, state_to_pad), dim=1)
            #print(a[:, -1].item())
            a = F.pad(a, (0, 0, 0, pad_length+1, 0, 0), value=-10)
            d = F.pad(d, (0, 0, 0, pad_length, 0, 0), value=2)
            timesteps = F.pad(timesteps, (0, 0, 0, pad_length, 0, 0), value=0)
            mask = F.pad(mask, (0, 0, 0, pad_length+1, 0, 0), value=0)
            rtg = F.pad(rtg, (0, 0, 0, pad_length, 0, 0), value=0)
        
        return s.squeeze(0), a.squeeze(0), rtg.squeeze(0), d.squeeze(), timesteps.squeeze(), mask.squeeze()
    
    def __len__(self):
        return len(self.data)
    
class CutDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.length = len(dataset) // 2

    def __getitem__(self, index):
        new_index = index * 2
        return self.dataset[new_index]

    def __len__(self):
        return self.length
    
class CombinedDataLoader:
    def __init__(self, n_init, n_final, multiplier, batch_size, cut_dataset, one_mixed_dataset=False, sampling_action=None, context_length=30):
        self.sampling_action = sampling_action
        self.n_init = n_init
        self.n_final = n_final
        self.multiplier = multiplier
        self.batch_size = batch_size
        self.cut_dataset = cut_dataset
        self.one_mixed_dataset = one_mixed_dataset
        self.context_length = context_length
        self.dataset = self._generate_combined_dataset()
        self.dataloader = self._generate_dataloader()
    
    def _get_dataloaders(self, data, gamma, max_length):
        train_dataset = AAR_dataset(data, gamma, max_length)
        return train_dataset

    def _get_segment_dataloaders(self, N, multiplier, maxN):
        data = generate_dict_with_trajectories(segments=N, multiplier=multiplier, sampling_action=self.sampling_action, context_length=self.context_length)
        train_dataset = self._get_dataloaders(data=data, gamma=1.0, max_length=self.context_length*maxN)

        return train_dataset

    def _generate_combined_dataset(self):
        if self.one_mixed_dataset == False:
            dataset = self._get_segment_dataloaders(N=self.n_init, multiplier=self.multiplier, maxN=self.n_final)
            for N in range(self.n_init+1, self.n_final + 1):

                if self.cut_dataset:
                    dataset = CutDataset(dataset)

                dataset_new = self._get_segment_dataloaders(N=N, multiplier=self.multiplier, maxN=self.n_final)
                dataset = ConcatDataset([dataset, dataset_new])
                
        elif self.one_mixed_dataset == True:
            dataset = self._get_segment_dataloaders(N=self.n_init, multiplier=self.multiplier, maxN=self.n_final)
            for N in range(2, self.n_final + 1):
                dataset_new = self._get_segment_dataloaders(N=N, multiplier=self.multiplier, maxN=self.n_final)
                dataset = ConcatDataset([dataset, dataset_new])

        return dataset

    def _generate_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)