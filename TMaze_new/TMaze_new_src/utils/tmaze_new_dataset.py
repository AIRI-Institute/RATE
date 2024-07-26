import numpy as np
from TMaze_new.TMaze_new_src.utils.tmaze import TMazeClassicPassive
import glob
import pickle
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

OMP_NUM_THREADS = '1'
os.environ['OMP_NUM_THREADS'] = OMP_NUM_THREADS 

def TMaze_data_generator(max_segments, multiplier, hint_steps, desired_reward=1.0, win_only=True, segment_length=30):
    current_directory = glob.os.getcwd()
    gen_flag= False
    for i in range(1, max_segments+1):
        name = f'new_tmaze_data_segment_{i}_multiplier_{multiplier}_hint_steps_{hint_steps}_desired_reward_{desired_reward}_win_only_{win_only}_segment_length_{segment_length}'
        data_path = current_directory + '/TMaze_new/TMaze_new_data/'
        save_path = data_path + f"{name}.pickle"
        if not os.path.exists(save_path):
            if i == 1: 
                print("Data is not available. Generating...")
                gen_flag = True
            
            generate_dict_with_trajectories(segments=i, multiplier=multiplier, hint_steps=hint_steps, 
                                            desired_reward=desired_reward, win_only=win_only, segment_length=segment_length)
        else:
            if i == 1: print("Data is available.")
        if i == max_segments and gen_flag == True: print("Data successfully generated.")
            

class TMaze_dataset(Dataset):
    def __init__(self, path, gamma, mode, max_length):
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
            
        self.gamma = gamma
        self.max_length = max_length
        self.mode = mode
        """
        equal - 90/89/90
        diff - 50/49/50 or 76/77/76 padded to 90/89/90
        """
        
    def discount_cumsum(self, x):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0]-1)):
            discount_cumsum[t] = x[t] + self.gamma * discount_cumsum[t+1]
        return discount_cumsum

    def __getitem__(self, index):
        channels = self.data[index]['obs'].shape[1]
        # channels = 4
        if self.mode == "equal":
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
            
        elif self.mode == "diff":
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
    def __init__(self, n_init, n_final, multiplier, hint_steps, 
                 batch_size, mode, cut_dataset, one_mixed_dataset=False, 
                 desired_reward=1.0, win_only=True, segment_length=30):
        self.n_init = n_init
        self.n_final = n_final
        self.multiplier = multiplier
        self.hint_steps = hint_steps
        self.batch_size = batch_size
        self.mode = mode
        self.desired_reward = desired_reward
        self.win_only = win_only
        self.cut_dataset = cut_dataset
        self.one_mixed_dataset = one_mixed_dataset
        self.segment_length = segment_length
        self.dataset = self._generate_combined_dataset()
        self.dataloader = self._generate_dataloader()
    
    def _get_dataloaders(self, path, gamma, mode, max_length):
        train_dataset = TMaze_dataset(path, gamma, mode, max_length)
        return train_dataset

    def _get_segment_dataloaders(self, N, multiplier, hint_steps, maxN, mode, desired_reward=1.0, win_only=True):
        name = f'new_tmaze_data_segment_{N}_multiplier_{multiplier}_hint_steps_{hint_steps}_desired_reward_{desired_reward}_win_only_{win_only}_segment_length_{self.segment_length}'
        data_path = f'TMaze_new/TMaze_new_data/{name}.pickle'

        train_dataset = self._get_dataloaders(path=data_path, gamma=1.0, mode="diff", max_length=self.segment_length*maxN)

        return train_dataset

    def _generate_combined_dataset(self):
        if self.one_mixed_dataset == False:
            dataset = self._get_segment_dataloaders(N=self.n_init, multiplier=self.multiplier, hint_steps=self.hint_steps, 
                                                    maxN=self.n_final, mode=self.mode, desired_reward=self.desired_reward,
                                                    win_only=self.win_only)
            for N in range(self.n_init+1, self.n_final + 1):

                if self.cut_dataset:
                    dataset = CutDataset(dataset)

                dataset_new = self._get_segment_dataloaders(N=N, multiplier=self.multiplier, hint_steps=self.hint_steps, 
                                                            maxN=self.n_final, mode=self.mode, desired_reward=self.desired_reward,
                                                            win_only=self.win_only)
                dataset = ConcatDataset([dataset, dataset_new])
                
        elif self.one_mixed_dataset == True:
            dataset = self._get_segment_dataloaders(N=self.n_init, multiplier=self.multiplier, hint_steps=self.hint_steps, 
                                                    maxN=self.n_final, mode=self.mode, desired_reward=self.desired_reward,
                                                    win_only=self.win_only)
            for N in range(2, self.n_final + 1):
                dataset_new = self._get_segment_dataloaders(N=N, multiplier=self.multiplier, hint_steps=self.hint_steps, 
                                                            maxN=self.n_final, mode=self.mode, desired_reward=self.desired_reward,
                                                            win_only=self.win_only)
                dataset = ConcatDataset([dataset, dataset_new])

        return dataset

    def _generate_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    
# # Example of usage:
# combined_dataloader = CombinedDataLoader(n_init=1, n_final=2, multiplier=10, hint_steps=1, batch_size=2, mode=None, cut_dataset=False) #modes: "" for full and "mini_" for mini #jlast for last el junction
# train_dataloader = combined_dataloader.dataloader
# dataset = combined_dataloader.dataset
# len(train_dataloader), len(dataset)




def generate_trajectory(episode_timeout, corridor_length, hint_steps, win, seed_env, seed_noise, desired_reward=1.0): 
    """
    seed_env: 0 -> down, 1 -> up
    Returns:
        - states: {y, hint, flag, noise}
        - actions: {act}, act in [0, 1, 2, 3]
        - returns_to_go: {rtg}, rtg in [0, 1]
    """
    # Initialization:
    np.random.seed(seed_env)
    env = TMazeClassicPassive(episode_length=episode_timeout, corridor_length=corridor_length, penalty=0, goal_reward=desired_reward) # return {x, y, hint}
    obs = env.reset() # {x, y, hint}
    np.random.seed(seed_noise)
    obs = np.concatenate((obs, np.array([0]))) # {x, y, hint, flag}
    obs = np.concatenate((obs, np.array([np.random.randint(low=-1, high=1+1)]))) # {x, y, hint, flag, noise}
    done = False
    DESIRED_REWARD = desired_reward
    rtg = DESIRED_REWARD
    obss, acts, rtgs, dones, trajectories = [], [], [rtg], [], []
    
    # Agent's motion:
    for t in range(episode_timeout):
        #act = env.action_space.sample()
        trajectories.append(obs)
        obss.append(obs)
        
        # Optimal policy:
        if t == 0:
            act = 0
        elif obs[0] == corridor_length:
            if win == True:
                if trajectories[0][2] == -1:
                    act = 3 # down
                else:
                    act = 1 # up
            else:
                if trajectories[0][2] == -1:
                    act = 1
                else:
                    act = 3
        else:
            act = 0 # right
            
        obs, rew, done, info = env.step(act)
        rtg = rtg - (rew)
        """ HINT N STEPS """
        if t <= hint_steps - 2:
            obs[2] = trajectories[0][2]
        """ ============ """
        acts.append(act)
        rtgs.append(rtg)
        dones.append(int(done))
        
        # {x, y, hint} -> {x, y, hint, flag}
        if t != corridor_length-1:
            obs = np.concatenate((obs, np.array([0])))
        else:
            obs = np.concatenate((obs, np.array([1])))
        # {x, y, hint, flag} -> {x, y, hint, flag, noise}
        obs = np.concatenate((obs, np.array([np.random.randint(low=-1, high=1+1)])))
        
        if done:
            obss.append(obs)
            break

    return np.array(obss)[:, 1:], np.array(acts), np.array(rtgs), np.array(dones) #[:, 1:]


def generate_dict_with_trajectories(segments, multiplier, hint_steps, desired_reward=1.0, win_only=True, segment_length=30):
    current_directory = glob.os.getcwd()
    name = f'new_tmaze_data_segment_{segments}_multiplier_{multiplier}_hint_steps_{hint_steps}_desired_reward_{desired_reward}_win_only_{win_only}_segment_length_{segment_length}'
    data_path = current_directory + '/TMaze_new/TMaze_new_data/'
    isExist = os.path.exists(data_path)
    save_path = data_path + f"{name}.pickle"
    if not isExist:
        os.makedirs(data_path)

    files = glob.glob(save_path + '*')
    for f in files:
        os.remove(f)

    data = {}
    iteration = 0
    if win_only:
        for seed_env in [0, 1]:
            for seed_noise in tqdm(range(multiplier)):
                obss, acts, rtgs, dones = generate_trajectory(episode_timeout=segment_length*segments, corridor_length=segment_length*segments-2, 
                                                            hint_steps=hint_steps, win=True, seed_env=seed_env, seed_noise=seed_noise, desired_reward=desired_reward)
                data[iteration] = {'obs': obss, 'action': acts, 'rtg': rtgs, 'done': dones}
                iteration += 1
    else:
        for seed_env in [0, 1]:
            for seed_noise in tqdm(range(multiplier//2)):
                obss, acts, rtgs, dones = generate_trajectory(episode_timeout=segment_length*segments, corridor_length=segment_length*segments-2, 
                                                            hint_steps=hint_steps, win=True, seed_env=seed_env, seed_noise=seed_noise, desired_reward=desired_reward)
                data[iteration] = {'obs': obss, 'action': acts, 'rtg': rtgs, 'done': dones}
                iteration += 1
            for seed_noise in tqdm(range(multiplier//2, multiplier)):
                obss, acts, rtgs, dones = generate_trajectory(episode_timeout=segment_length*segments, corridor_length=segment_length*segments-2, 
                                                            hint_steps=hint_steps, win=False, seed_env=seed_env, seed_noise=seed_noise, desired_reward=desired_reward)
                data[iteration] = {'obs': obss, 'action': acts, 'rtg': rtgs, 'done': dones}
                iteration += 1

    with open(save_path,'wb') as f:
            pickle.dump(data, f)