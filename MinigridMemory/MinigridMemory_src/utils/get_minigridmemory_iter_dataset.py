import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F

class MinigridMemoryIterDataset(Dataset):
    def __init__(self, directory, gamma, max_length, normalize):
        """_summary_

        Args:
            directory (str): path to the directory with data files
            gamma (float): discount factor
            max_length (int): maximum number of timesteps used in batch generation
                                (max in dataset: 1001)
            only_non_zero_rewards (bool): if True then use only trajectories
                                            with non-zero reward in the first
                                            max_length timesteps
        """
        self.directory = directory
        self.file_list = os.listdir(directory)
        self.gamma = gamma
        self.max_length = max_length
        self.normalize = normalize
        self.filtered_list = []
        print('Filtering data...')
        self.filter_trajectories()

    def discount_cumsum(self, x):
        """
        Compute the discount cumulative sum of a 1D array.

        Args:
            x (ndarray): 1D array of values.

        Returns:
            ndarray: Discount cumulative sum of the input array.
        """
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + self.gamma * discount_cumsum[t+1]
        return discount_cumsum
    
    def filter_trajectories(self):
        for idx in tqdm(range(len(self.file_list))):
            #for idx in range(len(self.file_list)):
            file_path = os.path.join(self.directory, self.file_list[idx])
            data = np.load(file_path)
            if data['obs'].shape[0] <= self.max_length:
                self.filtered_list.append(self.file_list[idx])

    def __len__(self):
        return len(self.filtered_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filtered_list[idx])
        data = np.load(file_path)

        s = data['obs'] * 255.0 # because data is already divided by 255
        a = data['action']
        r = data['reward']
        d = data['done']

        length = s.shape[0]
        
        s = torch.from_numpy(s).float()
        if self.normalize == 1:
            s = s / 255.0

        
        s = s.unsqueeze(0)
        

        # if not ohe:
        a = torch.from_numpy(a).unsqueeze(0).unsqueeze(-1)
        # IF OHE: a = torch.from_numpy(a).unsqueeze(0)#.unsqueeze(-1)
        rtg = torch.from_numpy(self.discount_cumsum(r)).unsqueeze(0).unsqueeze(-1)
        d = torch.from_numpy(d).unsqueeze(0).unsqueeze(-1).to(dtype=torch.long)
       
        timesteps = torch.from_numpy(np.arange(0, self.max_length).reshape(1, -1, 1))
        mask = torch.ones_like(a)
        
        pad_length = self.max_length - length
        state_to_pad = s[:, -1:, :, :, :].expand(1, pad_length, 3, 84, 84)
        s = torch.cat((s, state_to_pad), dim=1)

        a = F.pad(a, (0, 0, 0, pad_length, 0, 0), value=-10)
        d = F.pad(d, (0, 0, 0, pad_length, 0, 0), value=2)
        mask = F.pad(mask, (0, 0, 0, pad_length, 0, 0), value=0)
        rtg = F.pad(rtg, (0, 0, 0, pad_length, 0, 0), value=0)

        return s.squeeze(), a.squeeze().unsqueeze(-1), rtg.squeeze().unsqueeze(-1), d.squeeze().unsqueeze(-1), timesteps.squeeze().unsqueeze(-1), mask.squeeze().unsqueeze(-1)


# Assuming 'directory_path' is the path to the directory containing .npz files
# dataset = ViZDoomIterDataset('../VizDoom_data/iterative_data/', gamma=1.0, max_length=90, normalize=1)
# dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)