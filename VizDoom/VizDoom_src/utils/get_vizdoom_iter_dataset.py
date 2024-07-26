import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

class ViZDoomIterDataset(Dataset):
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
            file_path = os.path.join(self.directory, self.file_list[idx])
            data = np.load(file_path)
            if data['obs'].shape[0] == self.max_length:
                self.filtered_list.append(self.file_list[idx])

    def __len__(self):
        return len(self.filtered_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.filtered_list[idx])
        #print(file_path)
        data = np.load(file_path)

        s = data['obs']
        a = data['action']
        r = data['reward']
        d = data['done']
        
        s = torch.from_numpy(s).float()

        if self.normalize == 1:
            s = s / 255.0
        
        s = s.unsqueeze(0)

        a = torch.from_numpy(a).unsqueeze(0).unsqueeze(-1)
        rtg = torch.from_numpy(self.discount_cumsum(r)).unsqueeze(0).unsqueeze(-1)
        d = torch.from_numpy(d).unsqueeze(0).unsqueeze(-1).to(dtype=torch.long)
       
        timesteps = torch.from_numpy(np.arange(0, self.max_length).reshape(1, -1, 1))
        mask = torch.ones_like(a)
        
        # * from beginning of trajectory
        s = s[:, :self.max_length, :, :, :]
        a = a[:, :self.max_length, :]
        rtg = rtg[:, :self.max_length, :]
        d = d[:, :self.max_length, :]
        mask = mask[:, :self.max_length, :]

        return s.squeeze(0), a.squeeze(0), rtg.squeeze(0), d.squeeze(), timesteps.squeeze(), mask.squeeze()


# Assuming 'directory_path' is the path to the directory containing .npz files
# dataset = ViZDoomIterDataset('../VizDoom_data/iterative_data/', gamma=1.0, max_length=90, normalize=1)
# dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)