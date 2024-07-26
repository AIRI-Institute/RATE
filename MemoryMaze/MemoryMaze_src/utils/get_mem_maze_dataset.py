import torch
import os
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class MemoryMazeDataset(Dataset):
    def __init__(self, directory, gamma, max_length, only_non_zero_rewards):
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
        self.filtered_list = []
        self.list_of_start_indexes = []
        self.only_non_zero_rewards = only_non_zero_rewards
        if self.only_non_zero_rewards:
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
        for idx in tqdm(range(len(self.file_list)), total=len(self.file_list)):
            file_path = os.path.join(self.directory, self.file_list[idx])
            data = np.load(file_path)
            
            # * step=5: >= 2 -> 69177 / >=3 -> 3103 | v4 step=30: >=2 -> 11800 v5 60->6111 v6 90->4249
            for i in range(0, 1000-self.max_length, 90):
                if data['reward'][i:i+self.max_length].sum() >= 2:
                    self.filtered_list.append(self.file_list[idx])
                    self.list_of_start_indexes.append(i)
    

    def __len__(self):
        if self.only_non_zero_rewards:
            return len(self.filtered_list)
        else:
            return len(self.file_list)

    def __getitem__(self, idx):
        if self.only_non_zero_rewards:
            file_path = os.path.join(self.directory, self.filtered_list[idx])
            start_idx = self.list_of_start_indexes[idx]
        else:
            file_path = os.path.join(self.directory, self.file_list[idx])
        data = np.load(file_path)

        image = torch.from_numpy(data['image']).permute(0, 3, 1, 2).float()
        action = torch.from_numpy(data['action'])
        rtg = torch.from_numpy(self.discount_cumsum(data['reward'])).unsqueeze(-1)
        timesteps = torch.from_numpy(np.arange(start_idx, start_idx+self.max_length).reshape(-1))
        done = torch.zeros_like(timesteps)
        done[-1] = 1
        mask = torch.ones_like(timesteps)
        
        image = image[start_idx:start_idx+self.max_length, :, :, :]
        image = image / 255.
        action = action[start_idx:start_idx+self.max_length, :]
        rtg = rtg[start_idx:start_idx+self.max_length, :]

        action = torch.argmax(action, dim=-1).unsqueeze(-1)

        return image, action, rtg, done, timesteps, mask