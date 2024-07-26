import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class VisDoomDataset(Dataset):
    def __init__(self, data, gamma, max_length, normalize):
        """
        Custom dataset for VisDoom data.

        Args:
            data (dict): Dictionary containing the data with keys 'obs', 'action', 'reward', 'done', 'is_red'.
            gamma (float): Discount factor for reward discounting.
            max_length (int): Maximum length of the data to consider.
            normalize (bool): Flag indicating whether to normalize the state data.
        """
        self.data = data
            
        self.gamma = gamma
        self.max_length = max_length
        self.normalize = normalize
        
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

    def __getitem__(self, index):
        """
        Get a single sample from the dataset at the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple of tensors containing the state, action, discounted cumulative sum of rewards, done flag,
                   timesteps, and mask.
        """
        s = self.data['obs'][index]
        a = self.data['action'][index]
        r = self.data['reward'][index]
        d = self.data['done'][index]
        
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
    
    def __len__(self):
        return len(self.data['obs'])
    
def get_dataset(data, gamma, max_length, normalize):
    """
    Create data loaders for the given data.

    Args:
        data (dict): Dictionary containing the data with keys 'obs', 'action', 'reward', 'done', 'is_red'.
        gamma (float): Discount factor for reward discounting.
        max_length (int): Maximum length of the data to consider during training / validation.
        normalize (bool): Flag indicating whether to normalize the state data.

    Returns:
        DataLoader: Training data loader.
    """
    dataset = VisDoomDataset(data, gamma, max_length, normalize)
    
    return dataset