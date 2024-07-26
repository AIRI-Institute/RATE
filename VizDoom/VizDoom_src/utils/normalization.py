import torch
from tqdm import tqdm

def batch_mean_and_std(loader):
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    print("Calculating dataset statistics...")
    for images, _, _, _, _, _ in tqdm(loader, total=len(loader)):
        b, l, c, h, w = images.shape
        images = images.reshape(-1, c, h, w)
        nb_pixels = b * l * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

    return mean, std

def z_normalize(data, mean, std):
    """_summary_

    traj_norm.shape = torch.Size([5760, 3, 64, 112])
    """
    return (data - mean[None, :, None, None]) / std[None, :, None, None]


def inverse_z_normalize(data, mean, std):
    return data * std[None, :, None, None] + mean[None, :, None, None]
  
# mean, std = batch_mean_and_std(val_dataloader)
# print("mean and std:", mean, std)