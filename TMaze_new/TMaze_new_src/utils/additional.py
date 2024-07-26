import numpy as np
import torch
import random
import os

seeds_list = [0,2,3,4,6,9,13,15,18,24,25,31,3,40,41,42,43,44,48,49,50,
              51,62,63,64,65,66,69,70,72,73,74,75,83,84,85,86,87,88,91,
              92,95,96,97,98,100,102,105,106,107,1,5,7,8,10,11,12,14,16,
              17,19,20,21,22,23,26,27,28,29,30,32,34,35,36,37,38,39,45,
              46,47,52,53,54,55,56,57,58,59,60,61,67,68,71,76,77,78,79,80,81,82]

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
