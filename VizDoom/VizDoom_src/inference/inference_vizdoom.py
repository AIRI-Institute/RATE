import numpy as np
import yaml
import torch
from tqdm import tqdm
import argparse

import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from RATE_GTrXL import mem_transformer_v2_GTrXL
from VizDoom.VizDoom_src.inference.val_vizdoom import get_returns_VizDoom
from TMaze_new.TMaze_new_src.utils import seeds_list, get_intro2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# python3 VizDoom/VizDoom_src/inference/inference_vizdoom.py --model_mode 'RATE' --ckpt_name 'nmt15_arch_mode_TrXL_RATE_RUN_1' --ckpt_folder 'RATE_speep' --arch_mode 'TrXL' --ckpt_chooser 0

def create_args():
    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('--model_mode', type=str, default='RATE', help='Description of model_name argument')
    parser.add_argument('--ckpt_name', type=str, default='checkpoint_name', help='Description of name argument')
    parser.add_argument('--ckpt_folder', type=str, default='', help='0 if last else int')
    parser.add_argument('--arch_mode', type=str, default='TrXL', help='Description of model_name argument')
    parser.add_argument('--ckpt_chooser', type=int, default=0, help='0 if last else int')

    return parser

with open("VizDoom/VizDoom_src/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == '__main__':
    get_intro2()

    args = create_args().parse_args()

    model_mode = args.model_mode
    ckpt_name = args.ckpt_name
    ckpt_folder = args.ckpt_folder
    ckpt_chooser = args.ckpt_chooser
    arch_mode = args.arch_mode

    episode_timeout = config["online_inference_config"]["episode_timeout"]
    use_argmax = config["online_inference_config"]["use_argmax"]
    
    MEAN = torch.tensor([13.6313, 19.6772, 14.7505])#.to(device)
    STD  = torch.tensor([16.7388, 20.3475, 10.3455])#.to(device)

    config["model_mode"] = model_mode
    config["arctitecture_mode"] = arch_mode
    config["training_config"]["sections"] = 3

    """ ARCHITECTURE MODE """
    if config["arctitecture_mode"] == "TrXL":
        config["model_config"]["use_gate"] = False
        config["model_config"]["use_stable_version"] = False
    elif config["arctitecture_mode"] == "TrXL-I":
        config["model_config"]["use_gate"] = False
        config["model_config"]["use_stable_version"] = True
    elif config["arctitecture_mode"] == "GTrXL":
        config["model_config"]["use_gate"] = True
        config["model_config"]["use_stable_version"] = True     

    print(f"Selected Architecture: {config['arctitecture_mode']}")  

    """ MODEL MODE """
    if config["model_mode"] == "RATE": 
        config["model_config"]["mem_len"] = 2 ########################### 2 FOR DTXL 0
        config["model_config"]["mem_at_end"] = True ########################### True FOR DTXL False
    elif config["model_mode"] == "DT":
        config["model_config"]["mem_len"] = 0 ########################### 2 FOR DTXL 0
        config["model_config"]["mem_at_end"] = False ########################### True FOR DTXL False
        config["model_config"]["num_mem_tokens"] = 0
    elif config["model_mode"] == "DTXL":
        config["model_config"]["mem_len"] = 2
        config["model_config"]["mem_at_end"] = False
        config["model_config"]["num_mem_tokens"] = 0
    elif config["model_mode"] == "RATEM":
        config["model_config"]["mem_len"] = 0
        config["model_config"]["mem_at_end"] = True

    print(f"Selected Model: {config['model_mode']}")

    model = mem_transformer_v2_GTrXL.MemTransformerLM(**config["model_config"])

    folder_name = f'VizDoom/VizDoom_checkpoints/{ckpt_folder}/{ckpt_name}/'

    files = os.listdir(folder_name)
    files = [f for f in files if f.endswith('_KTD.pth') and '_' in f]
    if files[0].split('_')[1] != 'save':
        files = sorted(files, key=lambda x: int(x.split('_')[1]))
    last_file = files[-1]

    if ckpt_chooser == 0:
        ckpt_num = last_file
        ckpt_path = f'VizDoom/VizDoom_checkpoints/{ckpt_folder}/{ckpt_name}/{ckpt_num}'
        print(f"Ckeckpoint: {ckpt_name}/{ckpt_num}")
    else:
        ckpt_num = ckpt_chooser
        ckpt_path = f'VizDoom/VizDoom_checkpoints/{ckpt_folder}/{ckpt_name}/_{ckpt_num}_KTD.pth'
        print(f"Ckeckpoint: {ckpt_name}/_{ckpt_num}_KTD.pth")

    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    _ = model.eval()

    # for ret in [1.01, 3.9, 11.1, 56.5]:
    for ret in [config["online_inference_config"]["desired_return"]]:
        print("TARGET RETURN:", ret)
        goods, bads = 0, 0
        pbar = tqdm(range(len(seeds_list)))
        returns = []
        ts = []
        for i in pbar:
            episode_return, act_list, t, _, _ = get_returns_VizDoom(model=model, ret=ret, seed=seeds_list[i], 
                                                                    episode_timeout=episode_timeout, 
                                                                    context_length=config["training_config"]["context_length"], 
                                                                    device=device, act_dim=config["model_config"]["ACTION_DIM"], 
                                                                    config=config,
                                                                    mean=MEAN,
                                                                    std=STD,
                                                                    use_argmax=use_argmax, create_video=False)
            returns.append(episode_return)
            ts.append(t)
            pbar.set_description(f"Time: {t}, Return: {episode_return:.2f}")
            
        print(f"Mean reward: {np.mean(returns):.2f}")
        print(f"STD reward: {np.std(returns):.2f}")
        print(f"Mean T: {np.mean(ts):.2f}")
        print(f"STD T: {np.std(ts):.2f}")