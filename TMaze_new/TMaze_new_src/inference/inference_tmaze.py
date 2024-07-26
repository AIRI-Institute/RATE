import torch
import numpy as np
from tqdm import tqdm
import torch
import argparse
import pandas as pd
import re
import yaml

import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from RATE_GTrXL import mem_transformer_v2_GTrXL
from TMaze_new.TMaze_new_src.inference.val_tmaze import get_returns_TMaze
from TMaze_new.TMaze_new_src.utils import seeds_list, get_intro2

with open("TMaze_new/TMaze_new_src/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
def create_args():
    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('--model_mode', type=str, default='RATE', help='Description of model_name argument')
    parser.add_argument('--max_n_final', type=int, default=3, help='Description of max_n_final argument')
    parser.add_argument('--ckpt_name', type=str, default='checkpoint_name', help='Description of name argument')
    parser.add_argument('--ckpt_chooser', type=int, default=0, help='0 if last else int')
    parser.add_argument('--ckpt_folder', type=str, default='', help='0 if last else int')
    parser.add_argument('--arch_mode', type=str, default='TrXL', help='Description of model_name argument')

    return parser

if __name__ == '__main__':
    get_intro2()
    
    args = create_args().parse_args()

    model_mode = args.model_mode
    min_n_final = 1
    max_n_final = args.max_n_final
    ckpt_name = args.ckpt_name
    ckpt_chooser = args.ckpt_chooser
    ckpt_folder = args.ckpt_folder
    arch_mode = args.arch_mode

    config["training_config"]["max_segments"] = max_n_final
    config["training_config"]["sections"] = max_n_final
    config["model_mode"] = model_mode
    config["arctitecture_mode"] = arch_mode

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
        config["model_config"]["mem_len"] = 2
        config["model_config"]["mem_at_end"] = True

    elif config["model_mode"] == "DT":
        config["model_config"]["mem_len"] = 0
        config["model_config"]["mem_at_end"] = False
        config["model_config"]["num_mem_tokens"] = 0
        config["training_config"]["context_length"] = config["training_config"]["context_length"] * config["training_config"]["sections"]
        config["training_config"]["sections"] = 1

    elif config["model_mode"] == "DTXL":
        config["model_config"]["mem_len"] = 2
        config["model_config"]["mem_at_end"] = False
        config["model_config"]["num_mem_tokens"] = 0
        config["training_config"]["context_length"] = config["training_config"]["context_length"] * config["training_config"]["sections"]
        config["training_config"]["sections"] = 1

    elif config["model_mode"] == "RATEM":
        config["model_config"]["mem_len"] = 0
        config["model_config"]["mem_at_end"] = True

    elif config["model_mode"] == "RATE_wo_nmt":
        print("Custom Mode!!! RATE wo nmt")
        config["model_config"]["mem_len"] = 2
        config["model_config"]["mem_at_end"] = False
        config["model_config"]["num_mem_tokens"] = 0

    print(f"Selected Model: {config['model_mode']}, Context length: {config['training_config']['context_length']}") 

    model = mem_transformer_v2_GTrXL.MemTransformerLM(**config["model_config"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_dir = os.getcwd()
    current_folder = os.path.basename(current_dir)
    name = ckpt_name
    
    ckpt_path = f'../{current_folder}/TMaze_new/TMaze_new_checkpoints/{ckpt_folder}/{name}/'
    folder_name = f'TMaze_new_inference_{ckpt_folder}'

    folder_path = ckpt_path

    files = os.listdir(folder_path)
    files = [f for f in files if f.endswith('_KTD.pth') and '_' in f]
    last_file = files[-1]

    if ckpt_chooser == 0:
        ckpt_num = last_file
    else:
        ckpt_num = ckpt_chooser

    model.load_state_dict(torch.load(ckpt_path + last_file, map_location=device), strict=True)
    model.to(device)
    print(f"model parameters: {sum(p.numel() for p in list(model.parameters()))}")
    _ = model.eval()    

    #print("NAME:", name)
    print("Checkpoint:", ckpt_num)
    print("NAME:", name)

    segmentss, means, stds = [], [], []
    for segments in [1, 2, 3, 5, 7, 9, 12, 16, 20, 25, 30]:
        rets = []
        for seed in tqdm(seeds_list):
            episode_timeout = 30*segments
            corridor_length = 30*segments - 2
            create_video = False
    
            episode_return, act_list, t, states, _, attn_map = get_returns_TMaze(model=model, ret=config["data_config"]["desired_reward"], seed=seed, 
                                                                                episode_timeout=episode_timeout, corridor_length=corridor_length, 
                                                                                context_length=config["training_config"]["context_length"], 
                                                                                device=device, act_dim=config["model_config"]["ACTION_DIM"], 
                                                                                config=config, create_video=create_video)
            rets.append(int(episode_return == config["data_config"]["desired_reward"]))

        segmentss.append(segments)
        means.append(np.mean(rets))
        stds.append(np.std(rets))

        print("SEGMENTS", segments, np.mean(rets), np.std(rets), sep='\t')

    test = pd.read_csv('TMaze_new/TMaze_new_src/utils/table_empty.csv')
    match = re.search(r'RUN_(\d+)', name)
    if match:
        run_num = match.group(1)

    test.iloc[1, 0] = name
    test.iloc[1, 1] = run_num
    test.iloc[1, 2] = ckpt_num

    for i in range(len(means)):
        test.iloc[1, 3+i*2] = means[i]
        test.iloc[1, 4+i*2] = stds[i]

    string = name
    string_no_data = string.split('_')[:-6]
    new_string = ''
    for el in string_no_data:
        new_string += el + "_"

    new_string = new_string[:-1]

    new_string_no_run = new_string.split('_')[:-2]

    new_string2 = ''
    for el in new_string_no_run:
        new_string2 += el + "_"

    new_string2 = new_string2[:-1]

    save_path = f'../{current_folder}/TMaze_new/TMaze_new_inference/{folder_name}/{new_string2}/'
    isExist = os.path.exists(save_path)
    if not isExist:
        os.makedirs(save_path)

    test.to_csv(f'../{current_folder}/TMaze_new/TMaze_new_inference/{folder_name}/{new_string2}/{name}.csv', index=False)

