import os
import datetime
import wandb
import argparse
import yaml
from torch.utils.data import DataLoader

import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from VizDoom.VizDoom_src.train import train
from TMaze_new.TMaze_new_src.utils import set_seed, get_intro_vizdoom
from VizDoom.VizDoom_src.utils import get_dataset, batch_mean_and_std
from MinigridMemory.MinigridMemory_src.utils import MinigridMemoryIterDataset

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"  
os.environ["OMP_NUM_THREADS"] = "1" 

with open("wandb_config.yaml") as f:
    wandb_config = yaml.load(f, Loader=yaml.FullLoader)
os.environ['WANDB_API_KEY'] = wandb_config['wandb_api']


with open("MinigridMemory/MinigridMemory_src/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


# python3 MinigridMemory/MinigridMemory_src/train_minigridmemory.py --model_mode 'RATE' --arch_mode 'TrXL' --ckpt_folder 'RATE' --text 'RATE'
    


def create_args():
    parser = argparse.ArgumentParser(description='RATE MinigridMemory trainer') 

    parser.add_argument('--model_mode',     type=str, default='RATE',  help='Model training mode. Available variants: "DT, DTXL, RATE (Ours), RATEM (RMT)"')    
    parser.add_argument('--arch_mode',      type=str, default='TrXL',  help='Model architecture mode. Available variants: "TrXL", "TrXL-I", "GTrXL"')
    parser.add_argument('--start_seed',     type=int, default=1,       help='Start seed')
    parser.add_argument('--end_seed',       type=int, default=3,       help='End seed')
    parser.add_argument('--ckpt_folder',    type=str, default='ckpt',  help='Checkpoints directory')
    parser.add_argument('--text',           type=str, default='',      help='Short text description of rouns group')

    return parser

if __name__ == '__main__':
    get_intro_vizdoom()
    
    args = create_args().parse_args()
    #================================================== DATA LOADING =============================================================#

    model_mode = args.model_mode
    start_seed = args.start_seed
    end_seed = args.end_seed
    arch_mode = args.arch_mode
    ckpt_folder = args.ckpt_folder
    TEXT_DESCRIPTION = args.text

    config["model_mode"] = model_mode
    config["arctitecture_mode"] = arch_mode

    for RUN in range(start_seed, end_seed+1):
        set_seed(RUN)
        print(f"Random seed set as {RUN}") 

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

        max_segments = config["training_config"]["sections"]

        """ MODEL MODE """
        if config["model_mode"] == "RATE": 
            config["model_config"]["mem_len"] = 2
            config["model_config"]["mem_at_end"] = True
            max_length = config["training_config"]["sections"]*config["training_config"]["context_length"]
        

        elif config["model_mode"] == "DT":
            config["model_config"]["mem_len"] = 0
            config["model_config"]["mem_at_end"] = False
            config["model_config"]["num_mem_tokens"] = 0
            config["training_config"]["context_length"] = config["training_config"]["context_length"] * config["training_config"]["sections"]
            config["training_config"]["sections"] = 1
            max_length = config["training_config"]["context_length"]

        elif config["model_mode"] == "DTXL":
            config["model_config"]["mem_len"] = 2
            config["model_config"]["mem_at_end"] = False
            config["model_config"]["num_mem_tokens"] = 0 
            config["training_config"]["context_length"] = config["training_config"]["context_length"] * config["training_config"]["sections"]
            config["training_config"]["sections"] = 1
            max_length = config["training_config"]["context_length"]

        elif config["model_mode"] == "RATEM":
            config["model_config"]["mem_len"] = 0
            config["model_config"]["mem_at_end"] = True
            max_length = config["training_config"]["sections"]*config["training_config"]["context_length"]

        elif config["model_mode"] == "RATE_wo_nmt":
            print("Custom Mode!!! RATE wo nmt")
            config["model_config"]["mem_len"] = 2
            config["model_config"]["mem_at_end"] = False
            config["model_config"]["num_mem_tokens"] = 0
            max_length = config["training_config"]["sections"]*config["training_config"]["context_length"]
        

        print(f"Selected Model: {config['model_mode']}, max length: {max_length}")  

        mini_text = f"arch_mode_{config['arctitecture_mode']}"
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S").replace('-', '_')
        group = f'{TEXT_DESCRIPTION}_{mini_text}_{config["model_mode"]}'
        name = f'{TEXT_DESCRIPTION}_{mini_text}_{config["model_mode"]}_RUN_{RUN}_{date_time}'
        current_dir = os.getcwd()
        current_folder = os.path.basename(current_dir)
        ckpt_path = f'../{current_folder}/MinigridMemory/MinigridMemory_checkpoints/{ckpt_folder}/{name}/'
        isExist = os.path.exists(ckpt_path)
        if not isExist:
            os.makedirs(ckpt_path)

        if config["wandb_config"]["wwandb"]:
            run = wandb.init(project=config['wandb_config']['project_name'], name=name, group=group, config=config, save_code=True, reinit=True)

        #================================================== DATALOADERS CREATION ======================================================#
        
        # * IF USE ITER DATASET (3K TRAJECTORIES)
        path_to_splitted_dataset = 'MinigridMemory/MinigridMemory_data/'
        train_dataset = MinigridMemoryIterDataset(path_to_splitted_dataset, 
                                         gamma=config["data_config"]["gamma"], 
                                         max_length=max_length, 
                                         normalize=config["data_config"]["normalize"])
        
        train_dataloader = DataLoader(train_dataset, 
                                     batch_size=config["training_config"]["batch_size"], 
                                     shuffle=True, 
                                     num_workers=8)

        print(f"Train: {len(train_dataloader) * config['training_config']['batch_size']} trajectories (first {max_length} steps)")

        if config["data_config"]["normalize"] == 0:
            type_norm = 'Without normalization'
        elif config["data_config"]["normalize"] == 1:
            type_norm = '/255.'
        elif config["data_config"]["normalize"] == 2:
            type_norm = 'Standard scaling'

        print(f'Normalization mode: {config["data_config"]["normalize"]}: {type_norm}')
        if config["data_config"]["normalize"] == 2:
            mean, std = batch_mean_and_std(train_dataloader)
            print("mean and std:", mean, std)
        else:
            mean, std = None, None
        #==============================================================================================================================#
        wandb_step = 0
        model = train(ckpt_path, config, train_dataloader, mean, std, max_segments)
                
        if config["wandb_config"]["wwandb"]:
            run.finish()