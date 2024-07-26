import sys
sys.path.insert(0, '../')
sys.path.insert(0, './')
import os
os.environ["LD_LIBRARY_PATH"]="/.mujoco/mujoco210/bin"
os.environ["LD_LIBRARY_PATH"]="/home/jovyan/.mujoco/mujoco210/bin"
sys.path.append("/home/jovyan/.mujoco/mujoco210/bin")
sys.path.append("/home/jovyan/.mujoco/mujoco210")
sys.path.append("/home/jovyan/")

import gym
import numpy as np
import torch
import wandb
from RATE_GTrXL import mem_transformer_v2_GTrXL
from tqdm import tqdm

import argparse
import pickle
import os
import glob
from colabgymrender.recorder import Recorder

import d4rl
import gym
import mujoco_py

import yaml
with open("wandb_config.yaml") as f:
    wandb_config = yaml.load(f, Loader=yaml.FullLoader)

os.environ['WANDB_API_KEY'] = wandb_config['wandb_api']

"""
python3 MuJoCo/train_rate_mujoco_ca.py --env_id 0 --number_of_segments 3 --segment_length 20 --num_mem_tokens 15 --n_head_ca 2 --seed 123
"""


from utils.eval_functions import get_batch, get_returns

ENVS = [['halfcheetah-medium-v2','checkpoints_Cheetah_Med',17,6,1000,6000], # ? 0
        ['halfcheetah-medium-replay-v2','checkpoints_Cheetah_Med_Repl',17,6,1000,6000], # ? 1
        ['halfcheetah-expert-v2','checkpoints_Cheetah_Expert',17,6,1000,6000], # ! 2
       
        ['walker2d-medium-v2','checkpoints_Walker_Med',17,6,1000,5000], # ? 3
        ['walker2d-medium-replay-v2','checkpoints_Walker_Med_Repl',17,6,1000,5000], # ? 4
        ['walker2d-expert-v2','checkpoints_Walker_Expert',17,6,1000,5000], # ! 5
        
        ['hopper-medium-v2','checkpoints_Hopper_Med',11,3,1000,3600], # ? 6
        ['hopper-medium-replay-v2','checkpoints_Hopper_Med_Repl',11,3,1000,3600], # ? 7
        ['hopper-expert-v2','checkpoints_Hopper_Expert',11,3,1000,3600], # ! 8 
        
        ['halfcheetah-medium-expert-v2','checkpoints_Cheetah_Med_Exp',17,6,1000,12000], # ? 9
        ['walker2d-medium-expert-v2','checkpoints_Walker_Med_Exp',17,6,1000,5000], # ? 10
        ['hopper-medium-expert-v2','checkpoints_Hopper_Med_Exp',11,3,1000,3600], # ? 11
       ]

INIT_ENV = True

class Args:
    def __init__(self):
        self.num_steps= 500000
        self.num_buffers=50
        self.batch_size=128
        self.nemb=128
        self.data_dir_prefix=wandb_config['mujoco']['data_dir_prefix']
        self.trajectories_per_buffer=10
        self.use_scheduler=True

        self.vocab_size = 100
        self.n_layer = 3
        self.n_head = 1
        self.d_model = 128
        self.d_head = 128
        self.d_inner = 128
        self.dropout = 0.2
        self.dropatt = 0.05
        self.MEM_LEN = 2         
        self.ext_len = 1
        self.tie_weight = False
        self.num_mem_tokens = 3*5
        self.mem_at_end = True   
        self.learning_rate = 1e-4
        self.weight_decay = 0.1
        self.betas = (0.9, 0.95)
        self.warmup_steps = 100
        self.grad_norm_clip = 1.0
        self.max_timestep = 10000
        self.context_length = 20
        self.sections = 3
        self.num_spets_per_epoch = 1000
        self.is_train = True
        self.wwandb = False
        
        self.batch_size = 2048*2
        self.num_eval_episodes = 10
        self.pct_traj = 1.
        self.max_ep_len = 10000
        self.scale = 1000.


class Agent:
    def __init__(self,args):
        self.args = args
        self.device = torch.device('cuda')

    def load_dataset(self):
        args = self.args

        gym_name = ENVS[args.env_id]
        
        game_gym_name = gym_name[0]
        env_name = game_gym_name.split('-')[0]
        dataset = '-'.join(game_gym_name.split('-')[1:-1])
        state_dim = gym_name[2]
        act_dim = gym_name[3]
        self.max_ep_len = gym_name[4]
        ret_global = gym_name[5]
        use_recorder = False
        
        if INIT_ENV:

            if args.env_id in [0, 1, 2, 9]:
                game_gym_name = 'HalfCheetah-v3'
            elif args.env_id in [3, 4, 5, 10]:
                game_gym_name = 'Walker2d-v3'
            elif args.env_id in [6, 7, 8, 11]:
                game_gym_name = 'Hopper-v3'

            self.env = gym.make(game_gym_name)
            directory = 'video'
            if use_recorder:
                self.env = Recorder(self.env, directory, fps=30)
    
        # load dataset
        dataset_path = f"{wandb_config['mujoco']['data_dir_prefix']}{env_name}-{dataset}-v2.pkl"
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)
        states, traj_lens, returns = [], [], []
        for path in self.trajectories:
            states.append(path['observations'])
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)  
        num_timesteps = sum(traj_lens)
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        self.state_mean_torch, self.state_std_torch = torch.from_numpy(self.state_mean).to(self.device), torch.from_numpy(self.state_std).to(self.device) 
        
        print('=' * 50)
        print(f'Starting new experiment: {env_name} {dataset}')
        print(f'{len(traj_lens)} self.trajectories, {num_timesteps} timesteps found')
        print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
        print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
        print('=' * 50)


        if args.env_id in [0, 1, 2, 9]:
            # * game_gym_name = 'HalfCheetah-v3'
            self.env.ref_max_score = 12135.0
            self.env.ref_min_score = -280.178953
        elif args.env_id in [3, 4, 5, 10]:
            # * game_gym_name = 'Walker2d-v3'
            self.env.ref_max_score = 4592.3
            self.env.ref_min_score = 1.629008
        elif args.env_id in [6, 7, 8, 11]:
            # * game_gym_name = 'Hopper-v3'
            self.env.ref_max_score = 3234.3
            self.env.ref_min_score = -20.272305
        
        """
        HOPPER_RANDOM_SCORE = -20.272305 # hop med -> 315.87| hop med-rep -> -1.44 | hop med-exp -> 315.87
        HALFCHEETAH_RANDOM_SCORE = -280.178953 # hc med -> -310.23 | hc med-rep -> -638.49 | hc med-exp -> -310.23
        WALKER_RANDOM_SCORE = 1.629008 # wal med -> -6.61 | wal med-rep -> -50.20 | wal med-exp -> -6.61
        ANT_RANDOM_SCORE = -325.6

        HOPPER_EXPERT_SCORE = 3234.3 # hop med -> 3222.36 | hop med-rep -> 3192.93 | hop med-exp -> 3759.08
        HALFCHEETAH_EXPERT_SCORE = 12135.0 # hc med -> 5309.38 | hc med-rep -> 4985.14 | hc med-exp -> 11252.04
        WALKER_EXPERT_SCORE = 4592.3 # wal med -> 4226.94 | wal med-rep -> 4132.00 | wal med-exp -> 5011.69
        ANT_EXPERT_SCORE = 3879.7
        """
    
        # only train on top pct_traj trajectories (for %BC experiment)
        num_timesteps = max(int(args.pct_traj*num_timesteps), 1)
        self.sorted_inds = np.argsort(returns)  # lowest to highest
        self.num_trajectories = 1
        timesteps = traj_lens[self.sorted_inds[-1]]
        ind = len(self.trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[self.sorted_inds[ind]] <= num_timesteps:
            timesteps += traj_lens[self.sorted_inds[ind]]
            self.num_trajectories += 1
            ind -= 1
        self.sorted_inds = self.sorted_inds[-self.num_trajectories:]
        # used to reweight sampling so we sample according to timesteps instead of trajectories
        self.p_sample = traj_lens[self.sorted_inds] / sum(traj_lens[self.sorted_inds])
        
        s, a, r, d, rtg, timesteps, mask = get_batch(self.trajectories,self.state_mean,self.state_std,self.num_trajectories,self.max_ep_len,args.state_dim,args.act_dim,self.sorted_inds,self.p_sample,self.device,batch_size=args.batch_size,max_len=args.EFFECTIVE_SIZE_BLOCKS)

    def load_model(self):
        args = self.args

        # self.model = mem_transformer_ca_mujoco.MemTransformerLM(STATE_DIM=args.state_dim,ACTION_DIM=args.act_dim,n_token=args.vocab_size, n_layer=args.n_layer, n_head=args.n_head, d_model=args.d_model,
        # d_head=args.d_head, d_inner=args.d_inner, dropout=args.dropout, dropatt=args.dropatt, mem_len=args.MEM_LEN, ext_len=args.ext_len, tie_weight=args.tie_weight, num_mem_tokens=args.num_mem_tokens, mem_at_end=args.mem_at_end)
        self.model = mem_transformer_v2_GTrXL.MemTransformerLM(
                                              STATE_DIM=args.state_dim,
                                              ACTION_DIM=args.act_dim,
                                              n_token=args.vocab_size, 
                                              n_layer=args.n_layer, 
                                              n_head=args.n_head, 
                                              n_head_ca=args.n_head_ca,
                                              d_model=args.d_model,
                                              d_head=args.d_head, 
                                              d_inner=args.d_inner, 
                                              dropout=args.dropout, 
                                              dropatt=args.dropatt,
                                              mem_len=args.MEM_LEN, 
                                              ext_len=args.ext_len, 
                                              tie_weight=args.tie_weight, 
                                              num_mem_tokens=args.num_mem_tokens, 
                                              mem_at_end=args.mem_at_end,
                                              mode='mujoco')

        torch.nn.init.xavier_uniform_(self.model.r_w_bias);
        torch.nn.init.xavier_uniform_(self.model.r_r_bias);
        self.model.train()
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate,  weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda steps: min((steps+1)/args.warmup_steps, 1))
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model

    def train(self):
        args = self.args
        print(f"model parameters: {sum(p.numel() for p in list(self.model.parameters()))}")
        
        INTERVAL = 200
        losses = []
        wandb_step  = 0
        for epoch in range(args.max_epochs):
            pbar = tqdm(enumerate(list(range(args.num_spets_per_epoch))), total=args.num_spets_per_epoch)
            for it, i in pbar:
                s, a, r, d, rtg, timesteps, mask = get_batch(self.trajectories,self.state_mean,self.state_std,self.num_trajectories,self.max_ep_len,args.state_dim,args.act_dim,self.sorted_inds,self.p_sample,self.device,batch_size=args.batch_size,max_len=args.EFFECTIVE_SIZE_BLOCKS)
                s = s[torch.all(mask==1,dim=1)]
                a = a[torch.all(mask==1,dim=1)]
                rtg = rtg[torch.all(mask==1,dim=1)]
                timesteps = timesteps[torch.all(mask==1,dim=1)]

                memory = None
                mem_tokens = None

                for block_part in range(args.EFFECTIVE_SIZE_BLOCKS//args.BLOCKS_CONTEXT):

                    from_idx = block_part*(args.BLOCKS_CONTEXT)
                    to_idx = (block_part+1)*(args.BLOCKS_CONTEXT)
                    x1 = s[:, from_idx:to_idx, :].to(self.device)
                    y1 = a[:, from_idx:to_idx, :].to(self.device)
                    r1 = rtg[:,:-1,:][:, from_idx:to_idx, :].to(self.device)
                    t1 = timesteps[:, from_idx:to_idx].to(self.device)

                    if mem_tokens is not None:
                        mem_tokens = mem_tokens.detach()
                    elif self.raw_model.mem_tokens is not None:
                        mem_tokens = self.raw_model.mem_tokens.repeat(1, x1.shape[0], 1)

                    with torch.set_grad_enabled(args.is_train):
                        if memory is not None:
                            res = self.model(x1, y1, r1, y1,t1, *memory, mem_tokens=mem_tokens)
                        else:
                            res = self.model(x1, y1, r1, y1,t1, mem_tokens=mem_tokens)
                        memory = res[0][2:]
                        mem_tokens = res[1]
                        logits, loss = res[0][0], res[0][1]
                        
                        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                        #print('Loss: ',loss.item())
                        if args.wandb:
                            wandb.log({"train_loss":  loss.item()})
                            
                    if args.is_train:
                        # backprop and update the parameters
                        self.model.zero_grad()
                        # loss.backward(retain_graph=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_norm_clip)
                        self.optimizer.step()
                        self.scheduler.step()
                        # decay the learning rate based on our progress
                        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                        # report progress
                        pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")     
                
                if it % INTERVAL == 0:
                    
                    if INIT_ENV:
                        rews = []
                        steps = 10
                        prompt_steps = 1
                        pbar = tqdm(enumerate(list(range(steps))), total=steps)
                        for it, i in pbar:
                            eval_return, self.env = get_returns(self.model, self.env,args.ret_global, args.context_length, args.state_dim, args.act_dim, self.state_mean_torch, self.state_std_torch, self.max_ep_len, self.device, prompt_steps=prompt_steps, memory_20step=True, without_memory=False)
                            rews.append(eval_return)

                        print(f"Prompt_steps: {prompt_steps}  Mean Reward: {np.mean(np.array(rews)):.5f}. STD Reward: {np.std(np.array(rews)):.5f}")        
                        if args.wandb:            
                            wandb.log({"episode_STD":  np.std(rews)})
                            wandb.log({"episode_MEAN":  np.mean(rews)})

                        rews = []
                        steps = 10
                        prompt_steps = args.context_length
                        pbar = tqdm(enumerate(list(range(steps))), total=steps)
                        for it, i in pbar:
                            eval_return, self.env = get_returns(self.model, self.env,args.ret_global, args.context_length, args.state_dim, args.act_dim, self.state_mean_torch, self.state_std_torch, self.max_ep_len, self.device, prompt_steps=prompt_steps, memory_20step=True, without_memory=False)
                            rews.append(eval_return)

                        print(f"Prompt_steps: {prompt_steps}  Mean Reward: {np.mean(np.array(rews)):.5f}. STD Reward: {np.std(np.array(rews)):.5f}")        
                        if args.wandb:            
                            wandb.log({"{}_episode_STD".format(args.context_length):  np.std(rews)})
                            wandb.log({"{}_episode_MEAN".format(args.context_length):  np.mean(rews)})

                    wandb_step += 1     
                    wandb.log({"ckpt": wandb_step})
                    torch.save(self.model.state_dict(), args.ckpt_path+'/'+str(args.seed)+'/'+str(wandb_step)+'.pth')
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=int)
    parser.add_argument("--number_of_segments", type=int)
    parser.add_argument("--segment_length", type=int)
    parser.add_argument("--num_mem_tokens", type=int, default=5)
    parser.add_argument("--n_head_ca", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    
    args_INPUT = parser.parse_args()
    print(args_INPUT.env_id, 
          args_INPUT.number_of_segments, 
          args_INPUT.segment_length, 
          args_INPUT.num_mem_tokens, 
          args_INPUT.n_head_ca,
          args_INPUT.seed)
    
    index = int(args_INPUT.env_id)
    
    gym_name = ENVS[index]

    args = Args()
    args.env_id = int(args_INPUT.env_id)
    args.context_length = int(args_INPUT.segment_length)
    args.sections = int(args_INPUT.number_of_segments)
    args.num_mem_tokens = int(args_INPUT.num_mem_tokens)
    args.n_head_ca = int(args_INPUT.n_head_ca)
    args.seed = int(args_INPUT.seed)
    
    args.block_size = 3*args.context_length
    args.EFFECTIVE_SIZE_BLOCKS = args.context_length * args.sections
    args.BLOCKS_CONTEXT = args.block_size//3
    args.init = 'uniform'
    args.init_range = 1
    args.init_std = 1
    args.game_gym_name = gym_name[0]
    args.env_name = args.game_gym_name.split('-')[0]
    args.dataset = '-'.join(args.game_gym_name.split('-')[1:-1])
    args.state_dim = gym_name[2]
    args.act_dim = gym_name[3]
    args.max_ep_len = gym_name[4]
    args.ret_global = gym_name[5]
    args.use_recorder = False
    args.ckpt_path = 'MuJoCo/MuJoCo_checkpoints/checkpoints_mujoco_rate_v3/{}_ns_{}_sl_{}_nt_{}_nhca_{}'.format('_'.join(args.game_gym_name.split('-')[:-1]),args_INPUT.number_of_segments,args_INPUT.segment_length,args_INPUT.num_mem_tokens, args_INPUT.n_head_ca)
    args.max_epochs = 10
    args.lr_decay = False
    args.wandb = True # !!! use_wandb

    isExist = os.path.exists(args.ckpt_path+'/'+str(args.seed))
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(args.ckpt_path+'/'+str(args.seed))

    files = glob.glob(args.ckpt_path+'/'+str(args.seed)+'/*')
    for f in files:
        os.remove(f) 

    if args.wandb:
        # idd = 'tzuzlv1i' #
        wandb.init(project="RATE-MuJoCo", name='v3_nmt15_'+str(args.env_id)+'_'+str(args.seed)+args.ckpt_path.split('/')[-1],
                   group='v3_nmt15_'+str(args.env_id),
                   save_code=True, resume="allow")


    agent = Agent(args)

    agent.load_dataset()
    agent.load_model()
    agent.train()


    print(args.ckpt_path)