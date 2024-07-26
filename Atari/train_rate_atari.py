import argparse
import os

import sys
sys.path.insert(0, '../')
sys.path.insert(0, './')
from RATE_GTrXL import mem_transformer_v2_GTrXL

import torch.nn as nn
import sys
import logging
import numpy as np
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
import random
import argparse
sys.path.insert(0, '../')
sys.path.insert(0, './')
from utils.create_dataset import create_dataset
import wandb
import sys
import cv2
from utils import gym_env
import random
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, 'YOUR PATH TO deep_rl_zoo (IF IT IS NOT INSTALLED)')
sys.path.insert(0, '../')

import yaml
with open("wandb_config.yaml") as f:
    wandb_config = yaml.load(f, Loader=yaml.FullLoader)
os.environ['WANDB_API_KEY'] = wandb_config['wandb_api']


# python3 Atari/train_rate_atari.py --game Seaquest --mem_len 2 --num_mem_tokens 15 --seed 123



parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help='Your GPU idx')
parser.add_argument("--game",type=str, default='Breakout', help='The game used to train the model.')
parser.add_argument("--mem_len",type=int, default=2, help='How many contexts do we need to remember (see TransformerXL)')
parser.add_argument("--context_length",type=int, default=30, help='The length of a section')
parser.add_argument("--sections",type=int, default=3, help='How many steps to use memory')
parser.add_argument("--n_layer",type=int, default=6, help='Number of layers in the transformer')
parser.add_argument("--n_head",type=int, default=8, help='Number of heads in the transformer')
parser.add_argument("--d_model",type=int, default=128, help='Hidden dimension of the model')
parser.add_argument("--d_head",type=int, default=128, help='Dimension of the head')
parser.add_argument("--d_inner",type=int, default=128, help='Inner dimension')
parser.add_argument("--dropout",type=float, default=0.2, help='Dropout rate')
parser.add_argument("--dropatt",type=int, default=0.05, help='Dropout rate in attention')
parser.add_argument("--ext_len",type=int, default=1, help='External cache length')
parser.add_argument("--tie_weight",type=bool, default=False, help='Using of weigths tying')
parser.add_argument("--num_mem_tokens",type=int, default=3*5, help='The length of memory segment')
parser.add_argument("--mem_at_end",type=bool, default=False, help='Whether to place memory segment at the end of section')
parser.add_argument("--save_path", type=str, default="./Atari/Atari_checkpoints/", help="Checkpoint path to save")
parser.add_argument("--epochs", type=int, default=10, help='Number of epochs')
parser.add_argument("--n_head_ca", type=int, default=2, help='Number of MRV attention heads')

parser.add_argument("--seed", type=int, default=123, help='seed')

user_config = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(user_config.device)

MAP_TO_TARGET_REWARD = {
    'Breakout' : 180,
    'Qbert' : 14000,
    'Seaquest' : 1150,
    'Pong' : 20
}

class Args:
    def __init__(self):
        self.seed=user_config.seed # 123
        self.context_length=user_config.context_length
        self.epochs=user_config.epochs
        self.num_steps= 500000 # 500000
        self.num_buffers=50
        self.game=user_config.game
        self.batch_size=128
        self.nemb=128
        self.data_dir_prefix=wandb_config['atari']['data'] #'data/' #'../data/'
        self.trajectories_per_buffer=10
        self.use_scheduler=True
        self.ckpt_path = user_config.save_path+user_config.game+'_'+str(user_config.seed)+'_'+str(user_config.mem_len)+'_'+str(user_config.context_length)+'_'+str(user_config.sections)+'_'+str(user_config.num_mem_tokens)+'/'
        
args = Args()
MEM_SEGMENTS = user_config.mem_len # how many contexts do we need to remember
CONTEXT_LEN = args.context_length
args = Args()
class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.vocab_size = max(actions) + 1
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        while True:
            done_idx = idx + block_size
            for i in self.done_idxs:
                if i > idx: # first done_idx greater than idx
                    done_idx = min(int(i), done_idx)
                    break
            if done_idx - block_size > 0:
                break
            idx = done_idx
            
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)
        
        return states, actions, rtgs, timesteps
    
obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer)

EFFECTIVE_SIZE_BLOCKS = CONTEXT_LEN * user_config.sections # we took 3 times larger context size

train_dataset = StateActionReturnDataset(obss, EFFECTIVE_SIZE_BLOCKS*3, actions, done_idxs, rtgs, timesteps)

args.init = 'uniform'
args.init_range = 1
args.init_std = 1

def take_fire_action(env):
    """Some games requires the agent to press 'FIRE' to start the game once loss a life."""
    assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
    s_t, _, _, _ = env.step(1)
    return s_t

def check_atari_env(env):
    """Check if is atari env and has fire action."""
    has_fire_action = False
    lives = 0
    try:
        lives = env.ale.lives()
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            has_fire_action = True
    except AttributeError:
        pass

    return has_fire_action, lives

class Env():
    def __init__(self, args):
        self.device = args.device
        self.window = args.history_length  # Number of frames to concatenate
        self.eval_env = gym_env.create_atari_environment(
                env_name=user_config.game,
                frame_height=84,
                frame_width=84,
                frame_skip=4,
                frame_stack=4,
                max_episode_steps=28000,
                seed=1,
                noop_max=30,
                terminal_on_life_loss=False,
                clip_reward=False,
            )
    def _get_state(self):
        raise NotImplemented("Its not necessary")
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self): # OK
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(4, 84, 84, device=self.device))

    def reset(self): # OK
        #self._reset_buffer()
        observation = self.eval_env.reset()
        should_fire, lives = check_atari_env(self.eval_env)
        if should_fire:
            observation = take_fire_action(self.eval_env)

        observation = self.eval_env.reset()
        should_fire, lives = check_atari_env(self.eval_env)
        if should_fire:
            observation = take_fire_action(self.eval_env)
        self.num_actions = self.eval_env.action_space.n
        return torch.tensor(observation).float().to(self.device)/255.0
        self.state_buffer.append(torch.tensor(observation).float().to(self.device)/255.0)
        return torch.vstack(list(self.state_buffer))

    def step(self, action): # OK
        # Repeat action 4 times, max pool over last 2 frames
        observation, reward, done, info = self.eval_env.step(action)
        #print(observation.shape)
        observation = torch.tensor(observation).float().to(self.device)/(255.0)
        #print(observation.shape)
        return observation, int(reward), done, info
        #self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        
        # Return state, reward, done
        return torch.vstack(list(self.state_buffer)), reward, done

    # Uses loss of life as terminal signal
    def train(self): # OK
        self.training = True

    # Uses standard terminal signal
    def eval(self): # OK
        self.training = False

    def action_space(self): # OK
        return self.num_actions#len(self.actions)

    def render(self):
        raise NotImplemented("Its not necessary")
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        raise NotImplemented("Its not necessary")
        cv2.destroyAllWindows()


        
logger = logging.getLogger(__name__)
os.environ['WANDB_MODE'] = 'online'
# os.environ['WANDB_API_KEY'] = 'wandb_key'

wandb.init(project="RATE-Atari", name=str(user_config), save_code=True, resume="allow", group=args.game)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, block_size, steps, temperature=1.0, sample=False, top_k=None, actions=None, rtgs=None, timesteps=None, mem_tokens=1, saved_context=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size else actions[:, -block_size:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size else rtgs[:, -block_size:] # crop context if needed
        # print('val', mem_tokens.shape if mem_tokens is not None else 'None')
        if saved_context is not None:
            results = model(
                x_cond, actions, rtgs, None, None, *saved_context, mem_tokens=mem_tokens # Timesteps = None
            )
        else:
            results = model(
                x_cond, actions, rtgs, None, None, mem_tokens=mem_tokens)
        logits = results[0][0]
        logits = logits[:, -1, :] / temperature
        mem_tokens = results[1]
        memory = results[0][2:]
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        x = ix

    return x, mem_tokens, memory

import numpy as np
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)
            
            
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)
    
class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 128
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        
        isExist = os.path.exists(self.config.ckpt_path)
        if not isExist:
            os.makedirs(self.config.ckpt_path)

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, save_checkpoint, epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)        
        torch.save(raw_model.state_dict(), self.config.ckpt_path+str(epoch)+'_'+str(save_checkpoint)+'.ckpt')

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=config.betas) 

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            
            if epoch_num < 2:
                INTERVAL=150
            elif epoch_num < 4:
                INTERVAL = 300
            else:
                INTERVAL = 450
                
            BLOCKS_CONTEXT = self.config.block_size//3
            
            for it, (x, y, r, t) in pbar:

                # place data on the correct device
                memory = None
                mem_tokens=None
                for block_part in range(EFFECTIVE_SIZE_BLOCKS//BLOCKS_CONTEXT):
                    
                    from_idx = block_part*(BLOCKS_CONTEXT)
                    to_idx = (block_part+1)*(BLOCKS_CONTEXT)
                    x1 = x[:, from_idx:to_idx, :].to(self.device)
                    y1 = y[:, from_idx:to_idx, :].to(dtype=torch.float32, device=self.device)
                    r1 = r[:, from_idx:to_idx, :].to(self.device)
                    t1 = t.to(self.device)
                    
                    if mem_tokens is not None:
                        mem_tokens = mem_tokens.detach()
                    elif raw_model.mem_tokens is not None:
                        mem_tokens = raw_model.mem_tokens.repeat(1, x1.shape[0], 1)
                            
                    with torch.set_grad_enabled(is_train):
                        if memory is not None:
                            res = model(x1, y1, r1, y1, None, *memory, mem_tokens=mem_tokens) # timesteps = None
                        else:
                            res = model(x1, y1, r1, y1, None, mem_tokens=mem_tokens)
                        memory = res[0][2:]
                        mem_tokens = res[1]
                        logits, loss = res[0][0], res[0][1]
                        
                        mem_tokens = res[1]
                        
                        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                    if is_train:

                        # backprop and update the parameters
                        model.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()

                        # decay the learning rate based on our progress
                        if config.lr_decay:
                            self.tokens += (y1 >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                            if self.tokens < config.warmup_tokens:
                                # linear warmup
                                lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                            else:
                                # cosine learning rate decay
                                progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                            lr = config.learning_rate * lr_mult
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr
                        else:
                            lr = config.learning_rate

                        # report progress
                        pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
                        wandb.log({"train_loss":  loss.item(), "lr": lr})
                
                if (it % INTERVAL == 0 and it > 0):
                    eval_return = self.get_returns(MAP_TO_TARGET_REWARD[user_config.game])
                    self.save_checkpoint(it, epoch_num)
                    wandb.log({'it': it})
                
            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                wandb.log({"test_loss": test_loss})
                return test_loss
            
        best_return = -float('inf')

        self.tokens = 0 # counter used for learning rate decay

        for epoch in range(config.max_epochs):

            run_epoch('train', epoch_num=epoch)
            wandb.log({'epoch': epoch})
            # self.save_checkpoint(epoch)

            
    def get_returns(self, ret, return_frames=False):
        self.model.train(False)
        args=Args(self.config.game.lower(), self.config.seed)
        env = Env(args)
        env.eval()

        T_rewards, T_Qs = [], []
        done = True
        frames = []

        VIDEO_GENERATED = False

        HISTORY_LEN = CONTEXT_LEN
        N = 5
        for i in range(N):
            state = env.reset()
            should_fire, lives = check_atari_env(env.eval_env) # Added
            if should_fire: # Added
                state = torch.tensor(take_fire_action(env.eval_env)).to(args.device) # Added
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            if user_config.num_mem_tokens > 0:
                mem_tokens = model.mem_tokens.repeat(1, 1, 1).detach()
            else:
                mem_tokens = None
            saved_context = None
            
            sampled_action, _, _ = sample(
                model=self.model.module,
                x=state,
                block_size=HISTORY_LEN,
                steps=1,
                temperature=1.0,
                sample=True,
                actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device),
                mem_tokens=mem_tokens)
                
            j = 0
            all_states = state
            video_wandb = state
            actions = []
            cur_frames = []
            best_score = 0
            
            
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                    should_fire, lives = check_atari_env(env.eval_env) # Added
                    if should_fire: # Added
                        state = torch.tensor(take_fire_action(env.eval_env)).to(args.device) # Added
                    
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                state, reward, done, info = env.step(action)
                reward_sum += reward
                j += 1
                
                if should_fire and not done and lives != info['lives']:
                    lives = info['lives']
                    state = torch.tensor(take_fire_action(env.eval_env)).to(args.device)

                if done:
                    T_rewards.append(reward_sum)
                    break
                
                state = state.unsqueeze(0).unsqueeze(0).to(self.device)
                cur_frames.append(state[0, :, :].cpu().numpy())
                
                all_states = torch.cat([all_states, state], dim=0)
                video_wandb = torch.cat([video_wandb, state], dim=0)
                rtgs += [rtgs[-1] - reward]
                if len(actions) > HISTORY_LEN:
                    actions = actions[-1:]
                    all_states = all_states[-1:, :, :, :]
                    rtgs = rtgs[-1:]
                    mem_tokens = new_mem
                    saved_context = new_notes

                # all_states = all_states.squeeze(1) if len(all_states.shape) > 4 else all_states
                # print('state', all_states.shape)
                sampled_action, new_mem, new_notes = sample(model=self.model.module,  x=all_states.unsqueeze(0), block_size=HISTORY_LEN, steps=1, temperature=1.0, sample=True, 
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)), mem_tokens=mem_tokens, saved_context=saved_context)
            if T_rewards[-1] > best_score:
                best_score = T_rewards[-1]
                frames = cur_frames 

            if not VIDEO_GENERATED:
                #frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
                logged_video = video_wandb.squeeze().cpu().numpy()*255
                logged_video = logged_video.astype(np.uint8)
                # print(logged_video.shape)
                wandb.log({"video": wandb.Video(logged_video, fps=24, format="mp4")})
                VIDEO_GENERATED = True

        eval_return = sum(T_rewards)/float(N)
        wandb.log({"target_return":  ret, "eval_return":np.mean(T_rewards), "eval_std": np.std(T_rewards)})
        self.model.train(True)
        if return_frames:
            return eval_return, frames
        return eval_return


class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4
        self.ckpt_path = user_config.save_path
        
args.init = 'uniform'
args.init_range = 1
args.init_std = 1

MEM_LEN=MEM_SEGMENTS*CONTEXT_LEN*3

model = mem_transformer_v2_GTrXL.MemTransformerLM(
                                              STATE_DIM=None,
                                              ACTION_DIM=None,
                                              n_token=train_dataset.vocab_size, 
                                              n_layer=user_config.n_layer, 
                                              n_head=user_config.n_head, 
                                              n_head_ca=user_config.n_head_ca,
                                              d_model=user_config.d_model,
                                              d_head=user_config.d_head,
                                              d_inner=user_config.d_inner, 
                                              dropout=user_config.dropout, 
                                              dropatt=user_config.dropatt,
                                              mem_len=MEM_LEN, 
                                              ext_len=user_config.ext_len, 
                                              tie_weight=user_config.tie_weight, 
                                              num_mem_tokens=user_config.num_mem_tokens, 
                                              mem_at_end=user_config.mem_at_end,
                                              mode='atari')

print(f"model parameters: {sum(p.numel() for p in list(model.parameters()))}")

torch.nn.init.xavier_uniform(model.r_w_bias);
torch.nn.init.xavier_uniform(model.r_r_bias);


epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4, ckpt_path=args.ckpt_path,
                      lr_decay=args.use_scheduler, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                      num_workers=4, seed=args.seed, model_type=None, game=args.game, max_timestep=max(timesteps), block_size=3*args.context_length)

trainer = Trainer(model, train_dataset, None, tconf)
trainer.get_returns(MAP_TO_TARGET_REWARD[user_config.game])
trainer.train()