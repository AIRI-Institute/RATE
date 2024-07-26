import gym
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt

import argparse
import pickle
import random
import sys
import os
import glob
from colabgymrender.recorder import Recorder
import numpy as np
import torch


def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

K = 60
SCALE=1000.

def get_batch(trajectories, state_mean, state_std, num_trajectories, max_ep_len, state_dim, act_dim, sorted_inds, p_sample, device, batch_size=256, max_len=K):
    batch_inds = np.random.choice(
        np.arange(num_trajectories),
        size=batch_size,
        replace=True,
        p=p_sample,  # reweights so we sample according to timesteps
    )

    s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
    for i in range(batch_size):
        traj = trajectories[int(sorted_inds[batch_inds[i]])]
        si = random.randint(0, traj['rewards'].shape[0] - 1)

        # get sequences from dataset
        s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
        a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
        r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
        if 'terminals' in traj:
            d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
        else:
            d.append(traj['dones'][si:si + max_len].reshape(1, -1))
        timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
        rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
        if rtg[-1].shape[1] <= s[-1].shape[1]:
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

        # padding and state + reward normalization
        tlen = s[-1].shape[1]
        s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
        s[-1] = (s[-1] - state_mean) / state_std
        a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
        r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
        d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
        rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / SCALE
        timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
        mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

    s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
    d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

    return s, a, r, d, rtg, timesteps, mask

def eval_episodes(target_rew):
    def fn(model):
        returns, lengths = [], []
        for _ in range(num_eval_episodes):
            with torch.no_grad():
                if model_type == 'dt':
                    ret, length = evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=SCALE,
                        target_return=target_rew/SCALE,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
                else:
                    ret, length = evaluate_episode(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        target_return=target_rew/SCALE,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
            returns.append(ret)
            lengths.append(length)
        return {
            f'target_{target_rew}_return_mean': np.mean(returns),
            f'target_{target_rew}_return_std': np.std(returns),
            f'target_{target_rew}_length_mean': np.mean(lengths),
            f'target_{target_rew}_length_std': np.std(lengths),
        }
    return fn


@torch.no_grad()
def sample(model, x, block_size, steps, temperature=1.0, sample=False, top_k=None, actions=None, rtgs=None, timestep=None, mem_tokens=1, saved_context=None):
    
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size else actions[:, -block_size:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size else rtgs[:, -block_size:] # crop context if needed
        if saved_context is not None:
            results = model(x_cond, actions, rtgs,None, timestep, *saved_context, mem_tokens=mem_tokens)
        else:
            results = model(x_cond, actions, rtgs,None, timestep, mem_tokens=mem_tokens) 
            
        logits = results[0][0].detach()[:,-1,:]
        mem_tokens = results[1]
        memory = results[0][2:]
        
    return logits, mem_tokens, memory

def get_returns(model, env, ret, context_length, state_dim, act_dim, state_mean_torch, state_std_torch, max_ep_len, device, use_recorder=False, return_frames=False, prompt_steps=0, memory_20step=True, without_memory=False):

    ret = ret/SCALE


    done = True
    frames = []
    HISTORY_LEN = context_length
    state = env.reset(); rews = []
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(ret, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    sim_states = []
    episode_return, episode_length = 0, 0

    mem_tokens = model.mem_tokens.repeat(1, 1, 1).detach()
    saved_context = None
    segment = 0

    for t in range(max_ep_len):
        
        act_new_segment = False
        if actions.shape[0] > HISTORY_LEN -1:
            segment+=1
            
            if prompt_steps==0:
                actions = actions[-1:,:]
                states = states[-1:, :]
                target_return = torch.tensor(ret, device=device, dtype=torch.float32).reshape(1, 1)
                timesteps = timesteps[:, -1:]
                act_new_segment = True
            else:
                actions = actions[-prompt_steps:,:]
                states = states[-prompt_steps:, :]
                target_return = target_return[:,-prompt_steps:]
                timesteps = timesteps[:, -prompt_steps:]
                
            if not without_memory:
                if memory_20step:
                    if t%20==0 and t>5:
                        mem_tokens = new_mem
                        saved_context = new_notes
                else:
                    mem_tokens = new_mem
                    saved_context = new_notes
 
        if t==0 or act_new_segment:
            act_to_pass = None
        else:
            act_to_pass = actions.unsqueeze(0)[:,:,:] # :,1: for original
            if act_to_pass.shape[1]==0:
                act_to_pass = None       
                
        sampled_action, new_mem, new_notes = sample(model=model,  
                                                    x=(states.unsqueeze(0)- state_mean_torch) / state_std_torch,
                                                    block_size=HISTORY_LEN, 
                                                    steps=1, 
                                                    temperature=1.0, 
                                                    sample=True, 
                                                    actions=act_to_pass, 
                                                    rtgs=target_return.unsqueeze(-1), 
                                                    timestep=timesteps, 
                                                    mem_tokens=mem_tokens, 
                                                    saved_context=saved_context) 

        actions = torch.cat([actions, sampled_action])

        action = sampled_action.cpu().numpy()[0]
        state, reward, done, _ = env.step(action)
        rews.append(reward)
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim).float()
        states = torch.cat([states, cur_state], dim=0)
        rewards = torch.cat([rewards, torch.from_numpy(np.array([reward])).to(device)])
        pred_return = target_return[0,-1] - (reward/SCALE)
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat([timesteps,torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)
        episode_return += reward
        episode_length += 1

        if done:
            break  
    

    """
    HOPPER_RANDOM_SCORE = -20.272305
    HALFCHEETAH_RANDOM_SCORE = -280.178953
    WALKER_RANDOM_SCORE = 1.629008
    ANT_RANDOM_SCORE = -325.6

    HOPPER_EXPERT_SCORE = 3234.3
    HALFCHEETAH_EXPERT_SCORE = 12135.0
    WALKER_EXPERT_SCORE = 4592.3
    ANT_EXPERT_SCORE = 3879.7
    """
    # episode_return = env.get_normalized_score(episode_return)*100.
    episode_return = (episode_return - env.ref_min_score) / (env.ref_max_score - env.ref_min_score)
    # episode_return = env.get_normalized_score(episode_return)*100.
    episode_return = episode_return * 100.0
    
    #eval_return = sum(T_rewards)/float(N)
    # print("target return: %d, eval return: %d" % (ret, episode_return))
    
    return episode_return, env


