import numpy as np
import torch
from TMaze_new.TMaze_new_src.utils.tmaze import TMazeClassicPassive
#from TMaze_new.TMaze_new_src.utils import set_seed

@torch.no_grad()
def sample(model, x, block_size, steps, sample=False, top_k=None, actions=None, rtgs=None, timestep=None, mem_tokens=1, saved_context=None):
    
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size else actions[:, -block_size:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size else rtgs[:, -block_size:] # crop context if needed
        
        if saved_context is not None:
            results = model(x_cond, actions, rtgs, None, timestep, *saved_context, mem_tokens=mem_tokens)
        else:
            results = model(x_cond, actions, rtgs, None, timestep, mem_tokens=mem_tokens) 
        logits = results[0][0][:,-1,:]
        mem_tokens = results[1]
        memory = results[0][2:]
        attn_map = model.attn_map
        
    return logits, mem_tokens, memory, attn_map

def get_returns_TMaze(model, ret, seed, episode_timeout, corridor_length, context_length, device, act_dim, config, create_video=False):
    
    scale = 1
    channels = 5
    max_ep_len = episode_timeout

    env = TMazeClassicPassive(episode_length=episode_timeout, corridor_length=corridor_length, penalty=0, seed=seed, goal_reward=ret)
    state = env.reset() # {x, y, hint}
    np.random.seed(seed)
    where_i = state[0]
    mem_state = state[2]
    mem_state2 = state

    state = np.concatenate((state, np.array([0]))) # {x, y, hint, flag}
    state = np.concatenate((state, np.array([np.random.randint(low=-1, high=1+1)]))) # {x, y, hint, flag, noise}

    if create_video == True:
        print("down, required act: 3" if mem_state == -1.0 else "up,  required act: 1")

    state = torch.tensor(state).reshape(1, 1, channels)
    out_states = []
    out_states.append(state.cpu().numpy())
    done = True
    Flag = 0
    frames = []
    HISTORY_LEN = context_length
    
    rews = []
    attentions = []
    states = state.to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(ret, device=device, dtype=torch.float32).reshape(1, 1)
    sim_states = []
    episode_return, episode_length = 0, 0

    mem_tokens = model.mem_tokens.repeat(1, 1, 1).detach() if model.mem_tokens is not None else None
    model.cache = mem_tokens.clone() if mem_tokens is not None else None
    
    saved_context = None
    segment = 0
    prompt_steps = 0# 5
    act = None
    act_list= []

    switcher = False
    saved_mem = None
    
    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
        act_new_segment = False
        if config["model_mode"] != 'DT' and config["model_mode"] != 'DTXL':
            if actions.shape[0] > HISTORY_LEN:
                segment+=1
                
                if prompt_steps==0:
                    actions = actions[-1:,:]
                    states = states[:, -1:, :]
                    target_return = target_return[:,-1:]
                    
                if t%(context_length)==0:
                    if create_video:
                        out = torch.norm(mem_tokens).item() if mem_tokens is not None else None
                        print(f't: {t}, NEW MEMORY: {out}')
                        
                    mem_tokens = new_mem
                    saved_context = new_notes
                
        else:
            if actions.shape[0] > HISTORY_LEN:
                segment+=1
                
                if prompt_steps==0:
                    actions = actions[1:,:]
                    states = states[:, 1:, :]
                    target_return = target_return[:,1:]
                    
                if t%(context_length)==0:
                    if create_video:
                        out = torch.norm(mem_tokens).item() if mem_tokens is not None else None
                        print(f't: {t}, NEW MEMORY: {out}')
                    mem_tokens = new_mem
                    saved_context = new_notes
                
        if t==0:
            act_to_pass = None
        else:
            act_to_pass = actions.unsqueeze(0)[:, 1:, :]
            if act_to_pass.shape[1] == 0:
                act_to_pass = None 
        
        sampled_action, new_mem, new_notes, attn_map = sample(model=model,  
                                                        x=states[:, :, 1:],
                                                        block_size=HISTORY_LEN, 
                                                        steps=1, 
                                                        sample=True, 
                                                        actions=act_to_pass, 
                                                        rtgs=target_return.unsqueeze(-1), 
                                                        mem_tokens=mem_tokens,
                                                        saved_context=saved_context)
        
        if t > 0 and t % (context_length-1) == 0 and switcher == False:
            switcher = True
            saved_mem = new_mem

        
        act = torch.argmax(torch.softmax(sampled_action, dim=-1).squeeze()).item()
        if create_video:
            print(t, "act", act, np.round(torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy(), 3), "\tstate:", int(where_i), states[:, -1:, :].detach().cpu().numpy())
        actions[-1, :] = act
        act_list.append(act)
        state, reward, done, info = env.step(act)

        if t < config["data_config"]["hint_steps"]-1:
            state[2] = mem_state2[2]
        
         # {x, y, hint} -> {x, y, hint, flag}
        if state[0] != env.corridor_length:
            state = np.concatenate((state, np.array([0])))
        else:
            if Flag != 1:
                state = np.concatenate((state, np.array([1])))
                Flag = 1
            else:
                state = np.concatenate((state, np.array([0])))
                
        # {x, y, hint, flag} -> {x, y, hint, flag, noise}
        state = np.concatenate((state, np.array([np.random.randint(low=-1, high=1+1)])))
        
        
        delta_t = env.time_step - env.corridor_length - 1
        where_i = state[0]
        state = state.reshape(1, 1, channels)
        out_states.append(state)
        
        rews.append(reward)
        cur_state = torch.from_numpy(state).to(device=device).float()
        states = torch.cat([states, cur_state], dim=1)
        rewards[-1] = reward
        pred_return = target_return[0,-1] - (reward/scale)
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        episode_return += reward
        episode_length += 1
        
        if (t+1) % (context_length) == 0 and t > 0:
            attentions.append(attn_map)
            
        if done:
            if create_video == True:
                if np.round(where_i, 4) == np.round(corridor_length, 4):
                    print("Junction achieved 😀 ✅✅✅")
                    print("Chosen act:", "up" if act == 1 else "down" if act == 3 else "wrong")
                    if mem_state == -1 and act == 3:
                        print("Correct choice 😀 ✅✅✅")
                    elif mem_state == 1 and act == 1:
                        print("Correct choice 😀 ✅✅✅")
                    else:
                        print("Wrong choice 😭 ⛔️⛔️⛔️")
                else:
                    print("Junction is not achieved 😭 ⛔️⛔️⛔️")
                
                print(f"{np.round(torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy(),2)}")
            break  
    if create_video == True:
        print(f"Final position: [{int(where_i)}, {int(np.round(states.squeeze()[-1].tolist()[0 if channels == 3 else 1]))}] / [{int(corridor_length)}, {int(mem_state)}]")
        print("\n")
        
    return reward, act_list, t, np.array(out_states).squeeze(), delta_t, attentions