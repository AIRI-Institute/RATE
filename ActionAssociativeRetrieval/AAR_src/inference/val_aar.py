import numpy as np
import torch
from ActionAssociativeRetrieval.AAR_src.utils.aar_env import ActionAssociativeRetrieval

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

def get_returns_AAR(model, ret, seed, stay_number, context_length, device, act_dim, config, create_video=False):
    scale = 1.0

    env = ActionAssociativeRetrieval(stay_number=stay_number, seed=seed)
    state = env.reset() # {x, y, hint}
    np.random.seed(seed)

    state = torch.tensor(state).reshape(1, 1, 3)
    out_states = []
    out_states.append(state.cpu().numpy())
    done = True

    HISTORY_LEN = context_length
    
    rews = []
    attentions = []
    states = state.to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(ret, device=device, dtype=torch.float32).reshape(1, 1)
    episode_return, episode_length = 0, 0

    mem_tokens = model.mem_tokens.repeat(1, 1, 1).detach() if model.mem_tokens is not None else None
    model.cache = mem_tokens.clone() if mem_tokens is not None else None
    
    saved_context = None
    segment = 0
    prompt_steps = 0
    act = None
    act_list= []

    switcher = False
    

    for t in range(env.episode_timeout):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
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
                                                        x=states,
                                                        block_size=HISTORY_LEN, 
                                                        steps=1, 
                                                        sample=True, 
                                                        actions=act_to_pass, 
                                                        rtgs=target_return.unsqueeze(-1), 
                                                        mem_tokens=mem_tokens,
                                                        saved_context=saved_context)
        
        # !!!!!!!
        if t > 0 and t % (context_length-1) == 0 and switcher == False:
            switcher = True

        act = torch.argmax(torch.softmax(sampled_action, dim=-1).squeeze()).item()
        if create_video:
            print(t, "act", act, np.round(torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy(), 3), "\tstate:", int(where_i), states[:, -1:, :].detach().cpu().numpy())
        actions[-1, :] = act
        act_list.append(act)
        state, reward, done, info = env.step(act)        
        
        where_i = state[0]
        state = state.reshape(1, 1, 3)
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
                
                print(f"{np.round(torch.softmax(sampled_action, dim=-1).squeeze().detach().cpu().numpy(),2)}")
            break  
    if create_video == True:
        print("\n")

    delta_t = 0
        
    return reward, act_list, t, np.array(out_states).squeeze(), delta_t, attentions