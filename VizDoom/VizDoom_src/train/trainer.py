import numpy as np
import torch
import wandb
import comet_ml
from tqdm import tqdm
import math

from RATE_GTrXL import mem_transformer_v2_GTrXL

from VizDoom.VizDoom_src.utils import z_normalize, inverse_z_normalize
from VizDoom.VizDoom_src.inference.val_vizdoom import get_returns_VizDoom
from MemoryMaze.MemoryMaze_src.inference.val_mem_maze import get_returns_MemoryMaze 
from TMaze_new.TMaze_new_src.utils import seeds_list

from scipy.stats import sem

import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn

reds = [2, 3, 6, 8, 9, 10, 11, 14, 15, 16, 17, 18, 20, 21, 25, 26, 27, 28, 29, 31, 38, 40, 41, 42, 45,
        46, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60, 61, 63, 64, 67, 68, 70, 72, 73, 74, 77, 80, 82, 84, 
        86, 88, 89, 90, 91, 92, 97, 98, 99, 100, 101, 103, 106, 108, 109, 113, 115, 116, 117, 120, 
        123, 124, 125, 126, 127, 128, 129, 133, 134, 136, 139, 140, 142, 144, 145, 147, 148, 151, 152, 
        153, 154, 156, 157, 158, 159, 161, 164, 165, 170, 171, 173]

greens = [0, 1, 4, 5, 7, 12, 13, 19, 22, 23, 24, 30, 32, 33, 34, 35, 36, 37, 39, 43, 44, 47, 48, 56, 57,
          62, 65, 66, 69, 71, 75, 76, 78, 79, 81, 83, 85, 87, 93, 94, 95, 96, 102, 104, 105, 107, 110, 111, 
          112, 114, 118, 119, 121, 122, 130, 131, 132, 135, 137, 138, 141, 143, 146, 149, 150, 155, 160, 162, 
          163, 166, 167, 168, 169, 172, 175, 176, 177, 182, 183, 187, 190, 192, 193, 195, 199, 204, 206, 208, 
          209, 210, 212, 215, 216, 218, 219, 220, 221, 223, 224, 225]


def train(ckpt_path, config, train_dataloader, mean, std, max_segments, experiment):

    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    episode_timeout = config["online_inference_config"]["episode_timeout"]
    use_argmax = config["online_inference_config"]["use_argmax"]

    MEAN = mean
    STD = std

    model = mem_transformer_v2_GTrXL.MemTransformerLM(**config["model_config"])

    print(config)
    
    torch.nn.init.xavier_uniform_(model.r_w_bias);
    torch.nn.init.xavier_uniform_(model.r_r_bias);
    wandb_step  = 0

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config["training_config"]["learning_rate"], weight_decay=config["training_config"]["weight_decay"], 
                                  betas=(config["training_config"]["beta_1"], config["training_config"]["beta_2"]))

    raw_model = model.module if hasattr(model, "module") else model
        
    model.to(device)
    model.train()
    
    wwandb = config["wandb_config"]["wwandb"]
    wcomet = config['wandb_config']['wcomet']

    print(f"model parameters: {sum(p.numel() for p in list(model.parameters()))}")
    it_counter = 0
    switch = False

    EFFECTIVE_SIZE_BLOCKS = config["training_config"]["context_length"] * config["training_config"]["sections"]
    BLOCKS_CONTEXT = config["training_config"]["context_length"]
    
    pbar = tqdm(range(config["training_config"]["epochs"]))
    tokens_dict = {}
    tokens_dict[0] = None

    ckpt_dict = {}
    ckpt_dict[0] = None

    tokens = 0

    for epoch in pbar:
        train_imgs = []
        is_train = True
        model.train()
        for it, batch in enumerate(train_dataloader):
            s, a, rtg, d, timesteps, masks = batch

            if config["data_config"]["normalize"] == 2:
                _b, _l, _c, _h, _w = s.shape
                s = s.reshape(_b*_l, _c, _h, _w)
                s = z_normalize(s, MEAN, STD)
                s = s.reshape(_b, _l, _c, _h, _w)

            memory = None
            mem_tokens = None
            
            block_part_range = range(EFFECTIVE_SIZE_BLOCKS//BLOCKS_CONTEXT)
            if tokens_dict[0] is None:
                for block_part in block_part_range:
                    tokens_dict[block_part] = None
            if ckpt_dict[0] is None:
                for block_part in block_part_range:
                    ckpt_dict[block_part] = None
                
            for block_part in block_part_range:
                from_idx = block_part*(BLOCKS_CONTEXT)
                to_idx = (block_part+1)*(BLOCKS_CONTEXT)
                
                x1 = s[:, from_idx:to_idx, :].to(device)
                y1 = a[:, from_idx:to_idx, :].to(device).float()
                r1 = rtg[:,:,:][:, from_idx:to_idx, :].to(device).float() 
                t1 = timesteps[:, from_idx:to_idx].to(device)
                masks1 = masks[:, from_idx:to_idx].to(device)
                    
                model.flag = 1 if block_part == max(block_part_range) else 0
                if mem_tokens is not None:
                    mem_tokens = mem_tokens.detach()
                elif raw_model.mem_tokens is not None:
                    mem_tokens = raw_model.mem_tokens.repeat(1, r1.shape[0], 1)

                if config["model_config"]["num_mem_tokens"] > 0:
                    mean_tokens = torch.mean(mem_tokens, dim=1)[0]

                    if tokens_dict[block_part] is not None:
                        cos_sim = cos(mean_tokens, tokens_dict[block_part]).item()
                    else:
                        cos_sim = None

                with torch.set_grad_enabled(is_train):
                    optimizer.zero_grad()
                        
                    res = model(x1, y1, r1, y1, t1, *memory, mem_tokens=mem_tokens, masks=masks1) if memory is not None \
                    else model(x1, y1, r1, y1, t1, mem_tokens=mem_tokens, masks=masks1)
                    memory = res[0][2:]
                    logits, train_loss = res[0][0], res[0][1]
                    mem_tokens = res[1]
                
                    if wwandb:
                        wandb.log({"train_loss":  train_loss.item()})
                    elif wcomet:
                        experiment.log_metric("train_loss", train_loss.item(), step=it_counter)

                    if config["model_config"]["num_mem_tokens"] > 0:
                        if wwandb:
                            wandb.log({f"cos_1st_token_block_{block_part}": cos_sim})
                            tokens_dict[block_part] = mean_tokens
                        elif wcomet:
                            experiment.log_metric(f"cos_1st_token_block_{block_part}", cos_sim, step=it_counter)
                            tokens_dict[block_part] = mean_tokens

                        if ((epoch + 1) % int(config["training_config"]["ckpt_epoch"])) == 0 or epoch == config["training_config"]["epochs"] - 1 or (epoch + 1) == 1:
                            if it == len(train_dataloader)-1:
                                if ckpt_dict[block_part] is not None:
                                    cos_sim_ckpt= cos(mean_tokens, ckpt_dict[block_part]).item()
                                else:
                                    cos_sim_ckpt = None

                                if wwandb:
                                    wandb.log({f"ckpt_cos_1st_token_block_{block_part}": cos_sim_ckpt})
                                elif wcomet:
                                    experiment.log_metric(f"ckpt_cos_1st_token_block_{block_part}", cos_sim_ckpt, step=it_counter)
                            

                                ckpt_dict[block_part] = mean_tokens

                if is_train:
                    model.zero_grad()
                    train_loss.backward(retain_graph=True)
                    if config["training_config"]["grad_norm_clip"] != 'None':
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training_config"]["grad_norm_clip"])
                    optimizer.step()

                    # * decay the learning rate based on our progress
                    tokens += (y1 >= 0).sum()
                    if tokens < config["training_config"]["warmup_steps"]:
                        # linear warmup
                        lr_mult = float(tokens) / float(max(1, config["training_config"]["warmup_steps"]))
                    else:
                        # cosine learning rate decay
                        progress = float(tokens - config["training_config"]["warmup_steps"]) / float(max(1, config["training_config"]["final_tokens"] - config["training_config"]["warmup_steps"]))
                        lr_mult = max(config["training_config"]["lr_end_factor"], 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config["training_config"]["learning_rate"] * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr



                    lr = optimizer.state_dict()['param_groups'][0]['lr']
                    if wwandb:
                        wandb.log({"learning_rate": lr})
                    elif wcomet:
                        experiment.log_metric("learning_rate", lr, step=it_counter)

                it_counter += 1



            pbar.set_description(f"ep {epoch+1} it {it} tTotal {train_loss.item():.2f} lr {lr:e} tokens, M {(tokens/1e6):.2f}")

        if wwandb:
            wandb.log({"epochs": epoch+1})
        elif wcomet:
            experiment.log_metrics({"epochs": epoch+1}, step=it_counter)
        
        # ? Save
        if ((epoch + 1) % int(config["training_config"]["ckpt_epoch"])) == 0 or epoch == config["training_config"]["epochs"] - 1 or (epoch + 1) == 1:
            if config["training_config"]["online_inference"] == True:
                if config["model_config"]["mode"] == 'doom':
                    model.eval()
                    with torch.no_grad():
                        FRAME_SKIP = 2
                        def optimize_pillar(color, seeds, config, wwandb, wcomet):
                            for ret in [config["online_inference_config"]["desired_return_1"]]:
                                returns = []
                                ts = []
                                attn_map_received = False
                                for i in range(len(seeds)):
                                    episode_return, act_list, t, _, _, attn_map = get_returns_VizDoom(model=model, ret=ret, seed=seeds[i], 
                                                                                            episode_timeout=episode_timeout, 
                                                                                            context_length=config["training_config"]["context_length"], 
                                                                                            device=device, act_dim=config["model_config"]["ACTION_DIM"], 
                                                                                            config=config,
                                                                                            mean=MEAN,
                                                                                            std=STD,
                                                                                            use_argmax=config["online_inference_config"]["use_argmax"],
                                                                                            create_video=False)

                                    returns.append(episode_return)
                                    t *= FRAME_SKIP
                                    ts.append(t)

                                    pbar.set_description(f"Online inference {color} {ret}: [{i+1} / {len(seeds)}] Time: {t}, Return: {episode_return:.2f}")

                                returns_mean = np.mean(returns)
                                lifetime_mean = np.mean(ts)

                                if wwandb:
                                    wandb.log({f"LifeTimeMean_{color}_{ret}": lifetime_mean, 
                                            f"ReturnsMean_{color}_{ret}": returns_mean})
                                elif wcomet:
                                    experiment.log_metrics({f"LifeTimeMean_{color}_{ret}": lifetime_mean, 
                                                           f"ReturnsMean_{color}_{ret}": returns_mean}, step=it_counter)
    
                            return returns, ts

                        total_returns, total_ts = [], []
                        SKIP_RETURN = 4

                        # RED PILLAR
                        seeds_red = reds[::SKIP_RETURN]
                        red_returns, red_ts = optimize_pillar("red", seeds_red, config, wwandb, wcomet)
                        total_returns += red_returns
                        total_ts += red_ts

                        # GREEN PILLAR
                        seeds_green = greens[::SKIP_RETURN]
                        green_returns, green_ts = optimize_pillar("green", seeds_green, config, wwandb, wcomet)
                        total_returns += green_returns
                        total_ts += green_ts

                        total_returns = np.mean(total_returns)
                        total_ts = np.mean(total_ts)

                        if wwandb:
                            wandb.log({f"LifeTimeMean_{config['online_inference_config']['desired_return_1']}": total_ts, 
                                       f"ReturnsMean_{config['online_inference_config']['desired_return_1']}": total_returns})
                        elif wcomet:
                            experiment.log_metrics({f"LifeTimeMean_{config['online_inference_config']['desired_return_1']}": total_ts, 
                                                    f"ReturnsMean_{config['online_inference_config']['desired_return_1']}": total_returns}, step=it_counter)

                elif config["model_config"]["mode"] == 'memory_maze':
                    model.eval()
                    with torch.no_grad():
                        seeds = np.arange(0, 100).tolist()[::4]
                        for ret in [config["online_inference_config"]["desired_return_1"]]:
                            attn_map_received = False
                            returns = []
                            ts = []
                            for i in range(len(seeds)):
                                episode_return, act_list, t, _, _, attn_map = get_returns_MemoryMaze(model=model, ret=ret, seed=seeds[i], 
                                                                                        episode_timeout=episode_timeout, 
                                                                                        context_length=config["training_config"]["context_length"], 
                                                                                        device=device, act_dim=config["model_config"]["ACTION_DIM"], 
                                                                                        config=config,
                                                                                        mean=MEAN,
                                                                                        std=STD,
                                                                                        use_argmax=config["online_inference_config"]["use_argmax"],
                                                                                        create_video=False)

                                returns.append(episode_return)
                                ts.append(t)
                                pbar.set_description(f"Online inference_{ret}: [{i+1} / {len(seeds)}] Time: {t}, Return: {episode_return:.2f}")

                            returns_mean = np.mean(returns)
                            returns_max = np.max(returns)
                            lifetime_mean = np.mean(ts)

                            if wwandb:
                                wandb.log({f"LifeTimeMean_{ret}":   lifetime_mean,
                                           f"ReturnsMax_{ret}":     returns_max,
                                           f"ReturnsMean_{ret}":    returns_mean})
                            elif wcomet:
                                experiment.log_metrics({f"LifeTimeMean_{ret}":   lifetime_mean,
                                           f"ReturnsMax_{ret}":     returns_max,
                                           f"ReturnsMean_{ret}":    returns_mean}, step=it_counter)
                                
            model.train()
            wandb_step += 1 
            if wwandb:
                wandb.log({"checkpoint_step": epoch+1})
            elif wcomet:
                experiment.log_metrics({"checkpoint_step": epoch+1}, step=it_counter)
            torch.save(model.state_dict(), ckpt_path + str(epoch+1) + '_KTD.pth')
            
    return model