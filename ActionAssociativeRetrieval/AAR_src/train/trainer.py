import torch
import wandb
from tqdm import tqdm
import numpy as np

from RATE_GTrXL import mem_transformer_v2_GTrXL
from ActionAssociativeRetrieval.AAR_src.inference.val_aar import get_returns_AAR


def train(model, optimizer, scheduler, raw_model, new_segment, epochs_counter, segments_count, wandb_step, ckpt_path, config, train_dataloader, val_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use the config dictionary to initialize the model
    if model is None:
        model = mem_transformer_v2_GTrXL.MemTransformerLM(**config["model_config"])

        model.loss_last_coef = config["training_config"]["coef"]
        torch.nn.init.xavier_uniform_(model.r_w_bias);
        torch.nn.init.xavier_uniform_(model.r_r_bias);
        wandb_step  = 0
        epochs_counter = 0
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training_config"]["learning_rate"], 
                                      weight_decay=config["training_config"]["weight_decay"], 
                                      betas=(config["training_config"]["beta_1"], config["training_config"]["beta_2"]))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1)/config["training_config"]["warmup_steps"], 1))
        
        raw_model = model.module if hasattr(model, "module") else model

        print(config)
    

        
    model.to(device)
    model.train()
    
    wwandb = config["wandb_config"]["wwandb"]
    print(f"model parameters: {sum(p.numel() for p in list(model.parameters()))}")
    it_counter = 0
    best_val_loss = np.inf
    epochs_without_improvement = 0
    max_epochs_without_improvement = 50

    EFFECTIVE_SIZE_BLOCKS = config["training_config"]["context_length"] * config["training_config"]["sections"]
    BLOCKS_CONTEXT = config["training_config"]["context_length"]

    switch = False
    val_loss_all = torch.tensor([float('inf')])
    
    pbar = tqdm(range(config["training_config"]["epochs"]))
    for epoch in pbar:
        is_train = True
        model.train()
        for it, batch in enumerate(train_dataloader):
            s, a, rtg, d, timesteps, masks = batch

            memory = None
            mem_tokens = None
            model.cache = None
            
            block_part_range = range(EFFECTIVE_SIZE_BLOCKS//BLOCKS_CONTEXT)
                
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
                    model.cache = model.cache.detach()
                elif raw_model.mem_tokens is not None:
                    mem_tokens = raw_model.mem_tokens.repeat(1, r1.shape[0], 1)
                    model.cache = mem_tokens.permute(1,0,2).detach()
                with torch.set_grad_enabled(is_train):
                    optimizer.zero_grad()
                    res = model(x1, y1, r1, y1, t1, *memory, mem_tokens=mem_tokens, masks=masks1) if memory is not None else model(x1, y1, r1, y1, t1, mem_tokens=mem_tokens, masks=masks1)
                    memory = res[0][2:]
                    logits, loss = res[0][0], res[0][1]
                    mem_tokens = res[1]
        
                    train_loss_all = model.loss_all
                    if model.flag == 1:
                        train_loss_last = model.loss_last

                    if wwandb:
                        train_pr_0, train_acc_0, train_loss_0 = model.probs_0_mean, model.acc_0, model.loss_all_0
                        wandb.log({"train_pr_0": train_pr_0,
                                "train_acc_0": train_acc_0,
                                "train_loss_0": train_loss_0})
                        
                        train_pr_1, train_acc_1, train_loss_1 = model.probs_1_mean, model.acc_1, model.loss_all_1
                        wandb.log({"train_pr_1": train_pr_1,
                                "train_acc_1": train_acc_1,
                                "train_loss_1": train_loss_1})

                    if wwandb and model.flag == 1:
                        wandb.log({"train_last_loss": train_loss_last.item(), 
                                   "train_loss": train_loss_all.item(), 
                                   "train_accuracy": model.accuracy, 
                                   "train_last_acc": model.last_acc})

                if is_train:
                    model.zero_grad()
                    optimizer.zero_grad()
                    train_loss_all.backward(retain_graph=True)
                    if config["training_config"]["grad_norm_clip"] is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training_config"]["grad_norm_clip"])
                    optimizer.step()
                    scheduler.step()
                    lr = optimizer.state_dict()['param_groups'][0]['lr']

                    pbar.set_description(f"[train] ep {epoch+1} it {it} tTotal {train_loss_all.item():.2f} vTotal {val_loss_all.item():.2f} lr {lr:e}")
            it_counter += 1 
            epochs_counter += 1
        
        model.eval()
        is_train = False
        with torch.no_grad():
            for it, batch in enumerate(val_dataloader):        
                s, a, rtg, d, timesteps, masks = batch
                memory = None
                mem_tokens = None
                model.cache = None
                addition = 1 if config['model_mode'] in ['DT', 'DTXL'] else 3
                block_part_range = range((EFFECTIVE_SIZE_BLOCKS // BLOCKS_CONTEXT) + addition)
                    
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
                        model.cache = model.cache.detach()
                    elif raw_model.mem_tokens is not None:
                        mem_tokens = raw_model.mem_tokens.repeat(1, r1.shape[0], 1)
                        model.cache = mem_tokens.permute(1,0,2).detach()
                    with torch.set_grad_enabled(is_train):
                        optimizer.zero_grad()
                        res = model(x1, y1, r1, y1, t1, *memory, mem_tokens=mem_tokens, masks=masks1) if memory is not None else model(x1, y1, r1, y1, t1, mem_tokens=mem_tokens, masks=masks1)
                        memory = res[0][2:]
                        logits, loss = res[0][0], res[0][1]
                        mem_tokens = res[1]
                        
                        val_loss_all = model.loss_all
                        if model.flag == 1:
                            val_loss_last = model.loss_last
                    
                        if wwandb:
                            val_pr_0, val_acc_0, val_loss_0 = model.probs_0_mean, model.acc_0, model.loss_all_0
                            wandb.log({"val_pr_0": val_pr_0,
                                    "val_acc_0": val_acc_0,
                                    "val_loss_0": val_loss_0})
                            
                            val_pr_1, val_acc_1, val_loss_1 = model.probs_1_mean, model.acc_1, model.loss_all_1
                            wandb.log({"val_pr_1": val_pr_1,
                                    "val_acc_1": val_acc_1,
                                    "val_loss_1": val_loss_1})

                        if wwandb and model.flag == 1:
                            wandb.log({"val_last_loss": val_loss_last.item(), 
                                       "val_loss": val_loss_all.item(), 
                                       "val_accuracy": model.accuracy, 
                                       "val_last_acc": model.last_acc,
                                       "learning_rate": lr})
                    if model.flag == 1:
                        pbar.set_description(f"[val] ep {epoch+1} it {it} tTotal {train_loss_all.item():.2f} vTotal {val_loss_all.item():.2f} lr {lr:e}")

        if val_loss_all < best_val_loss:
            best_val_loss = val_loss_all
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= max_epochs_without_improvement and config["training_config"]["use_erl_stop"] == True:
            print("Early stopping!")
            break        

        if it_counter >= config["training_config"]["warmup_steps"] and switch == False: # !
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, 
                                                          end_factor=config['training_config']['lr_end_factor'], 
                                                          total_iters=len(train_dataloader)*config["training_config"]["epochs"]*config["training_config"]["max_segments"])
            switch = True
        
        if wwandb:
            wandb.log({"segments_count": segments_count})
        
        if ((epoch + 1) % int(config["training_config"]["ckpt_epoch"])) == 0 or epoch == config["training_config"]["epochs"] - 1 or epoch == 0:
            if config["training_config"]["online_inference"]:
                model.eval()
                with torch.no_grad():
                    goods, bads = 0, 0
                    rewards = []
                    seeds = [0, 1, 2, 3, 6, 8, 9, 10, 12, 15, 4, 5, 7, 11, 13, 14, 16, 18, 21, 23]
                    pbar2 = range(len(seeds))
                    for indx, iii in enumerate(pbar2):
                        episode_return, act_list, t, _ , delta_t, attn_map = get_returns_AAR(model=model, ret=1.0, seed=seeds[iii], 
                                                                                             stay_number=config["online_inference_config"]["stay_number"],  
                                                                                             context_length=config["training_config"]["context_length"],
                                                                                             device=device, act_dim=config["model_config"]["ACTION_DIM"],
                                                                                             config=config, create_video=False)
                        if episode_return == 1.0:
                            goods += 1
                        else:
                            bads += 1
                        rewards.append(episode_return)
                        
                        pbar.set_description(f"[inference | {indx+1}/{len(seeds)}] ep {epoch+1} it {it} tTotal {train_loss_all.item():.2f} vTotal {val_loss_all.item():.2f} lr {lr:e}")
                            
                    suc_rate = goods / (goods + bads)

                    if wwandb:
                        wandb.log({"Success_rate": suc_rate})
        
            model.train()
            wandb_step += 1 
            if wwandb:
                wandb.log({"checkpoint_step": wandb_step})
            torch.save(model.state_dict(), ckpt_path + '_save' + '_KTD.pth')

    return model, wandb_step, optimizer, scheduler, raw_model, epochs_counter