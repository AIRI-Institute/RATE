#!/bin/bash

python3 VizDoom/VizDoom_src/train_vizdoom.py \
        --model_mode 'RATE' \
        --arch_mode 'TrXL' \
        --ckpt_folder 'RATE' \
        --start_seed 1 \
        --end_seed 6
