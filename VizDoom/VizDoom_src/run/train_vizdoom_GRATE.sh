#!/bin/bash

python3 VizDoom/VizDoom_src/train_vizdoom.py \
        --model_mode 'RATE' \
        --arch_mode 'GTrXL' \
        --ckpt_folder 'GRATE' \
        --start_seed 1 \
        --end_seed 6
