#!/bin/bash

python3 VizDoom/VizDoom_src/train_vizdoom.py \
        --model_mode 'DT' \
        --arch_mode 'TrXL' \
        --ckpt_folder 'DT' \
        --start_seed 1 \
        --end_seed 6
