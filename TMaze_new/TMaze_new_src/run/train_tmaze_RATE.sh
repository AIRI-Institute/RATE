#!/bin/bash

python3 TMaze_new/TMaze_new_src/train_tmaze.py \
        --model_mode 'RATE' \
        --max_n_final 1 \
        --end_seed 10 \
        --arch_mode 'TrXL' \
        --curr 'true' \
        --ckpt_folder 'RATE_max_1'

python3 TMaze_new/TMaze_new_src/train_tmaze.py \
        --model_mode 'RATE' \
        --max_n_final 3 \
        --end_seed 10 \
        --arch_mode 'TrXL' \
        --curr 'true' \
        --ckpt_folder 'RATE_max_3'

python3 TMaze_new/TMaze_new_src/train_tmaze.py \
        --model_mode 'RATE' \
        --max_n_final 5 \
        --end_seed 10 \
        --arch_mode 'TrXL' \
        --curr 'true' \
        --ckpt_folder 'RATE_max_5'