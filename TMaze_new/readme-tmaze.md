# T-Maze

## Dataset

The dataset is generated automatically and stored in the `TMaze_new/TMaze_new_data` directory.

## Example usage

```python
python3 TMaze_new/TMaze_new_src/train_tmaze.py --model_mode 'RATE' --arch_mode 'TrXL' --curr 'true' --ckpt_folder 'RATE_max_3' --max_n_final 3 --text 'my_comment'
```

Where:

1. `model_mode` - select model:
    - 'RATE' -- our model
    - 'DT' -- Decision Transformer model
    - 'DTXL' -- Decision Transformer woth caching hidden states
    - 'RATEM' -- RATE without caching hidden states
    - 'RATE_wo_nmt' -- RATE without memory embeddings
2. `arch_mode` - select backbone model:
    - 'TrXL'
    - 'TrXL-I'
    - 'GTrXL'
3. `curr` - use of curriculum learning:
    - 'true' - use curriculum learning
    - 'false' - use standatd trajectories sampling
4. `max_n_final` - maximum number of segments used during training

