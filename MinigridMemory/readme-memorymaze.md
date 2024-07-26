# Minigrid.Memory

## Dataset

Then a dataset can be downloaded by executing the `MinigridMemory/get_data/collect_traj.py` file.

## Example usage

```python
python3 MinigridMemory/MinigridMemory_src/train_minigridmemory.py --model_mode 'RATE' --arch_mode 'TrXL' --ckpt_folder 'RATE_ckpt' --text 'my_comment'
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

