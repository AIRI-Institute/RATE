# Memory Maze

## Dataset

Then a dataset can be downloaded by executing the `MemoryMaze/MemoryMaze_src/get_data.py` file.

## Example usage

```python
python3 -W ignore MemoryMaze/MemoryMaze_src/train_mem_maze.py --model_mode 'RATE' --arch_mode 'TrXL' --ckpt_folder 'RATE_ckpt' --text 'my_comment'
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

