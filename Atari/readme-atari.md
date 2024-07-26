
# Atari

## Installation

Dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
```

## Downloading datasets


Create a directory for the dataset and load the dataset using [gsutil](https://cloud.google.com/storage/docs/gsutil_install#install). Replace `[DIRECTORY_NAME]` and `[GAME_NAME]` accordingly (e.g., `./dqn_replay` for `[DIRECTORY_NAME]` and `Breakout` for `[GAME_NAME]`)
```
mkdir [DIRECTORY_NAME]
gsutil -m cp -R gs://atari-replay-datasets/dqn/[GAME_NAME] [DIRECTORY_NAME]
```

In the `wandb_config.yaml` in the main directory add the following lines to specify the directory with Atari data:

```python
atari:
  data: '/path/to/atari/data/'
```

## Example usage

```python
python3 Atari/train_rate_atari.py --game Seaquest --mem_len 2 --num_mem_tokens 15 --seed 123 --n_head_ca 2
```

Where:

1. `game` - Atari game name:
    - 'Breakout'
    - 'Qbert'
    - 'Seaquest'
    - 'Pong'
2. `n_head_ca` - number of MRV attention heads