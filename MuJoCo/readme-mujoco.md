# MuJoCo

## Installation

Experiments require MuJoCo.
Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.
Then, dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
```

## Downloading datasets

Datasets are stored in the `data` directory.
Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.
Then, run the following script in order to download the datasets and save them in our format:

```
python download_d4rl_datasets.py
```

In the `wandb_config.yaml` in the main directory add the following lines:

```python
mujoco:
  data_dir_prefix: '/path/to/mujoco/data/'
```

## Example usage

```python
python3 MuJoCo/train_rate_mujoco_ca.py --env_id 0 --number_of_segments 3 --segment_length 20 --num_mem_tokens 15 --n_head_ca 2 --seed 123
```

Where:

1. `env_id` - MuJoCo environment id:
    - 0 → `halfcheetah-medium`
    - 1 → `halfcheetah-medium-replay`
    - 2 → `halfcheetah-expert`
    - 3 → `walker2d-medium`
    - 4 → `walker2d-medium-replay`
    - 5 → `walker2d-expert`
    - 6 → `hopper-medium`
    - 7 → `hopper-medium-replay`
    - 8 → `hopper-expert`
    - 9 → `halfcheetah-medium-expert`
    - 10 → `walker2d-medium-expert`
    - 11 → `hopper-medium-expert`
2. `n_head_ca` - number of MRV attention heads

