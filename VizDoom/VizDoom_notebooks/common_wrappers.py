import gym
import torch
import numpy as np
from collections import deque
from gym import spaces

def batchify(input, add_time_dim=False, device=None, dtype=None):

    if not isinstance(input, list):
        input = [input]

    def convert(e):
        e = torch.as_tensor(e, device=device, dtype=dtype)
        if add_time_dim:
            e = e.unsqueeze(dim=0)
        return e

    return  [convert(e) for e in input]


class FrameStack(gym.Wrapper):
    def __init__(self, env, n_frames, greedy=False):
        """Stack n_frames last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        stable_baselines.common.atari_wrappers.LazyFrames
        :param env: (Gym Environment) the environment
        :param n_frames: (int) the number of frames to stack
        """
        #assert env.observation_space.dtype == np.uint8

        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        D, *extra_dims = env.observation_space.shape
        new_shape = (D*n_frames,) + tuple(extra_dims)

        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=new_shape,
            dtype=env.observation_space.dtype
        )

        self.greedy = greedy
        if self.greedy:
            self._get_ob = self._get_greedy_ob
        else:
            self._get_ob = self._get_lazy_ob


    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_greedy_ob(self):
        return np.concatenate(self.frames, axis=0)

    def _get_lazy_ob(self):
        assert len(self.frames) == self.n_frames
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
        self.dtype = frames[0].dtype

    def _force(self):
        return np.concatenate(
            np.array(self._frames, dtype=self.dtype), axis=0
        )

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class PrevActionAndReward(gym.Wrapper):
    """
    Adds prev reward and action to observation.
    Observation becomes a dict with two additional keys: "prev_reward", "prev_action"
    """
    def __init__(self, env, padding=0.):
        assert isinstance(env.action_space, spaces.Discrete), 'This implementation works only with discrete action spaces'
        super().__init__(env)
        self.padding = padding
        self.observation_space = self._make_obs_space()
        self.prev_r = 0.
        self.prev_a = None
        self.empty_act = np.zeros(self.action_space.n, dtype=np.float32)

        if isinstance(self.env.observation_space, spaces.Dict):
            self._make_obs = self._add_to_dict_obs
        else:
            self._make_obs = self._make_dict_for_vector_obs

    def _one_hot(self, a):
        vec_a = self.empty_act.copy()
        if a is None:
            return vec_a
        else:
            vec_a[a] = 1.
            return vec_a

    def _add_to_dict_obs(self, obs):
        obs['prev_reward'] = np.full(1, self.prev_r, dtype=np.float32)
        obs['prev_action'] = self._one_hot(self.prev_a)
        return obs

    def _make_dict_for_vector_obs(self, obs):
        return {
            'observation': obs,
            'prev_reward': np.full(1, self.prev_r, dtype=np.float32), #self.prev_r,
            'prev_action': self._one_hot(self.prev_a),
        }

    def _make_obs_space(self):

        wrapped_space = self.env.observation_space

        if isinstance(wrapped_space, spaces.Dict):
            env_spaces = dict(wrapped_space.spaces)
        else:
            env_spaces = dict(observation=wrapped_space)

        env_spaces['prev_reward'] = spaces.Box(
            low=float('-inf'),
            high=float('inf'),
            shape=(1,),
            dtype=np.float32
        )

        env_spaces['prev_action'] = spaces.Box(
            low=0.,
            high=1.,
            shape=(self.action_space.n,),
            dtype=np.float32,
        )

        return spaces.Dict(env_spaces)
    def print_kwargs(self, **kwargs):
        print(kwargs)

    def reset(self, **kwargs):
        self.prev_r = 0.
        self.prev_a = None

        obs = self.env.reset(**kwargs)
        return self._make_obs(obs)

    def step(self, action):
        self.prev_a = action
        obs, reward, done, info = self.env.step(action)
        self.prev_a = action
        self.prev_r = reward
        return self._make_obs(obs), reward, done, info


class AddMemory(gym.Wrapper):
    """
    Adds state of a learned memory model as one of observations.
    """
    def __init__(self, env, memory_model):
        super(AddMemory, self).__init__(env)
        self.is_vec_env = hasattr(env, 'num_envs')
        self.num_envs = getattr(env, 'num_envs', 1)

        self.model = memory_model
        self.device = self.model.device
        self.model.eval()

        self.memory_state = None

        # number of input channels may depends on framestack
        # self._input_channels = self.model.input_shape[0]
        self.observation_space = self._make_obs_space()

        if isinstance(self.env.observation_space, spaces.Dict):
            self._make_obs = self._update_dict_obs
            self._add_done = self._add_done_to_dict_obs
        else:
            self._make_obs = self._create_dict_obs_from_vector
            self._add_done = self._add_done_to_vector_obs

    def _update_dict_obs(self, obs, memory):
        obs['memory'] = memory
        return obs

    def _create_dict_obs_from_vector(self, obs, memory):
        return {
            'observation': obs,
            'memory': memory,
        }

    def _make_obs_space(self):
        wrapped_space = self.env.observation_space

        if isinstance(wrapped_space, spaces.Dict):
            env_spaces = dict(wrapped_space.spaces)
        else:
            env_spaces = dict(observation=wrapped_space)

        env_spaces['memory'] = spaces.Box(
            low=float('-inf'),
            high=float('inf'),
            #can be updated to use outputs from all itermediate rnn-layers if needed
            shape=(self.model.hidden_dim,),
            dtype=np.float32
        )

        return spaces.Dict(env_spaces)

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        init_done = [False]*self.num_envs if self.is_vec_env else False
        self.memory_state = self.model.init_state(self.num_envs)
        h_t = self._update_memory(obs, init_done)
        return self._make_obs(obs, h_t)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        h_t = self._update_memory(obs, done)
        return self._make_obs(obs, h_t), reward, done, info

    @torch.no_grad()
    def _update_memory(self, obs, done=False):
        out, self.memory_state = self.model(
            self._preproc_obs(obs, done),
            mem_state=self.memory_state,
        )
        return self._prepare_mem_for_obs(out)

    def _prepare_mem_for_obs(self, memory):
        memory = memory.cpu().squeeze(1)  # squeeze time dimension
        if self.is_vec_env:
            return memory
        else:
            return memory.squeeze(0)

    def _add_done_to_dict_obs(self, obs, done):
        obs = dict(obs)
        obs['done'] = self._preproc_done(done)
        return obs

    def _add_done_to_vector_obs(self, obs, done):
        return dict(
            observation=obs,
            done=self._preproc_done(done)
        )

    def _preproc_obs(self, obs, done):
        obs = self._add_done(obs, done)
        if not self.is_vec_env:
            obs = {k:v[np.newaxis,:] for k,v in obs.items()}

        return {
            k: torch.as_tensor(v).unsqueeze(1).to(self.device)
            for k,v in obs.items()
        }

    def _preproc_done(self, done):
        if self.is_vec_env:
            return np.asarray(done).reshape(-1,1)
        else:
            return np.full((1), done)
