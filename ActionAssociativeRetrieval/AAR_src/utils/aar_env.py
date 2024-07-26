import numpy as np

class ActionAssociativeRetrieval:
    def __init__(self, stay_number: int, seed: int = 42):
        """
        NTR: need to return next step

        states: 0, 1
        obs: [state, NTR]
        actions: 0, 1, 2
        rewards: 1, -1

        1. S0(act=0|1) = S1, first_act = act
        2. while t < stay_number:
                S1(act=2) = S1
        3. if t == stay_number + 1:
                if act == first_act:
                    return reward = 1
                else:
                    return reward = -1
        """
        np.random.seed(seed)
        self.stay_number = stay_number
        self.episode_timeout = self.stay_number + 2
        self.reset()

    def reset(self):
        self.obs = [0, 0, np.random.randint(low=-1, high=1+1)]
        self.time_step = 0
        self.goal_state = self.obs
        self.state_changed = False

        return np.array(self.obs)

    def reward_fn(self, state, action):
        if self.time_step == self.episode_timeout:
            if state[0] == self.goal_state[0] and action == self.goal_action:
                if self.state_changed:
                    return 1
                else:
                    return 0
            else:
                return 0
        else:
            return 0

    def step(self, action):
        if self.time_step == 0:
            self.goal_action = action
        if self.time_step == self.stay_number:
            self.obs[1] = 1
        if self.time_step > self.stay_number:
            self.obs[1] = 0
        self.time_step += 1

        if self.obs[0] == 0 and action in (0, 1):
            self.obs[0] = 1
            self.state_changed = True
        elif self.obs[0] == 0 and action == 2:
            self.obs[0] = 0
            self.state_changed = False
        elif self.obs[0] == 1 and action in (0, 1):
            self.obs[0] = 0
            self.state_changed = True
        elif self.obs[0] == 1 and action == 2:
            self.obs[0] = 1
            self.state_changed = False

        self.obs[2] = np.random.randint(low=-1, high=1+1)

        reward = self.reward_fn(self.obs, action)

        if self.time_step >= self.episode_timeout or reward == 1:
            done = True
        else:
            done = False
            
        return np.array(self.obs), reward, done, self.time_step