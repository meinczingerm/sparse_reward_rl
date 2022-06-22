import copy
import random

import h5py
import numpy as np
from sb3_contrib import TQC
from stable_baselines3 import HerReplayBuffer


class HerReplayBufferWithDemonstrationGoals(HerReplayBuffer):
    def __init__(self, demonstration_hdf5, env, buffer_size=int(1e5), **kwargs):
        self.demonstration_hdf5 = demonstration_hdf5
        self.demonstrations = []

        for _env in env.envs:
            assert _env.use_engineered_observation_encoding
        with h5py.File(demonstration_hdf5, "r") as f:
            for demo_id in f["data"].keys():
                self.demonstrations.append({'actions': np.array(f["data"][demo_id]["actions"]),
                                            'engineered_encodings':
                                                np.array(f["data"][demo_id]["engineered_encodings"])})

        self.goal_buffer = [demo["engineered_encodings"][-1] for demo in self.demonstrations]
        for _env in (_env_monitor.env for _env_monitor in env.envs):
            _env.get_desired_goal_fn = self.get_random_goal
        super().__init__(env, buffer_size, **kwargs)

    def get_random_goal(self):
        return copy.deepcopy(self.goal_buffer[random.randint(0, len(self.goal_buffer)-1)])


class HinDRLReplayBuffer(HerReplayBuffer):
    def __init__(self, demonstration_hdf5, env, buffer_size=int(1e5), **kwargs):
        self.demonstration_hdf5 = demonstration_hdf5
        self.demonstrations = []

        for _env in env.envs:
            assert _env.use_engineered_observation_encoding
        with h5py.File(demonstration_hdf5, "r") as f:
            for demo_id in f["data"].keys():
                self.demonstrations.append({'actions': np.array(f["data"][demo_id]["actions"]),
                                            'engineered_encodings':
                                                np.array(f["data"][demo_id]["engineered_encodings"])})

        self.goal_buffer = [demo["engineered_encodings"][-1] for demo in self.demonstrations]
        for _env in (_env_monitor.env for _env_monitor in env.envs):
            _env.get_desired_goal_fn = self.get_random_goal
        super().__init__(env, buffer_size, **kwargs)

    def get_random_goal(self):
        return copy.deepcopy(self.goal_buffer[random.randint(0, len(self.goal_buffer)-1)])


class HinDRLTQC(TQC):
    def __init__(self, replay_buffer: HinDRLReplayBuffer, **kwargs):

        super(HinDRLTQC, self).__init__(**kwargs)
        self.replay_buffer = replay_buffer


