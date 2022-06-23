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


class HinDRLReplayBuffer(HerReplayBuffer):
    def __init__(self, demonstration_hdf5, env, replay_strategy="uniform_demonstration",
                 buffer_size=int(1e5), **kwargs):
        self.demonstration_hdf5 = demonstration_hdf5
        self.replay_strategy = replay_strategy

        self.demonstrations = {"actions": [],
                               "observations": []}
        self.online_goal_buffer = []

        for _env in env.envs:
            assert _env.use_engineered_observation_encoding
        with h5py.File(demonstration_hdf5, "r") as f:
            for demo_id in f["data"].keys():
                actions = np.array(f["data"][demo_id]["actions"])
                observations = np.array(f["data"][demo_id]["observations"])
                self.demonstrations["actions"].append(actions)
                self.demonstrations["observations"].append(observations)
                self.online_goal_buffer.append(observations[-1])

        for _env in (_env_monitor.env for _env_monitor in env.envs):
            _env.get_desired_goal_fn = self.get_online_goal
        super().__init__(env, buffer_size, **kwargs)

    def get_online_goal(self):
        return copy.deepcopy(self.online_goal_buffer[random.randint(0, len(self.online_goal_buffer) - 1)])

    def sample_goals(self, episode_indices: np.ndarray, her_indices: np.ndarray, transitions_indices: np.ndarray,
    ) -> np.ndarray:
        if self.replay_strategy == "uniform_demonstration":
            return self._sample_uniform_demonstration_goals(num_of_goals=her_indices.shape[0])
        else:
            raise NotImplementedError

    def _sample_uniform_demonstration_goals(self, num_of_goals: int):
        """
        Get random states uniformly from all demonstration.
        :param num_of_goals: Number of sampled goals
        :return: Sampled goals (np.array)
        """
        demonstrations = self.demonstrations["observations"]
        num_of_demonstrations = len(demonstrations)
        demonstration_indices = np.random.randint(low=0, high=num_of_demonstrations, size=num_of_goals)

        goals = []
        for i, demo in enumerate(demonstrations):
            goal_num_from_demo = (demonstration_indices == i).sum()
            goal_indices = np.random.randint(0, demo.shape[0]-1, goal_num_from_demo)
            goal_array = np.take(demo, goal_indices, axis=0)
            goals.append(goal_array)

        goals = np.vstack(goals)
        goals = goals.reshape([goals.shape[0], 1, goals.shape[1]])
        return goals










class HinDRLTQC(TQC):
    def __init__(self, replay_buffer: HinDRLReplayBuffer, **kwargs):

        super(HinDRLTQC, self).__init__(**kwargs)
        self.replay_buffer = replay_buffer


