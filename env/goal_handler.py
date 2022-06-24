from typing import List

import h5py
import numpy as np


class HinDRLGoalHandler:
    """ The HinDRL method described in https://arxiv.org/pdf/2112.00597.pdf uses environments with specific
    way for calculating the goal conditioned reward. This class is responsible for implementing this functionality
    by leveraging the use of a demonstration."""
    def __init__(self, demonstration_hdf5, m, k):
        """
        Init.
        :param env_to_wrap: environment to wrap with HinDRL functionality
        :param demonstration_hdf5: absolute path to hdf5 file containing the demonstrations
        :param m: m parameter used for calculating the goal conditioned reward threshold epsilon
                    (more info appendix A.8, https://arxiv.org/pdf/2112.00597.pdf)
        :param k: k parameter used for calculating the goal conditioned reward threshold epsilon
                    (more info appendix A.8, https://arxiv.org/pdf/2112.00597.pdf)
        """
        demonstrations = []
        with h5py.File(demonstration_hdf5, "r") as f:
            for demo_id in f["data"].keys():
                demonstrations.append(np.array(f["data"][demo_id]["observations"]))

        self.goal_buffer = [demo[-1] for demo in demonstrations]
        self.epsilon = self._calc_epsilon(demonstrations, m, k)

    @staticmethod
    def _calc_epsilon(demonstrations: List[np.array], m: int, k: int):
        """
        Calculates the goal conditioned reward threshold epsilon
        (more info appendix A.8, https://arxiv.org/pdf/2112.00597.pdf)
        :param demonstrations: list of episode observations
        :param m: m parameter
        :param k: k parameter
        """
        distances = []
        for episode in demonstrations:
            rolling_window_left = episode[m:]
            rolling_window_right = episode[:-m]
            distance = np.linalg.norm(rolling_window_left - rolling_window_right, axis=1)
            distances.append(distance)
        distances = np.hstack(distances)

        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        epsilon = mean_distance + k * std_distance
        return epsilon

    def compute_reward(self, achieved_goal, goal, info):
        """
        Goal conditioned reward calculation based on self.epsilon threshold
        :param achieved_goal: batched achieved goal
        :param goal: batched goal
        :param info:
        :return:
        """
        # Sparse reward
        distance = np.linalg.norm(achieved_goal - goal)
        reward = (distance < self.epsilon).astype(float)
        return reward

    def get_desired_goal(self):
        return self.goal_buffer[np.random.randint(0, len(self.goal_buffer)-1)]