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
        self.epsilon_params = {"m": m, "k": k}
        achieved_goal = []
        with h5py.File(demonstration_hdf5, "r") as f:
            for demo_id in f["data"].keys():
                if "desired_goal" in f["data"][demo_id]:
                    raise NotImplementedError
                else:
                    achieved_goal.append(np.array(f["data"][demo_id]["achieved_goal"]))

        self.goal_buffer = [demo[-1] for demo in achieved_goal]
        if m == 0:
            self.epsilon = 0
        else:
            self.epsilon = self._calc_epsilon(achieved_goal, m, k)

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
            distance = np.abs(rolling_window_left - rolling_window_right)
            distances.append(distance)
        distances = np.vstack(distances)

        mean_distance = np.mean(distances, axis=0)
        std_distance = np.std(distances, axis=0)
        epsilon = mean_distance + k * std_distance
        return epsilon

    def compute_reward(self, achieved_goal, goal, infos):
        """
        Goal conditioned reward calculation based on self.epsilon threshold
        :param achieved_goal: batched achieved goal
        :param goal: batched goal
        :param info:
        :return:
        """
        # Sparse reward
        distance = np.abs(achieved_goal - goal)
        reward = np.all(distance < self.epsilon, axis=1).astype(float)

        for info in infos:
            info['is_demonstration'] = False
        return reward

    def get_desired_goal(self):
        if len(self.goal_buffer) == 1:
            return self.goal_buffer[0]
        return self.goal_buffer[np.random.randint(0, len(self.goal_buffer)-1)]


class DefinedDistanceGoalHandler:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def compute_reward(self, achieved_goal, goal, infos):
        """
        Goal conditioned reward calculation based on self.epsilon threshold
        :param achieved_goal: batched achieved goal
        :param goal: batched goal
        :param info:
        :return:
        """
        # Sparse reward
        distance = np.linalg.norm(achieved_goal - goal, axis=1)
        reward = (distance <= self.epsilon).astype(float)
        for info in infos:
            info['is_demonstration'] = False
        return reward