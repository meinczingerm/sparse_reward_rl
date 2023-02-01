import numpy as np
import robosuite.utils.transform_utils as T
from gym import spaces

from env.robot_envs.parameterized_reach import ParameterizedReachEnv


class FixedParameterizedReachEnv(ParameterizedReachEnv):
    def __init__(self, waypoints=None, **kwargs):
        super(FixedParameterizedReachEnv, self).__init__(**kwargs)
        self.name = f"FixedParameterizedReach_{kwargs['number_of_waypoints']}Waypoint"
        if waypoints is not None:
            assert len(waypoints) == kwargs['number_of_waypoints']
            self.goal_poses = waypoints

    def reset(self):
        # get new goals and reset progress
        self.idx_of_reached_waypoint = -1
        obs = super(ParameterizedReachEnv, self).reset()
        return obs

    def add_goal_handler(self, goal_handler):
        self.goal_handler = goal_handler

    def compute_reward(self, achieved_goal, goal, info):
        """

        :param achieved_goal: batched
        :param goal:
        :param info:
        :return:
        """
        reward = self.goal_handler.compute_reward(achieved_goal, goal, info)
        return reward

    def _get_desired_goal(self):
        desired_pose = self.goal_poses[-1]
        desired_mask = np.ones(self.number_of_waypoints-1)
        return np.hstack([desired_pose, desired_mask])

    def get_target_pose(self):
        target_pose = self._get_next_goal()[:6]
        return target_pose
