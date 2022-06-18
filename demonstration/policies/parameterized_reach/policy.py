from typing import List

import numpy as np
from robosuite import load_controller_config
import robosuite.utils.transform_utils as T

from env.parameterized_reach import ParameterizedReachEnv

class DemonstrationPolicy:
    """
    Demonstration policy for the ParameterizedReachEnv with IK control.
    """
    def __init__(self):
        self.progress = 0

    def reset(self):
        self.progress = 0

    def step(self, observation):
        """
        Policy step.
        :param observation: observation from the env
        :return: action (np.array) for the robotarm
        """
        cur_pos = observation['robot0_eef_pos']
        cur_quat = observation['robot0_eef_quat']

        target_pos = observation["desired_goal"][:3]
        target_axis_angle = observation["desired_goal"][3:6]
        target_quat = T.axisangle2quat(target_axis_angle)

        target_pos_dif = target_pos - cur_pos
        cur_mat = T.quat2mat(cur_quat)
        cur_mat = cur_mat @ T.euler2mat(np.array([0, 0, -np.pi / 2]))

        rot_vector = np.matmul(cur_mat.T, T.get_orientation_error(target_quat, cur_quat))
        if observation['observation'][6:].sum() != self.progress:
            self.progress += 1
        action = np.hstack([target_pos_dif, rot_vector, 1])
        return action

if __name__ == '__main__':
    env = ParameterizedReachEnv()

    demonstration_policy = DemonstrationPolicy()

    action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
    obs, _, done, _ = env.step(action)

    while True:
        action = demonstration_policy.step(obs)
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            print(done)
            env.reset()
            demonstration_policy.reset()