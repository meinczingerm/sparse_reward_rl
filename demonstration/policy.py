import math
from typing import Union

import numpy as np
import robosuite
from robosuite import load_controller_config
from robosuite.controllers import OperationalSpaceController
from robosuite.utils.transform_utils import mat2euler, matrix_inverse, quat2mat, quat_distance, quat2axisangle

from env.cable_insertion_env import CableInsertionEnv

class DemonstrationPolicy:
    def __init__(self):
        self.mother_grabbed = False
        self.father_grabbed = False
        self.i = 0
        self.reached = False

    def step(self, observation):
        action = []
        action.append(self.right_arm_step(observation))
        action.append(self.left_arm_step(observation))
        self.i += 1
        return np.hstack(action)

    def left_arm_step(self, observation):
        if self.i < 200:
            left_action = np.hstack([-0.3, 0.6, 1.3, np.array([np.pi / 2, 0, 0, -1])])
        elif self.i<300:
            target_pos = observation['father_grip_pos'] + np.array([0, 0.1, 0])
            left_action = np.hstack([target_pos, np.array([np.pi/2, 0, 0, -1])])
        elif self.i<400:
            target_pos = observation['father_grip_pos']
            self.saved_target_left = target_pos
            left_action = np.hstack([target_pos, np.array([np.pi/2, 0, 0, -1])])
        elif self.i < 500:
            target_pos = self.saved_target_left
            left_action = np.hstack([self.saved_target_left, np.array([np.pi / 2, 0, 0, 1])])
        elif self.i < 600:
            target_pos = self.saved_target_left + np.array([0, 0.1, 0.1])
            left_action = np.hstack([target_pos, np.array([0, 0, 0, 1])])
        elif self.i < 700:
            target_pos = observation['mother_grip_pos'] + np.array([-0.001, 0.15, 0])
            left_action = np.hstack([target_pos, np.array([0, 0, 0, 1])])
        elif self.i < 800:
            target_pos = observation['mother_grip_pos'] + np.array([-0.001, 0.112, 0])
            left_action = np.hstack([target_pos, np.array([0, 0, 0, 1])])
        elif self.i < 810:
            target_pos = observation['mother_grip_pos'] + np.array([-0.001, 0.112, 0])
            left_action = np.hstack([target_pos, np.array([0, 0, 0, -1])])
        else:
            target_pos = observation['mother_grip_pos'] + np.array([-0.001, 0.112, -0.1])
            left_action = np.hstack([target_pos, np.array([np.pi/2, 0, 0, -1])])

        return left_action

    def right_arm_step(self, observation):
        if self.i<300:
            target_pos = observation['mother_grip_pos'] + np.array([0, -0.1, 0])
            right_action = np.hstack([target_pos, np.array([-np.pi/2, 0, 0, -1])])
        elif self.i<400:
            target_pos = observation['mother_grip_pos']
            self.saved_target_right = target_pos
            right_action = np.hstack([target_pos, np.array([-np.pi/2, 0, 0, -1])])
        elif self.i < 500:
            right_action = np.hstack([self.saved_target_right, np.array([-np.pi / 2, 0, 0, 1])])
        elif self.i<800:
            target_pos = self.saved_target_right + np.array([0, 0.1, 0.2])
            right_action = np.hstack([target_pos, np.array([0, 0, 0, 1])])
        elif self.i<810:
            target_pos = self.saved_target_right + np.array([0, 0.1, 0.2])
            right_action = np.hstack([target_pos, np.array([0, 0, 0, -1])])
        elif self.i < 820:
            target_pos = self.saved_target_right + np.array([0, 0.1, 0.1])
            right_action = np.hstack([target_pos, np.array([-np.pi/2, 0, 0, -1])])
        else:
            target_pos = self.saved_target_right + np.array([0, 0.1, -0.1])
            right_action = np.hstack([target_pos, np.array([-np.pi/2, 0, 0, -1])])
        return right_action


    def grab_mother(self, observation):
        raise NotImplementedError


    def distance_to_mother(self):
        raise NotImplementedError


if __name__ == '__main__':
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config['control_delta'] = False
    controller_config['kp'] = 100

    env = CableInsertionEnv(robots=["Panda", "Panda"],  # load a Sawyer robot and a Panda robot
                            gripper_types="default",  # use default grippers per robot arm
                            controller_configs=controller_config,  # each arm is controlled using OSC
                            env_configuration="single-arm-parallel",
                            render_camera=None,# (two-arm envs only) arms face each other
                            has_renderer=True,  # no on-screen rendering
                            has_offscreen_renderer=False,  # no off-screen rendering
                            control_freq=20,  # 20 hz control for applied actions
                            horizon=10000,  # each episode terminates after 200 steps
                            use_object_obs=True,  # provide object observations to agent
                            use_camera_obs=False,  # don't provide image observations to agent
                            reward_shaping=True)  # use a dense reward signal for learning)

    demonstration_policy = DemonstrationPolicy()

    action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
    obs, _, _, _ = env.step(action)

    while True:
        action = demonstration_policy.step(obs)
        obs, _, _, _ = env.step(action)
        env.render()