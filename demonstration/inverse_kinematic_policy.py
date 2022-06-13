import math

import numpy as np

from env.cable_insertion_env import CableInsertionEnv
from robosuite import load_controller_config
from robosuite.utils.transform_utils import pose2mat, quat2mat, get_orientation_error, quat2axisangle, \
    mat2quat, mat2euler, euler2mat


class DemonstrationPolicy:
    def __init__(self):
        self.mother_grabbed = False
        self.father_grabbed = False
        self.i = 0
        self.reached = False
        self.prev_rot = np.zeros(3)

    def step(self, observation):
        action = []
        action.append(np.array([0, 0, 0, 0, 0, 0, 0]))
        action.append(self.left_arm_step(observation))
        self.i += 1
        return np.hstack(action)

    def left_arm_step(self, observation):
        if self.i < 100:
            self.saved_target_left = observation['father_grip_pos']
            target_pos = np.array([-0.35, 0.5, 1.35])
            target_ori = np.array([1, 0, 0, 0])
            target_pose = pose2mat((target_pos, target_ori))
            cur_ori = observation['robot1_eef_quat']
            cur_pos = observation['robot1_eef_pos']
            cur_pose = pose2mat((cur_pos, cur_ori))
            target_mat = quat2mat(target_ori)

            target_pos_dif = target_pos - observation['robot1_eef_pos']
            # cur_ori = observation['robot1_eef_quat']
            cur_mat = quat2mat(cur_ori)
            # necessary_rotation = np.matmul(cur_mat.T, target_mat)
            rot_vector = np.matmul(cur_mat.T, get_orientation_error(target_ori, cur_ori))
            left_action = np.hstack([target_pos_dif, 0, 0, 0, np.array([-1])])
        if self.i < 10000:
            self.saved_target_left = observation['father_grip_pos']
            target_pos = np.array([-0.35, 0.5, 1.35])
            target_ori = np.array([0.5, -0.5, 0.5, 0.5])
            target_pose = pose2mat((target_pos, target_ori))
            cur_ori = observation['robot1_eef_quat']
            cur_pos = observation['robot1_eef_pos']
            cur_pose = pose2mat((cur_pos, cur_ori))
            target_mat = quat2mat(target_ori)

            target_pos_dif = target_pos - observation['robot1_eef_pos']
            # cur_ori = observation['robot1_eef_quat']
            cur_mat = quat2mat(cur_ori)
            cur_mat = cur_mat @ euler2mat(np.array([0, 0, -np.pi/2]))
            # necessary_rotation = np.matmul(cur_mat.T, target_mat)
            rot_vector = np.matmul(cur_mat.T, get_orientation_error(target_ori, cur_ori)) #- self.prev_rot*0.5
            # rot_vector[2] = rot_vector[2] + np.pi/2
            # rot_vector = quat2axisangle(mat2quat(cur_mat.T @ target_mat))
            # rot_vector = get_orientation_error(target_ori, cur_ori)
            # rot_vector = quat2mat(target_ori) @ cur_mat.T
            left_action = np.hstack([target_pos_dif, rot_vector, np.array([-1])])
        else:
            self.saved_target_left = observation['father_grip_pos']
            target_pos = np.array([-0.35, 0.7, 1.6])
            target_ori = np.array([1, 0, 0, 0])
            target_mat = quat2mat(target_ori)

            target_pos_dif = target_pos - observation['robot1_eef_pos']
            cur_ori = observation['robot1_eef_quat']
            cur_mat = quat2mat(cur_ori)
            necessary_rotation = np.matmul(cur_mat.T, target_mat)
            rot_vector = quat2axisangle(mat2quat(necessary_rotation))
            left_action = np.hstack([target_pos_dif, rot_vector, np.array([-1])])
        return left_action

    def right_arm_step(self, observation):
        right_action = np.zeros(7)
        return right_action

    @staticmethod
    def calculate_local_rot_error(mat_cur, mat_goal):
        transform_matrix = mat_cur.T @ mat_goal
        goal_x_axis_in_cur = transform_matrix @ np.array([[1], [0], [0]])
        goal_y_axis_in_cur = transform_matrix @ np.array([[0], [1], [0]])
        goal_z_axis_in_cur = transform_matrix @ np.array([[0], [0], [1]])

        rot_z_cos = np.dot(goal_x_axis_in_cur[:, 0], np.array([1, 0, 0]))
        rot_z_sin = np.dot(goal_y_axis_in_cur[:, 0], np.array([1, 0, 0]))
        rot_z = math.atan2(rot_z_sin, rot_z_cos)

        rot_y_cos = np.dot(goal_z_axis_in_cur[:, 0], np.array([0, 0, 1]))
        rot_y_sin = np.dot(goal_x_axis_in_cur[:, 0], np.array([0, 0, 1]))
        rot_y = math.atan2(rot_y_sin, rot_y_cos)

        rot_x_cos = np.dot(goal_y_axis_in_cur[:, 0], np.array([0, 1, 0]))
        rot_x_sin = np.dot(goal_z_axis_in_cur[:, 0], np.array([0, 1, 0]))
        rot_x = math.atan2(rot_x_sin, rot_x_cos)

        return np.hstack([rot_x, rot_y, rot_z])

if __name__ == '__main__':
    controller_config = load_controller_config(default_controller="IK_POSE")
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
    obs, _, done, _ = env.step(action)

    while True:
        action = demonstration_policy.step(obs)
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            print(done)
            env.reset()
            demonstration_policy = DemonstrationPolicy()
