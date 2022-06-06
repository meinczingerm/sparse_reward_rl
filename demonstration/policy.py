import numpy as np
from robosuite import load_controller_config

from env.cable_insertion_env import CableInsertionEnv


class DemonstrationPolicy:
    def __init__(self):
        self.i = 0

    def step(self, observation):
        action = []
        action.append(self.right_arm_step(observation))
        action.append(self.left_arm_step(observation))
        self.i += 1
        return np.hstack(action)

    def left_arm_step(self, observation):
        if self.i < 50:
            target_pos = observation['father_grip_pos']
            self.saved_target_left = target_pos
            target = np.array([-0.45, 0.7, 1.35])
            left_action = np.hstack([target, np.array([3/4 * np.pi, 0, 0, -1])])
        elif self.i < 75:
            target = self.saved_target_left + np.array([0, 0.1, 0])
            left_action = np.hstack([target, np.array([3/4*np.pi, 0, 0, -1])])
        elif self.i < 125:
            left_action = np.hstack([self.saved_target_left, np.array([3/4 * np.pi, 0, 0, -1])])
        elif self.i < 200:
            left_action = np.hstack([self.saved_target_left, np.array([3/4 * np.pi, 0, 0, 1])])
        elif self.i < 300:
            target_pos = observation['mother_grip_pos'] + np.array([0, 0.15, 0])
            left_action = np.hstack([target_pos, np.array([np.pi/4, 0, 0, 1])])
        elif self.i < 325:
            target_pos = observation['mother_grip_pos'] + np.array([0, 0.11, 0])
            left_action = np.hstack([target_pos, np.array([np.pi/4, 0, 0, 1])])
        elif self.i < 350:
            target_pos = observation['mother_grip_pos'] + np.array([0, 0.11, 0])
            left_action = np.hstack([target_pos, np.array([np.pi/4, 0, 0, -1])])
        else:
            target_pos = observation['mother_grip_pos'] + np.array([0, 0.116, -0.2])
            left_action = np.hstack([target_pos, np.array([np.pi/2, 0, 0, -1])])

        return left_action

    def right_arm_step(self, observation):
        if self.i<50:
            self.saved_target_right = observation['mother_grip_pos']
            target_pos = self.saved_target_right + np.array([0, -0.3, 0])
            right_action = np.hstack([target_pos, np.array([-3/4*np.pi, 0, 0, -1])])
        elif self.i<75:
            target_pos = self.saved_target_right + np.array([0, -0.1, 0])
            right_action = np.hstack([target_pos, np.array([-3/4*np.pi, 0, 0, -1])])
        elif self.i < 125:
            right_action = np.hstack([self.saved_target_right, np.array([-3/4*np.pi, 0, 0, -1])])
        elif self.i < 150:
            right_action = np.hstack([self.saved_target_right, np.array([-3/4*np.pi, 0, 0, 1])])
        elif self.i < 325:
            target_pos = self.saved_target_right + np.array([0, 0.1, 0.2])
            right_action = np.hstack([target_pos, np.array([-np.pi/4, 0, 0, 1])])
        elif self.i < 350:
            target_pos = self.saved_target_right + np.array([0, 0.1, 0.2])
            right_action = np.hstack([target_pos, np.array([-np.pi/4, 0, 0, -1])])
        else:
            target_pos = self.saved_target_right + np.array([0, 0.1, -0.1])
            right_action = np.hstack([target_pos, np.array([-np.pi/2, 0, 0, -1])])
        return right_action

    def reset(self):
        self.i = 0


if __name__ == '__main__':
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config['control_delta'] = False
    controller_config['kp'] = 150

    env = CableInsertionEnv(robots=["Panda", "Panda"],  # load a Sawyer robot and a Panda robot
                            gripper_types="default",  # use default grippers per robot arm
                            controller_configs=controller_config,  # each arm is controlled using OSC
                            env_configuration="single-arm-parallel",
                            render_camera=None,# (two-arm envs only) arms face each other
                            has_renderer=True,  # no on-screen rendering
                            has_offscreen_renderer=False,  # no off-screen rendering
                            control_freq=20,  # 20 hz control for applied actions
                            horizon=1000,  # each episode terminates after 200 steps
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
