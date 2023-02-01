import numpy as np
import robosuite.utils.transform_utils as T

from env.robot_envs.fixed_parameterized_reach import FixedParameterizedReachEnv
from env.robot_envs.parameterized_reach import ParameterizedReachEnv

class FixedParameterizedReachDemonstrationPolicy:
    """
    Demonstration policy for the FixedParameterizedReachEnv with IK control. The difference to the simple
    ParameterizedReachDemonstrationPolicy is that for the FixedParameterizedReachEnv the observations do not contain
    information about the required poses, so the required pose is extracted in another way for the demo policy.
    Note: the trained policy has to learn the required poses relying only on the demonstrations.
    """
    def __init__(self, env:FixedParameterizedReachEnv, randomness_scale=0.05):
        self.env = env
        self.randomness_scale = randomness_scale

    def reset(self):
        pass

    def add_env(self, env):
        assert self.env is None
        self.env = env

    def step(self, observation):
        """
        Policy step.
        :param observation: observation from the env
        :return: action (np.array) for the robotarm
        """
        cur_pos = observation['robot0_eef_pos']
        cur_quat = observation['robot0_eef_quat']

        target_pose = self.env.goal_poses[int(observation['observation'][-1])]
        target_pos = target_pose[:3]
        target_axis_angle = target_pose[3:]
        target_quat = T.axisangle2quat(target_axis_angle)

        target_pos_dif = target_pos - cur_pos
        move = np.random.normal(loc=target_pos_dif, scale=self.randomness_scale)


        cur_mat = T.quat2mat(cur_quat)
        cur_mat = cur_mat @ T.euler2mat(np.array([0, 0, -np.pi / 2]))
        rot_vector = np.matmul(cur_mat.T, T.get_orientation_error(target_quat, cur_quat))
        rotation = np.random.normal(loc=rot_vector, scale=self.randomness_scale)

        action = np.hstack([move, rotation, 1])
        return action


if __name__ == '__main__':
    env = FixedParameterizedReachEnv(has_renderer=True, render_camera=None, control_freq=20, number_of_waypoints=2,
                                     waypoints=[np.array([0.58766,0.26816,0.37820,2.89549,0.03567,-0.39348]),
                                             np.array([0.42493,0.07166,0.36318,2.88426,0.12777,0.35920])]
                                             )

    demonstration_policy = FixedParameterizedReachDemonstrationPolicy(env, randomness_scale=0.1)

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