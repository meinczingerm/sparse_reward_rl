from typing import List

import numpy as np
from robosuite.utils.transform_utils import quat2mat, get_orientation_error, mat2quat, euler2mat

from env.robot_envs.bring_near import BringNearEnv


class RobotTarget:
    """
    Class with data describing the milestone for the robotarm.
    """
    def __init__(self, target_pos, target_quat, target_gripper, wait=None, time_step_limit=None, relative=None):
        """
        Stages that define the movement of each robot arm
        :param type: "glob" or "rel" defining whether the goal stage is defined in global frame or relative to
                     the other arm
        :param target_pos: target position np.array(dim=3)
        :param target_quat: target quaternion
        :param target_gripper: target gripper value between -1 and 1
        :param wait: If defined, the given stage of the other arm has to be finished first
        :param time_step_limit: number of steps to keep the target (in this case there is no guarantee for reaching
                                target) Suggested for gripping, because there is no other way to know when the grabbing
                                is done.
        :param relative: if given as str: then the target pos is calculated relative to observation[$relative$]
                            if given as list: then the target pos is calculated as:
                                              observation[$relative[0]$]-observation[$relative[1]$] + cur_eef_pos
        """
        if relative is None:
            self.type = "glob"
        else:
            self.type = "rel"
        self.relative = relative

        self.target_pos = target_pos
        self.target_quat = target_quat
        self.target_gripper = target_gripper
        self.wait = wait
        self.time_step_limit = time_step_limit
        self.timestep = 0

        if self.time_step_limit is not None:
            assert wait is None

    def increase_timer(self):
        self.timestep += 1


class RobotStages:
    """
    Class for tracking the milestones for the robot.
    """
    def __init__(self, robot_targets: List[RobotTarget]):
        self.stage = 0
        self._targets = robot_targets


    def get_target(self):
        return self._targets[self.stage]

    target = property(get_target)

    def step(self):
        if len(self._targets) - 1 > self.stage:
            self.stage += 1


class BringNearDemonstrationPolicy:
    def __init__(self):
        self.robot_stages = None
        self.reset()

    def reset(self):
        robot_0_stages = RobotStages([RobotTarget(self.random_pos_around(np.array([-0.35, -0.5, 1.35])),
                                          self.random_quat_around(np.array([0.5, -0.5, -0.5, -0.5])), -1),
                                      RobotTarget(np.array([0, 0, 0]),
                                                  np.array([0.6532815, -0.6532815, -0.2705981, -0.2705981]),
                                                  -1, relative="mother_grip_pos"),
                                      RobotTarget(np.array([0, 0, 0]),
                                                  np.array([0.6532815, -0.6532815, -0.2705981, -0.2705981]),
                                                  1, relative="mother_grip_pos", time_step_limit=20),
                                      RobotTarget(self.random_pos_around(np.array([-0.35, -0.1, 1.45])),
                                                  self.random_quat_around(
                                                        np.array([0.2705981, -0.2705981, -0.6532815, -0.6532815]),
                                                        angle_limit=0.5),
                                                  1),
                                      ])

        robot_1_stages = RobotStages([RobotTarget(self.random_pos_around(np.array([-0.35, 0.3, 1.35])),
                                          self.random_quat_around(np.array([0.5, 0.5, -0.5, 0.5])), -1),
                                      RobotTarget(np.array([0, 0, 0]),
                                                  np.array([0.6532815, 0.6532815, -0.2705981, 0.2705981]),
                                                  -1, relative="father_grip_pos"),
                                      RobotTarget(np.array([0, 0, 0]),
                                                  np.array([0.6532815, 0.6532815, -0.2705981, 0.2705981]),
                                                  1, relative="father_grip_pos", time_step_limit=20),
                                      RobotTarget(np.array([0, 0, 0]),
                                                  np.array([0.6532815, 0.6532815, -0.2705981, 0.2705981]),
                                                  1, relative="father_grip_pos", wait=3),
                                      RobotTarget(np.array([0, 0, 0]),
                                                  self.random_quat_around(
                                                      np.array([0.2705981, 0.2705981, -0.6532815, 0.6532815]),
                                                      angle_limit=1),
                                                  1, relative=["mother_tip_pos", "father_tip_pos"])])
        self.robot_stages = [robot_0_stages, robot_1_stages]

    def step(self, observation):
        """
        Policy step.
        :param observation: observation from the env
        :return: action (np.array) for the robotarm
        """
        action = []
        action.append(self.right_arm_step(observation))
        action.append(self.left_arm_step(observation))
        return np.hstack(action)

    def right_arm_step(self, observation):
        """
        Calculates the action for the right robotarm
        :param observation: observation from the env
        :return: action (np.array) for the robotarm
        """
        self.check_current_stage(observation, 0)
        target = self.robot_stages[0].target
        if target.type == "rel":
            if type(target.relative) is str:
                target_pos = observation[target.relative] + target.target_pos
            else:
                assert len(target.relative) == 2
                target_pos = observation[target.relative[0]] - observation[target.relative[1]] +\
                             observation["robot0_eef_pos"]
        else:
            target_pos = target.target_pos
        target_quat = target.target_quat
        action = self.step_to_target(observation, target_pos, target_quat, target.target_gripper, 0)
        return action

    def left_arm_step(self, observation):
        """
        Calculates the action for the left robotarm
        :param observation: observation from the env
        :return: action (np.array) for the robotarm
        """
        self.check_current_stage(observation, 1)
        target = self.robot_stages[1].target
        if target.type == "rel":
            if type(target.relative) is str:
                target_pos = observation[target.relative] + target.target_pos
            else:
                assert len(target.relative) == 2
                target_pos = observation[target.relative[0]] - observation[target.relative[1]] +\
                             observation["robot1_eef_pos"]
        else:
            target_pos = target.target_pos
        target_quat = target.target_quat
        action = self.step_to_target(observation, target_pos, target_quat, target.target_gripper, 1)
        return action

    @staticmethod
    def step_to_target(observation, target_pos, target_quat, target_gripper, robot_idx):
        """
        Calculates the action for the roboat to reach the target.
        :param observation: observation from the env
        :param target_pos: target position
        :param target_quat: target quaternion
        :param target_gripper: target value (float[-1, 1]) for the gripper
        :param robot_idx: index for the actuated robot arm
        :return: action (np.array) for the robotarm
        """
        cur_ori = observation[f'robot{robot_idx}_eef_quat']

        target_pos_dif = target_pos - observation[f'robot{robot_idx}_eef_pos']
        cur_mat = quat2mat(cur_ori)
        cur_mat = cur_mat @ euler2mat(np.array([0, 0, -np.pi/2]))

        rot_vector = np.matmul(cur_mat.T, get_orientation_error(target_quat, cur_ori))
        action = np.hstack([target_pos_dif, rot_vector, target_gripper])

        return action

    def check_current_stage(self, observation, robot_idx):
        """
        Checks whether the current stage is already finished or not, and steps to the next stage if necessary.
        :param observation: observation from the environment
        :param robot_idx: index for the robot arm
        :return: None
        """
        target = self.robot_stages[robot_idx].target
        if target.time_step_limit is not None:
            if target.timestep == target.time_step_limit:
                self.robot_stages[robot_idx].step()
                return
            else:
                target.increase_timer()
                return

        _done = False
        waiting = False
        if target.wait is not None:
            other_robot_idx = (1 if robot_idx == 0 else 0)
            if target.wait != self.robot_stages[other_robot_idx].stage:
                waiting = True
        if target.type == "glob":
            pos_done = np.allclose(target.target_pos, observation[f'robot{robot_idx}_eef_pos'], atol=1e-3, rtol=0)
            ori_done = np.allclose(target.target_quat, observation[f'robot{robot_idx}_eef_quat'], atol=1e-2, rtol=0)

            _done = (pos_done and ori_done)
        else:
            if type(target.relative) is str:
                target_pos = observation[target.relative] + target.target_pos
            else:
                assert len(target.relative) == 2
                target_pos = observation[target.relative[0]] - observation[target.relative[1]] +\
                             observation[f"robot{robot_idx}_eef_pos"]
            pos_done = np.allclose(target_pos, observation[f'robot{robot_idx}_eef_pos'], atol=1e-3, rtol=0)
            ori_done = np.allclose(target.target_quat, observation[f'robot{robot_idx}_eef_quat'], atol=1e-2, rtol=0)
            _done = (pos_done and ori_done)

        if _done and not waiting:
            self.robot_stages[robot_idx].step()

    @staticmethod
    def random_pos_around(input_pos: np.array, distance_limit=0.1):
        """
        Get uniform random position from a cube around the input position.
        :param input_pos: np.array(dim=3)
        :param distance_limit: scales the uniform distribution in all direction
        :return: randomized position
        """
        noise = np.random.uniform(size=3) * distance_limit
        assert input_pos.shape==noise.shape
        return input_pos + noise

    @staticmethod
    def random_quat_around(input_quat: np.array, angle_limit=0.3):
        """
        Get uniform random position from a cube around the input position.
        :param input_quat: input quaternion
        :param angle_limit: defines the angle limit around the input quaternion
        :return: randomized position
        """
        random_angle = np.random.uniform(size=3) * angle_limit
        random_rotation = euler2mat(random_angle)
        input_mat = quat2mat(input_quat)

        random_quat = mat2quat(random_rotation @ input_mat)
        if random_quat[0] < 0:
            random_quat = random_quat * -1  # without -1 the goal quaternion is unstable
        return random_quat



if __name__ == '__main__':
    env = BringNearEnv(has_renderer=True, render_camera=None, control_freq=20)

    demonstration_policy = BringNearDemonstrationPolicy()



    action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
    obs, _, done, _ = env.step(action)

    while True:
        action = demonstration_policy.step(obs)
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            print(done)
            env.reset()
            demonstration_policy = BringNearDemonstrationPolicy()
