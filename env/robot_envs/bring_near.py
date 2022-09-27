import numpy as np
from robosuite.utils.transform_utils import mat2quat

from env.robot_envs.cable_manipulation_base import CableManipulationBase
from gym.vector.utils import spaces


class BringNearEnv(CableManipulationBase):
    pos_dist_limit = 0.02
    name = "BringNear"

    def _get_observation_space(self):
        custom_obs_low_limits = np.array([-2, -2, -2, -1, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1])
        custom_obs_high_limits = np.array([2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1])
        goal_low_limits = np.array([-2, -2, -2, -2, -2, -2, -2, -2, -2, 0, 0])
        goal_high_limits = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1])
        observation_space = spaces.Dict(
            dict(observation=spaces.Box(
                    custom_obs_low_limits, custom_obs_high_limits, shape=(14,), dtype="float32"
                ),
                desired_goal=spaces.Box(
                    goal_low_limits, goal_high_limits, shape=(11,), dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    goal_low_limits, goal_high_limits, shape=(11,), dtype="float32"
                )
            )
        )
        return observation_space

    def _get_custom_obs(self):
        # xmat could be reached directly, but there was an error, which is already fixed, but nut in new robosuite yet:
        # https://github.com/ARISE-Initiative/robosuite/pull/326/commits/9ce616023a3d53ccf3ecb98615158b478e4db823
        # this should be changed later, since the eef quat is "not consistent" see "def _eef1_xmat(self):" description
        robot0_gripper_pos = self._eef0_xpos
        robot0_gripper_quat = mat2quat(np.array(self.sim.data.site_xmat[self.robots[0].eef_site_id]).reshape(3, 3))
        robot1_gripper_pos = self._eef1_xpos
        robot1_gripper_quat = mat2quat(np.array(self.sim.data.site_xmat[self.robots[1].eef_site_id]).reshape(3, 3))

        obs = np.hstack([robot0_gripper_pos, robot0_gripper_quat, robot1_gripper_pos, robot1_gripper_quat])
        return obs

    def _get_engineered_encoding(self):
        # Important data
        robot0_gripper_pos = self._eef0_xpos
        robot1_gripper_pos = self._eef1_xpos

        mother_grip_id = self.sim.model.site_name2id(self._important_sites["mother_grip"])
        mother_grip_pos = np.array(self.sim.data.site_xpos[mother_grip_id])
        father_grip_id = self.sim.model.site_name2id(self._important_sites["father_grip"])
        father_grip_pos = np.array(self.sim.data.site_xpos[father_grip_id])

        mother_tip_id = self.sim.model.site_name2id(self._important_sites["mother_tip"])
        mother_tip_pos = np.array(self.sim.data.site_xpos[mother_tip_id])
        father_tip_id = self.sim.model.site_name2id(self._important_sites["father_tip"])
        father_tip_pos = np.array(self.sim.data.site_xpos[father_tip_id])

        # Calculate encoding based on the HinDRL article (https://arxiv.org/pdf/2112.00597.pdf)
        distance_to_mother_grip = robot0_gripper_pos - mother_grip_pos
        distance_to_father_grip = robot1_gripper_pos - father_grip_pos
        distance_between_cable_tips = father_tip_pos - mother_tip_pos
        grasp_mother = self._check_grasp(self.robots[0].gripper, "cable_stand_mother_contact")
        grasp_father = self._check_grasp(self.robots[1].gripper, "cable_stand_father_contact")

        encoding = np.hstack([distance_to_mother_grip, distance_to_father_grip, distance_between_cable_tips,
                              grasp_mother, grasp_father])
        return encoding

    def _check_success(self):
        """
        Check if cable is succesfully inserted.
        """
        father_tip_id = self.sim.model.site_name2id(self._important_sites["father_tip"])
        father_tip_pos = np.array(self.sim.data.site_xpos[father_tip_id])

        mother_tip_id = self.sim.model.site_name2id(self._important_sites["mother_tip"])
        mother_tip_pos = np.array(self.sim.data.site_xpos[mother_tip_id])


        pos_error = np.linalg.norm(father_tip_pos-mother_tip_pos)
        if pos_error < self.pos_dist_limit:
            return True
        else:
            return False