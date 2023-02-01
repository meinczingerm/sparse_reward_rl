import numpy as np

from env.robot_envs.cable_manipulation_base import CableManipulationBase
from gym.vector.utils import spaces


class CableInsertionEnv(CableManipulationBase):
    def _get_observation_space(self):
        observation_space = spaces.Dict(
            dict(observation=spaces.Box(
                    -np.inf, np.inf, shape=(7,), dtype="float32"
                ),
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=(7,), dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=(7,), dtype="float32"
                )
            )
        )
        return observation_space

    def get_custom_obs(self):
        return self.get_engineered_encoding()

    def get_engineered_encoding(self):
        # Important data
        robot0_gripper_pos = self._eef0_xpos
        robot1_gripper_pos = self._eef1_xpos

        mother_grip_id = self.sim.model.site_name2id(self._important_sites["mother_grip"])
        mother_grip_pos = np.array(self.sim.data.site_xpos[mother_grip_id])
        mother_grip_mat = np.array(self.sim.data.site_xmat[mother_grip_id]).reshape([3, 3])
        father_grip_id = self.sim.model.site_name2id(self._important_sites["father_grip"])
        father_grip_pos = np.array(self.sim.data.site_xpos[father_grip_id])
        father_grip_mat = np.array(self.sim.data.site_xmat[father_grip_id]).reshape([3, 3])

        mother_tip_id = self.sim.model.site_name2id(self._important_sites["mother_tip"])
        mother_tip_pos = np.array(self.sim.data.site_xpos[mother_tip_id])
        father_tip_id = self.sim.model.site_name2id(self._important_sites["father_tip"])
        father_tip_pos = np.array(self.sim.data.site_xpos[father_tip_id])
        mother_inner_tip_id = self.sim.model.site_name2id(self._important_sites["mother_inner_tip"])
        mother_inner_tip = np.array(self.sim.data.site_xpos[mother_inner_tip_id])

        # Calculate encoding based on the HinDRL article (https://arxiv.org/pdf/2112.00597.pdf)
        distance_to_mother_grip = np.linalg.norm(robot0_gripper_pos - mother_grip_pos)
        distance_to_father_grip = np.linalg.norm(robot1_gripper_pos - father_grip_pos)
        distance_between_cable_tips = np.linalg.norm(father_tip_pos - mother_tip_pos)
        grasp_mother = self._check_grasp(self.robots[0].gripper, "cable_stand_mother_contact")
        grasp_father = self._check_grasp(self.robots[1].gripper, "cable_stand_father_contact")

        # y and z axis is mixed in the respective mujoco sites,
        #  so we are calculating the dot product between the y axis
        cable_z_product = np.dot(mother_grip_mat @ np.array([[0], [1], [0]]).squeeze(),
                                 father_grip_mat @ np.array([[0], [1], [0]]).squeeze())

        z_mother = mother_grip_mat @ np.array([[0], [1], [0]]).squeeze()
        socket_distance_vector = father_tip_pos - mother_inner_tip
        z_distance_to_socket = np.dot(z_mother, socket_distance_vector)

        encoding = np.hstack([distance_to_mother_grip, distance_to_father_grip, distance_between_cable_tips,
                              grasp_mother, grasp_father, cable_z_product, z_distance_to_socket])
        return encoding

    def _check_success(self):
        """
        Check if cable is succesfully inserted.
        """
        father_tip_id = self.sim.model.site_name2id(self._important_sites["father_tip"])
        father_tip_pos = np.array(self.sim.data.site_xpos[father_tip_id])
        father_tip_ori = self.sim.data.site_xmat[father_tip_id].reshape([3, 3])
        father_tip_z_ori = np.matmul(father_tip_ori, np.array([[0], [1], [0]])) # Z and Y axis is changed in the xml site

        mother_tip_id = self.sim.model.site_name2id(self._important_sites["mother_inner_tip"])
        mother_tip_pos = np.array(self.sim.data.site_xpos[mother_tip_id])
        mother_tip_ori = self.sim.data.site_xmat[mother_tip_id].reshape([3, 3])
        mother_tip_z_ori = np.matmul(mother_tip_ori, np.array([[0], [1], [0]])) # Z and Y axis is changed in the xml site

        pos_error = np.linalg.norm(father_tip_pos-mother_tip_pos, 2)
        ori_error = np.linalg.norm(mother_tip_z_ori + father_tip_z_ori)  # the two z direction has to be oppsite
        if pos_error < 0.002 and ori_error < 0.05:
            return True
        else:
            return False

