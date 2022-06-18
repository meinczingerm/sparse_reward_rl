import itertools
import os.path
import warnings
from abc import abstractmethod
from collections import OrderedDict

import numpy as np
import robosuite.utils.transform_utils as T
from gym.vector.utils import spaces
from robosuite import load_controller_config
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import TableArena, EmptyArena
from robosuite.models.objects import MujocoXMLObject, CapsuleObject, BallObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler

from utils import get_project_root_path


class ParameterizedReachEnv(SingleArmEnv):
    """
    Parameterized reach env from https://arxiv.org/pdf/2112.00597.pdf.
    """

    def __init__(
        self,
        number_of_waypoints=20,
        robots="Panda",
        use_engineered_observation_encoding=True,  # special for HinDRL
        use_desired_goal=False,  # special for HinDRL
        env_configuration="default",
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=False,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):

        self.use_engineered_observation_encoding = use_engineered_observation_encoding
        self.number_of_waypoints = number_of_waypoints  # including the final goal pose
        self.idx_of_reached_waypoint = -1  # index for the last waypoint reached
        self.goal_poses = self.get_random_goals()
        self.placement_initializer = None


        controller_configs = load_controller_config(default_controller="IK_POSE")
        controller_configs['kp'] = 100

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

        warnings.warn("Observation space is not configured")
        self.action_space = spaces.Box(self.robots[0].controller.input_min, self.robots[0].controller.input_max,
                                       dtype="float32")

        warnings.warn("Observation space uses unlimited box. Should be updated.")
        self.observation_space = spaces.Dict(
                dict(observation=spaces.Box(
                        -np.inf, np.inf, shape=(6,), dtype="float32"
                    ),
                    desired_goal=spaces.Box(
                        -np.inf, np.inf, shape=(6,), dtype="float32"
                    ),
                    achieved_goal=spaces.Box(
                        -np.inf, np.inf, shape=(6,), dtype="float32"
                    )
                )
            )

        self.metadata = None
        self.spec = None


    # Quickfix for issue: https://github.com/ARISE-Initiative/robosuite/issues/321
    @property
    def _eef_xquat(self):
        eef_quat = T.convert_quat(self.sim.data.get_body_xquat(self.robots[0].robot_model.eef_name), to="xyzw")
        return eef_quat

    def _get_desired_goal(self):
        if self._check_success():
            return np.zeros(self.observation_space["observation"].shape)
        desired_pose = self.goal_poses[self.idx_of_reached_waypoint + 1]
        desired_mask = np.zeros(self.number_of_waypoints)
        desired_mask[:self.idx_of_reached_waypoint + 2] = 1
        return np.hstack([desired_pose, desired_mask])

    def _get_engineered_encoding(self):
        """
        Calculates the heuristically engineered encoding of states.
        :return: encoding (flattened np.array)
        """
        if self._check_reached_next_goal():
            self.idx_of_reached_waypoint += 1

        gripper_pos = self._eef_xpos
        gripper_axis_angle = T.quat2axisangle(self._eef_xquat)
        progress_mask = np.zeros(self.number_of_waypoints)
        progress_mask[:self.idx_of_reached_waypoint + 1] = 1
        return np.hstack([gripper_pos, gripper_axis_angle, progress_mask])

    def get_random_goals(self):
        random_positions = [np.random.uniform(low=np.array([0.4, -0.3, 0]), high=np.array([0.6, 0.3, 0.4]))
                            for _ in range(self.number_of_waypoints)]
        random_axis_angles = [T.random_axis_angle(np.pi/4) for _ in range(self.number_of_waypoints)]
        random_axis_angles = [rot_axis * rot_angle for rot_axis, rot_angle in random_axis_angles]
        random_mats = [T.quat2mat(T.axisangle2quat(random_axis_angle)) @ T.quat2mat([1, 0, 0, 0]) for
                      random_axis_angle in random_axis_angles]
        random_quats = [T.mat2quat(random_mat) for random_mat in random_mats]
        random_quats = [random_quat if random_quat[0] > 0 else random_quat * -1 for random_quat in random_quats]
        random_axis_angles = [T.quat2axisangle(random_quat) for random_quat in random_quats]
        for angle in random_axis_angles:
            assert np.allclose(T.quat2axisangle(T.axisangle2quat(angle)), angle)
        goal_poses = [np.hstack([goal_pos, goal_axis_angle]) for goal_pos, goal_axis_angle in
                      zip(random_positions, random_axis_angles)]
        return goal_poses

    def _check_reached_next_goal(self):
        gripper_pos = self._eef_xpos
        gripper_axis_angle = T.quat2axisangle(self._eef_xquat)
        desired_goal = self._get_desired_goal()
        desired_pos = desired_goal[0:3]
        desired_axis_angle = desired_goal[3:6]

        pos_dist = np.linalg.norm(gripper_pos - desired_pos)
        axis_angle_dist = np.linalg.norm(gripper_axis_angle - desired_axis_angle)

        print(f"Angle_dist: {axis_angle_dist}")

        return pos_dist < 0.005 and axis_angle_dist < 0.01

    def _check_success(self):
        """
        Checks succes of current state.
        :return: True if current state is succesful, False otherwise
        """
        if self.idx_of_reached_waypoint == self.number_of_waypoints - 1:
            return True
        else:
            return False

    def reward(self, action=None):
        """
        Sparse reward function for the task.

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        if self._check_success():
            reward = 1
        else:
            reward = 0

        return reward

    def reset(self):
        # get new goals and reset progress
        self.idx_of_reached_waypoint = -1
        self.goal_poses = self.get_random_goals()
        obs = super(ParameterizedReachEnv, self).reset()
        return obs

    def _load_model(self):
        """
        Loading the arena with table, the double-arm robots and the cable stand.
        """
        super()._load_model()

        # load model for table top workspace
        mujoco_arena = EmptyArena()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, -0.92])

        capsules = []
        for i, _goal_pose in enumerate(self.goal_poses):
            capsule = CapsuleObject(f"goal_{i}", size=np.array([0.01, 0.02]), joints=None, obj_type="visual",
                                    rgba=[0, 1, 0, i / self.number_of_waypoints])
            goal_quat = T.axisangle2quat(_goal_pose[3:])
            capsule._obj.attrib["quat"] = f"{goal_quat[3]} {goal_quat[0]} {goal_quat[1]} {goal_quat[2]}"
            print(f"{goal_quat[0]} {goal_quat[1]} {goal_quat[2]} {goal_quat[3]}")
            capsule._obj.attrib["pos"] = f"{_goal_pose[0]} {_goal_pose[1]} {_goal_pose[2]}"
            capsules.append(capsule)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=capsules
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment, namely the position and orientation for the
        gripping sites for both cable.

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # Get prefix from robot model to avoid naming clashes for multiple robots and define observables modality
        modality = f"objective"

        @sensor(modality=modality)
        def engineered_encoding(obs_cache):
            return self._get_engineered_encoding()

        @sensor(modality=modality)
        def get_desired_goal(obs_cache):
            return self._get_desired_goal()

        if self.use_engineered_observation_encoding:
            sensors = [engineered_encoding, engineered_encoding, get_desired_goal]
            names = ["observation", "achieved_goal", "desired_goal"]
        else:
            raise NotImplementedError

        # Create observables for this robot
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    def _post_action(self, action):
        """
        Do any housekeeping after taking an action.
        Args:
            action (np.array): Action to execute within the environment
        Returns:
            3-tuple:
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) empty dict to be filled with information by subclassed method
        """
        reward = self.reward(action)

        # done if number of elapsed timesteps is greater than horizon
        self.done = (self.timestep >= self.horizon) or self._check_success()

        return reward, self.done, {}

if __name__ == '__main__':
    env = ParameterizedReachEnv()

    while True:
        action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            print(done)
            env.reset()

