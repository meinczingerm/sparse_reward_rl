import os.path
import warnings
from collections import OrderedDict

import numpy as np
import robosuite.utils.transform_utils as T
from gym.vector.utils import spaces
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor

from utils import get_project_root_path


class CableInsertionEnv(TwoArmEnv):
    """
    This class corresponds to the lifting task for two robot arms.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be either 2 single single-arm robots or 1 bimanual robot!

        use_engineered_observation_encoding (bool): True if observation is the engineered encoding, False if raw
                                                    observation is used.

        use_desired_goal (bool): If True then "desired_goal" will be added to the observation, specified by
                                 get_desired_goal_fn.

        get_desired_goal_fn (function): Function with no input which returns desired_goal as flattened
                                        np.array with dimensions of "achieved_goal".  This function can be used to can
                                        pass goals from demonstrations.

        env_configuration (str): Specifies how to position the robots within the environment. Can be either:

            :`'bimanual'`: Only applicable for bimanual robot setups. Sets up the (single) bimanual robot on the -x
                side of the table
            :`'single-arm-parallel'`: Only applicable for multi single arm setups. Sets up the (two) single armed
                robots next to each other on the -x side of the table
            :`'single-arm-opposed'`: Only applicable for multi single arm setups. Sets up the (two) single armed
                robots opposed from each others on the opposite +/-y sides of the table.

        Note that "default" corresponds to either "bimanual" if a bimanual robot is used or "single-arm-opposed" if two
        single-arm robots are used.

        controller_configs (str or list of dict): If set, contains relevant demonstration parameters for creating a
            custom demonstration. Else, uses the default demonstration for this specific task. Should either be single
            dict if same demonstration is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        ValueError: [Invalid number of robots specified]
        ValueError: [Invalid env configuration]
        ValueError: [Invalid robots for specified env configuration]
    """

    def __init__(
        self,
        robots,
        use_engineered_observation_encoding=True,  # special for HinDRL
        use_desired_goal=False,  # special for HinDRL
        get_desired_goal_fn=None,  # special for HinDRL
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(1.5, 2, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
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
        self.use_desired_goal = use_desired_goal
        self.get_desired_goal_fn = get_desired_goal_fn
        self.desired_goal = None

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.9))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

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
        self.action_space = spaces.Box(np.hstack([self.robots[0].controller.input_min,
                                                 self.robots[1].controller.input_min]),
                                       np.hstack([self.robots[0].controller.input_max,
                                                 self.robots[1].controller.input_max]), dtype="float32")

        warnings.warn("Observation space uses unlimited box. Should be updated.")
        if self.use_desired_goal:
            self.observation_space = spaces.Dict(
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
        else:
            self.observation_space = spaces.Dict(
                dict(observation=spaces.Box(
                        -np.inf, np.inf, shape=(7,), dtype="float32"
                    )
                )
            )

        self.metadata = None
        self.spec = None

    def reward(self, action=None):
        """
        Sparse reward function for the task.

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        if self._check_success():
            reward = 5000
        else:
            reward = -1

        return reward

    def reset(self):
        obs = super(CableInsertionEnv, self).reset()
        if self.get_desired_goal_fn is not None:
            self.desired_goal = self.get_desired_goal_fn()
            assert self.desired_goal.shape == self.observation_space["desired_goal"].shape

        if self.use_desired_goal:
            assert self.get_desired_goal_fn is not None

        return obs


    def _load_model(self):
        """
        Loading the arena with table, the double-arm robots and the cable stand.
        """
        super()._load_model()

        # Adjust base pose(s) accordingly
        if self.env_configuration == "bimanual":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "single-arm-opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:  # "single-arm-parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        self.cable_stand = MujocoXMLObject(os.path.join(get_project_root_path(),
                                                         "env/assets/cable_stand/cable_stand_object.xml"),
                                           "cable_stand", joints=None)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cable_stand
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

        # cable grips
        @sensor(modality=modality)
        def mother_grip_pos(obs_cache):
            mother_grip_id = self.sim.model.site_name2id(self._important_sites["mother_grip"])
            return np.array(self.sim.data.site_xpos[mother_grip_id])

        @sensor(modality=modality)
        def mother_grip_mat(obs_cache):
            mother_grip_id = self.sim.model.site_name2id(self._important_sites["mother_grip"])
            return np.array(self.sim.data.site_xmat[mother_grip_id])

        @sensor(modality=modality)
        def father_grip_pos(obs_cache):
            father_grip_id = self.sim.model.site_name2id(self._important_sites["father_grip"])
            return np.array(self.sim.data.site_xpos[father_grip_id])

        @sensor(modality=modality)
        def father_grip_mat(obs_cache):
            father_grip_id = self.sim.model.site_name2id(self._important_sites["father_grip"])
            return np.array(self.sim.data.site_xmat[father_grip_id])

        @sensor(modality=modality)
        def engineered_encoding(obs_cache):
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

        @sensor(modality=modality)
        def get_desired_goal(obs_cache):
            return self.desired_goal

        if self.use_engineered_observation_encoding:
            sensors = [mother_grip_pos, mother_grip_mat, father_grip_pos, father_grip_mat, engineered_encoding,
                       engineered_encoding]
            names = [f"mother_grip_pos", "mother_grip_mat", "father_grip_pos", "father_grip_mat", "observation",
                     "achieved_goal"]
        else:
            raise NotImplementedError

        if self.use_desired_goal:
            sensors.append(get_desired_goal)
            names.append("desired_goal")

        # Create observables for this robot
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    def _check_success(self):
        """
        Check if cable is succesfully inserted.
        """
        father_tip_id = self.sim.model.site_name2id(self._important_sites["father_tip"])
        father_tip_pos = np.array(self.sim.data.site_xpos[father_tip_id])
        father_tip_ori = self.sim.data.site_xmat[father_tip_id].reshape([3, 3])
        father_tip_z_ori = np.matmul(father_tip_ori, np.array([[0], [1], [0]])) # Z and Y axis is changed in the xml site

        mother_tip_id = self.sim.model.site_name2id(self._important_sites["mother_tip"])
        mother_tip_pos = np.array(self.sim.data.site_xpos[mother_tip_id])
        mother_tip_ori = self.sim.data.site_xmat[mother_tip_id].reshape([3, 3])
        mother_tip_z_ori = np.matmul(mother_tip_ori, np.array([[0], [1], [0]])) # Z and Y axis is changed in the xml site


        pos_error = np.linalg.norm(father_tip_pos-mother_tip_pos, 2)
        ori_error = np.linalg.norm(mother_tip_z_ori + father_tip_z_ori) # the two z direction has to be oppsite
        if pos_error < 0.003 and ori_error < 0.03:
            return True
        else:
            return False

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
        self.done = (self.timestep >= self.horizon) or reward == 5000

        return reward, self.done, {}

    @property
    def _important_sites(self):
        """
        Sites used to aid visualization by human. (usually "grip_site" and "grip_cylinder")
        (and should be hidden from robots)
        """
        return {
            "father_grip": "cable_stand_father_grip",
            "father_tip": "cable_stand_father_tip",
            "mother_grip": "cable_stand_mother_grip",
            "mother_tip": "cable_stand_mother_tip",
            "mother_inner_tip": "cable_stand_mother_inner_tip"
        }

