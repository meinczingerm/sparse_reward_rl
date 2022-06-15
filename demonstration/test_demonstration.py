import h5py
import numpy as np
from robosuite import load_controller_config

from env.base import CableManipulationBase


class ReplayedJointVelPolicy:
    def __init__(self, hdf5_file):
        with h5py.File(hdf5_file, "r") as f:
            self.actions = np.array(f["data"]["demo_1"]["observations"])

        self.i = 1

    def step(self, observation):
        action = self.actions[self.i]
        self.i += 1
        return action

    def reset(self):
        self.i = 0


class ReplayedActionPolicy:
    def __init__(self, hdf5_file):
        with h5py.File(hdf5_file, "r") as f:
            self.actions = np.array(f["data"]["demo_1"]["actions"])

        self.i = 0

    def step(self, observation):
        action = self.actions[self.i]
        self.i += 1
        return action

    def reset(self):
        self.i = 0


class ReplayedTorquePolicy:
    def __init__(self, hdf5_file):
        with h5py.File(hdf5_file, "r") as f:
            self.actions = np.array(f["data"]["demo_1"]["torque_actions"])

        self.i = 0

    def step(self, observation):
        action = self.actions[self.i]
        self.i += 1
        return action

    def reset(self):
        self.i = 0


def test_ik_replay(hdf5_file="/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/1655196314_930382/demo.hdf5"):
    controller_config = load_controller_config(default_controller="IK_POSE")
    controller_config['kp'] = 100

    env = CableManipulationBase(robots=["Panda", "Panda"],  # load a Sawyer robot and a Panda robot
                                gripper_types="default",  # use default grippers per robot arm
                                controller_configs=controller_config,  # each arm is controlled using OSC
                                env_configuration="single-arm-parallel",
                                render_camera=None,  # (two-arm envs only) arms face each other
                                has_renderer=True,  # no on-screen rendering
                                has_offscreen_renderer=False,  # no off-screen rendering
                                control_freq=20,  # 20 hz control for applied actions
                                horizon=1000,  # each episode terminates after 200 steps
                                use_object_obs=True,  # provide object observations to agent
                                use_camera_obs=False,  # don't provide image observations to agent
                                reward_shaping=True)  # use a dense reward signal for learning)

    demonstration_policy = ReplayedActionPolicy(hdf5_file)

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


def test_osc_replay(hdf5_file="/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/1654522800_8797252/demo.hdf5"):
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config['control_delta'] = False
    controller_config['kp'] = 150

    env = CableManipulationBase(robots=["Panda", "Panda"],  # load a Sawyer robot and a Panda robot
                                gripper_types="default",  # use default grippers per robot arm
                                controller_configs=controller_config,  # each arm is controlled using OSC
                                env_configuration="single-arm-parallel",
                                render_camera=None,  # (two-arm envs only) arms face each other
                                has_renderer=True,  # no on-screen rendering
                                has_offscreen_renderer=False,  # no off-screen rendering
                                control_freq=20,  # 20 hz control for applied actions
                                horizon=1000,  # each episode terminates after 200 steps
                                use_object_obs=True,  # provide object observations to agent
                                use_camera_obs=False,  # don't provide image observations to agent
                                reward_shaping=True)  # use a dense reward signal for learning)

    demonstration_policy = ReplayedActionPolicy(hdf5_file)

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


def test_torque_replay(hdf5_file="/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/1654522800_8797252/demo.hdf5"):
    controller_config = load_controller_config(default_controller="JOINT_TORQUE")
    controller_config['input_max'] = 80
    controller_config['input_min'] = -80
    controller_config['output_max'] = 80
    controller_config['output_min'] = -80

    env = CableManipulationBase(robots=["Panda", "Panda"],  # load a Sawyer robot and a Panda robot
                                gripper_types="default",  # use default grippers per robot arm
                                controller_configs=controller_config,  # each arm is controlled using OSC
                                env_configuration="single-arm-parallel",
                                render_camera=None,  # (two-arm envs only) arms face each other
                                has_renderer=True,  # no on-screen rendering
                                has_offscreen_renderer=False,  # no off-screen rendering
                                control_freq=20,  # 20 hz control for applied actions
                                horizon=1000,  # each episode terminates after 200 steps
                                use_object_obs=True,  # provide object observations to agent
                                use_camera_obs=False,  # don't provide image observations to agent
                                reward_shaping=True)  # use a dense reward signal for learning)

    demonstration_policy = ReplayedTorquePolicy(hdf5_file)

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

def test_joint_vel_replay(hdf5_file="/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/1654605742_6722271/demo.hdf5"):
    controller_config = load_controller_config(default_controller="JOINT_VELOCITY")
    controller_config["output_max"] = 1
    controller_config["output_min"] = -1

    env = CableManipulationBase(robots=["Panda", "Panda"],  # load a Sawyer robot and a Panda robot
                                gripper_types="default",  # use default grippers per robot arm
                                controller_configs=controller_config,  # each arm is controlled using OSC
                                env_configuration="single-arm-parallel",
                                render_camera=None,  # (two-arm envs only) arms face each other
                                has_renderer=True,  # no on-screen rendering
                                has_offscreen_renderer=False,  # no off-screen rendering
                                control_freq=20,  # 20 hz control for applied actions
                                horizon=1000,  # each episode terminates after 200 steps
                                use_object_obs=True,  # provide object observations to agent
                                use_camera_obs=False,  # don't provide image observations to agent
                                reward_shaping=True)  # use a dense reward signal for learning)

    demonstration_policy = ReplayedJointVelPolicy(hdf5_file)

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

if __name__ == '__main__':
    test_ik_replay()
    # test_joint_vel_replay()
    # test_torque_replay()
    # test_osc_replay("/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/1654605742_6722271/demo.hdf5")