"""
A script for collecting demonstrations as .hdf5 using the demonstration/policy.py as a policy.
The .hdf5 file can not be played back the standard way with DemoPlaybackCameraMover, because the
env is not part of robosuite.
"""

import datetime
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.camera_utils import DemoPlaybackCameraMover
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

from demonstration.inverse_kinematic_policy import IKDemonstrationPolicy
from demonstration.observation_collection_wrapper import ObservationCollectionWrapper
from demonstration.policy import DemonstrationPolicy
from env.cable_insertion_env import CableInsertionEnv
from utils import get_project_root_path


def run_demonstration_episode(env, policy, render=True):
    """
    Runs environment with given demonstration policy.
    :param env: env wrapped with ObservationCollectionWrapper
    :param policy: policy for demonstration collection
    :param render: (bool) True, if rendering should be shown.
    :return: succesful (bool): True, if the demonstration reached to goal
    """

    env.reset()
    policy.reset()
    env.render()
    random_action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
    observation, _, done, _ = env.step(random_action)

    # Loop until we get a reset from the input or the task completes
    while True:

        action = policy.step(observation)
        observation, reward, done, info = env.step(action)
        if render:
            env.render()
        if done:
            succesful = reward == 5000  # TODO check original reward used
            break

    # cleanup for end of data collection episodes
    env.close()

    return succesful


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration
            observations (dataset) - observations during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        engineered_encodings = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
                engineered_encodings.append(ai["engineered_encoding"])

        if len(states) == 0:
            continue

        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        ep_data_grp.create_dataset("engineered_encodings", data=np.array(engineered_encodings))

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


def collect_demonstrations(episode_num=10):
    """
    Collect demonstrations in demonstration/collection folder.
    :param episode_num: number of succesful episodes to collect
    :return: None
    """
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
                            horizon=500,  # each episode terminates after 200 steps
                            use_object_obs=True,  # provide object observations to agent
                            use_camera_obs=False,  # don't provide image observations to agent
                            reward_shaping=True)  # use a dense reward signal for learning)

    env_config = {
        "env_name": "CableInsertionEnv",
        "robots": ["Panda", "Panda"],
        "controller_configs": controller_config,
    }
    env_config = json.dumps(env_config)
    demonstration_policy = IKDemonstrationPolicy()

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = ObservationCollectionWrapper(env, tmp_directory)

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(get_project_root_path(), 'demonstration', 'collection', f"{t1}_{t2}")
    os.makedirs(new_dir)

    # collect demonstrations
    number_of_succesful_demonstrations = 0
    unsuccesful_episode_dirs = []
    while number_of_succesful_demonstrations != episode_num:
        succesful = run_demonstration_episode(env, demonstration_policy)
        if succesful:
            number_of_succesful_demonstrations += 1
        else:
            unsuccesful_episode_dirs.append(env.ep_directory)

        print(f"Progress: {number_of_succesful_demonstrations}/{episode_num}")

    for directory in unsuccesful_episode_dirs:
        shutil.rmtree(directory)

    gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_config)

def read_hdf5_file(file_path):
    with h5py.File(file_path, "r") as f:
        demo_1 = f["data"]["demo_1"]
        print("k")

if __name__ == '__main__':
    collect_demonstrations(2)
    # read_hdf5_file("/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/1654522800_8797252/demo.hdf5")