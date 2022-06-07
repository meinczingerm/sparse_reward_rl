import os

from robosuite import load_controller_config
from sb3_contrib import TQC
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3.common.env_util import make_vec_env

from env.cable_insertion_env import CableInsertionEnv
from model.hindrl_buffer import HinDRLReplayBuffer, HinDRLTQC
from utils import create_log_dir, save_dict, get_baseline_model_with_name

config = {
    "demonstration_hdf5": "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/1654605742_6722271/demo.hdf5",
    "model_config":{
            "policy": "MultiInputPolicy",
            "buffer_size": 1000000,
            "batch_size": 2048,
            "gamma": 0.95,
            "learning_rate": float(1e-3),
            "tau": 0.05,
            "verbose": 1,
            "learning_starts": 10000,
            "policy_kwargs": {"net_arch":[512, 512, 512], "n_critics":2},
        }
}


def _setup_training(demonstration_hdf5, model_config):
    print("Creating env...")
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config['control_delta'] = False
    controller_config['kp'] = 150
    horizon = 1000

    env = make_vec_env(CableInsertionEnv,
                       env_kwargs={
                           "robots": ["Panda", "Panda"],
                           "gripper_types": "default",
                           "controller_configs": controller_config,
                           "env_configuration": "single-arm-parallel",
                           "render_camera": None,
                           "has_renderer": True,
                           "has_offscreen_renderer": False,
                           "control_freq": 20,
                           "horizon": horizon,
                           "use_object_obs": True,
                           "use_camera_obs": False,
                           "reward_shaping": True}
                       )

    model_config["env"] = env
    log_dir = create_log_dir("cable_insertion")
    model_config['tensorboard_log'] = log_dir
    save_dict(config, os.path.join(log_dir, 'config.json'))

    replay_buffer = HinDRLReplayBuffer(demonstration_hdf5, env, max_episode_length=horizon)
    print("Env ready")
    model = HinDRLTQC(replay_buffer, **model_config)
    return env, model


def train(_config):
    env, model = _setup_training(_config["demonstration_hdf5"], _config["model_config"])
    model.learn(50000000)


if __name__ == '__main__':
    train(config)
    # run_parallel(_configs=)