import os

from robosuite import load_controller_config
from sb3_contrib import TQC
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env

from env.cable_manipulation_base import CableManipulationBase
from env.parameterized_reach import ParameterizedReachEnv
from eval import EvalVideoCallback
from model.hindrl_buffer import HinDRLReplayBuffer, HinDRLTQC, HerReplayBufferWithDemonstrationGoals
from utils import create_log_dir, save_dict, get_baseline_model_with_name

config = {
    "demonstration_hdf5": "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/1655215933_278294/demo.hdf5",
    "model_config":{
            "policy": "MultiInputPolicy",
            "buffer_size": 1000000,
            "batch_size": 2048,
            "gamma": 0.95,
            "learning_rate": float(1e-3),
            "tau": 0.05,
            "verbose": 1,
            "learning_starts": 10000,
            "policy_kwargs": {"net_arch": [512, 512, 512], "n_critics": 2},
        },
    "env": {"name": "parameterised_reach"}
}


def _setup_training(demonstration_hdf5, model_config):
    print("Creating env...")
    controller_config = load_controller_config(default_controller="IK_POSE")
    controller_config['kp'] = 100
    horizon = 1000

    env = make_vec_env(ParameterizedReachEnv)

    model_config["env"] = env
    log_dir = create_log_dir(config["env"]["name"])
    model_config['tensorboard_log'] = log_dir
    save_dict(config, os.path.join(log_dir, 'config.json'))

    replay_buffer = HerReplayBufferWithDemonstrationGoals(demonstration_hdf5, env, max_episode_length=horizon)
    print("Env ready")
    model = HinDRLTQC(replay_buffer, **model_config)
    return env, model, log_dir


def train(_config):
    env, model, log_dir = _setup_training(_config["demonstration_hdf5"], _config["model_config"])

    eval_env = make_vec_env(ParameterizedReachEnv)
    # Use deterministic actions for evaluation
    eval_path = os.path.join(log_dir, 'train_eval')

    video_callback = EvalVideoCallback(eval_env, best_model_save_path=eval_path,
                                       log_path=eval_path, eval_freq=10000,
                                       deterministic=True, render=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path=eval_path,
                                 log_path=eval_path, eval_freq=1000, n_eval_episodes=10, deterministic=True,
                                 render=False)
    eval_callbacks = CallbackList([video_callback, eval_callback])
    model.learn(50000000)


if __name__ == '__main__':
    train(config)
    # run_parallel(_configs=)