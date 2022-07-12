import os
import warnings
from multiprocessing import Pool

import torch.autograd
from robosuite import load_controller_config
from sb3_contrib import TQC
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from demonstration.collect import collect_demonstrations
from demonstration.policies.bring_near.bring_near_policy import BringNearDemonstrationPolicy
from demonstration.policies.parameterized_reach.policy import ParameterizedReachDemonstrationPolicy
from env.bring_near import BringNearEnv
from env.cable_manipulation_base import CableManipulationBase
from env.goal_handler import HinDRLGoalHandler
from env.parameterized_reach import ParameterizedReachEnv
from eval import EvalVideoCallback
from model.hindrl_buffer import HinDRLReplayBuffer, HinDRLTQC
from utils import create_log_dir, save_dict, get_baseline_model_with_name, get_controller_config

# configs = [{
#     "demonstration_hdf5": "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/ParameterizedReach_2Waypoint/1656412922_273637/demo.hdf5",
#     "demonstration_policy": ParameterizedReachDemonstrationPolicy(),
#     "replay_buffer_type": 'HER',
#     "model_config":{
#             "policy": "MultiInputPolicy",
#             "buffer_size": 1000000,
#             "batch_size": 2048,
#             "gamma": 0.95,
#             "learning_rate": float(1e-3),
#             "tau": 0.05,
#             "verbose": 1,
#             "learning_starts": 250,
#             "policy_kwargs": {"net_arch": [512, 512, 512], "n_critics": 2},
#         },
#     "env_class": ParameterizedReachEnv,
#     "env_kwargs": {"horizon": 250,
#                    "goal_handler": HinDRLGoalHandler(
#                        "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/ParameterizedReach_2Waypoint/1656412922_273637/demo.hdf5",
#                        m=10, k=1)},
# },
#     {
#     "demonstration_hdf5": "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/ParameterizedReach_2Waypoint/1656412922_273637/demo.hdf5",
#     "demonstration_policy": ParameterizedReachDemonstrationPolicy(),
#     "replay_buffer_type": 'HinDRL',
#     "model_config": {
#         "policy": "MultiInputPolicy",
#         "buffer_size": 1000000,
#         "batch_size": 2048,
#         "gamma": 0.95,
#         "learning_rate": float(1e-3),
#         "tau": 0.05,
#         "verbose": 1,
#         "learning_starts": 1000,
#         "policy_kwargs": {"net_arch": [512, 512, 512], "n_critics": 2},
#     },
#     "env_class": ParameterizedReachEnv,
#     "env_kwargs": {"horizon": 250,
#                    "goal_handler": HinDRLGoalHandler(
#                        "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/ParameterizedReach_2Waypoint/1656412922_273637/demo.hdf5",
#                        m=10, k=1)},
#
#     }]


def _setup_training(demonstration_hdf5, config):
    print("Creating env...")
    env = make_vec_env(config['env_class'], env_kwargs=config['env_kwargs'])

    config['model_config']["env"] = env
    log_dir = create_log_dir(env.envs[0].name)
    config['model_config']['tensorboard_log'] = log_dir
    save_dict(config, os.path.join(log_dir, 'config.json'))

    if config['replay_buffer_type'] == "HinDRL":
        replay_buffer = HinDRLReplayBuffer(demonstration_hdf5, env,
                                           max_episode_length=config['env_kwargs']["horizon"],
                                           device="cuda")
    elif config['replay_buffer_type'] == "HER":
        replay_buffer = HerReplayBuffer(env, buffer_size=int(1e5), max_episode_length=config['env_kwargs']["horizon"],
                                        device="cuda")
    print("Env ready")
    model = HinDRLTQC(replay_buffer, **config['model_config'])
    return env, model, log_dir


def _collect_demonstration(env, demonstration_policy, episode_num):
    env_config = {
        'env_name': env.name
    }
    demo_path = collect_demonstrations(env, env_config=env_config, demonstration_policy=demonstration_policy,
                                 episode_num=episode_num)
    return demo_path


def train(_config):
    env, model, log_dir = _setup_training(_config["demonstration_hdf5"], _config)

    eval_env_config = _config['env_kwargs']
    eval_env_config['has_renderer'] = False
    eval_env_config['has_offscreen_renderer'] = True
    eval_env_config['use_camera_obs'] = False
    eval_env = Monitor(_config['env_class'](**eval_env_config))
    env.reset()
    eval_env.reset()
    # Use deterministic actions for evaluation
    eval_path = os.path.join(log_dir, 'train_eval')

    video_callback = EvalVideoCallback(eval_env, best_model_save_path=eval_path,
                                       log_path=eval_path, eval_freq=10000,
                                       deterministic=True, render=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path=eval_path,
                                 log_path=eval_path, eval_freq=1000, n_eval_episodes=10, deterministic=True,
                                 render=False)
    eval_callbacks = CallbackList([video_callback, eval_callback])

    model.learn(50000000, callback=eval_callbacks)


def run_parallel(_configs):
    pool = Pool(processes=len(_configs))

    # map the function to the list and pass
    # function and list_ranges as arguments
    pool.map(train, _configs)


if __name__ == '__main__':
    config = configs[0]
    # _collect_demonstration(config["env_class"](), config["demonstration_policy"], episode_num=30)
    train(config)
    # run_parallel(_configs=configs)