import os
import warnings

from robosuite import load_controller_config
from sb3_contrib import TQC
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from demonstration.collect import collect_demonstrations
from demonstration.policies.parameterized_reach.policy import ParameterizedReachDemonstrationPolicy
from env.cable_manipulation_base import CableManipulationBase
from env.parameterized_reach import ParameterizedReachEnv
from eval import EvalVideoCallback
from model.hindrl_buffer import HinDRLReplayBuffer, HinDRLTQC, HerReplayBufferWithDemonstrationGoals
from utils import create_log_dir, save_dict, get_baseline_model_with_name, get_controller_config

config = {
    "demonstration_hdf5": "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/ParameterizedReach_2Waypoint/1655978351_2683008/demo.hdf5",
    "model_config":{
            "policy": "MultiInputPolicy",
            "buffer_size": 1000000,
            "batch_size": 2048,
            "gamma": 0.95,
            "learning_rate": float(1e-3),
            "tau": 0.05,
            "verbose": 1,
            "learning_starts": 200,
            "policy_kwargs": {"net_arch": [512, 512, 512], "n_critics": 2},
        },
    "env_class": ParameterizedReachEnv,
    "env_kwargs": {"horizon": 200},
    "demonstration_policy": ParameterizedReachDemonstrationPolicy()
}


def _setup_training(demonstration_hdf5, config):
    print("Creating env...")
    env = make_vec_env(config['env_class'], env_kwargs=config['env_kwargs'])

    config['model_config']["env"] = env
    log_dir = create_log_dir(env.envs[0].name)
    config['model_config']['tensorboard_log'] = log_dir
    save_dict(config, os.path.join(log_dir, 'config.json'))

    warnings.warn("This is HNDRL now. Train with the other !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    replay_buffer = HinDRLReplayBuffer(demonstration_hdf5, env,
                                       max_episode_length=config['env_kwargs']["horizon"],
                                       device="cuda")
    print("Env ready")
    model = HinDRLTQC(replay_buffer, **config['model_config'])
    return env, model, log_dir


def _collect_demonstration(env, demonstration_policy):
    env_config = {
        'env_name': env.name
    }
    collect_demonstrations(env, env_config=env_config, demonstration_policy=demonstration_policy, episode_num=10)


def train(_config):
    env, model, log_dir = _setup_training(_config["demonstration_hdf5"], _config)

    eval_env_config = _config['env_kwargs']
    eval_env_config['has_renderer'] = False
    eval_env_config['has_offscreen_renderer'] = True
    eval_env = Monitor(_config['env_class'](**eval_env_config))
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


if __name__ == '__main__':
    # _collect_demonstration()
    train(config)
    # run_parallel(_configs=)