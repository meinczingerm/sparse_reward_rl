import os
from multiprocessing import Pool

from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from tools.eval import EvalVideoCallback
from tools.utils import create_log_dir, save_dict, get_baseline_model_with_name


# Example configuration
configs =[
    {"model": {
        "name": "TQC",
        "kwargs": {
            "policy": "MultiInputPolicy",
            "buffer_size": 1000000,
            "batch_size": 2048,
            "gamma": 0.95,
            "learning_rate": float(1e-3),
            "tau": 0.05,
            "verbose": 1,
            "learning_starts": 10000,
            "replay_buffer_class": HerReplayBuffer,
            "replay_buffer_kwargs":{
                "online_sampling": True,
                "goal_selection_strategy": 'future',
                "n_sampled_goal": 4},
            "policy_kwargs": {"net_arch": [512, 512, 512], "n_critics":2},
        }

    },
    "env": {
        'name': "FetchSlide-v1",
        'env_num': 1,
        'env_wrapper': TimeFeatureWrapper,
        'env_wrapper_kwargs': {"max_steps": 50},
    }},
    {"model": {
        "name": "TQC",
        "kwargs": {
            "policy": "MultiInputPolicy",
            "buffer_size": 1000000,
            "batch_size": 2048,
            "gamma": 0.95,
            "learning_rate": float(1e-3),
            "tau": 0.05,
            "verbose": 1,
            "learning_starts": 10000,
            "replay_buffer_class": HerReplayBuffer,
            "replay_buffer_kwargs":{
                "online_sampling": True,
                "goal_selection_strategy": 'future',
                "n_sampled_goal": 4},
            "policy_kwargs": {"net_arch":[512, 512, 512], "n_critics":2},
        }

    },
    "env": {
        'name': "FetchSlide-v1",
        'env_num': 1,
        'env_wrapper': TimeFeatureWrapper,
        'env_wrapper_kwargs': {"max_steps": 50},
    }},
    {"model": {
        "name": "TQC",
        "kwargs": {
            "policy": "MultiInputPolicy",
            "buffer_size": 1000000,
            "batch_size": 2048,
            "gamma": 0.95,
            "learning_rate": float(1e-3),
            "tau": 0.05,
            "verbose": 1,
            "learning_starts": 10000,
            "replay_buffer_class": HerReplayBuffer,
            "replay_buffer_kwargs":{
                "online_sampling": True,
                "goal_selection_strategy": 'future',
                "n_sampled_goal": 4},
            "policy_kwargs": {"net_arch":[512, 512, 512], "n_critics":2},
        }
    },
    "env": {
        'name': "FetchSlide-v1",
        'env_num': 1,
        'env_wrapper': TimeFeatureWrapper,
        'env_wrapper_kwargs': {"max_steps": 50},
    }},
]


def train(_config):
    env, model, log_dir = setup_training(_config)

    eval_env = make_vec_env(_config['env']['name'], n_envs=_config['env']['env_num'], wrapper_class=_config['env']['env_wrapper'],
                       wrapper_kwargs=_config['env']['env_wrapper_kwargs'])
    # Use deterministic actions for evaluation
    eval_path = os.path.join(log_dir, 'train_eval')

    video_callback = EvalVideoCallback(None, eval_env, best_model_save_path=eval_path,
                                      log_path=eval_path, eval_freq=100000,
                                      deterministic=True, render=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path=eval_path,
                             log_path=eval_path, eval_freq=1000, n_eval_episodes=10, deterministic=True, render=False)
    eval_callbacks = CallbackList([video_callback, eval_callback])

    model.learn(50000000, callback=eval_callbacks)


def run_parallel(_configs):
    pool = Pool(processes=len(_configs))

    # map the function to the list and pass
    # function and list_ranges as arguments
    pool.map(train, _configs)


def setup_training(config):
    print("creating env")
    env = make_vec_env(config['env']['name'], n_envs=config['env']['env_num'], wrapper_class=config['env']['env_wrapper'],
                       wrapper_kwargs=config['env']['env_wrapper_kwargs'])

    log_dir = create_log_dir(config['env']['name'])
    config['model']['kwargs']['tensorboard_log'] = log_dir
    save_dict(config, os.path.join(log_dir, 'config.json'))

    print("env ready")
    model = get_baseline_model_with_name(config["model"]["name"], config["model"]["kwargs"], env=env)
    return env, model, log_dir



if __name__ == '__main__':
    train(configs[0])
    # run_parallel(_configs=configs)