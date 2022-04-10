import os

import gym

from stable_baselines3 import A2C, HerReplayBuffer, DQN, SAC, DDPG
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from eval import EvalVideoCallback
from utils import save_result_gif, setup_training

from multiprocessing import Pool

configs =[
    {"model": {
        "name": "DDPG",
        "kwargs": {
            "policy": "MultiInputPolicy",
            "buffer_size": 1000000,
            "batch_size": 2048,
            "gamma": 0.95,
            "learning_rate": float(1e-3),
            "tau": 0.05,
            "verbose": 1,
            "replay_buffer_class": HerReplayBuffer,
            "replay_buffer_kwargs":{
                "online_sampling": True,
                "goal_selection_strategy": 'future',
                "n_sampled_goal": 4},
            "policy_kwargs": {"net_arch":[512, 512, 512], "n_critics":2}
        }

    },
    "env": {
        'name': "FetchPush-v1",
        'env_num': 1}},
    {"model": {
        "name": "DDPG",
        "kwargs": {
            "policy": "MultiInputPolicy",
            "buffer_size": 1000000,
            "batch_size": 2048,
            "gamma": 0.95,
            "learning_rate": float(1e-3),
            "tau": 0.05,
            "verbose": 1,
            "replay_buffer_class": HerReplayBuffer,
            "replay_buffer_kwargs":{
                "online_sampling": True,
                "goal_selection_strategy": 'future',
                "n_sampled_goal": 4},
            "policy_kwargs": {"net_arch":[512, 512, 512], "n_critics":2}
        }
    },
    "env": {
        'name': "FetchPickAndPlace-v1",
        'env_num': 1}},
    {"model": {
        "name": "DDPG",
        "kwargs": {
            "policy": "MultiInputPolicy",
            "buffer_size": 1000000,
            "batch_size": 2048,
            "gamma": 0.95,
            "learning_rate": float(1e-3),
            "tau": 0.05,
            "verbose": 1,
            "replay_buffer_class": HerReplayBuffer,
            "replay_buffer_kwargs":{
                "online_sampling": True,
                "goal_selection_strategy": 'future',
                "n_sampled_goal": 4},
            "policy_kwargs": {"net_arch":[512, 512, 512], "n_critics":2}
        }
    },
    "env": {
        'name': "FetchSlide-v1",
        'env_num': 1}},
]


def train(_config):
    env, model, log_dir = setup_training(_config)

    eval_env = gym.make(_config['env']['name'])
    # Use deterministic actions for evaluation
    eval_path = os.path.join(log_dir, 'train_eval')

    video_callback = EvalVideoCallback(eval_env, best_model_save_path=eval_path,
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



if __name__ == '__main__':
    train(configs[0])
    # run_parallel(_configs=configs)