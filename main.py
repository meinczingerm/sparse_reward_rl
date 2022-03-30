import os

import gym

from stable_baselines3 import A2C, HerReplayBuffer, DQN, SAC

from eval import CustomEvalCallback
from utils import save_result_gif, setup_training

from multiprocessing import Pool

configs =[
    {"model": {
        "name": "DDPG",
        "kwargs": {
            "policy": "MultiInputPolicy",
            "verbose": 1,
            "buffer_size": 100000,
            "replay_buffer_class": HerReplayBuffer,
            "replay_buffer_kwargs": {"goal_selection_strategy": "final",
                                     "n_sampled_goal": 1}
        }
    },
    "env": {
        'name': "FetchPush-v1",
        'env_num': 1}},
    {"model": {
        "name": "DDPG",
        "kwargs": {
            "policy": "MultiInputPolicy",
            "verbose": 1,
            "buffer_size": 100000,
            "replay_buffer_class": HerReplayBuffer,
            "replay_buffer_kwargs": {"goal_selection_strategy": "final",
                                     "n_sampled_goal": 2}
        }
    },
    "env": {
        'name': "FetchPickAndPlace-v1",
        'env_num': 1}},
    {"model": {
        "name": "DDPG",
        "kwargs": {
            "policy": "MultiInputPolicy",
            "verbose": 1,
            "buffer_size": 100000,
            "replay_buffer_class": HerReplayBuffer,
            "replay_buffer_kwargs": {"goal_selection_strategy": "final"}
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
    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=eval_path,
                                 log_path=eval_path, eval_freq=100000,
                                 deterministic=True, render=False)

    model.learn(50000000, callback=eval_callback)

def run_parallel(_configs):
    pool = Pool(processes=len(_configs))

    # map the function to the list and pass
    # function and list_ranges as arguments
    pool.map(train, _configs)



if __name__ == '__main__':
    run_parallel(_configs=configs)