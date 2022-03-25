import os

import gym

from stable_baselines3 import A2C, HER, DQN, SAC

from eval import CustomEvalCallback
from utils import save_result_gif, setup_training

config = {
    "model": {
        "name": "DQN",
        "kwargs": {
            "policy": "MlpPolicy",
            "verbose": 1,
            "buffer_size": 10000
        }
    },
    "env": "LunarLander-v2"
}


def train():
    env, model, log_dir = setup_training(config)

    eval_env = gym.make(config['env'])
    # Use deterministic actions for evaluation
    eval_path = os.path.join(log_dir, 'train_eval')
    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=eval_path,
                                 log_path=eval_path, eval_freq=100000,
                                 deterministic=True, render=False)

    model.learn(5000000, callback=eval_callback)

if __name__ == '__main__':
    train()