import os

import gym

from stable_baselines3 import A2C, HerReplayBuffer, DQN, SAC

from eval import CustomEvalCallback
from utils import save_result_gif, setup_training

config = {
    "model": {
        "name": "SAC",
        "kwargs": {
            "policy": "MultiInputPolicy",
            "verbose": 1,
            "buffer_size": 10000,
            "replay_buffer_class": HerReplayBuffer,
            "replay_buffer_kwargs": {"goal_selection_strategy": "final"}
        }
    },
    "env": {
        'name': "FetchSlide-v1",
        'env_num': 1}
}


def train():
    env, model, log_dir = setup_training(config)

    eval_env = gym.make(config['env']['name'])
    # Use deterministic actions for evaluation
    eval_path = os.path.join(log_dir, 'train_eval')
    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=eval_path,
                                 log_path=eval_path, eval_freq=1000,
                                 deterministic=True, render=False)

    model.learn(50000000, callback=eval_callback)

if __name__ == '__main__':
    train()