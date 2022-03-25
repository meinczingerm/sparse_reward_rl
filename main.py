import os

import gym

from stable_baselines3 import A2C, HER, DQN, SAC

from utils import save_result_gif, setup_training

config = {
    "model": {
        "name": "SAC",
        "kwargs": {
            "policy": "MlpPolicy",
            "verbose": 1,
            "buffer_size": 100
        }
    },
    "env": "Reacher-v2"
}


def train():
    env, model, log_dir = setup_training(config)

    print("Starting to learn")
    model.learn(total_timesteps=int(100))
    print("Finished learning")

    save_result_gif(env, model, log_dir, 'result_video.gif')




if __name__ == '__main__':
    train()