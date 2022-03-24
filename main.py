import gym

from stable_baselines3 import A2C, HER, DQN, SAC

from utils import save_result_gif, get_project_root_path


def main():
    env = gym.make('Reacher-v2')


    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log="/home/mark/tum/2022ss/master_thesis/playground/logs")
    model.learn(total_timesteps=int(100000))

    save_result_gif(env, model, '')




if __name__ == '__main__':

    main()