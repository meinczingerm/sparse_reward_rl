import os

import gym
from robosuite import load_controller_config
from sb3_contrib import TQC
from sb3_contrib.common.wrappers import TimeFeatureWrapper

from matplotlib import animation
import matplotlib.pyplot as plt
from stable_baselines3 import *
from stable_baselines3.common.callbacks import EventCallback
from stable_baselines3.common.utils import safe_mean


def save_result_gif(env, model, path, filename, frames_to_save=100, deterministic=True):
    """
    Saving result example as gif.
    :param env: gym environment
    :param model:
    :param path: path for saving the gif
    :param filename: name for the gif
    :param frames_to_save: length in frames for the gif
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
    print("Saving results.")
    obs = env.reset()
    frames = []
    for t in range(frames_to_save):
        # Render to frames buffer
        frames.append(env.render(mode='rgb'))
        action, _state = model.predict(obs, deterministic=deterministic)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    save_frames_as_gif(frames, path=path, filename=filename)


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(os.path.join(path, filename), writer='imagemagick', fps=60)
    plt.close()


def get_project_root_path():
    """
    Get the absolute path to the project root: .../playground
    :return: absolute path (str)
    """
    import pathlib
    import os
    work_dir = pathlib.Path().resolve()
    root_dir = os.path.join(str(work_dir).split('master_thesis')[0], 'master_thesis')

    return root_dir


def get_baseline_model_with_name(model_name, model_kwargs, env):
    model_class = globals()[model_name]
    return model_class(env=env, **model_kwargs)


def fix_env_creation():
    """Starting the env creation in debug mode results in a lock file within the mujoco_py package. This lock file
    blocks the creation of envs in later attempts so it has to be removed. This function is responsible for this forced
    removal. This might be only the case for my environment setup. The lock file path has to be set manually.
    (otherwise mujoco import is already hanging)
    Might be solved by https://github.com/openai/mujoco-py/issues/544#issuecomment-964267742
    """
    lock_file_path = \
        "/home/mark/anaconda3/envs/thesis/lib/python3.10/site-packages/mujoco_py/generated/mujocopy-buildlock.lock"
    if os.path.exists(lock_file_path):
        os.remove(lock_file_path)


def create_log_dir(directory_name):
    """Creates a directory with defined name and added version number ({directory_name}_v{x) inside the logs folder"""
    logs_path = os.path.join(get_project_root_path(), 'training_logs')
    current_dirs = os.listdir(logs_path)
    current_dirs_with_same_name = [dir for dir in current_dirs if dir.startswith(directory_name)]
    versions = [int(dir.split('_')[-1]) for dir in current_dirs_with_same_name]
    if len(versions) == 0:
        latest_version = 0
    else:
        latest_version = max(versions)
    directory_name_with_version = f"{directory_name}_{latest_version+1}"
    new_directory_path = os.path.join(get_project_root_path(), 'training_logs', directory_name_with_version)
    os.makedirs(new_directory_path)
    return new_directory_path


def save_dict(data_dict, path):
    import json
    from pathlib import Path

    path = Path(path)
    # create dir if necessary
    parent_path = path.parent.absolute()
    os.makedirs(parent_path, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(str(data_dict), f)


def get_controller_config(controller_type="IK"):
    """
    Returns default controller config dict.
    :param controller_type: type of controller to use, currently only "IK" (inverse kinematic) is supported
    :return:
    """
    if controller_type == "IK":
        controller_config = load_controller_config(default_controller="IK_POSE")
        controller_config['kp'] = 100
    else:
        raise NotImplementedError


class SaveBestModelAccordingRollouts(EventCallback):
    """
    Callback class for saving the model, when a new performance peak is reached according to the rollout success rate.
    """
    def __init__(self, best_model_save_path):
        """
        Init.
        :param best_model_save_path: path where the model will be saved
        """
        super(SaveBestModelAccordingRollouts, self).__init__()
        self.best_rollout_reward = 0
        self.best_model_save_path = best_model_save_path

    def _on_rollout_end(self) -> None:
        mean_reward = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        if mean_reward > self.best_rollout_reward:
            self.best_rollout_reward = mean_reward
            self.model.save(os.path.join(self.best_model_save_path, "best_rollout_model"))






