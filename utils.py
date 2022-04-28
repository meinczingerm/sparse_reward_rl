import os

from stable_baselines3 import *
from stable_baselines3.common.env_util import make_vec_env


def save_result_gif(env, model, path, filename, frames_to_save=100):
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
        frames.append(env.render(mode="rgb_array"))
        action, _state = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    save_frames_as_gif(frames, path=path, filename=filename)


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    from mujoco_py import GlfwContext
    GlfwContext(offscreen=True)
    from matplotlib import animation
    import matplotlib.pyplot as plt

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(os.path.join(path, filename), writer='imagemagick', fps=60)


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
    logs_path = os.path.join(get_project_root_path(), 'logs')
    current_dirs = os.listdir(logs_path)
    current_dirs_with_same_name = [dir for dir in current_dirs if dir.startswith(directory_name)]
    versions = [int(dir.split('_')[-1]) for dir in current_dirs_with_same_name]
    if len(versions) == 0:
        latest_version = 0
    else:
        latest_version = max(versions)
    directory_name_with_version = f"{directory_name}_{latest_version+1}"
    new_directory_path = os.path.join(get_project_root_path(), 'logs', directory_name_with_version)
    os.makedirs(new_directory_path)
    return new_directory_path


def save_dict(data_dict, path):
    import json

    with open(path, 'w') as f:
        json.dump(str(data_dict), f)


def setup_training(config):
    print("creating env")
    env = make_vec_env(config['env']['name'], n_envs=config['env']['env_num'])

    log_dir = create_log_dir('fetch_slide')
    config['model']['kwargs']['tensorboard_log'] = log_dir
    save_dict(config, os.path.join(log_dir, 'config.json'))

    print("env ready")
    model = get_baseline_model_with_name(config["model"]["name"], config["model"]["kwargs"], env=env)
    return env, model, log_dir





