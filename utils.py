def save_result_gif(env, model, path, filename):
    """
    Saving result example as gif.
    :param env: gym environment
    :param model:
    :param path: path for saving the gif
    :param filename: name for the gif
    :return:
    """
    obs = env.reset()
    frames = []
    for t in range(500):
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
    anim.save(path + filename, writer='imagemagick', fps=60)


def get_project_root_path():
    """
    Get the absolute path to the project root: .../playground
    :return: absolute path (str)
    """
    import pathlib
    import os
    work_dir = pathlib.Path().resolve()
    root_dir = os.path.join(str(work_dir).split('playground')[0], 'playground')

    return root_dir
