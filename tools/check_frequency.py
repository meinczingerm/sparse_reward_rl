import shutil
from pathlib import Path

import h5py
import numpy as np

from demonstration.collect import gather_demonstrations
from env.robot_envs.parameterized_reach import ParameterizedReachEnv
from demonstration.policies.parameterized_reach.policy import ParameterizedReachDemonstrationPolicy


def check_avg_len(env, demonstration_policy, num_of_averaging_steps=2):
    demo_path = gather_demonstrations(env, demonstration_policy=demonstration_policy,
                           episode_num=num_of_averaging_steps)
    demo_path = Path(demo_path)
    demo_dir_path = demo_path.parent.absolute()

    demo_lengths = []
    with h5py.File(demo_path, "r") as f:
        for demo_id in f["data"].keys():
            actions = np.array(f["data"][demo_id]["actions"])
            demo_lengths.append(actions.shape[0])

    shutil.rmtree(demo_dir_path)

    avg_len = sum(demo_lengths) / len(demo_lengths)
    return avg_len


def check_frequency():
    """
    Returns the average len (number of steps) for a demonstration with given config.
    :return:
    """
    env = ParameterizedReachEnv()
    demonstration_policy = ParameterizedReachDemonstrationPolicy()

    avg_len_20 = check_avg_len(env, demonstration_policy, num_of_averaging_steps=200)


    env = ParameterizedReachEnv(control_freq=10)
    demonstration_policy = ParameterizedReachDemonstrationPolicy()

    avg_len_10 = check_avg_len(env, demonstration_policy, num_of_averaging_steps=200)

    print(f"With control freq=20: {avg_len_20}")
    print(f"With control freq=10: {avg_len_10}")


if __name__ == '__main__':
    check_frequency()