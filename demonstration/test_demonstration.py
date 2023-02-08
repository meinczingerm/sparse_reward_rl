import h5py
import numpy as np
from robosuite import load_controller_config

from env.robot_envs.cable_insertion import CableInsertionEnv
from env.robot_envs.cable_manipulation_base import CableManipulationBase


class ReplayedActionPolicy:
    """
    Policy replaying actions from a demonstration (independendent of observation).
    Note: because of the not entirely deterministic behavior of Mujoco, it might shift from the actual recorded
    observation, and so the actions can be irrelevant. It is only used for checking the demonstrations.
    """
    def __init__(self, hdf5_file):
        with h5py.File(hdf5_file, "r") as f:
            self.actions = np.array(f["data"]["demo_1"]["actions"])

        self.i = 0

    def step(self, observation):
        action = self.actions[self.i]
        self.i += 1
        return action

    def reset(self):
        self.i = 0


def test_ik_replay(hdf5_file="/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/1655196314_930382/demo.hdf5"):
    env = CableInsertionEnv(has_renderer=True)

    demonstration_policy = ReplayedActionPolicy(hdf5_file)

    action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
    obs, _, done, _ = env.step(action)

    while True:
        action = demonstration_policy.step(obs)
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            print(done)
            env.reset()
            demonstration_policy.reset()


if __name__ == '__main__':
    test_ik_replay("/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/BringNear/1655990392_916894/demo.hdf5")