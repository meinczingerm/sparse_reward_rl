from sb3_contrib import TQC
from stable_baselines3 import HerReplayBuffer


class HinDRLReplayBuffer(HerReplayBuffer):
    def __init__(self, demonstration_hdf5, env, buffer_size=int(1e5), **kwargs):
        self.demonstration_hdf5 = demonstration_hdf5
        super().__init__(env, buffer_size, **kwargs)


class HinDRLTQC(TQC):
    def __init__(self, replay_buffer: HinDRLReplayBuffer, **kwargs):

        self.replay_buffer = replay_buffer
        super(HinDRLTQC, self).__init__(**kwargs)


