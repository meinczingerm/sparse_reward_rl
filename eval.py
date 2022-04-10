from copy import deepcopy

import gym

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

# Separate evaluation env
from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize, sync_envs_normalization

from utils import save_result_gif


class EvalVideoCallback(EvalCallback):
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.custom_eval()

        return True

    def custom_eval(self):
        save_result_gif(self.eval_env, self.model, self.log_path, filename=f"progress_video{self.num_timesteps}.gif",
                        frames_to_save=200)