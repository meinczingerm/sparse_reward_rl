from copy import deepcopy

import gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

# Separate evaluation env
from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize, sync_envs_normalization

from env.goal_handler import GoalSelectionMethod
from utils import save_result_gif


class EvalVideoCallback(EvalCallback):
    """Callback used by the lightning module. It saves .gif-s during the evaluation."""
    def __init__(self, video_on_percentage_goal, eval_env, deterministic=True, frames_to_save=200, **kwargs):
        """
        Init.
        :param video_on_percentage_goal: If not None, then videos will be generated with goals given at percentages of
                                         the rollout. For example for the value 0.1, 10 video will be generated, where
                                         the first video is with a goal from a rollout at 10% the 2nd video with 20%
                                         etc.
        :param deterministic: if True, then the policy behaves determinstic
        :param frames_to_save: number of frames to save into the .gif
        :param eval_env: evaluation env instance
        :param kwargs: not used
        """
        super(EvalVideoCallback, self).__init__(eval_env, **kwargs)
        self.video_on_percentage_goal = video_on_percentage_goal
        self.deterministic = deterministic
        self.frames_to_save = frames_to_save

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.custom_eval()

        return True

    def custom_eval(self):
        if self.video_on_percentage_goal is None:
            save_result_gif(self.eval_env, self.model, self.log_path, filename=f"progress_video{self.num_timesteps}.gif",
                            frames_to_save=self.frames_to_save, deterministic=self.deterministic)
        else:
            assert len(self.eval_env.envs) == 1
            original_goal_handler_goal_selection_method = self.eval_env.envs[0].env.goal_handler.goal_selection
            self.eval_env.envs[0].env.goal_handler.goal_selection = GoalSelectionMethod.Percentage
            percentages = list(np.arange(self.video_on_percentage_goal, 1, self.video_on_percentage_goal))
            percentages.append(1)
            for percentage in percentages:
                self.eval_env.envs[0].env.goal_handler.percentage_for_rollout_goal = percentage
                save_result_gif(self.eval_env, self.model, self.log_path,
                                filename=f"progress_video{self.num_timesteps}_percentage{percentage:.2f}.gif",
                                frames_to_save=self.frames_to_save, deterministic=self.deterministic)

            self.eval_env.envs[0].env.goal_handler.goal_selection = original_goal_handler_goal_selection_method
            self.eval_env.envs[0].env.goal_handler.percentage_for_rollout_goal = -1

