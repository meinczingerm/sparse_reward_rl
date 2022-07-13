import os

import numpy as np
import torch as th
from sb3_contrib import TQC
from sb3_contrib.common.utils import quantile_huber_loss
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import polyak_update

from demonstration.policies.parameterized_reach.policy import ParameterizedReachDemonstrationPolicy
from env.parameterized_reach import ParameterizedReachEnv
from eval import EvalVideoCallback
from model.hindrl_buffer import HinDRLReplayBuffer
from train import _collect_demonstration
from utils import create_log_dir, save_dict


class DPGfD(TQC):
    def __init__(self, demonstration_hdf5, env, model_kwargs):

        self.lambda_bc = model_kwargs["lambda_bc"]
        del model_kwargs["lambda_bc"]
        super(DPGfD, self).__init__("MultiInputPolicy", env, **model_kwargs, device="cuda")
        self.replay_buffer = HinDRLReplayBuffer(demonstration_hdf5, env, n_sampled_goal=0,
                                                max_episode_length=env.envs[0].horizon, device="cuda")


    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, bc_losses, scaled_bc_losses, non_zero_bc_losses, num_from_demonstration =\
            [], [], [], [], [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())
            self.replay_buffer.ent_coef = ent_coef.item()

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute and cut quantiles at the next state
                # batch x nets x quantiles
                next_quantiles = self.critic_target(replay_data.next_observations, next_actions)

                # Sort and drop top k quantiles to control overestimation.
                n_target_quantiles = self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics
                next_quantiles, _ = th.sort(next_quantiles.reshape(batch_size, -1))
                next_quantiles = next_quantiles[:, :n_target_quantiles]

                # td error + entropy term
                target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_quantiles
                # Make target_quantiles broadcastable to (batch_size, n_critics, n_target_quantiles).
                target_quantiles.unsqueeze_(dim=1)

            # Get current Quantile estimates using action from the replay buffer
            current_quantiles = self.critic(replay_data.observations, replay_data.actions)
            # Compute critic loss, not summing over the quantile dimension as in the paper.
            critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            qf_pi = self.critic(replay_data.observations, actions_pi).mean(dim=2).mean(dim=1, keepdim=True)
            bc_loss = th.linalg.norm(actions_pi-replay_data.actions, axis=1, keepdims=True) * replay_data.is_demo
            non_zero_bc_losses.append(bc_loss[bc_loss > 0].mean().item())
            num_from_demonstration.append(bc_loss[bc_loss > 0].shape[0])
            scaling = 100/(self._n_updates + 1) # +1 is necessary to avoid "inf"
            scaling = 100
            scaled_bc_loss = bc_loss * self.lambda_bc * scaling
            actor_loss = (ent_coef * log_prob - qf_pi + scaled_bc_loss).mean()
            actor_losses.append(actor_loss.item())
            bc_losses.append(bc_loss.mean().item())
            scaled_bc_losses.append(scaled_bc_loss.mean().item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/bc_loss", np.mean(bc_losses))
        self.logger.record("train/non_zero_bc_loss", np.mean(non_zero_bc_losses))
        self.logger.record("train/num_of_demonstration_samples", np.mean(num_from_demonstration))
        self.logger.record("train/scaled_bc_loss", np.mean(scaled_bc_losses))
        self.logger.record("train/scaling_factor", scaling)
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


def train(_config):
    env = make_vec_env(config['env_class'], env_kwargs=config['env_kwargs'])
    if _config["regenerate_demonstrations"]:
        assert _config["demo_path"] is None
        expert_policy = _config["expert_policy"]
        demo_path = _collect_demonstration(env.envs[0].env, demonstration_policy=expert_policy,
                                           episode_num=_config["number_of_demonstrations"])
    else:
        demo_path = _config["demo_path"]

    log_dir = create_log_dir(env.envs[0].name)
    config['model_kwargs']['tensorboard_log'] = log_dir
    save_dict(config, os.path.join(log_dir, 'config.json'))

    model = DPGfD(demo_path, env, _config["model_kwargs"])

    eval_env_config = _config['env_kwargs']
    eval_env_config['has_renderer'] = False
    eval_env_config['has_offscreen_renderer'] = True
    eval_env_config['use_camera_obs'] = False
    eval_env = Monitor(_config['env_class'](**eval_env_config))
    env.reset()
    eval_env.reset()
    # Use deterministic actions for evaluation
    eval_path = os.path.join(log_dir, 'train_eval')

    video_callback = EvalVideoCallback(eval_env, best_model_save_path=eval_path,
                                       log_path=eval_path, eval_freq=10000,
                                       deterministic=True, render=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path=eval_path,
                                 log_path=eval_path, eval_freq=1000, n_eval_episodes=10, deterministic=True,
                                 render=False)
    eval_callbacks = CallbackList([video_callback, eval_callback])

    model.learn(50000000, callback=eval_callbacks)

if __name__ == '__main__':
    config = {"env_kwargs": {"number_of_waypoints": 1,
                             "horizon": 150},
              "env_class": ParameterizedReachEnv,
              "number_of_demonstrations": 100,
              "regenerate_demonstrations": False,
              "demo_path": "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/ParameterizedReach_1Waypoint/1657630758_2669017/demo.hdf5",
              "expert_policy": ParameterizedReachDemonstrationPolicy(),
              "model_kwargs": {"batch_size": 2048,
                        "lambda_bc": 1,
                        "policy_kwargs": {"net_arch": [32, 32]},
                        "learning_starts": 1,
                        }}

    train(config)