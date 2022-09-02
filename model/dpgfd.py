import os
from multiprocessing import Pool

import numpy as np
import torch as th
from sb3_contrib import TQC
from sb3_contrib.common.utils import quantile_huber_loss
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import polyak_update

from demonstration.policies.gridworld.grid_pick_and_place_policy import GridPickAndPlacePolicy
from demonstration.policies.parameterized_reach.policy import ParameterizedReachDemonstrationPolicy
from env.grid_world_envs.pick_and_place import GridPickAndPlace
from env.robot_envs.parameterized_reach import ParameterizedReachEnv
from eval import EvalVideoCallback
from model.hindrl_buffer import HinDRLReplayBuffer
from train import _collect_demonstration
from utils import create_log_dir, save_dict


class DPGfD(TQC):
    def __init__(self, demonstration_hdf5=None, env=None, model_kwargs=None, policy="MultiInputPolicy", device="cuda",
                 _init_setup_model=False):
        if _init_setup_model is not False:
            raise NotImplementedError

        if type(policy) is not str:
            # reloading from checkpoint without replay buffer
            super(DPGfD, self).__init__(policy, env, device=device)
        else:
            # normal init
            self.lambda_bc = model_kwargs["lambda_bc"]
            del model_kwargs["lambda_bc"]
            buffer_size = model_kwargs["buffer_size"]
            self.max_demo_ratio = model_kwargs["max_demo_ratio"]
            del model_kwargs["max_demo_ratio"]
            self.reach_zero = model_kwargs["reach_zero"]
            del model_kwargs["reach_zero"]
            goal_selection_strategy = model_kwargs["goal_selection_strategy"]
            del model_kwargs["goal_selection_strategy"]
            n_sampled_goal = model_kwargs["n_sampled_goal"]
            del model_kwargs["n_sampled_goal"]
            model_id = model_kwargs.get("model_id", -1)
            if "model_id" in model_kwargs.keys():
                del model_kwargs["model_id"]

            super(DPGfD, self).__init__(policy, env, **model_kwargs, device=device)
            self.replay_buffer = HinDRLReplayBuffer(demonstration_hdf5, env, n_sampled_goal=n_sampled_goal,
                                                    max_episode_length=env.envs[0].horizon, device="cuda",
                                                    buffer_size=int(buffer_size),
                                                    goal_selection_strategy=goal_selection_strategy)
        print(f"Model initialized {model_id}")


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
            scaling = self.lambda_bc
            scaled_bc_loss = bc_loss * scaling
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
        self.logger.record("train/non_zero_reward_sample", replay_data.rewards.sum().item())
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        demo_to_rollout_ratio = max([self.max_demo_ratio * (1 - self.num_timesteps / self.reach_zero), 0])
        self.replay_buffer.demo_to_rollout_sample_ratio = demo_to_rollout_ratio


def train(_config):
    env = make_vec_env(_config['env_class'], env_kwargs=_config['env_kwargs'], n_envs=1)
    if _config["regenerate_demonstrations"]:
        assert _config["demo_path"] is None
        expert_policy = _config["expert_policy"]
        demo_path = _collect_demonstration(env.envs[0].env, demonstration_policy=expert_policy,
                                           episode_num=_config["number_of_demonstrations"])
    else:
        demo_path = _config["demo_path"]

    log_dir = create_log_dir(env.envs[0].name)
    _config['model_kwargs']['tensorboard_log'] = log_dir
    save_dict(_config, os.path.join(log_dir, 'config.json'))

    model = DPGfD(demo_path, env, _config["model_kwargs"])

    eval_env_config = _config['env_kwargs']

    if not isinstance(env.envs[0].env, GridPickAndPlace):
        eval_env_config['has_renderer'] = False
        eval_env_config['has_offscreen_renderer'] = True
        eval_env_config['use_camera_obs'] = False
    eval_env = Monitor(_config['env_class'](**eval_env_config))
    env.reset()
    eval_env.reset()
    # Use deterministic actions for evaluation
    eval_path = os.path.join(log_dir, 'train_eval')

    # video_callback = EvalVideoCallback(eval_env, best_model_save_path=eval_path,
    #                                    log_path=eval_path, eval_freq=50000,
    #                                    deterministic=True, render=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path=eval_path,
                                 log_path=eval_path, eval_freq=10000, n_eval_episodes=50, deterministic=False,
                                 render=False)
    eval_callbacks = CallbackList([eval_callback])

    model.learn(50000000, callback=eval_callbacks)

def run_parallel(_configs):
    pool = Pool(processes=len(_configs))

    # map the function to the list and pass
    # function and list_ranges as arguments
    pool.map(train, _configs)


def load_and_eval(log_dir, eval_episodes=100):
    checkpoint_path = os.path.join(log_dir, "train_eval", "best_model.zip")
    env = ParameterizedReachEnv(number_of_waypoints=2, horizon=200)

    model = DPGfD.load(checkpoint_path, env=env)

    rewards, ep_lengths = evaluate_policy(model, env, n_eval_episodes=eval_episodes, return_episode_rewards=True)

    eval_result = {"mean_length": sum(ep_lengths)/len(ep_lengths),
                   "success_rate": sum(rewards)/len(rewards),
                   "n_eval_episodes": eval_episodes}
    save_dict(eval_result, os.path.join(log_dir, "eval_result2.json"))



if __name__ == '__main__':
    # configs = [{"env_kwargs": {"number_of_waypoints": 2,
    #                            "horizon": 200},
    #             "env_class": ParameterizedReachEnv,
    #             "number_of_demonstrations": 1000,
    #             "regenerate_demonstrations": False,
    #             "demo_path": "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/ParameterizedReach_2Waypoint/1000_1658869650_9963062/demo.hdf5",
    #             "expert_policy": ParameterizedReachDemonstrationPolicy(),
    #             "model_kwargs": {"batch_size": 8192,
    #                              "learning_rate": 1e-3,
    #                              "lambda_bc": 1,
    #                              "policy_kwargs": {"net_arch": [64, 64]},
    #                              "learning_starts": 5000,
    #                              "buffer_size": int(1e6),
    #                              "max_demo_ratio": 0.01,
    #                              "reach_zero": 5e5,
    #                              "n_sampled_goal": 0,
    #                              "goal_selection_strategy": None,
    #                              "tau": 0.005,
    #                              }},
    #            {"env_kwargs": {"number_of_waypoints": 2,
    #                            "horizon": 200},
    #             "env_class": ParameterizedReachEnv,
    #             "number_of_demonstrations": 1000,
    #             "regenerate_demonstrations": False,
    #             "demo_path": "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/ParameterizedReach_2Waypoint/1000_1658869650_9963062/demo.hdf5",
    #             "expert_policy": ParameterizedReachDemonstrationPolicy(),
    #             "model_kwargs": {"batch_size": 8192,
    #                              "learning_rate": 1e-3,
    #                              "lambda_bc": 1,
    #                              "policy_kwargs": {"net_arch": [64, 64]},
    #                              "learning_starts": 5000,
    #                              "buffer_size": int(1e6),
    #                              "max_demo_ratio": 0.025,
    #                              "reach_zero": 5e5,
    #                              "n_sampled_goal": 0,
    #                              "goal_selection_strategy": None,
    #                              "tau": 0.005,
    #                              }},
    #            ]

    configs = [{"env_kwargs": {"size": 5,
                               "number_of_objects": 1,
                               "horizon": 20},
                "env_class": GridPickAndPlace,
                "number_of_demonstrations": 1000,
                "regenerate_demonstrations": False,
                "demo_path": "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/GridPickAndPlace/1000_1661520571_830977/demo.hdf5",
                "expert_policy": GridPickAndPlacePolicy(),
                "model_kwargs": {"batch_size": 8192,
                                 "learning_rate": 1e-3,
                                 "lambda_bc": 1,
                                 "policy_kwargs": {"net_arch": [20, 20]},
                                 "learning_starts": 5000,
                                 "buffer_size": int(1e5),
                                 "max_demo_ratio": 0.1,
                                 "reach_zero": 1e6,
                                 "n_sampled_goal": 0,
                                 "goal_selection_strategy": None,
                                 "tau": 0.005,
                                 }},
               {"env_kwargs": {"size": 5,
                               "number_of_objects": 1,
                               "horizon": 20},
                "env_class": GridPickAndPlace,
                "number_of_demonstrations": 1000,
                "regenerate_demonstrations": False,
                "demo_path": "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/GridPickAndPlace/1000_1661520571_830977/demo.hdf5",
                "expert_policy": GridPickAndPlacePolicy(),
                "model_kwargs": {"batch_size": 8192,
                                 "learning_rate": 1e-3,
                                 "lambda_bc": 1,
                                 "policy_kwargs": {"net_arch": [32, 32]},
                                 "learning_starts": 5000,
                                 "buffer_size": int(1e5),
                                 "max_demo_ratio": 0.1,
                                 "reach_zero": 1e6,
                                 "n_sampled_goal": 0,
                                 "goal_selection_strategy": None,
                                 "tau": 0.005,
                                 }},
               ]

    run_parallel(configs)
    # train(configs[0])
    # load_and_eval("/home/mark/tum/2022ss/thesis/master_thesis/training_logs/ParameterizedReach_2Waypoint_279")