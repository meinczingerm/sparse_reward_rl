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

from demonstration.collect import gather_demonstrations
from demonstration.policies.parameterized_reach.fixed_parameterized_reach_policy import FixedParameterizedReachDemonstrationPolicy
from env.goal_handler import HinDRLGoalHandler
from env.grid_world_envs.pick_and_place import GridPickAndPlace
from env.robot_envs.fixed_parameterized_reach import FixedParameterizedReachEnv
from tools.eval import EvalVideoCallback
from model.HinDRL.hindrl_buffer import HinDRLReplayBuffer, HinDRLSamplingStrategy
from tools.utils import create_log_dir, save_dict, SaveBestModelAccordingRollouts


class HinDRL(TQC):
    """HinDRL (a Hindsight Goal Selection for Demo-Driven RL) model (based on the work of Davchev et al:
    https://arxiv.org/pdf/2112.00597.pdf). The model is a combination of a TQC RL algorithm and a modified
    HinDRL buffer. For clear difference to the original HinDRL model we suggest reading the Thesis itself."""
    def __init__(self, demonstration_hdf5=None, env=None, model_kwargs=None, policy="MultiInputPolicy", device="cuda",
                 _init_setup_model=False):
        if _init_setup_model is not False:
            raise NotImplementedError

        if type(policy) is not str:
            # reloading from checkpoint without replay buffer
            super(HinDRL, self).__init__(policy, env, device=device)
        else:
            # normal init
            self.lambda_bc = model_kwargs["lambda_bc"]
            del model_kwargs["lambda_bc"]
            buffer_size = model_kwargs["buffer_size"]
            self.max_demo_ratio = model_kwargs["max_demo_ratio"]
            del model_kwargs["max_demo_ratio"]
            self.reach_zero = model_kwargs["reach_zero"]
            del model_kwargs["reach_zero"]
            hindrl_sampling_strategy = model_kwargs["hindrl_sampling_strategy"]
            del model_kwargs["hindrl_sampling_strategy"]
            n_sampled_goal = model_kwargs["n_sampled_goal"]
            del model_kwargs["n_sampled_goal"]
            epsilon_filtering = model_kwargs["epsilon_filtering"]
            del model_kwargs["epsilon_filtering"]

            model_id = model_kwargs.get("model_id", -1)
            if "model_id" in model_kwargs.keys():
                del model_kwargs["model_id"]
            if "union_sampling_ratio" in model_kwargs.keys():
                union_sampling_ratio = model_kwargs["union_sampling_ratio"]
                del model_kwargs["union_sampling_ratio"]
            else:
                union_sampling_ratio = None

            super(HinDRL, self).__init__(policy, env, **model_kwargs, device=device)
            self.replay_buffer = HinDRLReplayBuffer(demonstration_hdf5, env, n_sampled_goal=n_sampled_goal,
                                                    max_episode_length=env.envs[0].horizon, device="cuda",
                                                    buffer_size=int(buffer_size),
                                                    hindrl_sampling_strategy=hindrl_sampling_strategy,
                                                    union_sampling_ratio=union_sampling_ratio,
                                                    epsilon_filtering=epsilon_filtering)
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

        sum_her_rewards = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data, her_indices = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # logging her rewards
            sum_her_rewards.append(replay_data.rewards[her_indices, 0].sum().item())

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
        self.logger.record("train/her_rewards_sum", np.mean(sum_her_rewards))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        demo_to_rollout_ratio = max([self.max_demo_ratio * (1 - self.num_timesteps / self.reach_zero), 0])
        self.replay_buffer.relabeling_ratio = demo_to_rollout_ratio


def train(_config):
    env = make_vec_env(_config['env_class'], env_kwargs=_config['env_kwargs'], n_envs=1)
    assert len(env.envs) == 1
    if _config["regenerate_demonstrations"]:
        assert _config["demo_path"] is None
        expert_policy = _config["expert_policy"]
        if hasattr(expert_policy, "add_env"):
            expert_policy.add_env(env.envs[0].env)
        demo_path = gather_demonstrations(env.envs[0].env, demonstration_policy=expert_policy,
                                          episode_num=_config["number_of_demonstrations"])
    else:
        demo_path = _config["demo_path"]

    eval_env_config = _config['env_kwargs']

    if "goal_handler" in _config.keys():
        if _config["goal_handler"] is not None:
            if "demonstration_hdf5" in _config["goal_handler_kwargs"] and\
                    _config["goal_handler_kwargs"]["demonstration_hdf5"] is None:
                _config["goal_handler_kwargs"]["demonstration_hdf5"] = demo_path
            goal_handler = _config["goal_handler"](**_config["goal_handler_kwargs"])
            env.envs[0].add_goal_handler(goal_handler)
            eval_env_config["goal_handler"] = goal_handler

    log_dir = create_log_dir(env.envs[0].name)
    _config['model_kwargs']['tensorboard_log'] = log_dir
    save_dict(_config, os.path.join(log_dir, 'config.json'))

    model = HinDRL(demo_path, env, _config["model_kwargs"])

    if not isinstance(env.envs[0].env, GridPickAndPlace):
        eval_env_config['has_renderer'] = False
        eval_env_config['has_offscreen_renderer'] = True
        eval_env_config['use_camera_obs'] = False
    eval_env = Monitor(_config['env_class'](**eval_env_config))
    env.reset()
    eval_env.reset()
    # Use deterministic actions for evaluation
    eval_path = os.path.join(log_dir, 'train_eval')

    video_callback = EvalVideoCallback(0.5, eval_env, best_model_save_path=eval_path,
                                       log_path=eval_path, eval_freq=_config['eval_freq'],
                                       deterministic=False, frames_to_save=eval_env_config["horizon"], render=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path=eval_path,
                                 log_path=eval_path, eval_freq=_config['eval_freq'],
                                 n_eval_episodes=_config['n_eval_episodes'], deterministic=True,
                                 render=False)
    best_rollout_reward_save_callback = SaveBestModelAccordingRollouts(best_model_save_path=eval_path)
    eval_callbacks = CallbackList([eval_callback, video_callback, best_rollout_reward_save_callback])

    model.learn(50000000, callback=eval_callbacks)

def run_parallel(_configs):
    pool = Pool(processes=len(_configs))

    # map the function to the list and pass
    # function and list_ranges as arguments
    pool.map(train, _configs)


def load_and_eval(log_dir, eval_episodes=100):
    checkpoint_path = os.path.join(log_dir, "train_eval", "best_rollout_model.zip")
    env = FixedParameterizedReachEnv(horizon=300, number_of_waypoints=2,
                       waypoints= [np.array([0.58766, 0.26816, 0.37820, 2.89549, 0.03567, -0.39348]),
                                     np.array([0.42493, 0.07166, 0.36318, 2.88426, 0.12777, 0.35920])])

    model = HinDRL.load(checkpoint_path, env=env)

    rewards, ep_lengths = evaluate_policy(model, env, n_eval_episodes=eval_episodes, return_episode_rewards=True, deterministic=False)

    eval_result = {"mean_length": sum(ep_lengths)/len(ep_lengths),
                   "success_rate": sum(rewards)/len(rewards),
                   "n_eval_episodes": eval_episodes}
    save_dict(eval_result, os.path.join(log_dir, "eval_result2.json"))



if __name__ == '__main__':

    # Example configs
    configs = [
        {"env_kwargs": {"horizon": 300, "number_of_waypoints": 2,
                        "waypoints": [np.array([0.58766, 0.26816, 0.37820, 2.89549, 0.03567, -0.39348]),
                                      np.array([0.42493, 0.07166, 0.36318, 2.88426, 0.12777, 0.35920])]
                        },
         "goal_handler": HinDRLGoalHandler,
         "goal_handler_kwargs": {
             "demonstration_hdf5": "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/FixedParameterizedReach_2Waypoint/1_1673455320_229703/demo.hdf5",
             "m": 5, "k": 0},
         "env_class": FixedParameterizedReachEnv,
         "number_of_demonstrations": 1,
         "regenerate_demonstrations": False,
         "demo_path": "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/FixedParameterizedReach_2Waypoint/1_1673455320_229703/demo.hdf5",
         "expert_policy": FixedParameterizedReachDemonstrationPolicy(env=None, randomness_scale=0.1),
         "eval_freq": 100000,
         "n_eval_episodes": 20,
         "model_kwargs": {"gradient_steps": 1,
                          "batch_size": 12000,
                          "learning_rate": 1e-3,
                          "lambda_bc": 0,
                          "policy_kwargs": {"net_arch": [32, 32, 32, 32, 32, 32, 32, 32]},
                          "top_quantiles_to_drop_per_net": 0,
                          "learning_starts": 5000,
                          "buffer_size": int(1e6),
                          "max_demo_ratio": 0.1,
                          "reach_zero": 1e6,
                          "n_sampled_goal": 4,
                          "hindrl_sampling_strategy": HinDRLSamplingStrategy.TaskConditioned,
                          "epsilon_filtering": True,
                          "tau": 0.005,
                          }},
        {"env_kwargs": {"horizon": 300, "number_of_waypoints": 2,
                        "waypoints": [np.array([0.58766, 0.26816, 0.37820, 2.89549, 0.03567, -0.39348]),
                                      np.array([0.42493, 0.07166, 0.36318, 2.88426, 0.12777, 0.35920])]
                        },
         "goal_handler": HinDRLGoalHandler,
         "goal_handler_kwargs": {
             "demonstration_hdf5": "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/FixedParameterizedReach_2Waypoint/1_1673455320_229703/demo.hdf5",
             "m": 5, "k": 0},
         "env_class": FixedParameterizedReachEnv,
         "number_of_demonstrations": 1,
         "regenerate_demonstrations": False,
         "demo_path": "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/FixedParameterizedReach_2Waypoint/1_1673455320_229703/demo.hdf5",
         "expert_policy": FixedParameterizedReachDemonstrationPolicy(env=None, randomness_scale=0.1),
         "eval_freq": 100000,
         "n_eval_episodes": 20,
         "model_kwargs": {"gradient_steps": 1,
                          "batch_size": 12000,
                          "learning_rate": 1e-3,
                          "lambda_bc": 0,
                          "policy_kwargs": {"net_arch": [32, 32, 32, 32, 32, 32, 32, 32]},
                          "top_quantiles_to_drop_per_net": 2,
                          "learning_starts": 5000,
                          "buffer_size": int(1e6),
                          "max_demo_ratio": 0.1,
                          "reach_zero": 1e6,
                          "n_sampled_goal": 4,
                          "hindrl_sampling_strategy": HinDRLSamplingStrategy.TaskConditioned,
                          "epsilon_filtering": True,
                          "tau": 0.005,
                          }},
        {"env_kwargs": {"horizon": 300, "number_of_waypoints": 2,
                        "waypoints": [np.array([0.58766, 0.26816, 0.37820, 2.89549, 0.03567, -0.39348]),
                                      np.array([0.42493, 0.07166, 0.36318, 2.88426, 0.12777, 0.35920])]
                        },
         "goal_handler": HinDRLGoalHandler,
         "goal_handler_kwargs": {
             "demonstration_hdf5": "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/FixedParameterizedReach_2Waypoint/1_1673455320_229703/demo.hdf5",
             "m": 5, "k": 0},
         "env_class": FixedParameterizedReachEnv,
         "number_of_demonstrations": 1,
         "regenerate_demonstrations": False,
         "demo_path": "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/FixedParameterizedReach_2Waypoint/1_1673455320_229703/demo.hdf5",
         "expert_policy": FixedParameterizedReachDemonstrationPolicy(env=None, randomness_scale=0.1),
         "eval_freq": 100000,
         "n_eval_episodes": 20,
         "model_kwargs": {"gradient_steps": 1,
                          "batch_size": 12000,
                          "learning_rate": 1e-3,
                          "lambda_bc": 0,
                          "policy_kwargs": {"net_arch": [32, 32, 32, 32, 32, 32, 32, 32]},
                          "top_quantiles_to_drop_per_net": 4,
                          "learning_starts": 5000,
                          "buffer_size": int(1e6),
                          "max_demo_ratio": 0.1,
                          "reach_zero": 1e6,
                          "n_sampled_goal": 4,
                          "hindrl_sampling_strategy": HinDRLSamplingStrategy.TaskConditioned,
                          "epsilon_filtering": True,
                          "tau": 0.005,
                          }},
               ]

    run_parallel(configs)
    # train(configs[0])
    # load_and_eval("/home/mark/tum/2022ss/thesis/master_thesis/training_logs/FixedParameterizedReach_2Waypoint_189")