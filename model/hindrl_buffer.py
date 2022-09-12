import warnings
from collections import deque
from enum import Enum
from typing import Optional, Union, NamedTuple, Dict, List, Any

import h5py
import numpy as np
import torch
import torch as th
from sb3_contrib import TQC
from sb3_contrib.common.utils import quantile_huber_loss
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.type_aliases import TensorDict, ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.her import GoalSelectionStrategy

class HinDRLSamplingStrategy(Enum):
    RolloutConditioned = 0
    TaskConditioned = 1
    JointUnion = 2
    JointIntersection = 3

class DemoDictReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    is_demo: th.Tensor


class HinDRLReplayBuffer(HerReplayBuffer):
    def __init__(self, demonstration_hdf5, env, goal_selection_strategy: [HinDRLSamplingStrategy, GoalSelectionStrategy],
                 demo_to_rollout_ratio=0.025,
                 buffer_size=int(1e5), **kwargs):
        self.demonstration_hdf5 = demonstration_hdf5

        self.demonstrations = {"actions": [],
                               "observations": [],
                               "desired_goals": []}
        self.online_goal_buffer = []

        if hasattr(env, "envs"):
            for _env in env.envs:
                assert _env.use_engineered_observation_encoding
        else:
            assert env.use_engineered_observation_encoding

        self.demo_episode_lengths = []
        with h5py.File(demonstration_hdf5, "r") as f:
            for demo_id in f["data"].keys():
                actions = np.array(f["data"][demo_id]["actions"])
                observations = np.array(f["data"][demo_id]["observations"])
                if "desired_goal" in f["data"][demo_id].keys():
                    desired_goal = np.array(f["data"][demo_id]["desired_goal"])
                else:
                    desired_goal = observations[-1]
                self.demonstrations["actions"].append(actions)
                self.demonstrations["observations"].append(observations)
                self.demonstrations["desired_goals"].append(desired_goal)
                self.online_goal_buffer.append(desired_goal)
                self.demo_episode_lengths.append(actions.shape[0])
        self.number_of_demonstrations = len(self.demonstrations["actions"])
        self.demo_episode_lengths = np.vstack(self.demo_episode_lengths)

        super().__init__(env, buffer_size, **kwargs)
        self.goal_selection_strategy = goal_selection_strategy

        self._demo_buffer = {
            key: np.zeros((self.number_of_demonstrations, self.max_episode_length, *buffer_item.shape[2:]), dtype=np.float32)
            for key, buffer_item in self._buffer.items()
        }
        self._load_demonstrations_to_buffer()
        self.demo_to_rollout_sample_ratio = demo_to_rollout_ratio


    def sample_goals(self, episode_indices: np.ndarray, her_indices: np.ndarray, transitions_indices: np.ndarray,
    ) -> np.ndarray:
        if isinstance(self.goal_selection_strategy, GoalSelectionStrategy):
            # use standard HER relabeling
            return super().sample_goals(episode_indices, her_indices, transitions_indices)
        else:
            if self.goal_selection_strategy == HinDRLSamplingStrategy.TaskConditioned:
                return self._sample_uniform_demonstration_goals(num_of_goals=her_indices.shape[0])
            else:
                raise NotImplementedError


    def _sample_uniform_demonstration_goals(self, num_of_goals: int):
        """
        Get random states uniformly from all demonstration.
        :param num_of_goals: Number of sampled goals
        :return: Sampled goals (np.array)
        """
        demonstrations = self.demonstrations["achieved_goal"]
        num_of_demonstrations = len(demonstrations)
        demonstration_indices = np.random.randint(low=0, high=num_of_demonstrations, size=num_of_goals)

        goals = []
        for i, demo in enumerate(demonstrations):
            goal_num_from_demo = (demonstration_indices == i).sum()
            goal_indices = np.random.randint(0, demo.shape[0]-1, goal_num_from_demo)
            goal_array = np.take(demo, goal_indices, axis=0)
            goals.append(goal_array)

        goals = np.vstack(goals)
        goals = goals.reshape([goals.shape[0], 1, goals.shape[1]])
        return goals

    def _load_demonstrations_to_buffer(self):
        for episode_idx in range(len(self.demonstrations["observations"])):
            episode_observations = self.demonstrations["observations"][episode_idx]
            episode_actions = self.demonstrations["actions"][episode_idx]
            episode_goals = self.demonstrations["desired_goals"][episode_idx]
            for timestep_idx in range(episode_observations.shape[0]):
                warnings.warn("Check desired goal, whether correct.")
                obs = {"observation": episode_observations[timestep_idx],
                       "achieved_goal": episode_observations[timestep_idx],
                       "desired_goal": episode_goals[timestep_idx]}
                action = episode_actions[timestep_idx]
                if timestep_idx != (episode_observations.shape[0] - 1):
                    next_obs = {"observation": episode_observations[timestep_idx + 1],
                                "achieved_goal": episode_observations[timestep_idx + 1],
                                "desired_goal": episode_goals[timestep_idx]}
                    reward = 0  # Reward is always 0 until the last timestep
                    done = 0
                else:
                    # Last step in demonstration will remain in the same position
                    next_obs = obs
                    reward = 1  # Last step reward in demonstration is always 1
                    done = 1
                # Empty infos
                infos = [{"is_demonstration": True}]

                self._demo_buffer["observation"][episode_idx][timestep_idx] = obs["observation"]
                self._demo_buffer["achieved_goal"][episode_idx][timestep_idx] = obs["achieved_goal"]
                self._demo_buffer["desired_goal"][episode_idx][timestep_idx] = obs["desired_goal"]
                self._demo_buffer["action"][episode_idx][timestep_idx] = action
                self._demo_buffer["done"][episode_idx][timestep_idx] = done
                self._demo_buffer["reward"][episode_idx][timestep_idx] = reward
                self._demo_buffer["next_obs"][episode_idx][timestep_idx] = next_obs["observation"]
                self._demo_buffer["next_achieved_goal"][episode_idx][timestep_idx] = next_obs["achieved_goal"]
                self._demo_buffer["next_desired_goal"][episode_idx][timestep_idx] = next_obs["desired_goal"]

    def _sample_transitions_from_demonstrations(self, batch_size: Optional[int], maybe_vec_env: Optional[VecNormalize]):
        episode_indices = np.random.randint(0, self.number_of_demonstrations, batch_size)
        ep_lengths = self.demo_episode_lengths[episode_indices]
        transitions_indices = np.random.randint(ep_lengths).squeeze(axis=1)
        transitions = {key: self._demo_buffer[key][episode_indices, transitions_indices].copy() for key in
                       self._buffer.keys()}

        # concatenate observation with (desired) goal
        observations = self._normalize_obs(transitions, maybe_vec_env)

        # HACK to make normalize obs and `add()` work with the next observation
        next_observations = {
            "observation": transitions["next_obs"],
            "achieved_goal": transitions["next_achieved_goal"],
            # The desired goal for the next observation must be the same as the previous one
            "desired_goal": transitions["desired_goal"],
        }
        next_observations = self._normalize_obs(next_observations, maybe_vec_env)

        next_obs = {key: self.to_torch(next_observations[key][:, 0, :]) for key in self._observation_keys}

        normalized_obs = {key: self.to_torch(observations[key][:, 0, :]) for key in self._observation_keys}

        is_demo = np.full_like(transitions["done"], 1)
        return DemoDictReplayBufferSamples(
            observations=normalized_obs,
            actions=self.to_torch(transitions["action"]),
            next_observations=next_obs,
            dones=self.to_torch(transitions["done"]),
            rewards=self.to_torch(self._normalize_reward(transitions["reward"], maybe_vec_env)),
            is_demo=self.to_torch(is_demo))



    def _sample_transitions(
        self,
        batch_size: Optional[int],
        maybe_vec_env: Optional[VecNormalize],
        online_sampling: bool,
        n_sampled_goal: Optional[int] = None,
    ) -> DemoDictReplayBufferSamples:
        """
        :param batch_size: Number of element to sample (only used for online sampling)
        :param env: associated gym VecEnv to normalize the observations/rewards
            Only valid when using online sampling
        :param online_sampling: Using online_sampling for HER or not.
        :param n_sampled_goal: Number of sampled goals for replay. (offline sampling)
        :return: Samples.
        """
        # Select which episodes to use
        batch_size_from_demo = int(self.demo_to_rollout_sample_ratio * batch_size)
        batch_size_from_rollout = batch_size - batch_size_from_demo
        if online_sampling:
            assert batch_size is not None, "No batch_size specified for online sampling of HER transitions"
            # Do not sample the episode with index `self.pos` as the episode is invalid
            if self.full:
                episode_indices = (
                    np.random.randint(1, self.n_episodes_stored, batch_size_from_rollout) + self.pos
                ) % self.n_episodes_stored
            else:
                episode_indices = np.random.randint(0, self.n_episodes_stored, batch_size_from_rollout)
            # A subset of the transitions will be relabeled using HER algorithm
            her_indices = np.arange(batch_size_from_rollout)[: int(self.her_ratio * batch_size_from_rollout)]
        else:
            raise NotImplementedError

        ep_lengths = self.episode_lengths[episode_indices]

        # Special case when using the "future" goal sampling strategy
        # we cannot sample all transitions, we have to remove the last timestep
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # restrict the sampling domain when ep_lengths > 1
            # otherwise filter out the indices
            her_indices = her_indices[ep_lengths[her_indices] > 1]
            ep_lengths[her_indices] -= 1

        if online_sampling:
            # Select which transitions to use
            transitions_indices = np.random.randint(ep_lengths)
        else:
            raise NotImplementedError

        # get selected transitions
        transitions = {key: self._buffer[key][episode_indices, transitions_indices].copy() for key in self._buffer.keys()}

        # sample new desired goals and relabel the transitions
        new_goals = self.sample_goals(episode_indices, her_indices, transitions_indices)
        transitions["desired_goal"][her_indices] = new_goals

        # Convert info buffer to numpy array
        transitions["info"] = np.array(
            [
                self.info_buffer[episode_idx][transition_idx]
                for episode_idx, transition_idx in zip(episode_indices, transitions_indices)
            ]
        )

        # Edge case: episode of one timesteps with the future strategy
        # no virtual transition can be created
        if len(her_indices) > 0:
            # Vectorized computation of the new reward
            transitions["reward"][her_indices, 0] = self.env.env_method(
                "compute_reward",
                # the new state depends on the previous state and action
                # s_{t+1} = f(s_t, a_t)
                # so the next_achieved_goal depends also on the previous state and action
                # because we are in a GoalEnv:
                # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
                # therefore we have to use "next_achieved_goal" and not "achieved_goal"
                transitions["next_achieved_goal"][her_indices, 0],
                # here we use the new desired goal
                transitions["desired_goal"][her_indices, 0],
                transitions["info"][her_indices, 0],
            )

        # concatenate observation with (desired) goal
        observations = self._normalize_obs(transitions, maybe_vec_env)

        # HACK to make normalize obs and `add()` work with the next observation
        next_observations = {
            "observation": transitions["next_obs"],
            "achieved_goal": transitions["next_achieved_goal"],
            # The desired goal for the next observation must be the same as the previous one
            "desired_goal": transitions["desired_goal"],
        }
        next_observations = self._normalize_obs(next_observations, maybe_vec_env)

        # a = self._sample_transitions_from_demonstrations(20, maybe_vec_env)

        if online_sampling:
            next_obs = {key: self.to_torch(next_observations[key][:, 0, :]) for key in self._observation_keys}

            normalized_obs = {key: self.to_torch(observations[key][:, 0, :]) for key in self._observation_keys}

            is_demo = np.array([int(transitions["info"][info_idx][0].get("is_demonstration", False))
                                for info_idx in range(transitions["info"].shape[0])]).reshape(transitions['done'].shape)
            rollout_samples = DemoDictReplayBufferSamples(
                observations=normalized_obs,
                actions=self.to_torch(transitions["action"]),
                next_observations=next_obs,
                dones=self.to_torch(transitions["done"]),
                rewards=self.to_torch(self._normalize_reward(transitions["reward"], maybe_vec_env)),
                is_demo=self.to_torch(is_demo))

            demo_samples = self._sample_transitions_from_demonstrations(batch_size_from_demo, maybe_vec_env)
            return self._merge_samples([rollout_samples, demo_samples])
        else:
            raise NotImplementedError

    def _merge_samples(self, sample_groups: List[DemoDictReplayBufferSamples]) -> DemoDictReplayBufferSamples:
        """
        Merges a list of DemoDictReplayBufferSamples to one DemoDictReplayBufferSamples, by stacking all data inside.
        :return: merged DemoDictReplayBufferSamples
        """
        attributes = {}
        for attrib_name in DemoDictReplayBufferSamples._fields:
            if type(sample_groups[0].__getattribute__(attrib_name)) is dict:
                attribute = {}
                for key in sample_groups[0].__getattribute__(attrib_name).keys():
                    attribute[key] = torch.cat([sample_group.__getattribute__(attrib_name)[key] for sample_group in
                                               sample_groups])
                attributes[attrib_name] = attribute
            else:
                attributes[attrib_name] = torch.cat([sample_group.__getattribute__(attrib_name) for
                                                     sample_group in sample_groups])
        return DemoDictReplayBufferSamples(**attributes)

