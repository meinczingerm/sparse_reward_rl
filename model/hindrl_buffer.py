from typing import Optional, Union, NamedTuple

import h5py
import numpy as np
import torch as th
from sb3_contrib import TQC
from sb3_contrib.common.utils import quantile_huber_loss
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.type_aliases import TensorDict, ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.her import GoalSelectionStrategy


class DemoDictReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    is_demo: th.Tensor


class HinDRLReplayBuffer(HerReplayBuffer):
    def __init__(self, demonstration_hdf5, env, replay_strategy="uniform_demonstration",
                 buffer_size=int(1e5), **kwargs):
        self.demonstration_hdf5 = demonstration_hdf5
        self.replay_strategy = replay_strategy

        self.demonstrations = {"actions": [],
                               "observations": []}
        self.online_goal_buffer = []

        if hasattr(env, "envs"):
            for _env in env.envs:
                assert _env.use_engineered_observation_encoding
        else:
            assert env.use_engineered_observation_encoding
        with h5py.File(demonstration_hdf5, "r") as f:
            for demo_id in f["data"].keys():
                actions = np.array(f["data"][demo_id]["actions"])
                observations = np.array(f["data"][demo_id]["observations"])
                self.demonstrations["actions"].append(actions)
                self.demonstrations["observations"].append(observations)
                self.online_goal_buffer.append(observations[-1])

        super().__init__(env, buffer_size, **kwargs)
        self._load_demonstrations_to_buffer()
        print("Done")


    def sample_goals(self, episode_indices: np.ndarray, her_indices: np.ndarray, transitions_indices: np.ndarray,
    ) -> np.ndarray:
        if self.replay_strategy == "uniform_demonstration":
            return self._sample_uniform_demonstration_goals(num_of_goals=her_indices.shape[0])
        else:
            raise NotImplementedError

    def _sample_uniform_demonstration_goals(self, num_of_goals: int):
        """
        Get random states uniformly from all demonstration.
        :param num_of_goals: Number of sampled goals
        :return: Sampled goals (np.array)
        """
        demonstrations = self.demonstrations["observations"]
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
            for timestep_idx in range(episode_observations.shape[0]):
                obs = {"observation": episode_observations[timestep_idx],
                       "achieved_goal": episode_observations[timestep_idx],
                       "desired_goal": episode_observations[-1]}
                action = episode_actions[timestep_idx]
                if timestep_idx != (episode_observations.shape[0] - 1):
                    next_obs = {"observation": episode_observations[timestep_idx + 1],
                                "achieved_goal": episode_observations[timestep_idx + 1],
                                "desired_goal": episode_observations[-1]}
                    reward = 0  # Reward is always 0 until the last timestep
                    done = 0
                else:
                    # Last step in demonstration will remain in the same position
                    next_obs = obs
                    reward = 1  # Last step reward in demonstration is always 1
                    done = 1
                # Empty infos
                infos = [{"is_demonstration": True}]
                self.add(obs, next_obs, action, reward, done, infos=infos)

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
        if online_sampling:
            assert batch_size is not None, "No batch_size specified for online sampling of HER transitions"
            # Do not sample the episode with index `self.pos` as the episode is invalid
            if self.full:
                episode_indices = (
                    np.random.randint(1, self.n_episodes_stored, batch_size) + self.pos
                ) % self.n_episodes_stored
            else:
                episode_indices = np.random.randint(0, self.n_episodes_stored, batch_size)
            # A subset of the transitions will be relabeled using HER algorithm
            her_indices = np.arange(batch_size)[: int(self.her_ratio * batch_size)]
        else:
            assert maybe_vec_env is None, "Transitions must be stored unnormalized in the replay buffer"
            assert n_sampled_goal is not None, "No n_sampled_goal specified for offline sampling of HER transitions"
            # Offline sampling: there is only one episode stored
            episode_length = self.episode_lengths[0]
            # we sample n_sampled_goal per timestep in the episode (only one is stored).
            episode_indices = np.tile(0, (episode_length * n_sampled_goal))
            # we only sample virtual transitions
            # as real transitions are already stored in the replay buffer
            her_indices = np.arange(len(episode_indices))

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
            if her_indices.size == 0:
                # Episode of one timestep, not enough for using the "future" strategy
                # no virtual transitions are created in that case
                return {}, {}, np.zeros(0), np.zeros(0)
            else:
                # Repeat every transition index n_sampled_goals times
                # to sample n_sampled_goal per timestep in the episode (only one is stored).
                # Now with the corrected episode length when using "future" strategy
                transitions_indices = np.tile(np.arange(ep_lengths[0]), n_sampled_goal)
                episode_indices = episode_indices[transitions_indices]
                her_indices = np.arange(len(episode_indices))

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

        if online_sampling:
            next_obs = {key: self.to_torch(next_observations[key][:, 0, :]) for key in self._observation_keys}

            normalized_obs = {key: self.to_torch(observations[key][:, 0, :]) for key in self._observation_keys}

            is_demo = np.array([int(transitions["info"][info_idx][0].get("is_demonstration", False))
                                for info_idx in range(transitions["info"].shape[0])]).reshape(transitions['done'].shape)
            return DemoDictReplayBufferSamples(
                observations=normalized_obs,
                actions=self.to_torch(transitions["action"]),
                next_observations=next_obs,
                dones=self.to_torch(transitions["done"]),
                rewards=self.to_torch(self._normalize_reward(transitions["reward"], maybe_vec_env)),
                is_demo=self.to_torch(is_demo))
        else:
            raise NotImplementedError

