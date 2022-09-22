from typing import Optional, NamedTuple

import gym
import numpy as np
from gym.spaces import Box

from env.goal_handler import HinDRLGoalHandler
from env.grid_world_envs.pick_and_place import GridPickAndPlace, KEY_TO_ACTION


class RectangleBox(NamedTuple):
    top_left: np.array
    bottom_right: np.array


class FixedGridPickAndPlace(GridPickAndPlace):
    """This env is a slight modification of the GridPickAndPlace environment. In this case the positions of the
    objects and the agent and goal positions are fixed with to places with small variance. This reduces the actually
    necessary state space, and thus making the demonstrations more informative about the task."""
    name = "FixedGridPickAndPlace"

    def __init__(self, random_box_size: int, number_of_objects=2, render_mode: Optional[str] = None, size: int = 10,
                 horizon=100, goal_handler: HinDRLGoalHandler = None):
        if number_of_objects != 2:
            raise NotImplementedError

        self.random_box_size = random_box_size
        self.starting_box = {
            "agent_pos": self._get_box_starting_from_corner(np.array([0, 0]), random_box_size),
            "goal_pos": self._get_box_starting_from_corner(np.array([0, size - random_box_size]),
                                                           random_box_size),
            "object_0_pos": self._get_box_starting_from_corner(np.array([size - random_box_size, 0]),
                                                               random_box_size),
            "object_1_pos": self._get_box_starting_from_corner(np.array([size - random_box_size,
                                                                         size - random_box_size]),
                                                               random_box_size),
        }

        super(FixedGridPickAndPlace, self).__init__(number_of_objects=number_of_objects,
                                                    render_mode=render_mode, size=size, horizon=horizon,
                                                    goal_handler=goal_handler)

    def reset(self):
        self.t = 0
        self._state = {}
        for i in range(self.number_of_objects):
            self._state[f"object_{i}_pos"] = self.starting_box[f"object_{i}_pos"].sample()

        self._state["agent_pos"] = self.starting_box[f"agent_pos"].sample()
        self._state["goal_pos"] = self.starting_box[f"goal_pos"].sample()
        self._state["grabbed_object"] = None
        self._state["next_object_to_transport"] = 0

        return self._get_obs_dict()

    @staticmethod
    def _get_box_starting_from_corner(top_left_corner: np.array, rectangle_size: int) -> Box:
        return Box(np.array([top_left_corner[0], top_left_corner[1]]),
                   np.array([top_left_corner[0] + rectangle_size - 1, top_left_corner[1] + rectangle_size - 1]),
                   dtype=int)

    def _sample_from_free_space_for_object(self, object_id):
        if self.random_box_size == 1:
            # in case of box_size 1 it can happen that the only available space is blocked by the agent
            used_position = np.empty([0, 2])
        else:
            used_position = self._state["agent_pos"]
            used_position = used_position.reshape([1, 2])
        pos_sample, _ = self._sample_from_not_used_position(self.starting_box[f"object_{object_id}_pos"], used_position)
        return pos_sample


if __name__ == '__main__':
    env = FixedGridPickAndPlace(random_box_size=2, render_mode="human", size=10)
    action = env.action_space.sample()
    while True:
        observation, reward, done, info = env.step(action)
        if done:
            env.reset()
        env._render_frame("human")
        while True:
            try:
                raw_action = input("Give next action")
                action = np.hstack([KEY_TO_ACTION[raw_action.lower()], int(not raw_action.islower())])
                break
            except KeyError:
                print("Use the following commands: wasd, WASD")




