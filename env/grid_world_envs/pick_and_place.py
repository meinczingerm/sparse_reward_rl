from typing import Optional

import gym
import numpy as np
from gym import spaces

from env.goal_handler import HinDRLGoalHandler

KEY_TO_ACTION = {
    "d": np.array([1, 0, 0, 0]),
    "s": np.array([0, 1, 0, 0]),
    "a": np.array([0, 0, 1, 0]),
    "w": np.array([0, 0, 0, 1])}


class GridPickAndPlace(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "single_rgb_array"], "render_fps": 4}
    name = "GridPickAndPlace"

    def __init__(self, number_of_objects=3, render_mode: Optional[str] = None, size: int = 5, horizon=100,
                 goal_handler: HinDRLGoalHandler = None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode  # Define the attribute render_mode in your environment
        self.number_of_objects = number_of_objects

        self.size = size  # The size of the square grid
        self.available_grid_space = spaces.Box(0, size - 1, shape=(2,), dtype=int)
        self.window_size = 512  # The size of the PyGame window
        self.horizon = horizon
        self.t = 0

        self.goal_handler = goal_handler  # necessary for HER compute_reward, for all other model could be skipped

        # All Robotic env uses engineered encoding, it would make sense there to change that behaviour
        # for this env engineered encoding is completely fine, experiments could still be interesting...
        self.use_engineered_observation_encoding = True

        # Observations are dictionaries with: distance to all object, distance to goal, mask with which object is
        # already transported
        obs_spaces_dict = {f"dist_agent_to_object_{i}": spaces.Box(0, size - 1, shape=(2,), dtype=int)
             for i in range(self.number_of_objects)}
        obs_spaces_dict["dist_agent_to_goal"] = spaces.Box(0, size - 1, shape=(2,), dtype=int)
        obs_spaces_dict["object_transported"] = spaces.MultiBinary(self.number_of_objects)
        obs_spaces_dict["object_grabbed"] = spaces.MultiBinary(self.number_of_objects)

        self.observation_space = spaces.Dict(
            obs_spaces_dict
        )
        observation_size = 4 * self.number_of_objects + 2
        self.observation_space = spaces.Dict(
            dict(observation=spaces.Box(
                    low=0,
                    high=size,
                    shape=(observation_size,),
                    dtype="float32"),
                 desired_goal=spaces.Box(
                    low=0,
                    high=size,
                    shape=(observation_size,),
                    dtype="float32"),
                 achieved_goal=spaces.Box(
                    low=0,
                    high=size,
                    shape=(observation_size,),
                    dtype="float32")))

        # We have 4 actions for moving corresponding to "right", "up", "left", "down" in a hot encoded fashion
        # and 1 action for grabbing (1: grab, 0: release)
        self.action_space = spaces.Box(0, 1, shape=(5,))

        # Action to direction conversion:
        # 0 is equal to the hot_encoded vector [1, 0, 0, 0] (after max operation) -> "right"
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        self._state = None
        self.reset()

        if self.render_mode == "human":
            import pygame  # import here to avoid pygame dependency with no render

            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

    def compute_reward(self, achieved_goal, goal, _info):
        """
        Calculates the reward given the achieved_goal and goal. It is used by HER for relabeling.
        :param achieved_goal: batched
        :param goal:
        :param _info:
        :return:
        """
        if self.goal_handler is not None:
            return self.goal_handler.compute_reward(achieved_goal, goal, _info)
        else:
            raise NotImplementedError

    def reset(self):
        self.t = 0
        used_positions = np.empty([0, 2])
        self._state = {}
        for i in range(self.number_of_objects):
            self._state[f"object_{i}_pos"], used_positions = \
                self._sample_from_not_used_position(self.available_grid_space, used_positions)

        self._state["agent_pos"], used_positions = self._sample_from_not_used_position(self.available_grid_space,
                                                                                       used_positions)
        self._state["goal_pos"], used_positions = self._sample_from_not_used_position(self.available_grid_space,
                                                                                      used_positions)
        self._state["grabbed_object"] = None
        self._state["next_object_to_transport"] = 0

        return self._get_obs_dict()

    def get_state(self):
        state = self._state
        state = np.hstack([val if val is not None else -1 for val in list(state.values())])

        return state

    def _sample_from_free_space(self, sampling_space: gym.spaces.Box):
        blocked_space_keys = [key for key in self._state.keys() if "pos" in key]
        _used_positions = np.empty([0, 2])
        for blocking_key in blocked_space_keys:
            _used_positions = np.vstack((_used_positions, self._state[blocking_key]))

        sample, _ = self._sample_from_not_used_position(sampling_space, _used_positions)
        return sample

    @staticmethod
    def _sample_from_not_used_position(sampling_space: gym.spaces.Box, _used_positions):
        while True:
            sample = sampling_space.sample()
            if np.equal(_used_positions, sample).any():
                continue
            else:
                _used_positions = np.vstack((_used_positions, sample))
                return sample, _used_positions

    def step(self, action):
        done = 0
        reward = 0
        move = action[:-1]
        direction = self._action_to_direction[int(np.argmax(move))]
        grab = action[-1] > 0.5
        # We use `np.clip` to make sure we don't leave the grid
        last_agent_pos = self._state["agent_pos"]
        self._state["agent_pos"] = np.clip(
            self._state["agent_pos"] + direction, 0, self.size - 1
        )

        if grab == 0:
            if self._state["grabbed_object"] is not None:
                released_obj = self._state["grabbed_object"]
                self._state["grabbed_object"] = None
                if np.allclose(self._state[f"object_{released_obj}_pos"], self._state["goal_pos"]) and released_obj == \
                        self._state["next_object_to_transport"]:
                        self._state["next_object_to_transport"] += 1

                else:
                    # Release object and teleport to new random free position
                    new_place = self._sample_from_free_space(self.available_grid_space)
                    self._state[f"object_{released_obj}_pos"] = new_place

        if grab == 1:

            if self._state["grabbed_object"] is None:
                # Grabbing if object is present
                for object_id in range(self._state["next_object_to_transport"], self.number_of_objects):
                    object_pos = self._state[f"object_{object_id}_pos"]
                    if np.allclose(object_pos, last_agent_pos):
                        self._state["grabbed_object"] = object_id

            if self._state["grabbed_object"] is not None:
                # Transporting the grabbed object
                self._state[f"object_{self._state['grabbed_object']}_pos"] = self._state["agent_pos"]
                # Object succesfully transported
                grabbed_object = self._state["grabbed_object"]
                if np.allclose(self._state[f"object_{grabbed_object}_pos"],
                               self._state["goal_pos"]) and grabbed_object == self.number_of_objects - 1:
                    done = 1
                    reward = 1

        if self.t == self.horizon-1:
            done = 1
        self.t += 1

        info = self._get_info()

        observation = self._get_obs_dict()

        return observation, reward, done, info

    def _get_obs_dict(self):
        observation = self._get_obs()
        observation = {
            "observation": observation,
            "achieved_goal": observation,
            "desired_goal": self._get_desired_goal()
        }
        return observation

    def _get_obs(self):
        obs = {f"dist_agent_to_object_{i}": self._state[f"object_{i}_pos"] - self._state["agent_pos"]
               for i in range(self.number_of_objects)}
        obs["agent_to_goal"] = self._state["goal_pos"] - self._state["agent_pos"]
        obs["object_transported"] = np.zeros(self.number_of_objects)
        obs["object_transported"][:self._state["next_object_to_transport"]] = 1
        obs["object_grabbed"] = np.zeros(self.number_of_objects)
        if self._state["grabbed_object"] is not None:
            obs["object_grabbed"][self._state["grabbed_object"]] = 1

        obs = np.hstack([value for value in obs.values()])
        return obs

    def _get_desired_goal(self):
        desired_goal = {f"dist_agent_to_object_{i}": np.array([0, 0])
               for i in range(self.number_of_objects)}
        desired_goal["agent_to_goal"] = np.array([0, 0])
        desired_goal["object_transported"] = np.zeros(self.number_of_objects)
        desired_goal["object_transported"][:-1] = 1
        desired_goal["object_grabbed"] = np.zeros(self.number_of_objects)
        desired_goal["object_grabbed"][-1] = 1

        desired_goal = np.hstack([value for value in desired_goal.values()])
        return desired_goal

    def _get_info(self):
        return {}

    def render(self, mode="human"):
        self._render_frame(mode)

    def _render_frame(self, mode: str):
        # This will be the function called by the Renderer to collect a single frame.
        assert mode is not None  # The renderer will not call this function with no-rendering.

        import pygame  # avoid global pygame dependency. This method is not called with no-render.

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._state["goal_pos"],
                (pix_square_size, pix_square_size),
            ),
        )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._state["agent_pos"] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        fontName = pygame.font.get_default_font()
        size = 20  # This means the text will be 10 pixels in height.
        # The width will be scaled automatically.
        font = pygame.font.Font(fontName, size)

        for i in range(self.number_of_objects):
            # Now we draw the agent
            pygame.draw.circle(
                canvas,
                (0, 255, 0),
                (self._state[f"object_{i}_pos"] + 0.5) * pix_square_size,
                pix_square_size / 4,
            )

            text = i
            antialias = True
            color = (0, 0, 0)
            surface = font.render(f"{text}", antialias, color)
            canvas.blit(surface, (self._state[f"object_{i}_pos"] + 0.5) * pix_square_size)

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if mode == "human":
            assert self.window is not None
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array or single_rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


if __name__ == '__main__':
    env = GridPickAndPlace(render_mode="human", size=5, number_of_objects=3)
    action = env.action_space.sample()
    while True:
        observation, reward, done, info = env.step(action)
        if done:
            env.reset()
        env._render_frame("human")
        while True:
            try:
                raw_action = input("Give next action")
                action = np.hstack([KEY_TO_ACTION[raw_action.lower()], int (not raw_action.islower())])
                break
            except KeyError:
                print("Use the following commands: wasd, WASD")

