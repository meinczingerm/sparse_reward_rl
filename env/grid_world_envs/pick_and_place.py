import time
from typing import Optional

import gym
import numpy as np
from gym import spaces

KEY_TO_ACTION = {
    "d": 0,
    "s": 1,
    "a": 2,
    "w": 3}


class GridPickAndPlace(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "single_rgb_array"], "render_fps": 4}

    def __init__(self, number_of_objects=3, render_mode: Optional[str] = None, size: int = 5):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode  # Define the attribute render_mode in your environment
        self.number_of_objects = number_of_objects

        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with: distance to all object, distance to goal, mask with which object is
        # already transported
        obs_spaces_dict = {f"dist_agent_to_object_{i}": spaces.Box(0, size - 1, shape=(2,), dtype=int)
             for i in range(self.number_of_objects)}
        obs_spaces_dict["dist_agent_to_goal"] = spaces.Box(0, size - 1, shape=(2,), dtype=int)
        obs_spaces_dict["object_transported"] = spaces.MultiBinary(self.number_of_objects)
        self.observation_space = spaces.Dict(
            obs_spaces_dict
        )

        # We have 4 actions for moving corresponding to "right", "up", "left", "down",
        # and 1 action for grabbing (1: grab, 0: release)
        self.action_space = spaces.MultiDiscrete([4, 1])
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

    def reset(self):
        available_space = self.observation_space["dist_agent_to_object_0"]
        used_positions = np.empty([0, 2])
        self._state = {}
        for i in range(self.number_of_objects):
            self._state[f"object_{i}_pos"], used_positions = \
                self._sample_from_not_used_position(available_space, used_positions)

        self._state["agent_pos"], used_positions = self._sample_from_not_used_position(available_space, used_positions)
        self._state["goal_pos"], used_positions = self._sample_from_not_used_position(available_space, used_positions)
        self._state["object_transported"] = np.zeros(self.number_of_objects)
        self._state["grabbed_object"] = None
        self._state["next_object_to_transport"] = 0

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
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        done = 0
        reward = 0
        move = action[0]
        grab = action[1]
        direction = self._action_to_direction[move]
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
                    # Object succesfully transported
                    if released_obj == self.number_of_objects - 1:
                        done = 1
                        reward = 1
                    else:
                        self._state["next_object_to_transport"] += 1

                else:
                    # Release object and teleport to new random free position
                    available_space = self.observation_space["dist_agent_to_object_0"]
                    new_place = self._sample_from_free_space(available_space)
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

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def _get_obs(self):
        obs = {f"dist_agent_to_object_{i}": self._state[f"object_{i}_pos"] - self._state["agent_pos"]
               for i in range(self.number_of_objects)}
        obs["agent_to_goal"] = self._state["goal_pos"] - self._state["agent_pos"]
        obs["object_transported"] = self._state["object_transported"]
        return obs

    def _get_info(self):
        return None

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
    env = GridPickAndPlace(render_mode="human")
    action = env.action_space.sample()
    while True:
        observation, reward, done, info = env.step(action)
        if done:
            env.reset()
        env._render_frame("human")
        while True:
            try:
                raw_action = input("Give next action")
                action = np.array([KEY_TO_ACTION[raw_action.lower()], int (not raw_action.islower())], dtype=np.int64)
                break
            except KeyError:
                print("Use the following commands: wasd, WASD")

