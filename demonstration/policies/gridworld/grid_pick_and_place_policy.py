import numpy as np

from env.grid_world_envs.fixed_pick_and_place import FixedGridPickAndPlace
from env.grid_world_envs.pick_and_place import GridPickAndPlace


class GridPickAndPlacePolicy:
    """
    Policy used for generating demonstrations for the GridPickAndPlace and the FixedGridPickAndPlace envs.
    """
    def __init__(self, random_action_probability=0.2):
        """
        Init.
        :param random_action_probability: The agent will perform a random move with the defined probability.
                                          The grabbing action will be always deterministic
        """
        self.random_action_probability = random_action_probability
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

    def step(self, observation):
        """
        Policy step.
        :param observation: observation from the env
        :return: action (np.array) for the robotarm
        """
        observation = self.convert_observation_to_dict(observation["observation"])
        next_object_to_transport = np.where(observation["object_transported"] == 0)[0][0]
        grabbed_object = np.where(observation["object_grabbed"] == 1)[0]
        if grabbed_object.shape[0] == 0:
            grabbed_object = None

        if grabbed_object is None:
            if np.allclose(observation[f"dist_agent_to_object_{next_object_to_transport}"], np.array([0, 0])):
                # Agent is already on the next object to grab
                grab_action = 1
                movement_action = self.stochastic_action_into_direction(observation['agent_to_goal'])
            else:
                grab_action = 0
                movement_action = self.stochastic_action_into_direction(
                    observation[f"dist_agent_to_object_{next_object_to_transport}"])
        else:
            if grabbed_object == next_object_to_transport:
                if np.allclose(observation['agent_to_goal'], np.array([0, 0])):
                    # Object arrived -> release and random movement
                    grab_action = 0
                    movement = self.random_movement()
                    movement_action = self.get_action_from_movement(movement)
                else:
                    # Agent is transporting the correct object
                    grab_action = 1
                    movement_action = self.stochastic_action_into_direction(observation['agent_to_goal'])
            else:
                grab_action = 0
                movement = self.random_movement()
                movement_action = self.get_action_from_movement(movement)

        hot_encoded_movement = np.zeros(4)
        hot_encoded_movement[movement_action] = 1
        return np.hstack([hot_encoded_movement, grab_action])

    def reset(self):
        pass

    def stochastic_action_into_direction(self, direction: np.array):
        movement = self.stochastic_movement_into_direction(direction)
        movement_action = self.get_action_from_movement(movement)
        return movement_action

    def stochastic_movement_into_direction(self, direction: np.array):
        if np.random.binomial(1, self.random_action_probability, 1)[0]:
            movement = self.random_movement()
        else:
            movement = self.random_movement_to_right_direction(direction)
        return movement

    def random_movement_to_right_direction(self, direction):
        if np.allclose(direction, np.array([0, 0])):
            movement = self.random_movement()
        else:
            movement_axis = np.random.choice(np.where(direction != 0)[0])
            movement = np.zeros_like(direction)
            movement[movement_axis] = direction[movement_axis]
            movement = np.clip(movement, -1, 1)

        return movement

    def random_movement(self):
        movement = np.zeros(2)
        movement_axis = np.random.choice(np.array([0, 1]))
        movement[movement_axis] = np.random.choice(np.array([-1, 1]))

        return movement

    def get_action_from_movement(self, movement):
        for key, value in self._action_to_direction.items():
            if np.allclose(value, movement):
                return key

    def convert_observation_to_dict(self, observation):
        """
        Reverting the _get_obs() function numpy.hstack step.
        :param observation: stacked observation array
        :return: observation as dict
        """
        number_of_objects = int((observation.shape[0] - 2) / 4)
        obs = {}
        slice_index = 0
        for i in range(number_of_objects):
            obs[f"dist_agent_to_object_{i}"] = observation[slice_index: slice_index + 2]
            slice_index += 2
        obs["agent_to_goal"] = observation[slice_index: slice_index + 2]
        slice_index += 2
        obs["object_transported"] = observation[slice_index: slice_index + number_of_objects]
        slice_index += number_of_objects
        obs["object_grabbed"] = observation[slice_index:]

        assert np.allclose(observation, np.hstack([value if value is not None else -1 for value in obs.values()]))

        return obs

if __name__ == '__main__':
    env = FixedGridPickAndPlace(render_mode="human", random_box_size=2)

    demonstration_policy = GridPickAndPlacePolicy(random_action_probability=0)

    action = env.action_space.sample()
    obs, _, done, _ = env.step(action)

    while True:
        action = demonstration_policy.step(obs)
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()
            demonstration_policy.reset()