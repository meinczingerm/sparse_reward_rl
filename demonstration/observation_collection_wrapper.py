import numpy as np
from robosuite.wrappers import DataCollectionWrapper


class ObservationCollectionWrapper(DataCollectionWrapper):
    def step(self, action):
        """
        Extends vanilla step() function call to accommodate observation collection.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ret = super(DataCollectionWrapper, self).step(action)
        self.t += 1

        # on the first time step, make directories for logging
        if not self.has_interaction:
            self._on_first_interaction()

        # collect the current simulation state if necessary
        if self.t % self.collect_freq == 0:
            state = self.env.sim.get_state().flatten()
            self.states.append(state)

            info = {}
            info["actions"] = np.array(action)
            info["observation"] = ret[0]["observation"]
            if "desired_goal" in ret[0].keys():
                info["desired_goal"] = ret[0]["desired_goal"]
            self.action_infos.append(info)

        # flush collected data to disk if necessary
        if self.t % self.flush_freq == 0:
            self._flush()

        return ret