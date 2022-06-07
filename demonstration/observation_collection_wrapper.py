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
            info["observations"] = np.hstack([ret[0]['robot0_joint_vel'], action[6], ret[0]['robot1_joint_vel'],
                                             action[13]])
            info["torque_actions"] = np.hstack([self.env.env.robots[0].recent_torques.last, action[6],
                                               self.env.env.robots[1].recent_torques.last, action[13]])
            self.action_infos.append(info)

        # flush collected data to disk if necessary
        if self.t % self.flush_freq == 0:
            self._flush()

        return ret