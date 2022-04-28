import numpy as np
import robosuite
from robosuite import load_controller_config

from env.cable_insertion_task import CableInsertionEnv

if __name__ == '__main__':
    controller_config = load_controller_config(default_controller="OSC_POSE")

    env = CableInsertionEnv(robots=["Panda", "Panda"],  # load a Sawyer robot and a Panda robot
                            gripper_types="default",  # use default grippers per robot arm
                            controller_configs=controller_config,  # each arm is controlled using OSC
                            env_configuration="single-arm-parallel",
                            render_camera=None,# (two-arm envs only) arms face each other
                            has_renderer=True,  # no on-screen rendering
                            has_offscreen_renderer=False,  # no off-screen rendering
                            control_freq=20,  # 20 hz control for applied actions
                            horizon=20000,  # each episode terminates after 200 steps
                            use_object_obs=True,  # provide object observations to agent
                            use_camera_obs=False,  # don't provide image observations to agent
                            reward_shaping=True)  # use a dense reward signal for learning)

    while True:
        action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
        asd = env.step(action)
        env.render()