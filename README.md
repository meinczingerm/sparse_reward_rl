<h1>Demonstration Utilization for Sparse Reward Deep Reinforcelment Learning</h1>

<h2>Project Description</h2>This project was implemented as part of my
Master's Thesis. The goal of this project is to implement and compare different
different methods of incorporating demonstrations to ease the sparse reward reinforcement learning
problem. The experiment results are described in the Thesis itself. This
repository can serve as a basis for future research by either using the models, the implemented environments, or the hardcoded policies used for gathering demosntrations.

<h2>Features</h2>
<h3>Environments</h3>
I present my environments below, but for a more complete description, I suggest reading the specific
part of the Thesis.
<h4>GridWorld Environments</h4>
As a simplified version of the robotic tasks implemented by Davchev et al. in 2022
(https://openreview.net/forum?id=FKp8-pIRo3y) I implemented some grid world environments.
This environment represents the main complexities of the original tasks.
<h5>PickAndPlace Env</h5>
In this environment the agent (blue dot) has to transfer all of the objects
(green dot) to the target grid (red rectangle) in the specified order represented
by the numbering of the objects. All the objects, the agent and the target is
initialized randomly. The agent has two type of actions moving
("up", "down", "left", "right") and grabbing (1="on" or 0="off"). If the agent
places the wrong object into the target or releases the object before reaching
the target the object will be teleported away. The reward for this task is a simple
sparse reward, meaning that the reward is "1" when the last object reaches
the target  rectangle and "0" otherwise.
<p align="center">
<img src=readme_imgs/orig_grid_world.jpg width="200">
</p>


<h5>FixedPickAndPlace Env</h5>
This is the constrained version of the environment described above. The only difference
is in the initialization of the environment. The objects, the agent and the target
grid is all randomized within predefined regions (marked with yellow rectangle). Thanks to
the constraint this environment is better at testing the positive effect of demonstrations,
because the task and the required movements are close to being the same and so the demonstration
provides more information during the training.
<p align="center">
<img src=readme_imgs/fixed_pick_and_place.jpg width="200">
</p>

<h4>Robotic Environments</h4>
I implemented similar robotic environments in Mujoco as it was used in the work of
Davchev et al. (2022) (https://openreview.net/forum?id=FKp8-pIRo3y). These environments
are not the exact same as it was used there, but I aimed to recreate their tasks as close
as possible.
<h4>FixedParameterizedReach</h4>
In this task the robotarm has to reach predefined poses (green capsulers) in the defined order. The returned reward is
a simple sparse reward, returning "1" when the last pose is reached and "0" otherwise.
<p align="center">
<img src=readme_imgs/parameterized_reach.jpg width="200"> 
</p>

<h4>BringNearEnv</h4>
In this task both the robotarm has to grab their respective cable and bring them close
to each other.
<p align="center">
<img src=readme_imgs/bring_near_solved.jpg width="200"> 
</p>

<h4>CableInsertion</h4>
In this task both the robotarm has to grab their respective cable and bring them close
to each other and finally insert them together.
<p align="center">
<img src=readme_imgs/cable_insertion_solved.jpg width="200"> 
</p>

<h3>Demonstration policy</h3>
There is a demonstration policy (https://github.com/meinczingerm/master_thesis/blob/19eef27a1fa55e10ecac9a5613d89aa3242a58e4/demonstration/policies) for each environment listed above, which can be used for
automatically generating (and recording) demonstrations. I have to note that these policies are not fail proof
in some cases they are unable to solve the more complex tasks. This problem is resolved in the
demonstration gathering phase, where only successful demonstrations are stored.

<h2>Setup</h2>
The project is dependent on the robosuite framework. To setup this
framework simply follow the installation guide at https://robosuite.ai/docs/installation.html. (The project
was tested with version 1.3.2 older versions can cause incompatibility issues, if you only manage to
install older versions checkout 2nd point in Known issues.)

<h3>Known issues</h3>

1. In some cases the mujoco renderer is not working on default and returns the error:
"ERROR: GLEW initalization error: Missing GL version".In this case the following environment 
variables has to be set properly:
LD_LIBRARY_PATH=$PATH_TO_MUJOCO$/mujoco200/bin <br />
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so <br />
(In case of running the code in terminal the variables
have to be set in the .bashrc or manually. If the code is started in pycharm, then the run_configurations have to be set.)

2. On my latest attempt I couldn't install the newest robosuite version (1.3.2) with the newest python version (3.10) because I ran into version incompatibilities between numpy and numba. This problem could be solved by using older version of python, the project was tested with python 3.8.12.
