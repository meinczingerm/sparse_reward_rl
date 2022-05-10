<h1>Master Thesis</h1>

<h2>Goal</h2>The goal of this project is to recreate the method "HinDRL" presented in  the article "FOR LONG-HORIZON DEXTEROUS MANIPULATION" (https://arxiv.org/pdf/2112.00597.pdf)

<h2>Features</h2>
<h3>Bimanual cable insertion environment</h3>The environment copies the task of bimanual
cable insertion presented in the paper:
<img src=readme_imgs/bimanual_cable_env.png> 

This environment can be tested with random actions by running
env/run.py.

<h3>Demonstration policy</h3>
There is also a simple controller implemented in demonstration/policy.py
which solves the insertion task. (Note: The controller is not 100% robust
because of the non-deterministic behaviour of mujoco the cable-s are slipping to
slightly different positions and the task fails around 10% of the time.)

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
