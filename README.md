# Reinforcement Learning Simulation Framework for PiH assembly task.

Simulation framework created based on Robosuite framework (https://github.com/ARISE-Initiative/robosuite). 
It utilizes mujoco engine to model phyiscs of the enviornment. This repository contains simulation framework used to learn impedance parameters using reinforcement learning, more specifcally PPO. 

## System Overview
* Robot: UR5e (6DOF)
* Controller: PD (in free space) and Impedance + PD (in contact)
* Trajectory planner: Minimum jerk trajectory
* RL: Utilizes PPO for learning impedance parameters based on stable baseline 3 implementation.

<div align="center">
  <img src="docs/images/env.png" width="500">
</div>

 <div align="center">
  <img src="docs/images/peg.png" width="500">
</div>

## Usage


### Description
Robosuite simulation for Peg-In-Hole (PiH) task. 
Simulation modified during M.Sc. studies and used for learning impedance parameters of impedance controller.
One set of impedance control parameters is learned per episode, thus the RL process is modified slightly, see `base.py` for modification.


Includes: 
* Custom environment for PiH includes peg, board with a hole.
Can be found in `robosuite/enviornments/manipulation/peg_in_hole_4_mm.py`
* Custom controllers including PD and Impedance Controllers for PiH, PiH with spiral search.
Can be found `robosuite/controllers/...`
* Main run files:
1. To run evaluation/ visualization or learning using one environment at a time use:
`main_model_learn.py`
2. For learning using multiple environments using stable-baselines-3 use:
`main_multi_learn.py`

### Important to note
1. When learning is completed two new folders will be created:
`/robosuite/robosuite`: contains various callbacks, best models, and network parameters
`/robosuite/learning_logs` contains tensorboard logs that can be used via` tensorboard --logdir=./learning_1` to display interactive plots.
2. To figure out multiprocessing go to sb3 github and look for issues with my username, I asked a lot of question so you will be able to figure it out based on responses I got :)

Good Luck !
