# Reinforcement Learning with Model-Agnostic Meta-Learning (MAML) for SUMO-RL
Repository containing code for running MAML-RL algorithm for SUMO-RL environment (version 1.3.0). Also supports 2 gym classic control environments:
* CartPole-v1
* MountainCar-v0

Forked from https://github.com/tristandeleu/pytorch-maml-rl, which hasn't been updated for a long time and took some time to figure out all the dependencies and fix some bugs.

## Installation and dependencies
Includes the .yml file with the needed conda environment. (tested on linux)

Essencially the most annoying requirements are:
* python==3.9.*
* gym==0.21.1
* protobuf<=3.20.*
* sumo-rl==1.3.0 (--no-deps)

To install, run:
```
conda env create -f conda_env.yml
conda activate maml
pip install sumo-rl==1.3.0 --no-deps
```

Might run for other requirements as well, but this is what I found that works.

## Logs
For every iteration generates a new folder that has the name of the environment and a current datetime in its name.

This folder stores the weights for the model, the plot of the reward function graph, and the text file with all the rewards that were used to construct the graph.

## Notes on the environments

### sumo-rl
To make the environment suitable for meta learning, by default, one of the 4 reward functions implemented in sumo-rl will be chosen in each task to optimize for. You can modify the .yaml config file for sumo-rl env to change it. You can also in theory add custom reward functions by adding them to `SumoEnvironmentMetaLearning` class in `sumo_rl.py` which extends the `SumoEnvironment` class from sumo-rl.

### mountain-car
No changes for meta learning environment. The task is always the same

### cart-pole
No changes for meta learning environment. The task is always the same


## When implementing a new environemnt
Refer to the original repo for more info
