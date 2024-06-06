# Reinforcement Learning with Model-Agnostic Meta-Learning (MAML) for SUMO-RL
Repository containing code for running MAML-RL algorithm for SUMO-RL environment (version 1.3.0). Also supports 2 gym classic control environments:
* CartPole-v1
* MountainCar-v0

Forked from https://github.com/tristandeleu/pytorch-maml-rl

## Installation and dependencies
Includes the .yml file with the needed conda environment. (tested on linux)

To install, run:
```
conda env create -f conda_env.yml
conda activate maml
pip install sumo-rl==1.3.0 --no-deps
```

## Logs
For every iteration generates a new folder that has the name of the environment and a current datetime in its name.

This folder stores the weights for the model, the plot of the reward function graph, and the text file with all the rewards that were used to construct the graph.
