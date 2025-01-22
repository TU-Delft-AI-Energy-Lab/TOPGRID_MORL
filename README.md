# Multi-Objective Reinforcement Learning for Power Grid Topology Control

This repository contains a collection of modules for implementing Multi-Objective Reinforcement Learning (MORL) algorithm specifically designed for multi-objective power grid topology control. This framework bases on morl_baselines for the MORL part and on Grid2Op for the power system environment.

## Table of Contents
- Installation
- Modules
  - MO_PPO.py
  - MO_BaselineAgents.py
  - GridRewards.py
  - CustomGymEnv.py
  - EnvSetup.py
  - MO_PPO_train_utils.py
  - MORL_analysis_utils.py
  - Dataloader.py
  - Grid2op_eval.py
  - env_start_up.py
  - generate_reward_stats.py
  - MORL_execution.py
- Usage
- Contributing
- License

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Modules

### MO_PPO.py
Contains the implementation of the Multi-Objective Proximal Policy Optimization (MO-PPO) algorithm, based on the MORL_baseline package

### GridRewards.py
Contains the implementation for calculating grid-based rewards.

#### Classes:
- GridRewards: Calculates rewards based on a grid of metrics.

### CustomGymEnv.py
Defines a custom Gym environment for MORL experiments.

### EnvSetup.py
Utility for setting up the custom Gym environment.

### MO_PPO_train_utils.py
Contains utility functions for training MO-PPO.

### MORL_analysis_utils.py
Contains utility functions for analyzing MORL experiments.

### case_studies.py 
Contains the analysis script and plotting for the case studies

### Grid2op_eval.py
Contains the evaluation script for Grid2Op environment.

### env_start_up.py
Sets up the environment for power grid topology control experiments.

### ols_DOL_exe.py 
Starts and proceeds the experiments including DOL and MOPPO Training and Evaluation. 


## Contributing
Contributions are welcome Please create an issue or submit a pull request for any changes.

## License
   
This work is licensed under a
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
