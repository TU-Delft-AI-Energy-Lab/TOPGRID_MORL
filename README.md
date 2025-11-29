# Multi-Objective Reinforcement Learning for Power Grid Topology Control

This repository contains a collection of modules for implementing Multi-Objective Reinforcement Learning (MORL) algorithms specifically designed for multi-objective power grid topology control. This framework is based on `morl_baselines` for the MORL part and on `Grid2Op` for the power system environment.

## Citation
This repository contains code for the paper:
*Thomas Lautenbacher, Ali Rajaei, Davide Barbieri, Jan Viebahn, Jochen L. Cremer, „Multi-Objective Reinforcement Learning for Power Grid Topology Control”, IEEE PES Powertech Kiel, Germany, 2025*

## Table of Contents
- [Prerequisites & Installation](#prerequisites--installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [1. Environment Initialization](#1-environment-initialization)
  - [2. Training](#2-training)
  - [3. HPC Execution](#3-hpc-execution)
- [Configuration](#configuration)
- [Modules Overview](#modules-overview)
- [License](#license)

## Prerequisites & Installation

The project requires Python 3.12 or higher. It is recommended to use a virtual environment (e.g., Conda).

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd TOPGRID_MORL
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n top python=3.12
    conda activate top
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r req.txt
    # OR
    pip install .
    ```

## Project Structure

```
TOPGRID_MORL/
├── configs/                # JSON configuration files for experiments
├── scripts/                # Execution scripts (training, initialization)
├── src/
│   └── topgrid_morl/       # Main package source code
│       ├── agent/          # Agent implementations (e.g., MO_PPO)
│       ├── envs/           # Custom Gym environments & wrappers
│       ├── utils/          # Utility functions
│       └── wrapper/        # OLS and other wrappers
├── morl_logs/              # Output logs and saved models
├── runjob.sh               # SLURM submission script
└── README.md               # This file
```

## Usage

> [!IMPORTANT]
> **You must initialize the environment before running any training scripts.**

### 1. Environment Initialization

Before running any experiments, you must generate the training, validation, and test sets. This is done using the `env_start_up.py` script.

**This step only needs to be performed once.**

1.  Open `scripts/env_start_up.py` and verify the `env_name` variable (default: `'l2rpn_case14_sandbox'`).
2.  Run the script:
    ```bash
    python scripts/env_start_up.py
    ```
    This will download the Grid2Op environment data and create the necessary splits.

### 2. Training

The main entry point for training with the OLS-DOL (Optimistic Linear Support - Dynamic Objective Learning) approach is `scripts/ols_DOL_exe.py`.

**Basic Usage:**
```bash
python scripts/ols_DOL_exe.py --seed 0 --config base_config.json
```

**Common Arguments:**
- `--seed`: Random seed (default: 0).
- `--config`: Path to configuration file (default: `base_config.json`).
- `--learning_rate`: Learning rate for the agent (default: 5e-5).
- `--update_epochs`: Number of update epochs (default: 10).
- `--vf_coef`: Value function coefficient.
- `--ent_coef`: Entropy coefficient.

### 3. HPC Execution

For running on a cluster with SLURM, a sample submission script is provided in `runjob.sh`.

1.  Ensure your Conda environment is named `top` (or update the script).
2.  Update the path in the `srun` command to point to your script location:
    ```bash
    srun python /path/to/TOPGRID_MORL/scripts/MORL_execution.py > morl.log
    ```
3.  Submit the job:
    ```bash
    sbatch runjob.sh
    ```

## Configuration

Experiments are configured using JSON files located in the `configs/` directory. A typical config file (`base_config.json`) includes:

- **project_name**: Name for WandB logging.
- **agent_params**: Hyperparameters for the PPO agent (gamma, lambda, etc.).
- **env_params**: Environment settings (max_rho, etc.).
- **network_params**: Neural network architecture.
- **ols_iterations**: Number of OLS iterations.
- **rewards**: Weights for different reward components.

## Modules Overview

- **`src/topgrid_morl/agent/MO_PPO.py`**: Implementation of the Multi-Objective Proximal Policy Optimization (MO-PPO) algorithm.
- **`src/topgrid_morl/envs/CustomGymEnv.py`**: Custom Gymnasium environment wrapping Grid2Op.
- **`src/topgrid_morl/envs/GridRewards.py`**: Definitions of various grid-based reward functions.
- **`scripts/ols_DOL_exe.py`**: Main execution script for the OLS-DOL algorithm.
- **`scripts/env_start_up.py`**: Initialization script for Grid2Op environments.

## License

This work is licensed under a [MIT License](https://opensource.org/licenses/MIT).
