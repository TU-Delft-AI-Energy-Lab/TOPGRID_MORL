Multi-Objective Reinforcement Learning for Power Grid Topology Control This repository contains a collection of modules for implementing Multi-Objective Reinforcement Learning (MORL) algorithms specifically designed for power grid topology control. The framework includes implementations of various components such as PPO for MORL, baseline agents, custom gym environments, reward calculations, data loaders, and evaluation scripts.

Table of Contents Installation Modules MO_PPO.py MO_BaselineAgents.py GridRewards.py CustomGymEnv.py EnvSetup.py MO_PPO_train_utils.py MORL_analysis_utils.py Dataloader.py Grid2op_eval.py env_start_up.py generate_reward_stats.py MORL_execution.py Usage Contributing License Installation To install the required dependencies, run:

pip install -r requirements.txt

Modules MO_PPO.py Contains the implementation of the Multi-Objective Proximal Policy Optimization (MO-PPO) algorithm.

Classes: MO_PPO: Implements the MO-PPO algorithm for training a policy network with multiple objectives. Methods: init(self, policy, value_function, optimizer, clip_epsilon=0.2, entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5): Initializes the MO-PPO instance. update(self, rollouts): Updates the policy using the provided rollouts. evaluate_actions(self, states, actions): Evaluates the actions taken in the given states. MO_BaselineAgents.py Contains baseline agents for comparison with MO-PPO.

Classes: BaselineAgent: A baseline agent for comparison purposes. Methods: init(self, policy, learning_rate=0.001): Initializes the baseline agent. select_action(self, state): Selects an action based on the current state. update(self, rollouts): Updates the policy based on the rollouts. GridRewards.py Contains the implementation for calculating grid-based rewards.

Classes: GridRewards: Calculates rewards based on a grid of metrics. Methods: init(self, reward_weights): Initializes the GridRewards instance with specified weights. calculate(self, metrics): Calculates the total reward based on provided metrics. CustomGymEnv.py Defines a custom Gym environment for MORL experiments.

Classes: CustomGymEnv: Custom Gym environment for MORL. Methods: init(self): Initializes the environment. step(self, action): Executes a step in the environment. reset(self): Resets the environment. render(self, mode=human): Renders the environment. close(self): Closes the environment. EnvSetup.py Utility for setting up the custom Gym environment.

Functions: setup_environment(): Sets up and returns the custom Gym environment. MO_PPO_train_utils.py Contains utility functions for training MO-PPO.

Functions: compute_advantages(rewards, values, gamma, lam): Computes the advantages using Generalized Advantage Estimation (GAE). MORL_analysis_utils.py Contains utility functions for analyzing MORL experiments.

Functions: calculate_pareto_front(metrics): Calculates the Pareto front from the given metrics. Dataloader.py Implements a dataloader for loading training and evaluation data.

Classes: Dataloader: Loads and manages data for training and evaluation. Methods: init(self, data): Initializes the dataloader with given data. get_next_batch(self, batch_size): Returns the next batch of data. Grid2op_eval.py Contains the evaluation script for Grid2Op environment.

Functions: evaluate_grid2op(agent, env): Evaluates the agent in the Grid2Op environment. env_start_up.py Sets up the environment for power grid topology control experiments.

Functions: initialize_environment(): Initializes the environment for experiments. generate_reward_stats.py Generates reward statistics for analysis.

Functions: generate_statistics(data): Generates and returns statistics from the given data. MORL_execution.py Main script for executing MORL experiments.

Functions: execute_experiment(config): Executes the experiment based on the given configuration. Usage To use the modules, import them into your scripts as needed. For example, to use the MO_PPO class:

from MO_PPO import MO_PPO

Initialize your policy, value function, and optimizer here ppo_agent = MO_PPO(policy, value_function, optimizer)

Use the ppo_agent to train or evaluate your model Contributing Contributions are welcome Please create an issue or submit a pull request for any changes.

License This project is licensed under the MIT License.
