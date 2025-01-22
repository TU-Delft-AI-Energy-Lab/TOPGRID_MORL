import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt     
from mpl_toolkits.mplot3d import Axes3D

from topgrid_morl.agent.MO_BaselineAgents import (  # Import the DoNothingAgent class
    DoNothingAgent,
)
from topgrid_morl.envs.EnvSetup import setup_environment
from topgrid_morl.envs.GridRewards import ScaledEpisodeDurationReward, ScaledLinesCapacityReward, LinesCapacityReward
from topgrid_morl.wrapper.monte_carlo import MOPPOTrainer

def sum_rewards(rewards):
        rewards_np = np.array(rewards)
        summed_rewards = rewards_np.sum(axis=0)
        return summed_rewards.tolist()
    
def mean_rewards(rewards1, rewards2):
        concat_rewards = np.concatenate(rewards1, rewards2)
    
        return concat_rewards

def main(seed: int, config: str) -> None:
    """
    Main function to set up the environment, initialize networks, define agent parameters, train the agent,
    and run a DoNothing benchmark.
    """
    env_name = "rte_case5_example"

    config_path = os.path.join(os.getcwd(), "configs", config)
    # Load configuration from file
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No such file or directory: '{config_path}'")

    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    config_name = config['config_name']
    project_name = config['project_name']
    agent_params = config["agent_params"]
    weight_vectors = config["weight_vectors"]
    weights = np.array(weight_vectors)  # Convert to numpy array for consistency
    max_gym_steps = config["max_gym_steps"]
    env_params = config["env_params"]
    max_rho = env_params["max_rho"]
    network_params = config["network_params"]
    net_arch = network_params["net_arch"]
    rewards = config["rewards"]
    reward_list = [rewards["second"], rewards["third"]]
    
    agent_params["log"] = False
    # Step 1: Setup Environment
    if env_name == "rte_case5_example":
        results_dir = "training_results_5bus_4094"
        action_dim = 53
        actions_file = "filtered_actions.json"
    elif env_name == "l2rpn_case14_sandbox":
        results_dir = "training_results_14bus_4096"
        action_dim = 134
        actions_file = "medha_actions.json"

    gym_env, obs_dim, action_dim, reward_dim, g2op_env = setup_environment(
        env_name=env_name,
        test=False,
        seed=seed,
        action_space=53,
        first_reward=ScaledLinesCapacityReward,
        rewards_list=reward_list,
        actions_file=actions_file,
        max_rho = max_rho
    )

    gym_env_val, _, _, _, g2op_env_val = setup_environment(
        env_name=env_name,
        test=False,
        seed=seed,
        action_space=53,
        first_reward=ScaledLinesCapacityReward,
        rewards_list=reward_list,
        actions_file=actions_file,
        env_type="_val",
        max_rho = max_rho
        
    )
    print(agent_params)
    # Reset the environment to verify dimensions
    gym_env.reset()
    gym_env_val.reset()
    weights = np.array([1,0,0])
    # Step 5: Train Agent
    trainer = MOPPOTrainer(
        iterations=5,
        max_gym_steps=max_gym_steps,
        seed=seed,
        results_dir=results_dir,
        env=gym_env,
        env_val=gym_env_val,
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        run_name="runMC",
        project_name=project_name,
        net_arch=net_arch,
        g2op_env=g2op_env, 
        g2op_env_val= g2op_env_val,
        reward_list = reward_list,
        **agent_params
    )
    eval_rewards, results_dict = trainer.runMC_MOPPO()
    print(results_dict)
    
    # Extracting weights and mean rewards
    weights = list(results_dict.keys())
    mean_rewards = list(results_dict.values())

    # Convert to numpy arrays for easier manipulation
    weights_array = np.array(weights)
    mean_rewards_array = np.array(mean_rewards)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each point with its corresponding label
    for i in range(len(weights_array)):
        x, y, z = mean_rewards_array[i]
        ax.scatter(x, y, z, marker='o')
        label = f"{weights_array[i]}"
        ax.text(x, y, z, label)

    # Set labels for axes
    ax.set_xlabel('Reward Dimension 1')
    ax.set_ylabel('Reward Dimension 2')
    ax.set_zlabel('Reward Dimension 3')

    plt.show()
    """
    iterations=2
        max_gym_steps=max_gym_steps,
        seed=seed,
        results_dir=results_dir,
        env=gym_env,
        env_val=gym_env_val,
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        run_name="runMC",
        project_name=project_name,
        net_arch=net_arch,
        g2op_env=g2op_env, 
        g2op_env_val= g2op_env_val,
        reward_list = reward_list,
        **agent_params
    print(eval_data)
    """
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiment with specific seed and weights"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="base_config.json",
        help="Path to the configuration file (default: configs/base_config.json)",
    )
    args = parser.parse_args()

    main(args.seed, args.config)
