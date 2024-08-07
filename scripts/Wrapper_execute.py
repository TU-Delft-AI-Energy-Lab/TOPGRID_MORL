import argparse
import json
import os

import numpy as np

from topgrid_morl.agent.MO_BaselineAgents import (  # Import the DoNothingAgent class
    DoNothingAgent,
)
from topgrid_morl.envs.EnvSetup import setup_environment
from topgrid_morl.envs.GridRewards import ScaledEpisodeDurationReward, ScaledLinesCapacityReward, LinesCapacityReward
from topgrid_morl.wrapper.monte_carlo import train_agent

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

    # Reset the environment to verify dimensions
    gym_env.reset()
    gym_env_val.reset()
    weights = np.array([1,0,0])
    # Step 5: Train Agent
    eval_data = train_agent(
        weights=weights,
        max_gym_steps=max_gym_steps,
        seed=seed,
        results_dir=results_dir,
        env=gym_env,
        env_val=gym_env_val,
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        run_name=config_name,
        project_name=project_name,
        net_arch=net_arch,
        g2op_env=g2op_env, 
        g2op_env_val= g2op_env_val,
        reward_list = reward_list,
        **agent_params
    )
    print(eval_data)
    # Access the rewards for eval_data_0
    rewards_eval_data_0 = eval_data['eval_data_0']['eval_rewards']

    # Access the rewards for eval_data_1
    rewards_eval_data_1 = eval_data['eval_data_1']['eval_rewards']
    
    summed_rewards1 = sum_rewards(rewards=rewards_eval_data_0)
    summed_rewards2 = sum_rewards(rewards=rewards_eval_data_1)
        
    print(summed_rewards1)
    
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
