import gymnasium as gym
import mo_gymnasium as mo_gym

from topgrid_morl.wrapper.ols import LinearSupport
from morl_baselines.single_policy.ser.mo_q_learning import MOQLearning

import torch as th
from datetime import datetime
import argparse
import json
import os
import pickle
from scipy.spatial import ConvexHull

import numpy as np
import matplotlib.pyplot as plt     
from mpl_toolkits.mplot3d import Axes3D
from topgrid_morl.agent.MO_PPO import MOPPO
from topgrid_morl.agent.MO_BaselineAgents import (  # Import the DoNothingAgent class
    DoNothingAgent,
)
from topgrid_morl.utils.MO_PPO_train_utils import initialize_agent
from topgrid_morl.envs.EnvSetup import setup_environment
from topgrid_morl.envs.GridRewards import ScaledEpisodeDurationReward, ScaledLinesCapacityReward, LinesCapacityReward
from topgrid_morl.wrapper.mc_moppo import MCMOPPOTrainer  # Use the MCMOPPOTrainer instead of OLS

# Recursive function to convert all numpy arrays to lists
def convert_ndarray_to_list(data):
    if isinstance(data, dict):
        return {key: convert_ndarray_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(element) for element in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, th.Tensor):  # Handle PyTorch tensors
        return data.tolist()
    else:
        return data

def sum_rewards(rewards):
    rewards_np = np.array(rewards)
    summed_rewards = rewards_np.sum(axis=0)
    return summed_rewards.tolist()
    
def mean_rewards(rewards1, rewards2):
    concat_rewards = np.concatenate(rewards1, rewards2)
    return concat_rewards
    
def save_agent(agent, weights, directory, index):
    # Save the entire agent object using pickle
    agent_filename = f"agent_{index}.pkl"
    agent_path = os.path.join(directory, agent_filename)
    with open(agent_path, "wb") as f:
        pickle.dump(agent, f)
    
    # Save the corresponding weights
    weights_filename = f"weights_{index}.json"
    weights_path = os.path.join(directory, weights_filename)
    with open(weights_path, "w") as f:
        json.dump(weights.tolist(), f, indent=4)
    
    print(f"Saved agent and weights at index {index}")
    
    return agent_filename, weights_filename

def is_point_in_hull(point, hull):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= 0.000001)  # Added tolerance for numerical stability
        for eq in hull.equations
    )
    
def main(seed: int, config: str, num_samples: int) -> None:
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
    max_gym_steps = config["max_gym_steps"]
    env_params = config["env_params"]
    max_rho = env_params["max_rho"]
    network_params = config["network_params"]
    net_arch = network_params["net_arch"]
    rewards = config["rewards"]
    reward_list = [rewards["second"], rewards["third"]]
    
    agent_params["log"] = True
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
        max_rho=max_rho
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
        max_rho=max_rho
    )
    
    gym_env_test, _, _, _, g2op_env_test = setup_environment(
        env_name=env_name,
        test=False,
        seed=seed,
        action_space=53,
        first_reward=ScaledLinesCapacityReward,
        rewards_list=reward_list,
        actions_file=actions_file,
        env_type="_test",
        max_rho=max_rho
    )
    print(agent_params)
    # Reset the environment to verify dimensions
    gym_env.reset()
    gym_env_val.reset()
  
    num_obj = 3 
    GAMMA = 0.99
    
    ccs_data = []  # This will store the agent and weights for each CCS entry
    values = []
    ccs_list = []

    # Initialize the necessary directories
    current_date = datetime.now().strftime("%Y-%m-%d")
    dir_path = os.path.join(
        "morl_logs",
        "MC",
        env_name,
        f"{current_date}",
        f"{reward_list}",
        f"seed_{seed}",
    )
    os.makedirs(dir_path, exist_ok=True)

    trainer = MCMOPPOTrainer(
        iterations=5,
        max_gym_steps=max_gym_steps,
        seed=seed,
        results_dir=results_dir,
        env=gym_env,
        env_val=gym_env_val,
        env_test=gym_env_test,
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        run_name="runMC",
        project_name=project_name,
        net_arch=net_arch,
        g2op_env=g2op_env, 
        g2op_env_val=g2op_env_val,
        g2op_env_test=g2op_env_test,
        reward_list=reward_list,
        **agent_params
    )
    
    # Initialize variables
    values = []
    ccs_list = []
    ccs_data = []

    # Sample weights and evaluate
    for i in range(num_samples):
        weights = trainer.sample_weights(1)[0]  # Sample one weight vector
        eval_data, test_data, agent = trainer.run_single(weights=weights)
        
        eval_rewards_1 = np.array(sum_rewards(eval_data['eval_data_0']['eval_rewards']))
        eval_rewards_2 = np.array(sum_rewards(eval_data['eval_data_1']['eval_rewards']))
        mean_rewards = (eval_rewards_1 + eval_rewards_2) / 2
        
        # Convert numpy arrays to lists where necessary
        weights_array = weights.tolist()
        mean_rewards_array = mean_rewards.tolist()
        
        # Check if the current mean_rewards should be added to the CCS
        points = np.array(values + [mean_rewards_array])  # Combine existing points with the new point
        
        if len(points) > 3:  # Only attempt ConvexHull if there are enough points
            hull = ConvexHull(points)
            if is_point_in_hull(mean_rewards, hull):
                values.append(mean_rewards_array)
                ccs_list.append(mean_rewards_array)  # Add to CCS list
            else:
                print(f"Point {mean_rewards_array} is not in the convex hull, skipping.")
                continue  # Skip adding to CCS
        else:
            values.append(mean_rewards_array)
            ccs_list.append(mean_rewards_array)  # Add to CCS list
        
        # Save the agent and corresponding weights for each CCS entry
        agent_filename, weights_filename = save_agent(agent, weights, dir_path, i)
        
        test_data_0 = test_data.get("test_data_0")

        # Convert the numpy arrays/tensors to lists
        test_data_0_conv = convert_ndarray_to_list(test_data_0)

        # Add the converted data to ccs_data only if it belongs to the CCS
        ccs_data.append({
            "weights": weights_array,
            "returns": mean_rewards_array,
            "agent_file": agent_filename,
            "weights_file": weights_filename,
            "test_chronic": test_data_0_conv.get("eval_chronic"),
            "test_rewards": test_data_0_conv.get("eval_rewards"),
            "test_actions": test_data_0_conv.get("eval_actions"),
            "test_states": test_data_0_conv.get("eval_states"),
            "test_steps": test_data_0_conv.get("eval_steps"),
        })


    # Ensure all numpy arrays are converted to lists
    values = [v.tolist() if isinstance(v, np.ndarray) else v for v in values]
    ccs_list = [ccs_item.tolist() if isinstance(ccs_item, np.ndarray) else ccs_item for ccs_item in ccs_list]

    data_dict = {
        "ccs_data": ccs_data,
        "values": values,  # Already converted to lists
        "ccs_list": ccs_list  # Already converted to lists
    } 

    filename = f"morl_logs_mc{seed}.json"
    filepath = os.path.join(dir_path, filename)

    with open(filepath, "w") as json_file:
        json.dump(data_dict, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo MOPPO with specific seed and weights"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="base_config.json",
        help="Path to the configuration file (default: configs/base_config.json)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of weight vectors to sample"
    )
    args = parser.parse_args()

    main(args.seed, args.config, args.num_samples)
