import gymnasium as gym
import mo_gymnasium as mo_gym

from topgrid_morl.wrapper.ols import LinearSupport
from morl_baselines.single_policy.ser.mo_q_learning import MOQLearning


from datetime import datetime
import argparse
import json
import os

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
from topgrid_morl.wrapper.monte_carlo import MOPPOTrainer

# Recursive function to convert all numpy arrays to lists
def convert_ndarray_to_list(data):
    if isinstance(data, dict):
        return {key: convert_ndarray_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(element) for element in data]
    elif isinstance(data, np.ndarray):
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
  
    num_obj = 3 
    GAMMA = 0.99
    
    ols = LinearSupport(num_objectives=num_obj, epsilon=0.01, verbose=True)
    agents = []
    weights = []
    values = []
    ccs_list = []
    while not ols.ended():
        w = ols.next_weight(algo='ols')
        print(f"this weights will be given to the MOPPO: {w}")
        if ols.ended(): 
            break
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
        eval_data, agent = trainer.train_agent(weights=w)
        eval_rewards_1 = np.array(sum_rewards(eval_data['eval_data_0']['eval_rewards']))
        eval_rewards_2 = np.array(sum_rewards(eval_data['eval_data_1']['eval_rewards']))
        mean_rewards = (eval_rewards_1 + eval_rewards_2) / 2
        weights_array = np.array(w)
        mean_rewards_array = np.array(mean_rewards)
        print(f"this is mean rewards{mean_rewards_array}")
        #policies.append(new_policy)
       
        removed_inds, ccs = ols.add_solution(mean_rewards_array, w)
        agents.append(agent)
        values.append(mean_rewards_array)
        weights.append(w)
        ccs_list.append(ccs)
    

    weights_array = np.array(weights)
    values = np.array(values)
    final_ccs = ccs_list[-1]
    print(final_ccs)
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    dir_path = os.path.join(
        "morl_logs",
        "OLS",
        env_name,
        f"{current_date}",
        f"{reward_list}",
        f"seed_{seed}",
    )
    os.makedirs(dir_path, exist_ok=True)
    data_dict = {
        "weights": weights,
        "values": [v.tolist() for v in values],  # Convert numpy arrays to lists
        "ccs_list": ccs_list
    }   
    data_dict = convert_ndarray_to_list(data_dict)
    filename = f"morl_logs_ols.json"
    filepath = os.path.join(dir_path, filename)


    with open(filepath, "w") as json_file:
        json.dump(data_dict, json_file, indent=4)



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