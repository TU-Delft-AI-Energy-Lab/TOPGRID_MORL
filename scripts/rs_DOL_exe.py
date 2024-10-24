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

import numpy as np
import matplotlib.pyplot as plt     
from mpl_toolkits.mplot3d import Axes3D
from topgrid_morl.agent.MO_PPO import MOPPO
from topgrid_morl.agent.MO_BaselineAgents import (  # Import the DoNothingAgent class
    DoNothingAgent,
)
from topgrid_morl.utils.MO_PPO_train_utils import initialize_agent
from topgrid_morl.envs.EnvSetup import setup_environment
from topgrid_morl.envs.GridRewards import L2RPNReward, TopoActionHourReward, EpisodeDurationReward, TopoActionDayReward, ScaledEpisodeDurationReward, ScaledLinesCapacityReward, LinesCapacityReward
from topgrid_morl.wrapper.ols_DOL import MOPPOTrainer

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
    
    print(f"Saved agent and  at index {index}")
    
    return agent_filename, weights_filename

def main(seed: int, config: str, learning_rate: float, vf_coef: float, ent_coef: float, clip_coef: float, update_epochs: int) -> None:
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
    ols_iterations = config['ols_iterations']
    reuse = config['reuse']
    case_study = config['case_study']
    name = config['name']
    use_opponent = config['use_opponent']
    rewards = config["rewards"]
    reward_list = [rewards["second"], rewards["third"]]
    
     # Override agent parameters with command-line arguments
    agent_params["learning_rate"] = learning_rate
    agent_params["vf_coef"] = vf_coef
    agent_params["ent_coef"] = ent_coef
    agent_params["clip_coef"] = clip_coef  # Adjust the clipping coefficient
    agent_params["update_epochs"] = update_epochs  # Set the number of epochs
    agent_params["log"] = True
    seed = seed
    
    
    config_data = {
        "config_name": config_name,
        "case_study": case_study,
        "project_name": project_name,
        "seed":seed,
        "agent_params": agent_params,
        "max_gym_steps": max_gym_steps,
        "env_params": env_params,
        "network_params": network_params,
        "reuse": reuse,
        "rewards": rewards,
        "use_opponent": use_opponent
    }
    # Step 1: Setup Environment
    if env_name == "rte_case5_example":
        results_dir = "training_results_5bus_4094"
        action_dim = 53
        actions_file = "filtered_actions.json"
    elif env_name == "l2rpn_case14_sandbox":
        results_dir = "training_results_14bus_4096"
        action_dim = 73
        actions_file = "tennet_actions.json"

    gym_env, obs_dim, action_dim, reward_dim, g2op_env = setup_environment(
        env_name=env_name,
        test=False,
        seed=seed,
        action_space=action_dim,
        first_reward=L2RPNReward,
        rewards_list=reward_list,
        actions_file=actions_file,
        max_rho=max_rho,
        use_opponent=use_opponent
    )

    gym_env_val, _, _, _, g2op_env_val = setup_environment(
        env_name=env_name,
        test=False,
        seed=seed,
        action_space=action_dim,
        first_reward=L2RPNReward,
        rewards_list=reward_list,
        actions_file=actions_file,
        env_type="_val",
        max_rho=max_rho, 
        use_opponent=use_opponent
    )
    
    gym_env_test, _, _, _, g2op_env_test = setup_environment(
        env_name=env_name,
        test=False,
        seed=seed,
        action_space=action_dim,
        first_reward=L2RPNReward,
        rewards_list=reward_list,
        actions_file=actions_file,
        env_type="_test",
        max_rho=max_rho,
        use_opponent=use_opponent
    )
    print(agent_params)
    # Reset the environment to verify dimensions
    gym_env.reset()
    gym_env_val.reset()
  
    num_obj = 3 
    GAMMA = 0.99
    
    ols = LinearSupport(num_objectives=num_obj, epsilon=0.00001, verbose=True)
    ccs_data = []  # This will store the agent and weights for each CCS entry
    values = []
    ccs_list = []

   
    # Initialize the necessary directories
    current_date = datetime.now().strftime("%Y-%m-%d")
    dir_path = os.path.join(
        "morl_logs",
        "3rd_trial",
        "RS",
        case_study,
        name,
    )
    os.makedirs(dir_path, exist_ok=True)
    i=0
    
    trainer = MOPPOTrainer(
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
            run_name="MC",
            project_name=project_name,
            net_arch=net_arch,
            g2op_env=g2op_env, 
            g2op_env_val=g2op_env_val,
            g2op_env_test=g2op_env_test,
            reward_list=reward_list,
            reuse=reuse,
            network_params=network_params, 
            env_params = env_params,
            **agent_params
        )
    while i<ols_iterations:
        w = ols.rs_sample_weights(reward_dim=3)
        print(f"this weights will be given to the MOPPO: {w}")
        
        eval_data, test_data, agent = trainer.run_single(weights=w)
        #print(test_data)
        
        eval_rewards_1 = np.array(sum_rewards(eval_data['eval_data_0']['eval_rewards']))
        eval_rewards_2 = np.array(sum_rewards(eval_data['eval_data_1']['eval_rewards']))
        mean_rewards = (eval_rewards_1 + eval_rewards_2) / 2
        
        # Convert numpy arrays to lists where necessary
        weights_array = w.tolist()
        mean_rewards_array = mean_rewards.tolist()

        removed_inds, ccs = ols.add_solution(mean_rewards_array, w)
        
        values.append(mean_rewards_array)
        ccs_list.append(ccs)  # ccs should already be a list
        # Save the agent and corresponding weights for each CCS entry
        agent_filename, weights_filename = save_agent(agent, w, dir_path, len(ccs_list) - 1)
        
        test_data_0 = test_data.get("test_data_0")
        test_data_1 = test_data.get('test_data_1')

        # Convert the numpy arrays/tensors to lists
        test_data_0_conv = convert_ndarray_to_list(test_data_0)
        test_data_1_conv = convert_ndarray_to_list(test_data_1)
        
        # Add the converted data to ccs_data
        ccs_data.append({
            
            "weights": weights_array,
            "returns": mean_rewards_array,
            "agent_file": agent_filename,
            "weights_file": weights_filename,
            "test_chronic_0": {
                "test_chronic": test_data_0_conv.get("eval_chronic"),
                "test_rewards": test_data_0_conv.get("eval_rewards"),
                "test_actions": test_data_0_conv.get("eval_actions"),
                'test_action_timestamp': test_data_0_conv.get('eval_action_timestamp'),
                'test_sub_ids': test_data_0_conv.get('sub_ids'),
                'test_topo_distance': test_data_0_conv.get('eval_topo_distance'),
                "test_states": test_data_0_conv.get("eval_states"),
                "test_steps": test_data_0_conv.get("eval_steps"),
            },
            "test_chronic_1": {
                "test_chronic": test_data_1_conv.get("eval_chronic"),
                "test_rewards": test_data_1_conv.get("eval_rewards"),
                "test_actions": test_data_1_conv.get("eval_actions"),
                'test_action_timestamp': test_data_1_conv.get('eval_action_timestamp'),
                'test_sub_ids': test_data_1_conv.get('sub_ids'),
                'test_topo_distance': test_data_1_conv.get('eval_topo_distance'),
                "test_states": test_data_1_conv.get("eval_states"),
                "test_steps": test_data_1_conv.get("eval_steps")
            }
        })
                
        #after 10 ols iterations: stop and save
        if ols_iterations == 20: 
            if i == 4: 
                # Ensure all numpy arrays are converted to lists
                values = [v.tolist() if isinstance(v, np.ndarray) else v for v in values]
                ccs_list = [ccs_item.tolist() if isinstance(ccs_item, np.ndarray) else ccs_item for ccs_item in ccs_list]

                
                
                data_dict = {
                    'config': config_data,
                    "ccs_data": ccs_data,
                    "values": values,  # Already converted to lists
                    "ccs_list": ccs_list  # Already converted to lists
                } 
                
                filename = f"5_iteration_morl_logs_seed_{seed}.json"
                filepath = os.path.join(dir_path, filename)
                
                with open(filepath, "w") as json_file:
                    json.dump(data_dict, json_file, indent=4)
            elif i==9: 
                # Ensure all numpy arrays are converted to lists
                values = [v.tolist() if isinstance(v, np.ndarray) else v for v in values]
                ccs_list = [ccs_item.tolist() if isinstance(ccs_item, np.ndarray) else ccs_item for ccs_item in ccs_list]

                
                
                data_dict = {
                    'config': config_data,
                    "ccs_data": ccs_data,
                    "values": values,  # Already converted to lists
                    "ccs_list": ccs_list  # Already converted to lists
                } 
                
                filename = f"10_iteration_morl_logs_seed_{seed}.json"
                filepath = os.path.join(dir_path, filename)
                
                with open(filepath, "w") as json_file:
                    json.dump(data_dict, json_file, indent=4)
                
        
        i+=1
        
    # Ensure all numpy arrays are converted to lists
    values = [v.tolist() if isinstance(v, np.ndarray) else v for v in values]
    ccs_list = [ccs_item.tolist() if isinstance(ccs_item, np.ndarray) else ccs_item for ccs_item in ccs_list]

    
    
    data_dict = {
        'config': config_data,
        "ccs_data": ccs_data,
        "values": values,  # Already converted to lists
        "ccs_list": ccs_list  # Already converted to lists
    } 
    
    if ols_iterations== 20: 
        filename = f"20_iteration_morl_logs_seed_{seed}.json"
    else:    
        filename = f"morl_logs_seed_{seed}.json"
    filepath = os.path.join(dir_path, filename)
    
    with open(filepath, "w") as json_file:
        json.dump(data_dict, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiment with specific seed and weights"
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="base_config.json",
        help="Path to the configuration file (default: configs/base_config.json)",
    )
    
    # Add arguments for learning rate, value function coefficient, entropy coefficient, clip coefficient, and update epochs
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate for the agent"
    )
    parser.add_argument(
        "--vf_coef", type=float, default=1.0, help="Value function coefficient"
    )
    parser.add_argument(
        "--ent_coef", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--clip_coef", type=float, default=0.3, help="Clip coefficient for PPO"
    )
    parser.add_argument(
        "--update_epochs", type=int, default=10, help="Number of update epochs for training"
    )
    
    
    args = parser.parse_args()

    # Pass the parsed arguments to the main function
    main(args.seed, args.config, args.learning_rate, args.vf_coef, args.ent_coef, args.clip_coef, args.update_epochs)