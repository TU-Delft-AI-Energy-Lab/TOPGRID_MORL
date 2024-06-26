# train_utils.py
import os
import numpy as np
import torch as th
from MO_PPO_ import MOPPONet, MOPPO
from BaselineAgents import DoNothingAgent
import json

# Function to initialize the neural network
def initialize_network(obs_dim, action_dim, reward_dim, net_arch = [64, 64]):
    # Example architecture (adjust as needed)
    return MOPPONet(obs_dim, action_dim, reward_dim, net_arch=net_arch)

def initialize_agent(env, weights, obs_dim, action_dim, reward_dim, **agent_params):
    networks = initialize_network(obs_dim, action_dim, reward_dim)
    agent = MOPPO(env=env, weights=weights, networks=networks, **agent_params)
    env.reset()
    return agent

# Function to pad the actions list
def pad_list(actions):
    max_length = max(len(sublist) for sublist in actions)
    padded_list = [sublist + [0] * (max_length - len(sublist)) for sublist in actions]
    return np.array(padded_list)

# Function to load a specific file based on weights and episodes
def load_saved_data(weights, num_episodes, seed, results_dir, donothing_prefix="DoNothing"):
    weights_str = "_".join(map(str, weights))
    filename_base = f"weights_{weights_str}_episodes_{num_episodes}_seed_{seed}"

    # Load the results
    result_filepath = os.path.join(results_dir, f"results_{filename_base}.npz")
    data = np.load(result_filepath)
    reward_matrix = data['reward_matrix']
    actions = data['actions']
    total_steps = data['total_steps']
    
    # Load the model (example, adjust as needed)
    model_filepath = os.path.join(results_dir, f"model_{filename_base}.pth")
    # agent.networks.load_state_dict(th.load(model_filepath))
    
    # Load the parameters (example, adjust as needed)
    params_filepath = os.path.join(results_dir, f"params_{filename_base}.json")
    with open(params_filepath, 'r') as json_file:
        params = json.load(json_file)
    
    # Load DoNothing results
    donothing_filename = f"{donothing_prefix}_reward_matrix_{num_episodes}_episodes_{seed}.npy"
    donothing_filepath = os.path.join(results_dir, donothing_filename)
    donothing_reward_matrix = np.load(donothing_filepath)

    donothing_total_steps_filename = f"{donothing_prefix}_total_steps_{num_episodes}_episodes_{seed}.npy"
    donothing_total_steps_filepath = os.path.join(results_dir, donothing_total_steps_filename)
    donothing_total_steps = np.load(donothing_total_steps_filepath)
    
    print(f"Loaded results from {result_filepath}")
    print(f"Loaded model from {model_filepath}")
    print(f"Loaded parameters from {params_filepath}")
    print(f"Loaded DoNothing reward matrix from {donothing_filepath}")
    print(f"Loaded DoNothing total steps from {donothing_total_steps_filepath}")
    
    return reward_matrix, actions, total_steps, params, donothing_reward_matrix, donothing_total_steps


# Function to train the agent using MO-PPO
def train_agent(weight_vectors, num_episodes, max_ep_steps, results_dir, seed, env, obs_dim, action_dim, reward_dim, **agent_params):
    os.makedirs(results_dir, exist_ok=True)
    
    for weights in weight_vectors:
        agent = initialize_agent(env, weights, obs_dim, action_dim, reward_dim, **agent_params)
        agent.weights = th.tensor(weights).float().to(agent.device)
        
        reward_matrix, actions, total_steps = agent.train(num_episodes=num_episodes, max_ep_steps=max_ep_steps, reward_dim=reward_dim)
        actions = pad_list(actions)
        
        weights_str = "_".join(map(str, weights))
        filename_base = f"weights_{weights_str}_episodes_{num_episodes}_seed_{seed}"
        
        result_filepath = os.path.join(results_dir, f"results_{filename_base}.npz")
        np.savez(result_filepath, reward_matrix=reward_matrix, actions=actions, total_steps=total_steps, weights=weights, num_episodes=num_episodes)
        
        model_filepath = os.path.join(results_dir, f"model_{filename_base}.pth")
        th.save(agent.networks.state_dict(), model_filepath)
        
        params = {
            "weights": weights.tolist(),
            "num_episodes": num_episodes,
            "seed": seed,
            "max_ep_steps": max_ep_steps,
            "reward_dim": reward_dim,
            "model_filepath": model_filepath,
            "result_filepath": result_filepath
        }
        params_filepath = os.path.join(results_dir, f"params_{filename_base}.json")
        with open(params_filepath, 'w') as json_file:
            json.dump(params, json_file, indent=4)
        
        print(f"Saved results for weights {weights} to {result_filepath}")
        print(f"Saved model for weights {weights} to {model_filepath}")
        print(f"Saved parameters for weights {weights} to {params_filepath}")
        

def train_and_save_donothing_agent( action_space, gym_env, num_episodes, max_ep_steps, seed, reward_dim, save_dir="results", file_prefix="DoNothing"):
    """
    Trains a DoNothing agent and saves the reward matrix to a file.

    Parameters:
    - agent_class: The class of the agent to be used for training.
    - action_space: The action space of the environment.
    - gym_env: The gym environment instance.
    - num_episodes: Number of episodes to train.
    - max_ep_steps: Maximum number of steps per episode.
    - reward_dim: The dimensionality of the reward.
    - save_dir: Directory where the reward matrix will be saved.
    - file_prefix: Prefix for the filename of the saved reward matrix.
    """
    # Create an instance of the agent
    agent = DoNothingAgent(action_space=action_space, gymenv=gym_env)

    # Train the agent
    reward_matrix, total_steps = agent.train(num_episodes=num_episodes, max_ep_steps=max_ep_steps, reward_dim=reward_dim)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Define the filename
    file_path_reward = os.path.join(save_dir, f"{file_prefix}_reward_matrix_{num_episodes}_episodes_{seed}.npy")
    file_path_steps = os.path.join(save_dir, f"{file_prefix}_total_steps_{num_episodes}_episodes_{seed}.npy")
    # Save the reward matrix
    np.save(file_path_reward, reward_matrix)
    np.save(file_path_steps, total_steps)
    print(f"Reward matrix saved to {file_path_reward}")