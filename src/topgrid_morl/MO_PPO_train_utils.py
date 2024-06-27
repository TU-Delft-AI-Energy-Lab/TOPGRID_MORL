# train_utils.py
import os
import numpy as np
import torch as th
from MO_PPO_ import MOPPONet, MOPPO
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
def load_saved_data(weights, num_episodes, results_dir):
    weights_str = "_".join(map(str, weights))
    filename_base = f"weights_{weights_str}_episodes_{num_episodes}"
    
    # Load the results
    result_filepath = os.path.join(results_dir, f"results_{filename_base}.npz")
    data = np.load(result_filepath)
    reward_matrix = data['reward_matrix']
    actions = data['actions']
    
    # Load the model
    model_filepath = os.path.join(results_dir, f"model_{filename_base}.pth")
    #agent.networks.load_state_dict(th.load(model_filepath))
    
    # Load the parameters
    params_filepath = os.path.join(results_dir, f"params_{filename_base}.json")
    with open(params_filepath, 'r') as json_file:
        params = json.load(json_file)
    
    print(f"Loaded results from {result_filepath}")
    print(f"Loaded model from {model_filepath}")
    print(f"Loaded parameters from {params_filepath}")
    
    return reward_matrix, actions, params


# Function to train the agent using MO-PPO
def train_agent(weight_vectors, num_episodes, max_ep_steps, results_dir, env, obs_dim, action_dim, reward_dim, **agent_params):
    os.makedirs(results_dir, exist_ok=True)
    
    for weights in weight_vectors:
        agent = initialize_agent(env, weights, obs_dim, action_dim, reward_dim, **agent_params)
        agent.weights = th.tensor(weights).float().to(agent.device)
        
        reward_matrix, actions = agent.train(num_episodes=num_episodes, max_ep_steps=max_ep_steps, reward_dim=reward_dim)
        actions = pad_list(actions)
        
        weights_str = "_".join(map(str, weights))
        filename_base = f"weights_{weights_str}_episodes_{num_episodes}"
        
        result_filepath = os.path.join(results_dir, f"results_{filename_base}.npz")
        np.savez(result_filepath, reward_matrix=reward_matrix, actions=actions, weights=weights, num_episodes=num_episodes)
        
        model_filepath = os.path.join(results_dir, f"model_{filename_base}.pth")
        th.save(agent.networks.state_dict(), model_filepath)
        
        params = {
            "weights": weights.tolist(),
            "num_episodes": num_episodes,
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