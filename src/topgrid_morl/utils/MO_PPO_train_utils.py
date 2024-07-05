import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch as th
import wandb

from topgrid_morl.agent.MO_BaselineAgents import DoNothingAgent
from topgrid_morl.agent.MO_PPO import MOPPO, MOPPONet
from topgrid_morl.utils.MORL_analysis_utils import generate_variable_name


def initialize_network(
    obs_dim: Tuple[int],
    action_dim: int,
    reward_dim: int,
    net_arch: List[int] = [64, 64],
) -> MOPPONet:
    """
    Initialize the neural network for the MO-PPO agent.

    Args:
        obs_dim (Tuple[int]): Observation dimension.
        action_dim (int): Action dimension.
        reward_dim (int): Reward dimension.
        net_arch (List[int]): Network architecture.

    Returns:
        MOPPONet: Initialized neural network.
    """
    return MOPPONet(obs_dim, action_dim, reward_dim, net_arch=net_arch)


def initialize_agent(
    env,
    weights: np.ndarray,
    obs_dim: Tuple[int],
    action_dim: int,
    reward_dim: int,
    **agent_params,
) -> MOPPO:
    """
    Initialize the MO-PPO agent.

    Args:
        env: The environment object.
        weights (np.ndarray): Weights for the objectives.
        obs_dim (Tuple[int]): Observation dimension.
        action_dim (int): Action dimension.
        reward_dim (int): Reward dimension.
        agent_params: Additional parameters for the agent.

    Returns:
        MOPPO: Initialized agent.
    """
    networks = initialize_network(obs_dim, action_dim, reward_dim)
    agent = MOPPO(env=env, weights=weights, networks=networks, **agent_params)
    env.reset()
    return agent


def pad_list(actions: List[List[int]]) -> np.ndarray:
    """
    Pad the actions list to have the same length for all sublists.

    Args:
        actions (List[List[int]]): List of action sublists.

    Returns:
        np.ndarray: Padded actions array.
    """
    max_length = max(len(sublist) for sublist in actions)
    padded_list = [sublist + [0] * (max_length - len(sublist)) for sublist in actions]
    return np.array(padded_list)


def load_saved_data(
    weights: List[float],
    num_episodes: int,
    seed: int,
    results_dir: str,
    donothing_prefix: str = "DoNothing",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, np.ndarray, np.ndarray]:
    """
    Load saved data based on weights and episodes.

    Args:
        weights (List[float]): List of weights.
        num_episodes (int): Number of episodes.
        seed (int): Random seed.
        results_dir (str): Directory of results.
        donothing_prefix (str): Prefix for DoNothing results.

    Returns:
        Tuple: Loaded reward matrix, actions, total steps, parameters, DoNothing reward matrix, and DoNothing total steps.
    """
    weights_str = "_".join(map(str, weights))
    filename_base = f"weights_{weights_str}_episodes_{num_episodes}_seed_{seed}"

    result_filepath = os.path.join(results_dir, f"results_{filename_base}.npz")
    data = np.load(result_filepath)
    reward_matrix = data["reward_matrix"]
    actions = data["actions"]
    total_steps = data["total_steps"]

    model_filepath = os.path.join(results_dir, f"model_{filename_base}.pth")
    # agent.networks.load_state_dict(th.load(model_filepath))

    params_filepath = os.path.join(results_dir, f"params_{filename_base}.json")
    with open(params_filepath, "r") as json_file:
        params = json.load(json_file)

    donothing_filename = (
        f"{donothing_prefix}_reward_matrix_{num_episodes}_episodes_{seed}.npy"
    )
    donothing_filepath = os.path.join(results_dir, donothing_filename)
    donothing_reward_matrix = np.load(donothing_filepath)

    donothing_total_steps_filename = (
        f"{donothing_prefix}_total_steps_{num_episodes}_episodes_{seed}.npy"
    )
    donothing_total_steps_filepath = os.path.join(
        results_dir, donothing_total_steps_filename
    )
    donothing_total_steps = np.load(donothing_total_steps_filepath)

    print(f"Loaded results from {result_filepath}")
    print(f"Loaded model from {model_filepath}")
    print(f"Loaded parameters from {params_filepath}")
    print(f"Loaded DoNothing reward matrix from {donothing_filepath}")
    print(f"Loaded DoNothing total steps from {donothing_total_steps_filepath}")

    return (
        reward_matrix,
        actions,
        total_steps,
        params,
        donothing_reward_matrix,
        donothing_total_steps,
    )


def train_agent(
    weight_vectors: List[np.ndarray],
    num_episodes: int,
    max_ep_steps: int,
    results_dir: str,
    seed: int,
    env,
    obs_dim: Tuple[int],
    action_dim: int,
    reward_dim: int,
    run_name: str,
    **agent_params,
) -> None:
    """
    Train the agent using MO-PPO.

    Args:
        weight_vectors (List[np.ndarray]): List of weight vectors.
        num_episodes (int): Number of episodes.
        max_ep_steps (int): Maximum steps per episode.
        results_dir (str): Directory to save results.
        seed (int): Random seed.
        env: The environment object.
        obs_dim (Tuple[int]): Observation dimension.
        action_dim (int): Action dimension.
        reward_dim (int): Reward dimension.
        run_name (str): Name of the run.
        agent_params: Additional parameters for the agent.
    """
    os.makedirs(results_dir, exist_ok=True)

    for weights in weight_vectors:
        agent = initialize_agent(
            env, weights, obs_dim, action_dim, reward_dim, **agent_params
        )
        agent.weights = th.tensor(weights).float().to(agent.device)
        run = wandb.init(
            project="TOPGrid_MORL",
            name=generate_variable_name(
                base_name=run_name,
                num_episodes=num_episodes,
                weights=weights,
                seed=seed,
            ),
        )
        reward_matrix, actions, total_steps = agent.train(
            num_episodes=num_episodes, max_ep_steps=max_ep_steps, reward_dim=reward_dim
        )
        run.finish()
        actions = pad_list(actions)

        weights_str = "_".join(map(str, weights))
        filename_base = f"weights_{weights_str}_episodes_{num_episodes}_seed_{seed}"

        result_filepath = os.path.join(results_dir, f"results_{filename_base}.npz")
        np.savez(
            result_filepath,
            reward_matrix=reward_matrix,
            actions=actions,
            total_steps=total_steps,
            weights=weights,
            num_episodes=num_episodes,
        )

        model_filepath = os.path.join(results_dir, f"model_{filename_base}.pth")
        th.save(agent.networks.state_dict(), model_filepath)

        params = {
            "weights": weights.tolist(),
            "num_episodes": num_episodes,
            "seed": seed,
            "max_ep_steps": max_ep_steps,
            "reward_dim": reward_dim,
            "model_filepath": model_filepath,
            "result_filepath": result_filepath,
        }
        params_filepath = os.path.join(results_dir, f"params_{filename_base}.json")
        with open(params_filepath, "w") as json_file:
            json.dump(params, json_file, indent=4)

        print(f"Saved results for weights {weights} to {result_filepath}")
        print(f"Saved model for weights {weights} to {model_filepath}")
        print(f"Saved parameters for weights {weights} to {params_filepath}")


def train_and_save_donothing_agent(
    action_space,
    gym_env,
    num_episodes: int,
    max_ep_steps: int,
    seed: int,
    reward_dim: int,
    save_dir: str = "results",
    file_prefix: str = "DoNothing",
) -> None:
    """
    Trains a DoNothing agent and saves the reward matrix to a file.

    Args:
        action_space: The action space of the environment.
        gym_env: The gym environment instance.
        num_episodes (int): Number of episodes to train.
        max_ep_steps (int): Maximum number of steps per episode.
        seed (int): Random seed.
        reward_dim (int): The dimensionality of the reward.
        save_dir (str): Directory where the reward matrix will be saved.
        file_prefix (str): Prefix for the filename of the saved reward matrix.
    """
    # Create an instance of the agent
    agent = DoNothingAgent(action_space=action_space, gymenv=gym_env)

    # Train the agent
    reward_matrix, total_steps = agent.train(
        num_episodes=num_episodes, max_ep_steps=max_ep_steps, reward_dim=reward_dim
    )

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Define the filenames
    file_path_reward = os.path.join(
        save_dir, f"{file_prefix}_reward_matrix_{num_episodes}_episodes_{seed}.npy"
    )
    file_path_steps = os.path.join(
        save_dir, f"{file_prefix}_total_steps_{num_episodes}_episodes_{seed}.npy"
    )

    # Save the reward matrix and total steps
    np.save(file_path_reward, reward_matrix)
    np.save(file_path_steps, total_steps)
    print(f"Reward matrix saved to {file_path_reward}")
    print(f"Total steps saved to {file_path_steps}")
