import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import torch as th
import wandb

from topgrid_morl.agent.MO_BaselineAgents import DoNothingAgent
from topgrid_morl.agent.MO_PPO import MOPPO, MOPPONet
#from topgrid_morl.utils.MORL_analysis_utils import generate_variable_name

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    return MOPPONet(obs_dim, action_dim, reward_dim, net_arch)


def initialize_agent(
    env: Any,
    env_val:Any,
    g2op_env: Any, 
    g2op_env_val: Any,
    weights: npt.NDArray[np.float64],
    obs_dim: Tuple[int],
    action_dim: int,
    reward_dim: int,
    net_arch: List[int] = [64, 64],
    seed: int = 42,
    generate_reward: int = False,
    **agent_params: Any,
) -> MOPPO:
    """
    Initialize the MO-PPO agent.

    Args:
        env: The environment object.
        weights (npt.NDArray[np.float64]): Weights for the objectives.
        obs_dim (Tuple[int]): Observation dimension.
        action_dim (int): Action dimension.
        reward_dim (int): Reward dimension.
        agent_params: Additional parameters for the agent.

    Returns:
        MOPPO: Initialized agent.
    """
    networks = initialize_network(obs_dim, action_dim, reward_dim, net_arch=net_arch)
    agent = MOPPO(env=env,env_val=env_val, g2op_env=g2op_env, g2op_env_val=g2op_env_val, weights=weights, networks=networks, seed=seed, generate_reward=generate_reward, **agent_params)
    env.reset()
    return agent


def pad_list(actions: List[List[int]]) -> npt.NDArray[np.int_]:
    """
    Pad the actions list to have the same length for all sublists.

    Args:
        actions (List[List[int]]): List of action sublists.

    Returns:
        npt.NDArray[np.int_]: Padded actions array.
    """
    max_length = max(len(sublist) for sublist in actions)
    padded_list = [sublist + [0] * (max_length - len(sublist)) for sublist in actions]
    return np.array(padded_list, dtype=np.int_)


def load_saved_data(
    weights: List[float],
    num_episodes: int,
    seed: int,
    results_dir: str,
    donothing_prefix: str = "DoNothing",
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
    Dict[str, Any],
    npt.NDArray[np.float64],
    npt.NDArray[np.int_],
]:
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
    reward_matrix: npt.NDArray[np.float64] = data["reward_matrix"]
    actions: npt.NDArray[np.int_] = data["actions"]
    total_steps: npt.NDArray[np.int_] = data["total_steps"]

    model_filepath = os.path.join(results_dir, f"model_{filename_base}.pth")
    # agent.networks.load_state_dict(th.load(model_filepath))

    params_filepath = os.path.join(results_dir, f"params_{filename_base}.json")
    with open(params_filepath, "r") as json_file:
        params = json.load(json_file)

    donothing_filename = (
        f"{donothing_prefix}_reward_matrix_{num_episodes}_episodes_{seed}.npy"
    )
    donothing_filepath = os.path.join(results_dir, donothing_filename)
    donothing_reward_matrix: npt.NDArray[np.float64] = np.load(donothing_filepath)

    donothing_total_steps_filename = (
        f"{donothing_prefix}_total_steps_{num_episodes}_episodes_{seed}.npy"
    )
    donothing_total_steps_filepath = os.path.join(
        results_dir, donothing_total_steps_filename
    )
    donothing_total_steps: npt.NDArray[np.int_] = np.load(
        donothing_total_steps_filepath
    )

    logger.info(f"Loaded results from {result_filepath}")
    logger.info(f"Loaded model from {model_filepath}")
    logger.info(f"Loaded parameters from {params_filepath}")
    logger.info(f"Loaded DoNothing reward matrix from {donothing_filepath}")
    logger.info(f"Loaded DoNothing total steps from {donothing_total_steps_filepath}")

    return (
        reward_matrix,
        actions,
        total_steps,
        params,
        donothing_reward_matrix,
        donothing_total_steps,
    )


def train_agent(
    weight_vectors: List[npt.NDArray[np.float64]],
    max_gym_steps: int,
    results_dir: str,
    seed: int,
    env: Any,
    env_val: Any,
    g2op_env: Any, 
    g2op_env_val: Any, 
    obs_dim: Tuple[int],
    action_dim: int,
    reward_dim: int,
    run_name: str,
    project_name: str = "TOPGrid_MORL_5",
    net_arch: List[int] = [64, 64],
    generate_reward: bool = False,
    reward_list: List = ["ScaledEpisodeDuration", "ScaledTopoAction"],
    **agent_params: Any,
) -> None:
    """
    Train the agent using MO-PPO.

    Args:
        weight_vectors (List[npt.NDArray[np.float64]]): List of weight vectors.
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
        weights_str = "_".join(map(str, weights))
        agent = initialize_agent(
            env,env_val, g2op_env, g2op_env_val, weights, obs_dim, action_dim, reward_dim, net_arch, seed, generate_reward, **agent_params
        )
        agent.weights = th.tensor(weights).cpu().to(agent.device)
        run = wandb.init(
            project=project_name,
            name=f"{run_name}_{reward_list[0]}_{reward_list[1]}_weights_{weights_str}_seed_{seed}",
            group=f"{reward_list[0]}_{reward_list[1]}",
            tags=[run_name]
        )
        agent.train(max_gym_steps=max_gym_steps, reward_dim=reward_dim, reward_list=reward_list)
        run.finish()
        """
        run = wandb.init(
            project="TOPGrid_MORL_5bus",
            name=generate_variable_name(
                base_name=run_name,
                max_gym_steps=max_gym_steps,
                weights=weights,
                seed=seed,
            )+'DoNothing',
            group=weights_str
            
        )
        do_nothing_agent = DoNothingAgent(env=env, env_val=env_val, log=agent_params["log"], device=agent_params["device"])
        do_nothing_agent.train(max_gym_steps=max_gym_steps, reward_dim=reward_dim)
        run.finish()
        """

def train_and_save_donothing_agent(
    action_space: Any,
    gym_env: Any,
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
    logger.info(f"Reward matrix saved to {file_path_reward}")
    logger.info(f"Total steps saved to {file_path_steps}")

