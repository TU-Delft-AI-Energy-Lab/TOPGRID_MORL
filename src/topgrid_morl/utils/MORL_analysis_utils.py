from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

import argparse
import json
import numpy as np
from grid2op.Reward import EpisodeDurationReward
from topgrid_morl.envs.GridRewards import ScaledEpisodeDurationReward, ScaledTopoActionReward
from topgrid_morl.envs.EnvSetup import setup_environment
from topgrid_morl.utils.MO_PPO_train_utils import train_agent
from grid2op.Chronics import ChronicsHandler

def plot_multiple_subplots(
    reward_matrices: List[npt.NDArray[np.float64]], summed_episodes: int
) -> None:
    """
    Plot multiple subplots of grouped bar plots for the provided reward matrices.

    Args:
        reward_matrices (List[npt.NDArray[np.float64]]): List of reward matrices.
        summed_episodes (int): Number of episodes to sum together for each bar.
    """
    num_matrices = len(reward_matrices)
    fig, axes = plt.subplots(1, num_matrices, figsize=(14, 6))

    if num_matrices == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one subplot

    for i, (ax, reward_matrix) in enumerate(zip(axes, reward_matrices)):
        plot_grouped_bar_plot(ax, reward_matrix, summed_episodes)
        ax.set_title(f"Subplot {i + 1}: Grouped Bar Plot for Reward Matrix {i + 1}")

    plt.tight_layout()
    plt.show()


def plot_grouped_bar_plot(
    ax: plt.Axes, reward_matrix: npt.NDArray[np.float64], summed_episodes: int
) -> None:
    """
    Plot a grouped bar plot for the given reward matrix.

    Args:
        ax (plt.Axes): Matplotlib axis to plot on.
        reward_matrix (npt.NDArray[np.float64]): Reward matrix to plot.
        summed_episodes (int): Number of episodes to sum together for each bar.
    """
    num_rows, num_cols = reward_matrix.shape
    rows_per_summary = summed_episodes
    num_summaries = num_rows // rows_per_summary

    summarized_matrix = np.array(
        [
            reward_matrix[i * rows_per_summary : (i + 1) * rows_per_summary].mean(
                axis=0
            )
            for i in range(num_summaries)
        ]
    )

    scaled_matrix = scale_columns_independently(summarized_matrix)

    bar_width = 0.2
    indices = np.arange(num_summaries)

    for col in range(num_cols):
        ax.bar(
            indices + col * bar_width,
            scaled_matrix[:, col],
            width=bar_width,
            label=f"Reward {col}",
        )

    ax.set_xlabel("Summary Index", fontsize=14)
    ax.set_ylabel("Average Value", fontsize=14)
    ax.set_title("Grouped Bar Plot for Summarized Rows", fontsize=16)
    ax.set_xticks(indices + bar_width * (num_cols - 1) / 2)
    ax.set_xticklabels([f"Part {i}" for i in range(num_summaries)], fontsize=12)
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)


def calculate_correlation(reward_matrix: npt.NDArray[np.float64]) -> None:
    """
    Calculate and plot the Pearson correlation coefficient between different reward columns.

    Args:
        reward_matrix (npt.NDArray[np.float64]): Reward matrix to analyze.
    """
    l2rpn = reward_matrix[:, 0]
    lines = reward_matrix[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(l2rpn, lines, color="blue", label="Episode Rewards")
    plt.title("Scatter Plot of L2RPN Reward against Lines Reward")
    plt.xlabel("L2RPN Reward")
    plt.ylabel("Lines Reward")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_total_sums(
    reward_matrices: List[npt.NDArray[np.float64]], labels: List[str]
) -> None:
    """
    Plot the total sums of rewards in a 3D scatter plot.

    Args:
        reward_matrices (List[npt.NDArray[np.float64]]): List of reward matrices.
        labels (List[str]): List of labels for the reward matrices.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for reward_matrix, label in zip(reward_matrices, labels):
        scaled_matrix = scale_columns_independently(reward_matrix)
        total_sums = np.sum(scaled_matrix, axis=0)

        total_reward_1 = total_sums[0]
        total_reward_2 = total_sums[1]
        total_reward_3 = total_sums[2]

        ax.scatter(
            total_reward_1, total_reward_2, total_reward_3, marker="o", label=label
        )
        ax.text(
            total_reward_1,
            total_reward_2,
            total_reward_3,
            f"({total_reward_1: .2f}, {total_reward_2: .2f}, {total_reward_3: .2f})",
            fontsize=10,
            color="blue",
        )

    ax.set_xlabel("Total Reward 1", fontsize=12)
    ax.set_ylabel("Total Reward 2", fontsize=12)
    ax.set_zlabel("Total Reward 3", fontsize=12)
    ax.set_title("Total Sums of Rewards", fontsize=14)
    ax.legend()

    plt.show()


def generate_variable_name(
    base_name: str, max_gym_steps: int, weights: List[float], seed: int
) -> str:
    """
    Generate a variable name based on the specifications.

    Args:
        base_name (str): Base name for the variable.
        num_episodes (int): Number of episodes.
        weights (List[float]): List of weights.
        seed (int): Seed value.

    Returns:
        str: Generated variable name.
    """
    weights_str = "_".join(map(str, weights))
    return f"{base_name}_episodes_{max_gym_steps}_weights_{weights_str}_seed_{seed}"


def scale_columns_independently(
    reward_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Scale the columns of the reward matrix independently to the range [0, 1].

    Args:
        reward_matrix (npt.NDArray[np.float64]): Reward matrix to scale.

    Returns:
        npt.NDArray[np.float64]: Scaled reward matrix.
    """
    scaler = MinMaxScaler()
    scaled_matrix = np.zeros_like(reward_matrix)

    for col in range(reward_matrix.shape[1]):
        scaled_matrix[:, col] = scaler.fit_transform(
            reward_matrix[:, col].reshape(-1, 1)
        ).flatten()

    return scaled_matrix


def plot_3d_mean_std(returns_dict: Dict[str, npt.NDArray[np.float64]]) -> None:
    """
    Plot a 3D scatter plot with mean and standard deviation of the rewards.

    Args:
        returns_dict (Dict[str, npt.NDArray[np.float64]]): Dictionary containing returns for different weight settings.
    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")

    for weight_setting, returns in returns_dict.items():
        sum_across_episodes = np.sum(returns, axis=1)
        means = np.mean(sum_across_episodes, axis=0)
        std_devs = np.std(sum_across_episodes, axis=0)

        ax.scatter(
            means[0],
            means[1],
            means[2],
            s=100,
            label=f"Mean {weight_setting}",
            marker="o",
        )
        ax.text(
            means[0],
            means[1],
            means[2],
            f"Mean: ({means[0]: .2f}, {means[1]: .2f}, {means[2]: .2f})",
        )
        ax.errorbar(
            means[0],
            means[1],
            means[2],
            xerr=std_devs[0],
            yerr=std_devs[1],
            zerr=std_devs[2],
            fmt="o",
            capsize=5,
            label=f"Std Dev {weight_setting}",
        )

    ax.set_xlabel("Total Reward 1", fontsize=12)
    ax.set_ylabel("Total Reward 2", fontsize=12)
    ax.set_zlabel("Total Reward 3", fontsize=12)
    ax.set_title("Total Sums of Rewards with Mean and Std Deviation", fontsize=14)
    ax.legend()

    plt.show()


def plot_mean_std_rewards(
    returns_dict: Dict[str, npt.NDArray[np.float64]], reward_dim: int
) -> None:
    """
    Plot mean and standard deviation of rewards across episodes.

    Args:
        returns_dict (Dict[str, npt.NDArray[np.float64]]): Dictionary containing returns for different weight settings.
        reward_dim (int): Number of reward dimensions.
    """
    colors = sns.color_palette("husl", len(returns_dict))
    fig, axes = plt.subplots(1, reward_dim, figsize=(18, 6), sharex=True, sharey=True)

    if reward_dim == 1:
        axes = [axes]

    for reward_idx in range(reward_dim):
        ax = axes[reward_idx]
        for (weight_setting, reward_matrix), color in zip(returns_dict.items(), colors):
            mean_rewards = np.mean(reward_matrix, axis=0)[:, reward_idx]
            std_rewards = np.std(reward_matrix, axis=0)[:, reward_idx]
            num_episodes = mean_rewards.shape[0]

            ax.errorbar(
                range(num_episodes),
                mean_rewards,
                yerr=std_rewards,
                label=f"Weights {weight_setting}",
                capsize=5,
                marker="o",
                linestyle="--",
                color=color,
            )

        ax.set_title(f"Reward {reward_idx + 1}", fontsize=16)
        ax.set_xlabel("Episodes", fontsize=14)
        ax.set_ylabel("Rewards", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)
        sns.despine(trim=True)

    plt.tight_layout()
    plt.show()


def plot_mean_std_total_steps(
    total_steps_dict: Dict[str, npt.NDArray[np.float64]], num_episodes: int
) -> None:
    """
    Plot mean and standard deviation of total steps across episodes.

    Args:
        total_steps_dict (Dict[str, npt.NDArray[np.float64]]):
        Dictionary containing total steps for different weight settings.
        num_episodes (int): Number of episodes.
    """
    colors = sns.color_palette("husl", len(total_steps_dict))
    fig, axes = plt.subplots(
        1, len(total_steps_dict), figsize=(18, 6), sharex=True, sharey=True
    )

    if len(total_steps_dict) == 1:
        axes = [axes]

    for idx, (weight_setting, total_steps) in enumerate(total_steps_dict.items()):
        ax = axes[idx]
        mean_total_steps = np.mean(total_steps, axis=0)
        std_total_steps = np.std(total_steps, axis=0)

        ax.errorbar(
            range(num_episodes),
            mean_total_steps,
            yerr=std_total_steps,
            label=f"Weights {weight_setting}",
            capsize=5,
            marker="o",
            linestyle="--",
            color=colors[idx],
        )

        ax.set_title(f"Weights {weight_setting}", fontsize=16)
        ax.set_xlabel("Episodes", fontsize=14)
        ax.set_ylabel("Total Steps", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)
        sns.despine()

    plt.tight_layout()
    plt.show()


def normalize_reward_matrix(
    reward_matrix: npt.NDArray[np.float64],
    total_steps: npt.NDArray[np.float64],
    num_seeds: int,
    episode_dur: bool = True,
) -> npt.NDArray[np.float64]:
    """
    Normalize the reward matrix by total steps.

    Args:
        reward_matrix (npt.NDArray[np.float64]): Reward matrix to normalize.
        total_steps (npt.NDArray[np.float64]): Total steps for normalization.
        num_seeds (int): Number of seeds.
        episode_dur (bool): Whether to include episode duration in normalization.

    Returns:
        npt.NDArray[np.float64]: Normalized reward matrix.
    """
    normalized_reward_matrix = np.zeros_like(reward_matrix)
    for seed in range(num_seeds):
        normalized_reward_matrix[seed] = (
            reward_matrix[seed] / total_steps[seed][:, np.newaxis]
        )
        if episode_dur:
            normalized_reward_matrix[seed][:, 0] = reward_matrix[seed][:, 0]
    return normalized_reward_matrix


def get_returns(
    reward_matrices: List[npt.NDArray[np.float64]], num_seeds: int, reward_dim: int
) -> npt.NDArray[np.float64]:
    """
    Get the returns from the reward matrices.

    Args:
        reward_matrices (List[npt.NDArray[np.float64]]): List of reward matrices.
        num_seeds (int): Number of seeds.
        reward_dim (int): Number of reward dimensions.

    Returns:
        npt.NDArray[np.float64]: Returns matrix.
    """
    return_matrix = np.zeros((num_seeds, reward_dim), dtype=np.float64)
    for seed in range(num_seeds):
        return_matrix[seed] = reward_matrices[seed].sum(axis=0)
    return return_matrix


def create_action_to_substation_mapping():
    env_name = "rte_case5_example"
    config_path = "configs/base_config.json"
    seed = 71

    # Load configuration from file
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    agent_params = config["agent_params"]
    weights = np.array(config["weight_vectors"])  # Convert to numpy array for consistency
    max_gym_steps = config["max_gym_steps"]

    # Step 1: Setup Environment
    if env_name == "rte_case5_example":
        results_dir = "training_results_5bus_4094"
        action_dim = 53
        test_flag = True
        actions_file = 'filtered_actions.json'

    gym_env, obs_dim, action_dim, reward_dim, g2op_env = setup_environment(
        env_name=env_name,
        test=False,
        seed=seed,
        action_space=53,
        first_reward=ScaledEpisodeDurationReward,
        rewards_list=["ScaledLinesCapacity", "ScaledTopoAction"],
        actions_file=actions_file
    )
    obs = gym_env.reset(options={"time serie id": "01"} )

    action_to_substation_mapping = {}
    for action in range(53):  # Actions from 0 to 52
        # Convert the discrete action to grid2op action
        g2op_act = gym_env.action_space.from_gym(action)
        
        # Extract the affected substation ID
        modif_subs_id = g2op_act.as_dict().get('set_bus_vect', {}).get('modif_subs_id', [None])
        substation_id = modif_subs_id[0] if modif_subs_id else None
        
        # Add the mapping to the dictionary
        action_to_substation_mapping[action] = substation_id

    #print(action_to_substation_mapping)
    return action_to_substation_mapping

# Example usage
mapping = create_action_to_substation_mapping()