import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import List, Dict, Tuple, Union

def plot_multiple_subplots(reward_matrices: List[np.ndarray], summed_episodes: int) -> None:
    """
    Plot multiple subplots of grouped bar plots for the provided reward matrices.

    Args:
        reward_matrices (List[np.ndarray]): List of reward matrices.
        summed_episodes (int): Number of episodes to sum together for each bar.
    """
    num_matrices = len(reward_matrices)
    fig, axes = plt.subplots(1, num_matrices, figsize=(14, 6))

    if num_matrices == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one subplot

    for i, (ax, reward_matrix) in enumerate(zip(axes, reward_matrices)):
        plot_grouped_bar_plot(ax, reward_matrix, summed_episodes)
        ax.set_title(f'Subplot {i + 1}: Grouped Bar Plot for Reward Matrix {i + 1}')

    plt.tight_layout()
    plt.show()

def plot_grouped_bar_plot(ax: plt.Axes, reward_matrix: np.ndarray, summed_episodes: int) -> None:
    """
    Plot a grouped bar plot for the given reward matrix.

    Args:
        ax (plt.Axes): Matplotlib axis to plot on.
        reward_matrix (np.ndarray): Reward matrix to plot.
        summed_episodes (int): Number of episodes to sum together for each bar.
    """
    num_rows, num_cols = reward_matrix.shape
    rows_per_summary = summed_episodes
    num_summaries = num_rows // rows_per_summary

    summarized_matrix = np.array([
        reward_matrix[i * rows_per_summary:(i + 1) * rows_per_summary].mean(axis=0)
        for i in range(num_summaries)
    ])

    scaled_matrix = scale_columns_independently(summarized_matrix)

    bar_width = 0.2
    indices = np.arange(num_summaries)

    for col in range(num_cols):
        ax.bar(indices + col * bar_width, scaled_matrix[:, col], width=bar_width, label=f'Reward {col}')

    ax.set_xlabel('Summary Index', fontsize=14)
    ax.set_ylabel('Average Value', fontsize=14)
    ax.set_title('Grouped Bar Plot for Summarized Rows', fontsize=16)
    ax.set_xticks(indices + bar_width * (num_cols - 1) / 2)
    ax.set_xticklabels([f'Part {i}' for i in range(num_summaries)], fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.5)

def calculate_correlation(reward_matrix: np.ndarray) -> None:
    """
    Calculate and plot the Pearson correlation coefficient between different reward columns.

    Args:
        reward_matrix (np.ndarray): Reward matrix to analyze.
    """
    L2RPN = reward_matrix[:, 0]
    Lines = reward_matrix[:, 1]
    Distance = reward_matrix[:, 2]

    corr_L2RPN_Lines, _ = pearsonr(L2RPN, Lines)
    corr_L2RPN_Distance, _ = pearsonr(L2RPN, Distance)

    print(f'Pearson correlation coefficient between L2RPN and Lines: {corr_L2RPN_Lines:.4f}')
    print(f'Pearson correlation coefficient between L2RPN and Distance: {corr_L2RPN_Distance:.4f}')

    plt.figure(figsize=(8, 6))
    plt.scatter(L2RPN, Lines, color='blue', label='Episode Rewards')
    plt.title('Scatter Plot of L2RPN Reward against Lines Reward')
    plt.xlabel('L2RPN Reward')
    plt.ylabel('Lines Reward')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_total_sums(reward_matrices: List[np.ndarray], labels: List[str]) -> None:
    """
    Plot the total sums of rewards in a 3D scatter plot.

    Args:
        reward_matrices (List[np.ndarray]): List of reward matrices.
        labels (List[str]): List of labels for the reward matrices.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for reward_matrix, label in zip(reward_matrices, labels):
        scaled_matrix = scale_columns_independently(reward_matrix)
        total_sums = np.sum(scaled_matrix, axis=0)

        total_reward_1 = total_sums[0]
        total_reward_2 = total_sums[1]
        total_reward_3 = total_sums[2]

        ax.scatter(total_reward_1, total_reward_2, total_reward_3, marker='o', label=label)
        ax.text(total_reward_1, total_reward_2, total_reward_3,
                f'({total_reward_1:.2f}, {total_reward_2:.2f}, {total_reward_3:.2f})',
                fontsize=10, color='blue')
        print(total_reward_1, total_reward_2, total_reward_3)

    ax.set_xlabel('Total Reward 1', fontsize=12)
    ax.set_ylabel('Total Reward 2', fontsize=12)
    ax.set_zlabel('Total Reward 3', fontsize=12)
    ax.set_title('Total Sums of Rewards', fontsize=14)
    ax.legend()

    plt.show()

def generate_variable_name(base_name: str, num_episodes: int, weights: List[float], seed: int) -> str:
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
    return f"{base_name}_episodes_{num_episodes}_weights_{weights_str}_seed_{seed}"

def scale_columns_independently(reward_matrix: np.ndarray) -> np.ndarray:
    """
    Scale the columns of the reward matrix independently to the range [0, 1].

    Args:
        reward_matrix (np.ndarray): Reward matrix to scale.

    Returns:
        np.ndarray: Scaled reward matrix.
    """
    print(f"Input reward_matrix shape: {reward_matrix.shape}")  # Debugging print

    scaler = MinMaxScaler()
    scaled_matrix = np.zeros_like(reward_matrix)  # Initialize the scaled matrix with the same shape

    for col in range(reward_matrix.shape[1]):
        scaled_matrix[:, col] = scaler.fit_transform(reward_matrix[:, col].reshape(-1, 1)).flatten()

    return scaled_matrix

def plot_3d_mean_std(returns_dict: Dict[str, np.ndarray]) -> None:
    """
    Plot a 3D scatter plot with mean and standard deviation of the rewards.

    Args:
        returns_dict (Dict[str, np.ndarray]): Dictionary containing returns for different weight settings.
    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    for weight_setting, returns in returns_dict.items():
        sum_across_episodes = np.sum(returns, axis=1)
        means = np.mean(sum_across_episodes, axis=0)
        std_devs = np.std(sum_across_episodes, axis=0)

        ax.scatter(means[0], means[1], means[2], s=100, label=f'Mean {weight_setting}', marker='o')
        ax.text(means[0], means[1], means[2],
                f'Mean: ({means[0]:.2f}, {means[1]:.2f}, {means[2]:.2f})',
                fontsize=9, color='black')
        ax.errorbar(means[0], means[1], means[2],
                    xerr=std_devs[0], yerr=std_devs[1], zerr=std_devs[2],
                    fmt='o', capsize=5, label=f'Std Dev {weight_setting}')

    ax.set_xlabel('Total Reward 1', fontsize=12)
    ax.set_ylabel('Total Reward 2', fontsize=12)
    ax.set_zlabel('Total Reward 3', fontsize=12)
    ax.set_title('Total Sums of Rewards with Mean and Std Deviation', fontsize=14)
    ax.legend()

    plt.show()

def plot_mean_std_rewards(returns_dict: Dict[str, np.ndarray], reward_dim: int) -> None:
    """
    Plot mean and standard deviation of rewards across episodes.

    Args:
        returns_dict (Dict[str, np.ndarray]): Dictionary containing returns for different weight settings.
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
                label=f'Weights {weight_setting}',
                capsize=5,
                marker='o',
                linestyle='--',
                color=color
            )

        ax.set_title(f'Reward {reward_idx + 1}', fontsize=16)
        ax.set_xlabel('Episodes', fontsize=14)
        ax.set_ylabel('Rewards', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)
        sns.despine(trim=True)

    plt.tight_layout()
    plt.show()

def plot_mean_std_total_steps(total_steps_dict: Dict[str, np.ndarray], num_episodes: int) -> None:
    """
    Plot mean and standard deviation of total steps across episodes.

    Args:
        total_steps_dict (Dict[str, np.ndarray]): Dictionary containing total steps for different weight settings.
        num_episodes (int): Number of episodes.
    """
    colors = sns.color_palette("husl", len(total_steps_dict))
    fig, axes = plt.subplots(1, len(total_steps_dict), figsize=(18, 6), sharex=True, sharey=True)

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
            label=f'Weights {weight_setting}',
            capsize=5,
            marker='o',
            linestyle='--',
            color=colors[idx]
        )

        ax.set_title(f'Weights {weight_setting}', fontsize=16)
        ax.set_xlabel('Episodes', fontsize=14)
        ax.set_ylabel('Total Steps', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)
        sns.despine(trim(True))

    plt.tight_layout()
    plt.show()

def normalize_reward_matrix(reward_matrix: np.ndarray, total_steps: np.ndarray, num_seeds: int, EpisodeDur: bool = True) -> np.ndarray:
    """
    Normalize the reward matrix by total steps.

    Args:
        reward_matrix (np.ndarray): Reward matrix to normalize.
        total_steps (np.ndarray): Total steps for normalization.
        num_seeds (int): Number of seeds.
        EpisodeDur (bool): Whether to include episode duration in normalization.

    Returns:
        np.ndarray: Normalized reward matrix.
    """
    normalized_reward_matrix = np.zeros_like(reward_matrix)
    for seed in range(num_seeds):
        normalized_reward_matrix[seed] = reward_matrix[seed] / total_steps[seed][:, np.newaxis]
        if EpisodeDur:
            normalized_reward_matrix[seed][:, 0] = reward_matrix[seed][:, 0]
    return normalized_reward_matrix

def get_returns(reward_matrices: List[np.ndarray], num_seeds: int, reward_dim: int) -> np.ndarray:
    """
    Get the returns from the reward matrices.

    Args:
        reward_matrices (List[np.ndarray]): List of reward matrices.
        num_seeds (int): Number of seeds.
        reward_dim (int): Number of reward dimensions.

    Returns:
        np.ndarray: Returns matrix.
    """
    return_matrix = np.zeros((num_seeds, reward_dim))
    for seed in range(num_seeds):
        return_matrix[seed] = reward_matrices[seed].sum(axis=0)
    print(return_matrix)
    return return_matrix
