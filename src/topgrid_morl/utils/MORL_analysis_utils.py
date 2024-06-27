import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def plot_multiple_subplots(reward_matrices, summed_episodes):
    num_matrices = len(reward_matrices)
    fig, axes = plt.subplots(1, num_matrices, figsize=(14, 6))

    if num_matrices == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one subplot

    for i, (ax, reward_matrix) in enumerate(zip(axes, reward_matrices)):
        plot_grouped_bar_plot(ax, reward_matrix, summed_episodes)
        ax.set_title(f'Subplot {i+1}: Grouped Bar Plot for Reward Matrix {i+1}')

    plt.tight_layout()
    plt.show()


def plot_grouped_bar_plot(ax, reward_matrix, summed_episodes):
    # Summarize every set of summed_episodes rows by computing the average
    num_rows, num_cols = reward_matrix.shape
    rows_per_summary = summed_episodes
    num_summaries = num_rows // rows_per_summary

    summarized_matrix = np.array([
        reward_matrix[i*rows_per_summary:(i+1)*rows_per_summary].mean(axis=0)
        for i in range(num_summaries)
    ])

    # Standardize the summarized matrix columns
    scaled_matrix = scale_columns_independently(summarized_matrix)

    # Define bar width
    bar_width = 0.2

    # Positions of the bars on the x-axis
    indices = np.arange(num_summaries)

    # Loop through each column and plot it
    for col in range(num_cols):
        ax.bar(indices + col * bar_width, scaled_matrix[:, col], width=bar_width, label=f'Reward {col}')

    # Adding labels and title
    ax.set_xlabel('Summary Index', fontsize=14)
    ax.set_ylabel('Average Value', fontsize=14)
    ax.set_title('Grouped Bar Plot for Summarized Rows', fontsize=16)
    ax.set_xticks(indices + bar_width * (num_cols - 1) / 2)
    ax.set_xticklabels([f'Part {i}' for i in range(num_summaries)], fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.5)

def calculate_correlation(reward_matrix):
    # Calculate Pearson correlation coefficient between the first two columns
    L2RPN = reward_matrix[:, 0]
    Lines = reward_matrix[:, 1]
    Distance = reward_matrix[:, 2]

    corr_L2RPN_Lines, _ = pearsonr(L2RPN, Lines)
    corr_L2RPN_Distance, _ = pearsonr(L2RPN, Distance)

    print(f'Pearson correlation coefficient between L2RPN and Lines: {corr_L2RPN_Lines:.4f}')
    print(f'Pearson correlation coefficient between L2RPN and Distance: {corr_L2RPN_Distance:.4f}')

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(L2RPN, Lines, color='blue', label='Episode Rewards')
    plt.title('Scatter Plot of L2RPN Reward against Lines Reward')
    plt.xlabel('L2RPN Reward')
    plt.ylabel('Lines Reward')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_total_sums(reward_matrices, labels):
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Loop through each reward matrix and plot the total sums
    for reward_matrix, label in zip(reward_matrices, labels):
        # Calculate the total sum for each column
        scaled_matrix = scale_columns_independently(reward_matrix)
        total_sums = np.sum(scaled_matrix, axis=0)

        # Extract the totals for each reward type
        total_reward_1 = total_sums[0]
        total_reward_2 = total_sums[1]
        total_reward_3 = total_sums[2]

        # Plot the total sums in 3D
        ax.scatter(total_reward_1, total_reward_2, total_reward_3, marker='o', label=label)

        # Annotate the point with its values
        ax.text(total_reward_1, total_reward_2, total_reward_3,
                f'({total_reward_1:.2f}, {total_reward_2:.2f}, {total_reward_3:.2f})',
                fontsize=10, color='blue')
        print(total_reward_1, total_reward_2, total_reward_3)
        
    # Adding labels and title
    ax.set_xlabel('Total Reward 1', fontsize=12)
    ax.set_ylabel('Total Reward 2', fontsize=12)
    ax.set_zlabel('Total Reward 3', fontsize=12)
    ax.set_title('Total Sums of Rewards', fontsize=14)
    ax.legend()
    
    # Show plot
    plt.show()

# Function to generate a variable name based on the specifications
def generate_variable_name(base_name, num_episodes, weights, seed):
    weights_str = "_".join(map(str, weights))
    return f"{base_name}_episodes_{num_episodes}_weights_{weights_str}_seed_{seed}"

def scale_columns_independently(reward_matrix):
    print(f"Input reward_matrix shape: {reward_matrix.shape}")  # Debugging print

    scaler = MinMaxScaler()
    scaled_matrix = np.zeros_like(reward_matrix)  # Initialize the scaled matrix with the same shape
    
    for col in range(reward_matrix.shape[1]):
        scaled_matrix[:, col] = scaler.fit_transform(reward_matrix[:, col].reshape(-1, 1)).flatten()
    
    return scaled_matrix

def plot_3d_mean_std(returns_dict):
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot data for each weight setting
    for weight_setting, returns in returns_dict.items():
        # Calculate the sum of rewards across episodes for each seed
        sum_across_episodes = np.sum(returns, axis=1)  # Shape: [num_seeds, reward_dim]

        # Calculate the mean and standard deviation across seeds for each reward type
        means = np.mean(sum_across_episodes, axis=0)
        std_devs = np.std(sum_across_episodes, axis=0)

        # Plot the mean point
        ax.scatter(means[0], means[1], means[2], s=100, label=f'Mean {weight_setting}', marker='o')

        # Annotate the mean point
        ax.text(means[0], means[1], means[2], 
                f'Mean: ({means[0]:.2f}, {means[1]:.2f}, {means[2]:.2f})', 
                fontsize=9, color='black')

        # Plot error bars for standard deviation
        ax.errorbar(means[0], means[1], means[2], 
                    xerr=std_devs[0], yerr=std_devs[1], zerr=std_devs[2], 
                    fmt='o', capsize=5, label=f'Std Dev {weight_setting}')

    # Adding labels and title
    ax.set_xlabel('Total Reward 1', fontsize=12)
    ax.set_ylabel('Total Reward 2', fontsize=12)
    ax.set_zlabel('Total Reward 3', fontsize=12)
    ax.set_title('Total Sums of Rewards with Mean and Std Deviation', fontsize=14)
    ax.legend()

    #plt.show()
    
def plot_mean_std_rewards(returns_dict, reward_dim):
    
    # Define color palette
    colors = sns.color_palette("husl", len(returns_dict))  # Different colors for each weight setting
    
    reward_dim = reward_dim  # Number of reward dimensions

    # Create subplots for each reward dimension
    fig, axes = plt.subplots(1, reward_dim, figsize=(18, 6), sharex=True, sharey=True)

    if reward_dim == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one reward dimension

    for reward_idx in range(reward_dim):
        ax = axes[reward_idx]
        for (weight_setting, reward_matrix), color in zip(returns_dict.items(), colors):
            # Calculate mean and standard deviation across seeds for each episode
            mean_rewards = np.mean(reward_matrix, axis=0)[:, reward_idx]
            std_rewards = np.std(reward_matrix, axis=0)[:, reward_idx]
            
            num_episodes = mean_rewards.shape[0]

            # Plotting
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
    #plt.show()


def plot_mean_std_total_steps(total_steps_dict, num_episodes):
    # Set the overall style of the plots

    # Define color palette
    colors = sns.color_palette("husl", len(total_steps_dict))  # Different colors for each weight setting
    
    # Determine the number of episodes based on the first entry in total_steps_dict
    num_episodes = num_episodes

    # Create subplots for each reward dimension
    fig, axes = plt.subplots(1, len(total_steps_dict), figsize=(18, 6), sharex=True, sharey=True)

    if len(total_steps_dict) == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one weight setting

    for idx, (weight_setting, total_steps) in enumerate(total_steps_dict.items()):
        ax = axes[idx]

        # Calculate mean and standard deviation across seeds for each episode
        mean_total_steps = np.mean(total_steps, axis=0)
        std_total_steps = np.std(total_steps, axis=0)

        # Plotting
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
        sns.despine(trim=True)

    plt.tight_layout()
    #plt.show()
    
    
def normalize_reward_matrix(reward_matrix,total_steps,num_seeds, EpisodeDur: bool=True):#EpisodeDuration: bool = True)
    normalized_reward_matrix = np.zeros_like(reward_matrix)
    for seed in range(num_seeds): 
        normalized_reward_matrix[seed] = reward_matrix[seed] / total_steps[seed][:, np.newaxis]
        if EpisodeDur==True: 
            normalized_reward_matrix[seed][:,0] = reward_matrix[seed][:,0]
    return normalized_reward_matrix

def get_returns(reward_matrices, num_seeds, num_episodes, reward_dim): 
    return_matrix = np.zeros((num_seeds, reward_dim))
    for seed in range(num_seeds): 
        return_matrix[seed] = reward_matrices[seed].sum(axis=0)
    print(return_matrix)    
    return return_matrix