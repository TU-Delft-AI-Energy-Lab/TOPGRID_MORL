import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D

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