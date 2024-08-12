import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def load_json_data(relative_path):
    """Loads JSON data from a given relative path."""
    absolute_path = os.path.abspath(relative_path)
    with open(absolute_path, 'r') as file:
        data = json.load(file)
    return data

def extract_coordinates(ccs_list):
    """Extracts x, y, z coordinates from a list of CCS points."""
    x_values = [coord[0] for coord in ccs_list]  # ScaledLinesCapacity
    y_values = [coord[1] for coord in ccs_list]  # ScaledL2RPN
    z_values = [coord[2] for coord in ccs_list]  # ScaledTopoDepth
    return x_values, y_values, z_values

def is_pareto_efficient(costs):
    """Finds the Pareto-efficient points with maximization in mind."""
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Maximize: look for points where all coordinates are less than or equal to the current point
            is_efficient[is_efficient] = np.any(costs[is_efficient] >= c, axis=1)
            is_efficient[i] = True
    return is_efficient

def pareto_frontier_2d(x_values, y_values):
    """Computes the Pareto frontier for 2D points considering maximization."""
    points = np.column_stack((x_values, y_values))
    is_efficient = is_pareto_efficient(points)
    x_pareto = np.array(x_values)[is_efficient]
    y_pareto = np.array(y_values)[is_efficient]
    
    # Sort the points by the first dimension to plot lines in order
    sorted_indices = np.argsort(x_pareto)
    return x_pareto[sorted_indices], y_pareto[sorted_indices], is_efficient

def plot_3d_scatter(x_values, y_values, z_values, label, ax=None, color=None):
    """Creates a 3D scatter plot for given x, y, z values with a specific label and color."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_values, y_values, z_values, label=label, color=color)
    
    # Label the axes
    ax.set_xlabel('ScaledLinesCapacity')
    ax.set_ylabel('ScaledL2RPN')
    ax.set_zlabel('ScaledTopoDepth')
    
    return ax

def plot_2d_projections_seeds(seed_paths):
    """Plots X vs Y, X vs Z, and Y vs Z in separate 2D plots with Pareto frontier, labeling seeds."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Colors for each seed

    for i, seed_path in enumerate(seed_paths):
        data = load_json_data(seed_path)
        ccs_list = data['ccs_list']

        # Only take the last array from the ccs_list
        last_ccs = ccs_list[-1]

        # Extract coordinates from the last CCS list
        x_all, y_all, z_all = extract_coordinates(last_ccs)

        # Plot X vs Y
        x_pareto, y_pareto, is_efficient = pareto_frontier_2d(x_all, y_all)
        axs[0].scatter(np.array(x_all)[~is_efficient], np.array(y_all)[~is_efficient], 
                       label=f'Seed {i+1}', color=colors[i % len(colors)], alpha=0.3)
        axs[0].scatter(x_pareto, y_pareto, color=colors[i % len(colors)], 
                       edgecolor='k', linewidth=1.5, s=80)
        axs[0].plot(x_pareto, y_pareto, color=colors[i % len(colors)], linestyle='dotted', linewidth=1)

        # Plot X vs Z
        x_pareto, z_pareto, is_efficient = pareto_frontier_2d(x_all, z_all)
        axs[1].scatter(np.array(x_all)[~is_efficient], np.array(z_all)[~is_efficient], 
                       label=f'Seed {i+1}', color=colors[i % len(colors)], alpha=0.3)
        axs[1].scatter(x_pareto, z_pareto, color=colors[i % len(colors)], 
                       edgecolor='k', linewidth=1.5, s=80)
        axs[1].plot(x_pareto, z_pareto, color=colors[i % len(colors)], linestyle='dotted', linewidth=1)

        # Plot Y vs Z
        y_pareto, z_pareto, is_efficient = pareto_frontier_2d(y_all, z_all)
        axs[2].scatter(np.array(y_all)[~is_efficient], np.array(z_all)[~is_efficient], 
                       label=f'Seed {i+1}', color=colors[i % len(colors)], alpha=0.3)
        axs[2].scatter(y_pareto, z_pareto, color=colors[i % len(colors)], 
                       edgecolor='k', linewidth=1.5, s=80)
        axs[2].plot(y_pareto, z_pareto, color=colors[i % len(colors)], linestyle='dotted', linewidth=1)

    # Set labels and titles
    axs[0].set_xlabel('ScaledLinesCapacity')
    axs[0].set_ylabel('ScaledL2RPN')
    axs[0].set_title('ScaledLinesCapacity vs ScaledL2RPN')

    axs[1].set_xlabel('ScaledLinesCapacity')
    axs[1].set_ylabel('ScaledTopoDepth')
    axs[1].set_title('ScaledLinesCapacity vs ScaledTopoDepth')

    axs[2].set_xlabel('ScaledL2RPN')
    axs[2].set_ylabel('ScaledTopoDepth')
    axs[2].set_title('ScaledL2RPN vs ScaledTopoDepth')

    # Add legends to each subplot
    for ax in axs:
        ax.legend()

    plt.show()

def plot_all_seeds(seed_paths):
    """Plots the data for all seeds in 3D and 2D projections."""
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Colors for each seed

    for i, seed_path in enumerate(seed_paths):
        data = load_json_data(seed_path)
        ccs_list = data['ccs_list']
        
        # Only take the last array from the ccs_list
        last_ccs = ccs_list[-1]

        # Extract coordinates from the last CCS list
        x_all, y_all, z_all = extract_coordinates(last_ccs)

        # Plot points for the current seed in 3D
        plot_3d_scatter(x_all, y_all, z_all, f'Seed {i+1}', ax_3d, color=colors[i % len(colors)])

    ax_3d.legend()
    plt.show()

    # Plot combined 2D projections for all seeds with labels
    plot_2d_projections_seeds(seed_paths)

def main():
    # Base path for the JSON files
    base_path = r"morl_logs\OLS\rte_case5_example\2024-08-11\['ScaledL2RPN', 'ScaledTopoDepth']"
    
    # List of seed directories
    seed_dirs = [f'seed_{i}' for i in range(0, 5)]  # Adjust the range as needed
    
    # Construct paths for each seed JSON file
    seed_paths = [os.path.join(base_path, seed_dir, f'morl_logs_ols{seed_dirs.index(seed_dir)}.json') for seed_dir in seed_dirs]
    
    # Plot combined data for all seeds
    plot_all_seeds(seed_paths)

if __name__ == "__main__":
    main()
