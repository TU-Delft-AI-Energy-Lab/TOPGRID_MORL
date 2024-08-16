import os
import json
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import plotly.express as px

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
            is_efficient[is_efficient] = np.any(costs[is_efficient] >= c, axis=1)
            is_efficient[i] = True
    return is_efficient

def pareto_frontier_2d(x_values, y_values):
    """Computes the Pareto frontier for 2D points considering maximization."""
    points = np.column_stack((x_values, y_values))
    is_efficient = is_pareto_efficient(points)
    x_pareto = np.array(x_values)[is_efficient]
    y_pareto = np.array(y_values)[is_efficient]
    
    sorted_indices = np.argsort(x_pareto)
    return x_pareto[sorted_indices], y_pareto[sorted_indices], is_efficient

def calculate_hypervolume(pareto_points, reference_point):
    """
    Calculate the hypervolume dominated by the Pareto frontier in 2D.
    
    Parameters:
    - pareto_points: List of tuples representing the Pareto points (sorted by the first objective).
    - reference_point: A tuple representing the reference point.
    
    Returns:
    - The calculated hypervolume.
    """
    pareto_points = np.array(pareto_points)
    hypervolume = 0.0

    # Ensure the points are sorted by the first objective (x-axis)
    pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]

    # Calculate the hypervolume
    for i in range(len(pareto_points)):
        if i == 0:
            width = pareto_points[i, 0] - reference_point[0]
        else:
            width = pareto_points[i, 0] - pareto_points[i - 1, 0]
        
        height = pareto_points[i, 1] - reference_point[1]
        hypervolume += width * height

    return hypervolume

def plot_3d_scatter(x_values, y_values, z_values, label, ax=None, color=None):
    """Creates a 3D scatter plot for given x, y, z values with a specific label and color."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_values, y_values, z_values, label=label, color=color)
    
    ax.set_xlabel('ScaledLinesCapacity')
    ax.set_ylabel('ScaledL2RPN')
    ax.set_zlabel('ScaledTopoDepth')
    
    return ax

def calculate_hypervolumes_for_all_projections(seed_paths,wrapper):
    """Calculates the hypervolume for X vs Y, X vs Z, and Y vs Z projections for each seed."""
    hypervolumes = []

    for seed_path in seed_paths:
        data = load_json_data(seed_path)
        ccs_list = data['ccs_list'][-1] 
        if wrapper == 'mc':
            ccs_list = data['ccs_list']
          # Use the last CCS list

        x_all, y_all, z_all = extract_coordinates(ccs_list)

        # Define reference points based on minimum values
        reference_point_xy = (min(x_all), min(y_all))
        reference_point_xz = (min(x_all), min(z_all))
        reference_point_yz = (min(y_all), min(z_all))

        # Calculate Pareto frontiers
        x_pareto_xy, y_pareto_xy, _ = pareto_frontier_2d(x_all, y_all)
        x_pareto_xz, z_pareto_xz, _ = pareto_frontier_2d(x_all, z_all)
        y_pareto_yz, z_pareto_yz, _ = pareto_frontier_2d(y_all, z_all)

        # Calculate hypervolumes for each 2D projection
        hv_xy = calculate_hypervolume(list(zip(x_pareto_xy, y_pareto_xy)), reference_point_xy)
        hv_xz = calculate_hypervolume(list(zip(x_pareto_xz, z_pareto_xz)), reference_point_xz)
        hv_yz = calculate_hypervolume(list(zip(y_pareto_yz, z_pareto_yz)), reference_point_yz)

        hypervolumes.append({
            "Hypervolume XY": hv_xy,
            "Hypervolume XZ": hv_xz,
            "Hypervolume YZ": hv_yz
        })

    return hypervolumes

def plot_2d_projections_seeds(seed_paths, wrapper):
    """Plots X vs Y, X vs Z, and Y vs Z in separate 2D plots, highlighting Pareto frontier points and calculating hypervolumes."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    colors = cm.tab10.colors  # A built-in colormap

    table_data = []
    hypervolumes = calculate_hypervolumes_for_all_projections(seed_paths, wrapper=wrapper)

    for i, seed_path in enumerate(seed_paths):
        data = load_json_data(seed_path)
        ccs_list = data['ccs_list'][-1]  # Use the last CCS list
        if wrapper == 'mc':
            ccs_list = data['ccs_list']

        x_all, y_all, z_all = extract_coordinates(ccs_list)

        # Calculate Pareto frontier for X vs Y
        x_pareto_xy, y_pareto_xy, is_efficient_xy = pareto_frontier_2d(x_all, y_all)
        pareto_points_xy = len(x_pareto_xy)

        # Calculate Pareto frontier for X vs Z
        x_pareto_xz, z_pareto_xz, is_efficient_xz = pareto_frontier_2d(x_all, z_all)
        pareto_points_xz = len(x_pareto_xz)

        # Calculate Pareto frontier for Y vs Z
        y_pareto_yz, z_pareto_yz, is_efficient_yz = pareto_frontier_2d(y_all, z_all)
        pareto_points_yz = len(y_pareto_yz)

        # Append hypervolume data to the table
        table_data.append({
            "Seed": f"Seed {i+1}",
            "Min ScaledLinesCapacity": min(x_all),
            "Max ScaledLinesCapacity": max(x_all),
            "Min ScaledL2RPN": min(y_all),
            "Max ScaledL2RPN": max(y_all),
            "Min ScaledTopoDepth": min(z_all),
            "Max ScaledTopoDepth": max(z_all),
            "Pareto Points XY": pareto_points_xy,
            "Pareto Points XZ": pareto_points_xz,
            "Pareto Points YZ": pareto_points_yz,
            "Hypervolume XY": hypervolumes[i]["Hypervolume XY"],
            "Hypervolume XZ": hypervolumes[i]["Hypervolume XZ"],
            "Hypervolume YZ": hypervolumes[i]["Hypervolume YZ"]
        })

        # Plot X vs Y, highlight Pareto points, and draw lines between them
        axs[0].scatter(x_all, y_all, label=f'Seed {i+1} (Non-Pareto)', color=colors[i % len(colors)], alpha=0.3)
        axs[0].scatter(x_pareto_xy, y_pareto_xy, color=colors[i % len(colors)], edgecolor='black', s=80, label=f'Seed {i+1} (Pareto)')
        axs[0].plot(x_pareto_xy, y_pareto_xy, color=colors[i % len(colors)], linestyle='dotted', linewidth=1)

        # Plot X vs Z, highlight Pareto points, and draw lines between them
        axs[1].scatter(x_all, z_all, label=f'Seed {i+1} (Non-Pareto)', color=colors[i % len(colors)], alpha=0.3)
        axs[1].scatter(x_pareto_xz, z_pareto_xz, color=colors[i % len(colors)], edgecolor='black', s=80, label=f'Seed {i+1} (Pareto)')
        axs[1].plot(x_pareto_xz, z_pareto_xz, color=colors[i % len(colors)], linestyle='dotted', linewidth=1)

        # Plot Y vs Z, highlight Pareto points, and draw lines between them
        axs[2].scatter(y_all, z_all, label=f'Seed {i+1} (Non-Pareto)', color=colors[i % len(colors)], alpha=0.3)
        axs[2].scatter(y_pareto_yz, z_pareto_yz, color=colors[i % len(colors)], edgecolor='black', s=80, label=f'Seed {i+1} (Pareto)')
        axs[2].plot(y_pareto_yz, z_pareto_yz, color=colors[i % len(colors)], linestyle='dotted', linewidth=1)

    axs[0].set_xlabel('ScaledLinesCapacity')
    axs[0].set_ylabel('ScaledL2RPN')
    axs[0].set_title('ScaledLinesCapacity vs ScaledL2RPN')

    axs[1].set_xlabel('ScaledLinesCapacity')
    axs[1].set_ylabel('ScaledTopoDepth')
    axs[1].set_title('ScaledLinesCapacity vs ScaledTopoDepth')

    axs[2].set_xlabel('ScaledL2RPN')
    axs[2].set_ylabel('ScaledTopoDepth')
    axs[2].set_title('ScaledL2RPN vs ScaledTopoDepth')

    for ax in axs:
        ax.legend()

    plt.show()

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(table_data)

    # Calculate mean and std for each metric across all seeds
    mean_df = df.mean(numeric_only=True).rename('Mean')
    std_df = df.std(numeric_only=True).rename('Std')
    
    # Add the mean and std as new rows to the DataFrame using pd.concat
    df = pd.concat([df, mean_df.to_frame().T, std_df.to_frame().T], ignore_index=True)
    
    # Label these rows as "Mean" and "Std"
    df.at[df.index[-2], 'Seed'] = 'Mean'
    df.at[df.index[-1], 'Seed'] = 'Std'

    # Display the DataFrame
    print(df)

    # Create an interactive table using Plotly
    """
    fig = px.imshow(df.transpose(), text_auto=True, aspect='auto', title="Seed Metrics with Mean and Std")
    fig.update_xaxes(side="top", tickangle=-45)
    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_white"
    )
    fig.show()
    """

def plot_all_seeds(seed_paths, wrapper):
    """Plots the data for all seeds in 3D and 2D projections, and shows a table with min and max values."""
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    colors = cm.tab10.colors # A built-in colormap

    for i, seed_path in enumerate(seed_paths):
        data = load_json_data(seed_path)
        
        ccs_list = data['ccs_list'][-1]  # Use the last CCS list
        if wrapper == 'mc':
            ccs_list = data['ccs_list']
        x_all, y_all, z_all = extract_coordinates(ccs_list)

        # Plot 3D scatter
        plot_3d_scatter(x_all, y_all, z_all, f'Seed {i+1}', ax_3d, color=colors[i % len(colors)])

    ax_3d.legend()
    plt.show()

    # Plot 2D projections and generate table
    plot_2d_projections_seeds(seed_paths, wrapper=wrapper)

def find_matching_weights_and_agent(ccs_list, ccs_data):
    matching_entries = []

    for ccs_entry in ccs_list:
        found_match = False
        for data_entry in ccs_data:
            print(data_entry['returns'])
            print(ccs_entry)
            # Ensure both are numpy arrays for consistent comparison
            ccs_entry_array = np.array(ccs_entry)
            returns_array = np.array(data_entry['returns'])

            if np.allclose(ccs_entry_array, returns_array, atol=1e-3):
                matching_entries.append({
                    "weights": data_entry['weights'], 
                    "returns": ccs_entry,
                    "agent_file": data_entry['agent_file'],
                    "test_steps": data_entry["test_steps"], 
                    "test_actions": data_entry['test_actions']
                })
                found_match = True
                break  # Assuming each returns value has a unique match, break after finding it
        
        if not found_match:
            print(f"No match found for CCS entry: {ccs_entry}")

    return matching_entries

def main():
    ols_base_path = r"morl_logs\OLS\rte_case5_example\2024-08-16\['ScaledL2RPN', 'ScaledTopoDepth']"
    mc_base_path = r"morl_logs\MC\rte_case5_example\2024-08-16\['ScaledL2RPN', 'ScaledTopoDepth']"
    
    seeds = [0,1]
    ols_seed_paths = [os.path.join(ols_base_path, f'seed_{seed}', f'morl_logs_ols{seed}.json') for seed in seeds]
    mc_seed_paths = [os.path.join(mc_base_path, f'seed_{seed}', f'morl_logs_mc{seed}.json') for seed in seeds]

    # Process OLS data
    print("Processing OLS Data...")
    process_data(ols_seed_paths, 'ols')
    
    # Process MC data
    print("Processing MC Data...")
    process_data(mc_seed_paths, 'mc')

def process_data(seed_paths, wrapper):
    all_data = []  # List to store all the data for the DataFrame
    
    for seed_path in seed_paths:
        if not os.path.exists(seed_path):
            print(f"File not found: {seed_path}")
            continue

        data = load_json_data(seed_path)
        ccs_list = data['ccs_list'][-1]
        if wrapper=='mc': 
            ccs_list = data['ccs_list']  # Use the last CCS list
        
        # Debugging: Print the contents of ccs_list
        #print(f"ccs_list contents: {ccs_list}")
        

        ccs_data = data['ccs_data']
        # Find matching weights and agent files
        matching_entries = find_matching_weights_and_agent(ccs_list, ccs_data)
        print(matching_entries)
        # Collect data for DataFrame
        for entry in matching_entries:
            row_data = {
                "Weights": entry['weights'],
                "Returns": entry['returns'],
                "Test Steps": entry['test_steps'],
                "Test Actions": entry['test_actions']
            }
            all_data.append(row_data)

    # Create the DataFrame
    if all_data:
        df_ccs_matching = pd.DataFrame(all_data)
    else:
        print("No data found for the given seeds.")
        df_ccs_matching = pd.DataFrame()

    # Display the DataFrame
    print(df_ccs_matching)

    # Further processing or saving the DataFrame
    if not df_ccs_matching.empty:
        df_ccs_matching.to_csv("ccs_matching_data.csv", index=False)  # Example of saving to a CSV file
    
    # Plot all seeds in 3D and 2D
    
    plot_all_seeds(seed_paths, wrapper=wrapper)

if __name__ == "__main__":
    main()
