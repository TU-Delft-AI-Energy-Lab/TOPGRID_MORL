import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D

from topgrid_morl.utils.MORL_analysis_utils import create_action_to_substation_mapping

def create_combined_dataframe(evaluation_data):
    df_steps = pd.DataFrame(evaluation_data["steps"])
    df_rewards = pd.DataFrame(evaluation_data["rewards"])
    df_actions = pd.DataFrame(evaluation_data["actions"])
    df_states = pd.DataFrame(evaluation_data["states"])

    df_combined = pd.concat([df_steps, df_rewards, df_actions, df_states], axis=1)
    return df_combined

def load_evaluation_data_with_details(dir_path: str) -> Dict[str, List[Dict[str, Any]]]:
    eval_data_list = {"steps": [], "actions": [], "rewards": [], "states": []}

    files_with_times = [
        (os.path.join(root, file), os.path.getmtime(os.path.join(root, file)))
        for root, _, files in os.walk(dir_path)
        for file in files
        if file.endswith(".json")
    ]

    files_with_times.sort(key=lambda x: x[1])

    for file_path, _ in files_with_times:
        with open(file_path, "r") as json_file:
            eval_data = json.load(json_file)
            eval_data_list["steps"].append(
                {"file_path": file_path, "eval_steps": eval_data["eval_steps"]}
            )
            eval_data_list["actions"].append(
                {"file_path": file_path, "eval_actions": eval_data["eval_actions"]}
            )
            eval_data_list["rewards"].append(
                {"file_path": file_path, "eval_rewards": eval_data["eval_rewards"]}
            )
            eval_data_list["states"].append(
                {"file_path": file_path, "eval_states": eval_data["eval_states"]}
            )

    return eval_data_list

def sum_rewards(rewards):
    rewards_np = np.array(rewards)
    summed_rewards = rewards_np.sum(axis=0)
    return summed_rewards.tolist()

def compute_row_pair_means(df):
    means = []
    for i in range(0, df.shape[0], 2):
        if i + 1 < df.shape[0]:
            row_mean = df.iloc[i : i + 2].mean()
        else:
            row_mean = df.iloc[i]
        means.append(row_mean)
    return pd.DataFrame(means)

def analyze_last_two_entries(seed: int, weight: str):
    dir_path = os.path.join(
        "eval_logs",
        "rte_case5_example_val_2",
        "2024-08-05",
        "['L2RPN', 'ScaledTopoDepth']",
        weight,
        f"seed_{seed}"
    )

    evaluation_data = load_evaluation_data_with_details(dir_path)
    print(f"Loaded evaluation data for seed {seed} and weight {weight} successfully.")

    df_rewards = pd.DataFrame(evaluation_data["rewards"]).tail(2)
    print(f"df_rewards content for seed {seed} and weight {weight}:\n", df_rewards.head())

    return df_rewards

weights = ["weights_1.00_0.00_0.00", "weights_0.00_1.00_0.00", "weights_0.00_0.00_1.00"]
colors = ['r', 'g', 'b']
labels = ['trained on LinesCapa', 'trained on L2RPN', 'trained on TopoDepth']

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for weight, color, label in zip(weights, colors, labels):
    # Initialize list to store mean rewards for the last two entries of each seed
    all_last_two_rewards = []

    # Iterate over 5 seeds and collect the rewards from the last two entries
    for seed in range(5):
        last_two_mean_rewards_df = analyze_last_two_entries(seed, weight)
        all_last_two_rewards.append(last_two_mean_rewards_df)

    # Concatenate all rewards from the last two entries
    concatenated_rewards = pd.concat(all_last_two_rewards)

    # Calculate cumulative rewards
    concatenated_rewards["cum_reward"] = concatenated_rewards["eval_rewards"].apply(sum_rewards)

    # Convert cum_reward to a DataFrame for further processing
    cum_last_two_rewards_df = pd.DataFrame(concatenated_rewards["cum_reward"].tolist())

    # Compute the mean of the rows and then accumulate the rewards
    mean_last_rewards_df = compute_row_pair_means(cum_last_two_rewards_df)
    print(f"mean last rows for weight {weight}")
    print(mean_last_rewards_df)

    mean_last_rewards_df.columns = ["ScaledLines", "L2RPN", "TopoDepth"]
    print(mean_last_rewards_df)

    # Calculate the mean and standard deviation over the rows
    mean_over_rows = mean_last_rewards_df.mean()
    std_over_rows = mean_last_rewards_df.std()

    # Scatter plot with error bars
    x = mean_over_rows['ScaledLines']
    y = mean_over_rows['L2RPN']
    z = mean_over_rows['TopoDepth']
    dx = std_over_rows['ScaledLines']
    dy = std_over_rows['L2RPN']
    dz = std_over_rows['TopoDepth']

    ax.scatter(x, y, z, c=color, marker='o', label=label)
    ax.errorbar(x, y, z, xerr=dx, yerr=dy, zerr=dz, fmt='o', color=color, alpha=0.5)

# Adding labels
ax.set_xlabel('ScaledLines')
ax.set_ylabel('L2RPN')
ax.set_zlabel('TopoDepth')

# Adding title
ax.set_title('3D Scatter Plot of Mean and Std Deviation Over Seeds')

# Adding legend
ax.legend()

plt.show()

def scatter_plot_rewards_for_weights(weights: List[str], colors: List[str], labels: List[str]):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for weight, color, label in zip(weights, colors, labels):
        for seed in range(5):
            last_two_rewards_df = analyze_last_two_entries(seed, weight)

            # Calculate cumulative rewards
            last_two_rewards_df["cum_reward"] = last_two_rewards_df["eval_rewards"].apply(sum_rewards)

            # Convert cum_reward to a DataFrame for further processing
            cum_rewards_df = pd.DataFrame(last_two_rewards_df["cum_reward"].tolist())
            cum_rewards_df.columns = ["ScaledLines", "L2RPN", "TopoDepth"]

            # Scatter plot for each reward
            x = cum_rewards_df["ScaledLines"]
            y = cum_rewards_df["L2RPN"]
            z = cum_rewards_df["TopoDepth"]

            ax.scatter(x, y, z, c=color, marker='o', label=label if seed == 0 else "")

    # Adding labels
    ax.set_xlabel('ScaledLines')
    ax.set_ylabel('L2RPN')
    ax.set_zlabel('TopoDepth')

    # Adding title
    ax.set_title('3D Scatter Plot of Rewards for Each Seed and Weight')

    # Adding legend
    ax.legend()

    plt.show()

# Usage
weights = ["weights_1.00_0.00_0.00", "weights_0.00_1.00_0.00", "weights_0.00_0.00_1.00"]
colors = ['r', 'g', 'b']
labels = ['trained on LinesCapa', 'trained on L2RPN', 'trained on TopoDepth']

scatter_plot_rewards_for_weights(weights, colors, labels)