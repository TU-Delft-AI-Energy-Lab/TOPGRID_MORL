import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from topgrid_morl.utils.MORL_analysis_utils import create_action_to_substation_mapping 

def create_combined_dataframe(evaluation_data):
    # Convert each type of data to a DataFrame
    df_steps = pd.DataFrame(evaluation_data["steps"])
    df_rewards = pd.DataFrame(evaluation_data["rewards"])
    df_actions = pd.DataFrame(evaluation_data["actions"])
    df_states = pd.DataFrame(evaluation_data["states"])

    df_combined = pd.concat([df_steps, df_rewards, df_actions, df_states], axis=1)

    return df_combined

def load_evaluation_data_with_details(dir_path: str) -> Dict[str, List[Dict[str, Any]]]:
    eval_data_list = {"steps": [], "actions": [], "rewards": [], "states": []}

    # Get list of all files in the directory with their modification times
    files_with_times = [
        (os.path.join(root, file), os.path.getmtime(os.path.join(root, file)))
        for root, _, files in os.walk(dir_path)
        for file in files
        if file.endswith(".json")
    ]

    # Sort files by modification time
    files_with_times.sort(key=lambda x: x[1])

    # Load JSON files in order and extract relevant details
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
    # Convert to numpy array for easier manipulation
    rewards_np = np.array(rewards)
    # Sum along the desired axis (e.g., sum along the first axis)
    summed_rewards = rewards_np.sum(axis=0)
    return summed_rewards.tolist()


def compute_row_pair_means(df):
    means = []
    for i in range(0, df.shape[0], 2):
        if i + 1 < df.shape[0]:
            row_mean = df.iloc[i : i + 2].mean()
        else:
            row_mean = df.iloc[i]  # If there is an odd number of rows, just take the last row
        means.append(row_mean)
    return pd.DataFrame(means)


# Specify the relative path to the directory containing the JSON files
dir_path = os.path.join(
    "eval_logs",
    "rte_case5_example_val",
    "2024-08-05",
    "['ScaledTopoDepth', 'SubstationSwitching']",
    "weights_0_1_0",
    "seed_42"
)

# Load all evaluation data with steps, actions, rewards, and states from the specified directory
evaluation_data = load_evaluation_data_with_details(dir_path)
print("Loaded evaluation data successfully.")

# Convert each type of data to a DataFrame for easier display and analysis
df_steps = pd.DataFrame(evaluation_data["steps"])
df_rewards = pd.DataFrame(evaluation_data["rewards"])

df_all = create_combined_dataframe(evaluation_data)
print(df_all)

# Add a new column with the summed rewards along the specified axis
df_rewards["cum_reward"] = df_rewards["eval_rewards"].apply(sum_rewards)

# Convert cum_reward to a DataFrame for plotting
cum_rewards_df = pd.DataFrame(df_rewards["cum_reward"].tolist())

# Rename the columns
cum_rewards_df.columns = [f"reward_{i}" for i in range(cum_rewards_df.shape[1])]

# Compute the means of each pair of rows
mean_rewards_df = compute_row_pair_means(cum_rewards_df)

# Compute the means of each pair of eval steps
eval_steps = [step["eval_steps"] for step in evaluation_data["steps"]]
eval_steps_df = pd.DataFrame(eval_steps, columns=["eval_steps"])
mean_eval_steps_df = compute_row_pair_means(eval_steps_df)

# Batch size
batch_size = 256

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot mean cumulative rewards
for i in range(mean_rewards_df.shape[1]):
    ax1.plot(mean_rewards_df.iloc[:, i], label=f"Mean Reward {i}")

# Adjust x-axis ticks to represent training steps
ax1.set_xticks(range(len(mean_rewards_df)))
ax1.set_xticklabels((mean_rewards_df.index * batch_size).tolist())
ax1.set_xlabel("Training Steps")
ax1.set_ylabel("Mean Cumulative Reward")
ax1.legend(loc="upper left")
ax1.grid(True)

# Create a secondary y-axis for eval steps
ax2 = ax1.twinx()
ax2.plot(mean_eval_steps_df, color="r", label="Eval Steps", linestyle="--")
ax2.set_ylabel("Evaluation Steps")

fig.suptitle("Mean Cumulative Rewards and Evaluation Steps Plot")
ax2.legend(loc="upper right")

#plt.show()
mapping = create_action_to_substation_mapping();

def map_to_substation(actions, mapping):
    return [mapping[action] for action in actions]

# Apply the function to create a new column 'substations'
df_all['substations'] = df_all['eval_actions'].apply(lambda x: map_to_substation(x, mapping))
print(df_all["substations"])