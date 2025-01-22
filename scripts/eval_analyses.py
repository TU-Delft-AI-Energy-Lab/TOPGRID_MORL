import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

def analyze_last_two_entries(seed: int):
    dir_path = os.path.join(
        "eval_logs",
        "rte_case5_example_val_2",
        "2024-08-05",
        "['L2RPN', 'ScaledTopoDepth']",
        "weights_1.00_0.00_0.00",
        f"seed_{seed}"
    )

    evaluation_data = load_evaluation_data_with_details(dir_path)
    print(f"Loaded evaluation data for seed {seed} successfully.")

    df_rewards = pd.DataFrame(evaluation_data["rewards"]).tail(2)
    print(f"df_rewards content for seed {seed}:\n", df_rewards.head())

    return df_rewards

# Initialize list to store mean rewards for the last two entries of each seed
all_last_two_rewards = []

# Iterate over 5 seeds and collect the rewards from the last two entries
for seed in range(5):
    last_two_mean_rewards_df = analyze_last_two_entries(seed)
    all_last_two_rewards.append(last_two_mean_rewards_df)

# Concatenate all rewards from the last two entries
concatenated_rewards = pd.concat(all_last_two_rewards)

# Calculate cumulative rewards
concatenated_rewards["cum_reward"] = concatenated_rewards["eval_rewards"].apply(sum_rewards)

# Convert cum_reward to a DataFrame for further processing
cum_last_two_rewards_df = pd.DataFrame(concatenated_rewards["cum_reward"].tolist())

# Compute the mean of the rows and then accumulate the rewards
mean_last_rewards_df = compute_row_pair_means(cum_last_two_rewards_df)
print("mean last rows")
print(mean_last_rewards_df)


mean_last_rewards_df.columns = ["ScaledLines", "L2RPN", "TopoDepth"]
print(mean_last_rewards_df)

# Calculate the mean and standard deviation over the rows
mean_over_rows = mean_last_rewards_df.mean()
std_over_rows = mean_last_rewards_df.std()

# Create a new DataFrame to store mean and std
mean_std_df = pd.DataFrame({
    'mean_ScaledLines': mean_over_rows[0],
    'std_ScaledLines': std_over_rows[0],
    'mean_L2RPN': mean_over_rows[1],
    'std_L2RPN': std_over_rows[1],
    'mean_TopoDepth': mean_over_rows[2],
    'std_TopoDepth': std_over_rows[2]
}, index=[0])

print("Mean and Std DataFrame")
print(mean_std_df)

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with error bars
x = mean_std_df['mean_ScaledLines']
y = mean_std_df['mean_L2RPN']
z = mean_std_df['mean_TopoDepth']
dx = mean_std_df['std_ScaledLines']
dy = mean_std_df['std_L2RPN']
dz = mean_std_df['std_TopoDepth']

ax.scatter(x, y, z, c='r', marker='o')
ax.errorbar(x, y, z, xerr=dx, yerr=dy, zerr=dz, fmt='o', color='b', alpha=0.5)

# Adding labels
ax.set_xlabel('ScaledLines')
ax.set_ylabel('L2RPN')
ax.set_zlabel('TopoDepth')

# Adding title
ax.set_title('3D Scatter Plot of Mean and Std Deviation Over Seeds')

plt.show()

def analyze_evaluation_data(seed: int):
    dir_path = os.path.join(
        "eval_logs",
        "rte_case5_example_val_2",
        "2024-08-05",
        "['L2RPN', 'ScaledTopoDepth']",
        "weights_1.00_0.00_0.00",
        f"seed_{seed}"
    )

    evaluation_data = load_evaluation_data_with_details(dir_path)
    print(f"Loaded evaluation data for seed {seed} successfully.")

    df_steps = pd.DataFrame(evaluation_data["steps"])
    df_rewards = pd.DataFrame(evaluation_data["rewards"])
    print(f"df_rewards content for seed {seed}:\n", df_rewards.head())

    df_all = create_combined_dataframe(evaluation_data)

    df_rewards["cum_reward"] = df_rewards["eval_rewards"].apply(sum_rewards)

    cum_rewards_df = pd.DataFrame(df_rewards["cum_reward"].tolist())
    cum_rewards_df.columns = [f"reward_{i}" for i in range(cum_rewards_df.shape[1])]

    scaler = MinMaxScaler()
    scaled_cum_rewards_df = pd.DataFrame(scaler.fit_transform(cum_rewards_df), columns=cum_rewards_df.columns)
    print(f"Scaled cumulative rewards for seed {seed}:\n", scaled_cum_rewards_df)

    mean_rewards_df = compute_row_pair_means(scaled_cum_rewards_df)

    eval_steps = [step["eval_steps"] for step in evaluation_data["steps"]]
    eval_steps_df = pd.DataFrame(eval_steps, columns=["eval_steps"])
    mean_eval_steps_df = compute_row_pair_means(eval_steps_df)

    batch_size = 256

    return mean_rewards_df

# Initialize lists to store mean rewards for each seed
all_mean_rewards = {i: [] for i in range(3)}

# Iterate over 5 seeds and collect the mean rewards for each reward index
for seed in range(5):
    mean_rewards_df = analyze_evaluation_data(seed)
    for i in range(3):
        all_mean_rewards[i].append(mean_rewards_df.iloc[:, i])

# Calculate mean and standard deviation over the seeds for each reward
mean_rewards_over_seeds = {i: pd.concat(all_mean_rewards[i], axis=1).mean(axis=1) for i in range(3)}
std_rewards_over_seeds = {i: pd.concat(all_mean_rewards[i], axis=1).std(axis=1) for i in range(3)}

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 8))
labels = ["Scaled Lines Capacity", "L2RPN", "TopoDepth"]
# Plot mean cumulative rewards for each reward index
for i in range(3):
    ax1.plot(mean_rewards_over_seeds[i], label=labels[i])
    ax1.fill_between(range(len(mean_rewards_over_seeds[i])), mean_rewards_over_seeds[i] - std_rewards_over_seeds[i], mean_rewards_over_seeds[i] + std_rewards_over_seeds[i], alpha=0.2)

ax1.set_xticks(range(len(mean_rewards_over_seeds[0])))
ax1.set_xticklabels((mean_rewards_over_seeds[0].index * 256).tolist())
ax1.set_xlabel("Training Steps")
ax1.set_ylabel("Mean Cumulative Reward")
ax1.legend(loc="upper left")
ax1.grid(True)

fig.suptitle("Mean Cumulative Rewards and Standard Deviation Over Seeds")
plt.show()
