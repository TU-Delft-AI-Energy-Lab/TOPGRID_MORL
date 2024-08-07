import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

def load_evaluation_data_with_details(base_dir_path: str, seeds: List[int]) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
    eval_data_by_seed = {}

    for seed in seeds:
        eval_data_list = {"steps": [], "actions": [], "rewards": [], "states": []}
        dir_path = os.path.join(base_dir_path, f"seed_{seed}")
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

        eval_data_by_seed[seed] = eval_data_list

    return eval_data_by_seed

def get_rewards_for_seed(base_dir_path: str, seed: int) -> pd.DataFrame:
    """
    Load evaluation rewards for a given seed and return them as a DataFrame.

    Parameters:
    - base_dir_path: str, base directory containing the evaluation data.
    - seed: int, specific seed to load the evaluation rewards for.

    Returns:
    - DataFrame containing the evaluation rewards with file names as row indices.
      The rewards are split into three columns: "Line", "L2RPN", and "TopoDepth".
    """
    eval_rewards_dict = {}
    dir_path = os.path.join(base_dir_path, f"seed_{seed}")
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
            eval_rewards_dict[file_path] = eval_data["eval_rewards"]

    # Convert to DataFrame and split the rewards into three separate columns
    eval_rewards_df = pd.DataFrame.from_dict(eval_rewards_dict, orient='index')
    
    # Flatten the nested lists into columns
    eval_rewards_split = pd.DataFrame(
        [(index, *reward) for index, rewards in eval_rewards_df.iterrows() for reward in rewards],
        columns=["File", "Line", "L2RPN", "TopoDepth"]
    )
    
    eval_rewards_split.set_index("File", inplace=True)
    
    return eval_rewards_split

# Example usage
base_dir_path = os.path.join(
    "eval_logs",
    "rte_case5_example_val_2",
    "2024-08-05",
    "['L2RPN', 'ScaledTopoDepth']",
    "weights_1.00_0.00_0.00"
)

# Specify the seed
seed = 0

# Get rewards for the specified seed
rewards_for_seed = get_rewards_for_seed(base_dir_path, seed)
print("Rewards for seed:", rewards_for_seed)

def compute_row_pair_means(df):
    means = []
    for i in range(0, df.shape[0], 2):
        if i + 1 < df.shape[0]:
            row_mean = df.iloc[i : i + 2].mean()
        else:
            row_mean = df.iloc[i]
        means.append(row_mean)
    return pd.DataFrame(means)

# Base directory path
base_dir_path = os.path.join(
    "eval_logs",
    "rte_case5_example_val_2",
    "2024-08-05",
    "['L2RPN', 'ScaledTopoDepth']",
    "weights_1.00_0.00_0.00"
)

# Seeds to iterate over
seeds = list(range(0, 5))

# Load evaluation data for all seeds
evaluation_data_by_seed = load_evaluation_data_with_details(base_dir_path, seeds)
print("Loaded evaluation data successfully.")

# Aggregate data for computing means and standard deviations for eval steps
all_eval_steps = []

for seed, evaluation_data in evaluation_data_by_seed.items():
    eval_steps = [step["eval_steps"] for step in evaluation_data["steps"]]
    eval_steps_df = pd.DataFrame(eval_steps, columns=["eval_steps"])
    
    # Compute the means of each pair of eval steps
    mean_eval_steps_df = compute_row_pair_means(eval_steps_df)
    all_eval_steps.append(mean_eval_steps_df)

# Concatenate all dataframes and compute the mean and standard deviation
concat_eval_steps_df = pd.concat(all_eval_steps, axis=1)
mean_eval_steps = concat_eval_steps_df.mean(axis=1)
std_eval_steps = concat_eval_steps_df.std(axis=1)

# Batch size
batch_size = 256

# Plotting mean evaluation steps with standard deviation error bars

# Aggregate data for computing means and standard deviations for rewards
all_eval_rewards = []

for seed, evaluation_data in evaluation_data_by_seed.items():
    eval_rewards = [reward["eval_rewards"] for reward in evaluation_data["rewards"]]
    eval_rewards_flattened = [item for sublist in eval_rewards for item in sublist]
    eval_rewards_df = pd.DataFrame(eval_rewards_flattened)
    
    # Compute the means of each pair of eval rewards
    mean_eval_rewards_df = compute_row_pair_means(eval_rewards_df)
    all_eval_rewards.append(mean_eval_rewards_df)

# Concatenate all dataframes and compute the mean and standard deviation for rewards
concat_eval_rewards_df = pd.concat(all_eval_rewards, axis=1)
mean_eval_rewards = concat_eval_rewards_df.mean(axis=1)
std_eval_rewards = concat_eval_rewards_df.std(axis=1)




print(concat_eval_rewards_df)
fig, ax = plt.subplots(figsize=(12, 8))

# Plot mean evaluation steps
ax.plot(mean_eval_steps, label="Mean Eval Steps")

# Plot standard deviation as error bars
ax.fill_between(range(len(mean_eval_steps)), 
                mean_eval_steps - std_eval_steps, 
                mean_eval_steps + std_eval_steps, 
                color='b', alpha=0.2, label="Std Dev")

ax.set_xticks(range(len(mean_eval_steps)))
ax.set_xticklabels((mean_eval_steps.index * batch_size).tolist())
ax.set_xlabel("Training Steps")
ax.set_ylabel("Evaluation Steps")
ax.legend(loc="upper left")
ax.grid(True)

fig.suptitle("Mean Evaluation Steps with Standard Deviation Across Seeds")

plt.show()
