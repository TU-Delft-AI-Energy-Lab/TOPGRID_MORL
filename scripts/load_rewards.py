import wandb
import pandas as pd
import matplotlib.pyplot as plt

# Initialize the wandb API
api = wandb.Api()

# Replace with your specific wandb project and entity (user/organization)
project_name = 'TOPGrid_MORL_5bus'
filter_tags = ['a']  # List of tags to filter for

# Fetch all runs from the specified project
runs = api.runs(f"{project_name}")

# Filter runs based on tags
filtered_runs = [run for run in runs if any(tag in run.tags for tag in filter_tags)]

# Collect the reward data from each filtered run
all_rewards = []

for run in filtered_runs:
    # Fetch the logged metrics dataframe
    metrics_df = run.history(keys=["reward"])

    # Extract the reward column (or whatever metric you are interested in)
    if "reward" in metrics_df:
        all_rewards.extend(metrics_df["reward"].dropna().values)

# Convert the list of rewards to a DataFrame for analysis
rewards_df = pd.DataFrame(all_rewards, columns=["reward"])



# Analyze the rewards (example: plotting the reward distribution)
plt.hist(rewards_df["reward"], bins=50, alpha=0.75)
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.title('Distribution of Rewards for Filtered Runs')
plt.show()
