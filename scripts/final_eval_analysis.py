import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from datetime import timedelta

# Folder path containing the JSON files
folder_path = r"final_logs\l2rpn_case14_sandbox_val\2024-09-23\['TopoActionDay', 'ScaledTopoDepth']\weights_1.00_0.00_0.00\seed_42"

# Initialize storage for timestamps, topological distances, and substations
timestamps = []
topo_distances = []
all_substations = []
all_topo_distances = []
max_steps_file = None
max_steps = -1

# Time step interval (5 minutes)
time_interval = timedelta(minutes=5)

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is a JSON file
    if file_name.endswith(".json"):
        file_path = os.path.join(folder_path, file_name)

        # Load the JSON data
        with open(file_path, 'r') as file:
            data = json.load(file)

            # Track the file with the highest total steps
            if data['eval_steps'] > max_steps:
                max_steps = data['eval_steps']
                max_steps_file = file_path

            # Collect timestamps and topology distances
            timestamps.extend(data['eval_action_timesamps'])
            topo_distances.extend(data['eval_topo_distance'])

            # Collect all substation IDs
            for sub in data['eval_sub_ids']:
                all_substations.append(str(sub[0]))  # Ensure substation IDs are strings

            # Collect all topological distances
            all_topo_distances.extend(data['eval_topo_distance'])

# Convert timestamps to actual time (starting from zero)
start_time = pd.Timestamp('2024-09-23 00:00:00')
time_series = [start_time + i * time_interval for i in timestamps]

# Create a DataFrame for the bar plot
df = pd.DataFrame({
    'Time': time_series,
    'Topological Distance': topo_distances
})

# 1. Bar Plot for Topological Distance Over Time
plt.figure(figsize=(10, 5))
plt.bar(df['Time'], df['Topological Distance'], color='blue', width=pd.Timedelta(minutes=4.5))
plt.title('Topological Distance Over Time', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Topological Distance', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('topological_distance_barplot.png')  # Save the figure to a file
plt.show()

# 2. Histogram for Topological Distances
plt.figure(figsize=(10, 5))
plt.hist(all_topo_distances, bins=range(min(all_topo_distances), max(all_topo_distances) + 2), edgecolor='black')
plt.title("Histogram of Topological Distances", fontsize=14)
plt.xlabel("Topological Distance", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('topological_distance_histogram.png')  # Save the figure to a file
plt.show()

# 3. Histogram for Substations Affected
substation_counts = Counter(all_substations)
plt.figure(figsize=(10, 5))
plt.bar(substation_counts.keys(), substation_counts.values(), edgecolor='black', color='orange')
plt.title("Histogram of Substations Affected", fontsize=14)
plt.xlabel("Substation ID", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('substations_histogram.png')  # Save the figure to a file
plt.show()

# ==== NEW TASK: In-depth analysis of the file with the highest total steps ====
print(f"\nPerforming in-depth analysis of file: {max_steps_file} with {max_steps} total steps\n")

# Load the JSON data of the file with the highest steps
with open(max_steps_file, 'r') as file:
    max_steps_data = json.load(file)

# Extract relevant data for analysis
max_steps_timestamps = max_steps_data['eval_action_timesamps']
max_steps_topo_distances = max_steps_data['eval_topo_distance']
max_steps_substations = max_steps_data['eval_sub_ids']  # Substations affected
start_time = pd.Timestamp('2024-09-23 00:00:00')
# Convert timestamps of this specific file
max_steps_time_series = [start_time + i * time_interval for i in max_steps_timestamps]

# Create a DataFrame for detailed analysis
df_max_steps = pd.DataFrame({
    'Time': max_steps_time_series,
    'Topological Distance': max_steps_topo_distances,
    'Substation': [sub[0] for sub in max_steps_substations]  # Get substation IDs
})

# 4. Detailed Bar Plot for the File with Highest Steps (Starting from Timestamp 0)
plt.figure(figsize=(15, 7))  # Increase figure size for a better view with extended time
bars = plt.bar(df_max_steps['Time'], df_max_steps['Topological Distance'], color='green', width=pd.Timedelta(minutes=4.5))
plt.title(f'Topological Distance Over Extended Time Period for {max_steps_file}', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Topological Distance', fontsize=12)
plt.xticks(rotation=45)

# Set the x-axis to start from timestamp = 0 (i.e., the start time) and extend it
time_min = start_time  # Start from timestamp 0 (start_time)
time_max = df_max_steps['Time'].max() + pd.Timedelta(minutes=30)  # End 30 minutes later for extended view
plt.xlim(time_min, time_max)  # Set the x-axis limits

plt.grid(True)

# Annotate each bar with the substation ID and action index
for i, bar in enumerate(bars):
    substation = df_max_steps['Substation'].iloc[i]
    action_time = df_max_steps['Time'].iloc[i]
    plt.annotate(f'Action {i}\nSub {substation}', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 ha='center', va='bottom', fontsize=10, color='black', rotation=90)

plt.tight_layout()
plt.savefig(f'{os.path.basename(max_steps_file)}_extended_annotated_barplot_from_0.png')  # Save the detailed figure to a file
plt.show()
