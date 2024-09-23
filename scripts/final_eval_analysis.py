import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
# Folder path containing the JSON files
folder_path = r"final_logs\l2rpn_case14_sandbox_val\2024-09-23\['TopoActionDay', 'ScaledTopoDepth']\weights_1.00_0.00_0.00\seed_42"

# Initialize storage for timestamps and topological distances
timestamps = []
topo_distances = []

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

            # Collect timestamps and topology distances
            timestamps.extend(data['eval_action_timesamps'])
            topo_distances.extend(data['eval_topo_distance'])

# Convert timestamps to actual time (starting from zero)
start_time = pd.Timestamp('2024-09-23 00:00:00')
time_series = [start_time + i * time_interval for i in timestamps]

# Create a DataFrame for plotting
df = pd.DataFrame({
    'Time': time_series,
    'Topological Distance': topo_distances
})

# Plot the time series of topological distance
plt.figure(figsize=(10, 5))
plt.bar(df['Time'], df['Topological Distance'], color='blue', width=pd.Timedelta(minutes=4.5))
plt.title('Topological Distance Over Time')
plt.xlabel('Time')
plt.ylabel('Topological Distance')
plt.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Initialize storage for aggregated data
all_substations = []
all_topo_distances = []

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is a JSON file
    if file_name.endswith(".json"):
        file_path = os.path.join(folder_path, file_name)

        # Load the JSON data
        with open(file_path, 'r') as file:
            data = json.load(file)

            # Collect all substation IDs
            for sub in data['eval_sub_ids']:
                all_substations.append(str(sub[0]))  # Ensure substation IDs are strings

            # Collect all topological distances
            all_topo_distances.extend(data['eval_topo_distance'])

# Histogram for topological distances
plt.figure(figsize=(10, 5))
plt.hist(all_topo_distances, bins=range(min(all_topo_distances), max(all_topo_distances) + 2), edgecolor='black')
plt.title("Histogram of Topological Distances")
plt.xlabel("Topological Distance")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig('topological_distance_histogram.png')  # Save the figure to a file

# Histogram for substations
substation_counts = Counter(all_substations)
plt.figure(figsize=(10, 5))
plt.bar(substation_counts.keys(), substation_counts.values(), edgecolor='black')
plt.title("Histogram of Substations Affected")
plt.xlabel("Substation ID")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig('substations_histogram.png')  # Save the figure to a file
