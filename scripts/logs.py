import json
import os

# Define the path to the folder containing JSON files
folder_path = os.path.join('eval_logs', 'rte_case5_example_val', 'weights_0.00_0.00_1.00')

# Initialize a dictionary to store the data from each file
data_dict = {}

# Iterate over all JSON files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.json'):
        file_path = os.path.join(folder_path, file_name)
        
        # Load the JSON data from the file
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        
        # Extract the relevant lists
        eval_rewards = json_data["eval_rewards"]
        eval_actions = json_data["eval_actions"]
        eval_states = json_data["eval_states"]
        eval_steps = json_data["eval_steps"]

        # Store the lists in a dictionary for the current file
        file_data = {
            "eval_rewards": eval_rewards,
            "eval_actions": eval_actions,
            "eval_states": eval_states,
            "eval_steps": eval_steps
        }

        # Store the file data dictionary in the main dictionary
        data_dict[file_name] = file_data

# Display the data from each file
for file_name, file_data in data_dict.items():
    print(f"Data from {file_name}:")
    for key, value in file_data.items():
        print(f"{key}: {value}")
    print("\n")
