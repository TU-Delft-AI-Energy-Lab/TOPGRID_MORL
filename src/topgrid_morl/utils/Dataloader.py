import os
import json
import numpy as np
import glob

class DataLoader:
    def __init__(self):
        self.state_action_log = []
        self.rewards_log = []

    def save_logs_json(self, path: str) -> None:
        """
        Save the state-action pairs and rewards to a file.

        Args:
            path (str): Path to save the logs.
        """
        logs = {
            "state_action_pairs": self.state_action_log,
            "rewards": self.rewards_log
        }
        with open(path, "w") as f:
            json.dump(logs, f)
    
    def load_logs_json(self, path: str) -> None:
        """
        Load the state-action pairs and rewards from a JSON file.

        Args:
            path (str): Path to load the logs from.
        """
        with open(path, "r") as f:
            logs = json.load(f)
            self.state_action_log = logs["state_action_pairs"]
            self.rewards_log = logs["rewards"]
            
    def load_multiple_logs_json(self, directory: str, seed_range: range) -> None:
        """
        Load multiple JSON logs from a directory based on a range of seeds.

        Args:
            directory (str): Directory containing the JSON logs.
            seed_range (range): Range of seed values to load.
        """
        for seed in seed_range:
            file_pattern = os.path.join(directory, f"training_logs_seed_{seed}_steps_*.json")
            files = glob.glob(file_pattern)
            
            for file in files:
                with open(file, "r") as f:
                    logs = json.load(f)
                    self.state_action_log.extend(logs["state_action_pairs"])
                    self.rewards_log.extend(logs["rewards"])

    def save_logs_npz(self, path: str) -> None:
        """
        Save the state-action pairs and rewards to a .npz file.

        Args:
            path (str): Path to save the logs.
        """
        np.savez(path, state_action_pairs=self.state_action_log, rewards=self.rewards_log)

    def load_logs_npz(self, path: str) -> None:
        """
        Load the state-action pairs and rewards from a .npz file.

        Args:
            path (str): Path to load the logs from.
        """
        data = np.load(path, allow_pickle=True)
        self.state_action_log = data['state_action_pairs'].tolist()
        self.rewards_log = data['rewards'].tolist()