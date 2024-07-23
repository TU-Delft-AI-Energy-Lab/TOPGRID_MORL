"""
import numpy as np

# Number of weight vectors
num_vectors = 5
# Number of elements in each weight vector
num_elements = 3

# List to store the weight vectors
weight_vectors = []

# Generate the weight vectors
for _ in range(num_vectors):
    weights = np.random.rand(num_elements)
    weights /= weights.sum()  # Normalize to sum to 1
    weights = np.round(weights, 2)  # Round to two decimal places
    weights[-1] = 1.0 - weights[:-1].sum()  # Adjust the last element to ensure the sum is exactly 1.0
    weights = np.round(weights, 2)
    weight_vectors.append(weights.tolist())

# Print the generated weight vectors
for i, wv in enumerate(weight_vectors):
    print(f"Weight vector {i+1}: {wv}")
"""
import argparse
import json

import numpy as np
from grid2op.Reward import EpisodeDurationReward

from topgrid_morl.envs.GridRewards import ScaledEpisodeDurationReward, ScaledTopoActionReward
from topgrid_morl.envs.EnvSetup import setup_environment
from topgrid_morl.utils.MO_PPO_train_utils import train_agent
#from topgrid_morl.agent.MO_BaselineAgents import DoNothingAgent, PPOAgent  # Import the DoNothingAgent class

env_name = "rte_case5_example"


config_path = "configs/base_config.json"
seed = 71
    # Load configuration from file
with open(config_path, 'r') as config_file:
    config = json.load(config_file)
    
agent_params = config["agent_params"]
weights = np.array(config["weight_vectors"])  # Convert to numpy array for consistency
max_gym_steps = config["max_gym_steps"]

    # Step 1: Setup Environment
if env_name == "rte_case5_example":
    results_dir = "training_results_5bus_4094"
    action_dim = 53
    test_flag = True
    actions_file = 'filtered_actions.json'


gym_env, obs_dim, action_dim, reward_dim = setup_environment(
        env_name=env_name,
        test=False,
        seed=seed,
        action_space=53,
        first_reward=ScaledEpisodeDurationReward,
        rewards_list=["ScaledLinesCapacity", "ScaledTopoAction"],
        actions_file=actions_file
    )

reward=0
penalty_factor = 10
gym_env.reset()
g2op_act = gym_env.action_space.from_gym(19)
action_dict = g2op_act.as_dict()
if action_dict == {}:
            print(reward) #no topo action
if list(action_dict.keys())[0] == 'set_bus_vect':
    #Modification of Topology
    nb_mod_objects = action_dict['set_bus_vect']['nb_modif_objects']
         #print("nb_mod_objects")
    reward = - penalty_factor * nb_mod_objects
print(reward)

