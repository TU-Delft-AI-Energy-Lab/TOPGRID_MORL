import argparse
import json
import numpy as np
from grid2op.Reward import EpisodeDurationReward
from topgrid_morl.envs.GridRewards import ScaledEpisodeDurationReward, ScaledTopoActionReward
from topgrid_morl.envs.EnvSetup import setup_environment
from topgrid_morl.utils.MO_PPO_train_utils import train_agent
from grid2op.Chronics import ChronicsHandler
#from topgrid_morl.agent.MO_BaselineAgents import DoNothingAgent, PPOAgent  # Import the DoNothingAgent class

def create_action_to_substation_mapping():
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

    gym_env, obs_dim, action_dim, reward_dim, g2op_env = setup_environment(
        env_name=env_name,
        test=False,
        seed=seed,
        action_space=53,
        first_reward=ScaledEpisodeDurationReward,
        rewards_list=["ScaledLinesCapacity", "ScaledTopoAction"],
        actions_file=actions_file
    )
    obs = gym_env.reset(options={"time serie id": "01"} )

    action_to_substation_mapping = {}
    for action in range(53):  # Actions from 0 to 52
        # Convert the discrete action to grid2op action
        g2op_act = gym_env.action_space.from_gym(action)
        
        # Extract the affected substation ID
        modif_subs_id = g2op_act.as_dict().get('set_bus_vect', {}).get('modif_subs_id', [None])
        substation_id = modif_subs_id[0] if modif_subs_id else None
        
        # Add the mapping to the dictionary
        action_to_substation_mapping[action] = substation_id

    print(action_to_substation_mapping)
    return action_to_substation_mapping

# Example usage
mapping = create_action_to_substation_mapping()


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

import argparse
import json

import numpy as np
from grid2op.Reward import EpisodeDurationReward

from topgrid_morl.envs.GridRewards import ScaledEpisodeDurationReward, ScaledTopoActionReward
from topgrid_morl.envs.EnvSetup import setup_environment
from topgrid_morl.utils.MO_PPO_train_utils import train_agent
from grid2op.Chronics import ChronicsHandler
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


gym_env, obs_dim, action_dim, reward_dim, g2op_env = setup_environment(
        env_name=env_name,
        test=False,
        seed=seed,
        action_space=53,
        first_reward=ScaledEpisodeDurationReward,
        rewards_list=["ScaledLinesCapacity", "ScaledTopoAction"],
        actions_file=actions_file
    )
obs = gym_env.reset(options={"time serie id": "01"} )
action_to_substation_mapping = {}
for action in range(53):  # Actions from 0 to 52
        # Convert the discrete action to grid2op action
        g2op_act = gym_env.action_space.from_gym(action)
        
        # Extract the affected substation ID
        modif_subs_id = g2op_act.as_dict().get('set_bus_vect', {}).get('modif_subs_id', [None])
        substation_id = modif_subs_id[0] if modif_subs_id else None
        
        # Add the mapping to the dictionary
        action_to_substation_mapping[action] = substation_id

print(action_to_substation_mapping)


obs, reward, done, info = gym_env.step(1)
action = 1
g2op_act = gym_env.action_space.from_gym(action)
print(g2op_act.as_dict()['set_bus_vect']['modif_subs_id'][0])
"""
"""
g2op_obs, reward1, done, info = g2op_env.step(g2op_act)
sub_topo = g2op_obs.sub_topology
busbar2 = np.sum(topo == 2)
print(busbar2)
action = 31
g2op_act = gym_env.action_space.from_gym(action)
g2op_obs, reward1, done, info = g2op_env.step(g2op_act)
topo = g2op_obs.topo_vect

busbar2 = np.sum(topo == 2)
print(busbar2)
"""
"""
chronics =g2op_env.chronics_handler.available_chronics()
for idx, chronic in enumerate(chronics):
    # Set the chronic to the environment
    g2op_env.set_id(chronic)

    # Reset the environment to the beginning of the chronic
    obs = gym_env.reset()

    # Print the chronic being processed
    print(f"Processing chronic {idx}: {chronic}")
"""