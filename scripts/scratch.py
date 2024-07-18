import argparse
import json

import numpy as np
from grid2op.Reward import EpisodeDurationReward

from topgrid_morl.envs.GridRewards import ScaledEpisodeDurationReward
from topgrid_morl.envs.EnvSetup import setup_environment
from topgrid_morl.utils.MO_PPO_train_utils import train_agent
from topgrid_morl.agent.MO_BaselineAgents import DoNothingAgent, PPOAgent  # Import the DoNothingAgent class

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
        frist_reward=ScaledEpisodeDurationReward,
        rewards_list=["ScaledLinesCapacity", "TopoAction"],
        actions_file=actions_file
    )
print(gym_env.step(31))