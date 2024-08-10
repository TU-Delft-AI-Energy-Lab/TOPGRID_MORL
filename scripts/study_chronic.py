import argparse
import json
import os
import numpy as np
from grid2op.Reward import EpisodeDurationReward

from topgrid_morl.envs.GridRewards import ScaledEpisodeDurationReward, ScaledTopoActionReward
from topgrid_morl.envs.EnvSetup import setup_environment
from topgrid_morl.utils.MO_PPO_train_utils import train_agent
from grid2op.Chronics import ChronicsHandler
from topgrid_morl.envs.GridRewards import ScaledEpisodeDurationReward, ScaledLinesCapacityReward, LinesCapacityReward
#from topgrid_morl.agent.MO_BaselineAgents import DoNothingAgent, PPOAgent  # Import the DoNothingAgent class
import matplotlib.pyplot as plt
env_name = "rte_case5_example"

config = "base_config.json"
config_path = os.path.join(os.getcwd(), "configs", config)
    # Load configuration from file
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"No such file or directory: '{config_path}'")

with open(config_path, "r") as config_file:
    config = json.load(config_file)

config_name = config['config_name']
project_name = config['project_name']
agent_params = config["agent_params"]
weight_vectors = config["weight_vectors"]
weights = np.array(weight_vectors)  # Convert to numpy array for consistency
max_gym_steps = config["max_gym_steps"]
env_params = config["env_params"]
max_rho = env_params["max_rho"]
network_params = config["network_params"]
net_arch = network_params["net_arch"]
rewards = config["rewards"]
reward_list = [rewards["second"], rewards["third"]]

seed= 2
    

actions_file = 'filtered_actions.json'

gym_env_val, _, _, _, g2op_env_val = setup_environment(
        env_name=env_name,
        test=False,
        seed=seed,
        action_space=53,
        first_reward=ScaledLinesCapacityReward,
        rewards_list=reward_list,
        actions_file=actions_file,
        env_type="_val",
        max_rho = max_rho
)
from grid2op.PlotGrid import PlotMatplot
plot_helper = PlotMatplot(g2op_env_val.observation_space)
obs = gym_env_val.reset(options={"time serie id": "12"} )

obs = g2op_env_val.reset(options={"time serie id": "12"} )
obs = g2op_env_val.observation_space
print(obs.line_status)



print(gym_env_val.action_space.from_gym(33))
obs, reward, done, info = g2op_env_val.step(gym_env_val.action_space.from_gym(33))
g2op_env_val
print(obs, info)
fig = plot_helper.plot_obs(obs)
plt.show()


