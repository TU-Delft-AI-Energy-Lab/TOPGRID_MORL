from BaselineAgents import DoNothingAgent

import numpy as np
from MO_PPO_EnvSetup import setup_environment  # Assuming this function sets up environment variables

import grid2op
from lightsim2grid import LightSimBackend

# Step 1: Setup Environment
# Assuming setup_environment() returns gym_env, obs_dim, action_dim, reward_dim
from grid2op.Reward import L2RPNReward
from grid2op.Reward import EpisodeDurationReward
from BaselineAgents import DoNothingAgent
from MO_PPO_train_utils import train_and_save_donothing_agent
gym_env, obs_dim, action_dim, reward_dim = setup_environment(frist_reward = EpisodeDurationReward, rewards_list=["LinesCapacity", "TopoAction"])

# Reset the environment to verify dimensions
gym_env.reset()
print(f"Action dimension: {action_dim}")
print(f"Observation dimension: {obs_dim}")

num_episodes=20
max_ep_steps = 50
reward_dim=3
results_dir = "training_results"
action_space = 219
#agent = DoNothingAgent(action_space=219, gymenv=gym_env)
#reward_matrix, total_steps = agent.train(num_episodes=num_episodes, max_ep_steps=max_ep_steps, reward_dim=reward_dim)
train_and_save_donothing_agent(agent_class=DoNothingAgent, action_space=action_space, gym_env=gym_env, num_episodes=10, max_ep_steps=max_ep_steps,reward_dim=reward_dim, save_dir=results_dir)





#Step 6: Do Nothing Agent Benchmark
