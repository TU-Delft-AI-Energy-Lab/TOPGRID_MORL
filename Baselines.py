from BaselineAgents import DoNothingAgent

import numpy as np
from MO_PPO_EnvSetup import setup_environment  # Assuming this function sets up environment variables

import grid2op
from lightsim2grid import LightSimBackend

# Step 1: Setup Environment
# Assuming setup_environment() returns gym_env, obs_dim, action_dim, reward_dim
from grid2op.Reward import L2RPNReward
from grid2op.Reward import EpisodeDurationReward
gym_env, obs_dim, action_dim, reward_dim = setup_environment(frist_reward = EpisodeDurationReward, rewards_list=["LinesCapacity", "TopoAction"])

# Reset the environment to verify dimensions
gym_env.reset()
print(f"Action dimension: {action_dim}")
print(f"Observation dimension: {obs_dim}")

DoNothingAgent.train_agent(
    weight_vectors=weight_vectors,
    num_episodes=num_episodes,
    max_ep_steps=max_ep_steps,
    results_dir=results_dir,
    env=gym_env,
    obs_dim=obs_dim,
    action_dim=action_dim,
    reward_dim=reward_dim,
    **agent_params
)

#Step 6: Do Nothing Agent Benchmark
from grid2op.Agent import DoNothingAgent
initial_obs = gym_env.reset()