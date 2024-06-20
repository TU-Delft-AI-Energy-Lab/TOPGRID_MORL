import numpy as np
from MO_PPO_EnvSetup import setup_environment  # Assuming this function sets up environment variables
from MO_PPO_train_utils import initialize_network, train_agent  # Functions for network initialization and training

# Step 1: Setup Environment
# Assuming setup_environment() returns gym_env, obs_dim, action_dim, reward_dim
gym_env, obs_dim, action_dim, reward_dim = setup_environment()

# Reset the environment to verify dimensions
gym_env.reset()
print(f"Action dimension: {action_dim}")
print(f"Observation dimension: {obs_dim}")

# Step 2: Initialize Neural Networks
# Example network architecture [64, 64]
networks = initialize_network(obs_dim=obs_dim, action_dim=action_dim, reward_dim=reward_dim, net_arch=[64, 64])

# Step 3: Define Agent Parameters
agent_params = {
    "id": 1,
    "log": False,
    "steps_per_iteration": 2048,
    "num_minibatches": 32,
    "update_epochs": 10,
    "learning_rate": 3e-4,
    "gamma": 0.995,
    "anneal_lr": False,
    "clip_coef": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "clip_vloss": True,
    "max_grad_norm": 0.5,
    "norm_adv": True,
    "target_kl": None,
    "gae": True,
    "gae_lambda": 0.95,
    "device": "cpu",
    "seed": 42,
}

# Step 4: Training Parameters
weight_vectors = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]
weight_vectors = np.array(weight_vectors)  # Convert to numpy array for consistency
num_episodes = 10
max_ep_steps = 20
results_dir = "training_results"



# Step 5: Train Agent
train_agent(
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