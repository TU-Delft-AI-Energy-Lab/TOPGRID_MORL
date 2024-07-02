import numpy as np
from topgrid_morl.envs.EnvSetup import setup_environment  # Assuming this function sets up environment variables
from topgrid_morl.utils.MO_PPO_train_utils import initialize_network, train_agent, train_and_save_donothing_agent  # Functions for network initialization and training
import wandb


# Step 1: Setup Environment
# Assuming setup_environment() returns gym_env, obs_dim, action_dim, reward_dim
from grid2op.Reward import L2RPNReward
from grid2op.Reward import EpisodeDurationReward

env_name = "rte_case5_example"
results_dir = "training_results_5bus"

num_seeds = 2
for seed in range(num_seeds):

    gym_env, obs_dim, action_dim, reward_dim = setup_environment(env_name=env_name, test=True, seed=seed, action_space=99, frist_reward = EpisodeDurationReward, rewards_list=["LinesCapacity", "TopoAction"])

    # Reset the environment to verify dimensions
    gym_env.reset()
    print(f"Action dimension: {action_dim}")
    print(f"Observation dimension: {obs_dim}")

    # Step 2: Initialize Neural Networks
    # Example network architecture [64, 64]
    networks = initialize_network(obs_dim=obs_dim, action_dim=action_dim, reward_dim=reward_dim, device="cpu", net_arch=[64, 64], )

    # Step 3: Define Agent Parameters
    agent_params = {
        "id": 1,
        "log": True,
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
        "gae_lambda": 0.95
    }

    # Step 4: Training Parameters
    weight_vectors = [
        [1, 0, 0],
        [0, 1, 0],  
        [0, 0, 1]
    ]
    
    weight_vectors = np.array(weight_vectors)  # Convert to numpy array for consistency
    num_episodes = 1
    max_ep_steps = 5

    # Step 5: Train Agent
    train_agent(
        weight_vectors=weight_vectors,
        num_episodes=num_episodes,
        max_ep_steps=max_ep_steps,
        seed=seed,
        results_dir=results_dir,
        env=gym_env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        run_name="Run",
        device="cpu",
        **agent_params
    )
    #wandb.finish()
    # Step 6: DoNothing Benchmark
    train_and_save_donothing_agent(action_space=99, gym_env=gym_env, num_episodes=num_episodes, seed=seed, max_ep_steps=max_ep_steps, reward_dim=reward_dim, save_dir=results_dir)