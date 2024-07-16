import argparse

import numpy as np
from grid2op.Reward import EpisodeDurationReward

from topgrid_morl.envs.EnvSetup import setup_environment
from topgrid_morl.utils.MO_PPO_train_utils import train_agent


def main(seed: int) -> None:
    """
    Main function to set up the environment, initialize networks, define agent parameters, train the agent,
    and run a DoNothing benchmark.
    """
    env_name="rte_case5_example"
    
    
    # Step 1: Setup Environment
    if env_name == "rte_case5_example":
        results_dir = "training_results_5bus_4094"
        action_dim= 53
        test_flag=True
        actions_file='filtered_actions.json'
    elif env_name == "l2rpn_case14_sandbox":
        results_dir = 'training_results_14bus_4096'
        action_dim =134
        test_flag = False
        actions_file = 'medha_actions.json'

    gym_env, obs_dim, action_dim, reward_dim = setup_environment(
        env_name=env_name,
        test=False,
        seed=seed,
        action_space=53,
        frist_reward=EpisodeDurationReward,
        rewards_list=["ScaledLinesCapacity", "TopoAction"],
        actions_file=actions_file
    )
    
    gym_env_val, _, _, _ = setup_environment(
        env_name=env_name,
        test=False,
        seed=seed,
        action_space=53,
        frist_reward=EpisodeDurationReward,
        rewards_list=["ScaledLinesCapacity", "TopoAction"],
        actions_file=actions_file,
        env_type='_val'
    )

    # Reset the environment to verify dimensions
    gym_env.reset()
    gym_env_val.reset()

    # Step 3: Define Agent Parameters
    agent_params = {
        "id": 1,
        "log": True,
        "steps_per_iteration": 16,
        "num_minibatches": 2,
        "update_epochs": 5,
        "learning_rate": 3e-4,
        "gamma": 0.995,
        "anneal_lr": True,
        "clip_coef": 0.2,
        "ent_coef": 0.5,
        "vf_coef": 1.0,
        "clip_vloss": True,
        "max_grad_norm": 0.5,
        "norm_adv": True,
        "target_kl": None,
        "gae": True,
        "gae_lambda": 0.95,
        "device": "cpu",
    }

    # Step 4: Training Parameters
    weight_vectors = [[1, 0, 0]]
    weight_vectors = np.array(weight_vectors)  # Convert to numpy array for consistency
    max_gym_steps = 256

    # Step 5: Train Agent
    train_agent(
        weight_vectors=weight_vectors,
        max_gym_steps=max_gym_steps,
        seed=seed,
        results_dir=results_dir,
        env=gym_env,
        env_val=gym_env_val,
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        run_name="Run",
        net_arch=[64, 128, 64],
        **agent_params
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with specific seed")
    parser.add_argument(
        "--seed", type=int, required=True, help="Seed for the experiment"
    )
    args = parser.parse_args()
    main(args.seed)
