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
    # Step 1: Setup Environment
    env_name = "rte_case5_example"
    results_dir = "training_results_5bus_4094"

    gym_env, obs_dim, action_dim, reward_dim = setup_environment(
        env_name=env_name,
        test=True,
        seed=seed,
        action_space=53,
        frist_reward=EpisodeDurationReward,
        rewards_list=["ScaledLinesCapacity", "TopoAction"],
    )

    # Reset the environment to verify dimensions
    gym_env.reset()

    # Step 3: Define Agent Parameters
    agent_params = {
        "id": 1,
        "log": True,
        "steps_per_iteration": 16,
        "num_minibatches": 2,
        "update_epochs": 5,
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
    }

    # Step 4: Training Parameters
    weight_vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    weight_vectors = np.array(weight_vectors)  # Convert to numpy array for consistency
    max_gym_steps = 32

    # Step 5: Train Agent
    train_agent(
        weight_vectors=weight_vectors,
        max_gym_steps=max_gym_steps,
        seed=seed,
        results_dir=results_dir,
        env=gym_env,
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
