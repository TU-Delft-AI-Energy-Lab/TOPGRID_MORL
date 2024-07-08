import numpy as np
import wandb
from grid2op.Reward import EpisodeDurationReward

from topgrid_morl.envs.EnvSetup import (  # Assuming this function sets up environment variables
    setup_environment,
)
from topgrid_morl.utils.MO_PPO_train_utils import (  # Functions for network initialization and training
    train_agent,
    train_and_save_donothing_agent,
)


def main() -> None:
    """
    Main function to set up the environment, initialize networks, define agent parameters, train the agent,
    and run a DoNothing benchmark.
    """
    # Step 1: Setup Environment
    env_name = "rte_case5_example"
    results_dir = "training_results_5bus"
    num_seeds = 2

    for seed in range(num_seeds):
        gym_env, obs_dim, action_dim, reward_dim = setup_environment(
            env_name=env_name,
            test=True,
            seed=seed,
            action_space=99,
            frist_reward=EpisodeDurationReward,
            rewards_list=["LinesCapacity", "TopoAction"],
        )

        # Reset the environment to verify dimensions
        gym_env.reset()
        wandb.log({"Action dimension": action_dim})
        wandb.log({"Observation dimension": obs_dim})

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
            "gae_lambda": 0.95,
            "device": "cpu",
        }

        # Step 4: Training Parameters
        weight_vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        weight_vectors = np.array(
            weight_vectors
        )  # Convert to numpy array for consistency
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
            **agent_params
        )

        # Step 6: DoNothing Benchmark
        train_and_save_donothing_agent(
            action_space=99,
            gym_env=gym_env,
            num_episodes=num_episodes,
            seed=seed,
            max_ep_steps=max_ep_steps,
            reward_dim=reward_dim,
            save_dir=results_dir,
        )


if __name__ == "__main__":
    main()
