import argparse
import json

import numpy as np
from grid2op.Reward import EpisodeDurationReward

from topgrid_morl.envs.GridRewards import ScaledEpisodeDurationReward
from topgrid_morl.envs.EnvSetup import setup_environment
from topgrid_morl.utils.MO_PPO_train_utils import train_agent
from topgrid_morl.agent.MO_BaselineAgents import DoNothingAgent, PPOAgent  # Import the DoNothingAgent class


def main(seed: int, config_path: str) -> None:
    """
    Main function to set up the environment, initialize networks, define agent parameters, train the agent,
    and run a DoNothing benchmark.
    """
    env_name = "rte_case5_example"
    
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
    elif env_name == "l2rpn_case14_sandbox":
        results_dir = 'training_results_14bus_4096'
        action_dim = 134
        test_flag = False
        actions_file = 'medha_actions.json'

    gym_env, obs_dim, action_dim, reward_dim = setup_environment(
        env_name=env_name,
        test=False,
        seed=seed,
        action_space=53,
        frist_reward=ScaledEpisodeDurationReward,
        rewards_list=["ScaledLinesCapacity", "TopoAction"],
        actions_file=actions_file
    )
    
    gym_env_val, _, _, _ = setup_environment(
        env_name=env_name,
        test=False,
        seed=seed,
        action_space=53,
        frist_reward=ScaledEpisodeDurationReward,
        rewards_list=["ScaledLinesCapacity", "TopoAction"],
        actions_file=actions_file,
        env_type='_val'
    )

    # Reset the environment to verify dimensions
    gym_env.reset()
    gym_env_val.reset()
    

    # Step 5: Train Agent
    train_agent(
        weight_vectors=weights,
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
        "--seed", type=int, default=42, help="Seed for the experiment"
    )
    parser.add_argument(
        "--config", type=str, default="configs/base_config.json", help="Path to the configuration file (default: config.json)"
    )
    args = parser.parse_args()
    main(args.seed, args.config)
