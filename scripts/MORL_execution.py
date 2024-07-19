import argparse
import json
import ast
import numpy as np
from grid2op.Reward import EpisodeDurationReward
import os
from topgrid_morl.envs.GridRewards import ScaledEpisodeDurationReward
from topgrid_morl.envs.EnvSetup import setup_environment
from topgrid_morl.utils.MO_PPO_train_utils import train_agent
from topgrid_morl.agent.MO_BaselineAgents import DoNothingAgent, PPOAgent  # Import the DoNothingAgent class


def main(seed: int, config: str) -> None:
    """
    Main function to set up the environment, initialize networks, define agent parameters, train the agent,
    and run a DoNothing benchmark.
    """
    env_name = "rte_case5_example"
    
    config_path = os.path.join(os.getcwd(), 'configs', config)
    # Load configuration from file
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No such file or directory: '{config_path}'")

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    agent_params = config["agent_params"]
    weight_vectors = config['weight_vectors']
    weights = np.array(weight_vectors)  # Convert to numpy array for consistency
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
        first_reward=ScaledEpisodeDurationReward,
        rewards_list=["ScaledLinesCapacity", "ScaledTopoAction"],
        actions_file=actions_file
    )
    
    gym_env_val, _, _, _ = setup_environment(
        env_name=env_name,
        test=False,
        seed=seed,
        action_space=53,
        first_reward=ScaledEpisodeDurationReward,
        rewards_list=["ScaledLinesCapacity", "ScaledTopoAction"],
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
    parser = argparse.ArgumentParser(description="Run experiment with specific seed and weights")
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the experiment"
    )
    parser.add_argument(
        "--config", type=str, default="base_config.json", help="Path to the configuration file (default: configs/base_config.json)"
    )
    args = parser.parse_args()
    
    # Use ast.literal_eval to safely parse the weights argument
    main(args.seed, args.config)