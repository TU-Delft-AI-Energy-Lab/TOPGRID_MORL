import json
import logging
import os
from typing import Any, List, Tuple
import gymnasium as gym

import grid2op
from grid2op.Action import BaseAction, PowerlineSetAction
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace, GymEnv
from grid2op.Reward import EpisodeDurationReward, LinesCapacityReward
from grid2op.Opponent import RandomLineOpponent, BaseActionBudget # Import the desired opponent class
from gymnasium.spaces import Discrete
from lightsim2grid import LightSimBackend

from topgrid_morl.envs.CustomGymEnv import CustomGymEnv
from topgrid_morl.envs.GridRewards import L2RPNReward, TopoActionDayReward, TopoActionHourReward, ScaledL2RPNReward, ScaledMaxTopoDepthReward, ScaledTopoDepthReward, SubstationSwitchingReward, MaxTopoDepthReward, TopoDepthReward, ScaledDistanceReward, DistanceReward, CloseToOverflowReward, N1Reward, ScaledEpisodeDurationReward, ScaledLinesCapacityReward, ScaledTopoActionReward, TopoActionReward, LinesCapacityReward


class CustomDiscreteActions(Discrete):
    """
    Class that customizes the action space.
    """

    def __init__(self, converter: Any):
        """init"""
        self.converter = converter
        Discrete.__init__(self, n=converter.n)

    def from_gym(self, gym_action: int) -> BaseAction:
        """from_gym"""
        return self.converter.convert_act(gym_action)

    def close(self) -> None:
        """close"""


def setup_environment(
    env_name: str = "l2rpn_case14_sandbox",
    test: bool = False,
    action_space: int = 73,
    seed: int = 0,
    first_reward: grid2op.Reward.BaseReward = L2RPNReward,
    rewards_list: List[str] = ["TopoActionDay", "ScaledTopoDepth"],
    actions_file: str = 'tennet_actions.json',
    env_type: str = '_train',
    max_rho: float = 0.95,
    use_opponent: bool = False
    ) -> Tuple[GymEnv, Tuple[int], int, int]:
    """
    Sets up the Grid2Op environment with the specified rewards, opponent,
    and returns the Gym-compatible environment and reward dim.
    """
    
    
    print(rewards_list)
    # Create environment
    # Initialize the base keyword arguments
    kwargs = {
        'test': test,
        'backend': LightSimBackend(),
        'reward_class': first_reward,
        'other_rewards': {
            reward_name: globals()[reward_name + "Reward"]
            for reward_name in rewards_list
        }
    }
    # Conditionally add opponent parameters
    if use_opponent == 'line_a':
        kwargs.update({
            'opponent_attack_cooldown': 144,  # Max 2 attacks per day
            'opponent_attack_duration': 48,   # 4 hours in a day
            'opponent_budget_per_ts': 0.333343333333,  # Taken from Blazej
            'opponent_init_budget': 144,
            'opponent_action_class': PowerlineSetAction,
            'opponent_class': RandomLineOpponent,
            'opponent_budget_class': BaseActionBudget,
            'kwargs_opponent': {
                "lines_attacked": [
                    '0_3_2'
                ]
            }
        })
    
    if use_opponent == 'line_b':
        kwargs.update({
            'opponent_attack_cooldown': 144,  # Max 2 attacks per day
            'opponent_attack_duration': 48,   # 4 hours in a day
            'opponent_budget_per_ts': 0.333343333333,  # Taken from Blazej
            'opponent_init_budget': 144,
            'opponent_action_class': PowerlineSetAction,
            'opponent_class': RandomLineOpponent,
            'opponent_budget_class': BaseActionBudget,
            'kwargs_opponent': {
                "lines_attacked": [
                    '0_4_3'
                ]
            }
        })  
    
    
    # Conditionally add opponent parameters
    if use_opponent == 'normal':
        kwargs.update({
            'opponent_attack_cooldown': 144,  # Max 2 attacks per day
            'opponent_attack_duration': 48,   # 4 hours in a day
            'opponent_budget_per_ts': 0.333343333333,  # Taken from Blazej
            'opponent_init_budget': 144,
            'opponent_action_class': PowerlineSetAction,
            'opponent_class': RandomLineOpponent,
            'opponent_budget_class': BaseActionBudget,
            'kwargs_opponent': {
                "lines_attacked": [
                    '0_3_2',
                    '0_4_3'
                ]
            }
        })
        
    # Conditionally add opponent parameters
    elif use_opponent == 'hard':
        kwargs.update({
            'opponent_attack_cooldown': 72,  # Max 2 attacks per day
            'opponent_attack_duration': 48,   # 4 hours in a day
            'opponent_budget_per_ts': 0.333343333333,  # Taken from Blazej
            'opponent_init_budget': 288,
            'opponent_action_class': PowerlineSetAction,
            'opponent_class': RandomLineOpponent,
            'opponent_budget_class': BaseActionBudget,
            'kwargs_opponent': {
                "lines_attacked": [
                    '0_3_2',
                    '0_4_3'
                ]
            }
        })

    # Now call the function with the combined arguments
    g2op_env = grid2op.make(
        env_name + env_type,
        **kwargs
    )

    g2op_env.seed(seed=seed)
    g2op_env.reset()
    
    # Use custom Gym environment
    gym_env = CustomGymEnv(g2op_env, safe_max_rho=max_rho)

    #gym_env = gym.wrappers.NormalizeObservation(gym_env)
    # Set rewards in Gym Environment
    gym_env.set_rewards(rewards_list=rewards_list)

    # Modify observation space
    obs_tennet = [
        "rho",
        "gen_p",
        "load_p",
        "topo_vect",
        "p_or",
        "p_ex",
        "timestep_overflow",
    ]
    gym_env.observation_space = BoxGymObsSpace(
        g2op_env.observation_space, attr_to_keep=obs_tennet
    )

    # Action space setup
    current_dir = os.getcwd()
    path = os.path.join(current_dir, "action_spaces", env_name, actions_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Action file not found: {path}")

    with open(path, "rt", encoding="utf-8") as action_set_file:
        all_actions = list(
            (
                g2op_env.action_space(action_dict)
                for action_dict in json.load(action_set_file)
            )
        )

    # Add do nothing action
    do_nothing_action = g2op_env.action_space({})
    all_actions.insert(0, do_nothing_action)

    gym_env.action_space = DiscreteActSpace(
        g2op_env.action_space, action_list=all_actions
    )

    # Calculate reward dimension
    reward_dim = len(rewards_list) + 1
    print(gym_env.action_space)

    return gym_env, gym_env.observation_space.shape, action_space, reward_dim, g2op_env
