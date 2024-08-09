import json
import logging
import os
from typing import Any, List, Tuple

import grid2op
from grid2op.Action import BaseAction
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace, GymEnv
from grid2op.Reward import EpisodeDurationReward, LinesCapacityReward
from gymnasium.spaces import Discrete
from lightsim2grid import LightSimBackend

from topgrid_morl.envs.CustomGymEnv import CustomGymEnv
from topgrid_morl.envs.GridRewards import L2RPNReward, ScaledL2RPNReward, ScaledMaxTopoDepthReward, ScaledTopoDepthReward, SubstationSwitchingReward, MaxTopoDepthReward, TopoDepthReward, ScaledDistanceReward, DistanceReward, CloseToOverflowReward, N1Reward, ScaledEpisodeDurationReward, ScaledLinesCapacityReward,  ScaledTopoActionReward, TopoActionReward, LinesCapacityReward





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
    action_space: int = 53,
    seed: int = 0,
    first_reward: grid2op.Reward.BaseReward = LinesCapacityReward,
    rewards_list: List[str] = ["EpisodeDuration", "TopoAction"],
    actions_file: str = 'filtered_actions.json',
    env_type: str = '_train',
    max_rho: float = 0.95
) -> Tuple[GymEnv, Tuple[int], int, int]:
    """
    Sets up the Grid2Op environment with the specified rewards and
    returns the Gym-compatible environment and reward dim

    Args:
        env_name (str): Name of the Grid2Op environment.
        test (bool): Whether to use the test version of the environment.
        action_space (int): Dimension of the action space.
        seed (int): Random seed for reproducibility.
        frist_reward (L2RPNReward): The primary reward class.
        rewards_list (List[str]): List of reward names to be included in the environment.

    Returns:
        Tuple[GymEnv, Tuple[int], int, int]: Gym-compatible environment instance,
        observation space shape, action space dimension, and reward dimension.
    """
    
   
    print(rewards_list)
    # Create environment
    g2op_env = grid2op.make(
        env_name+env_type,
        test=test,
        backend=LightSimBackend(),
        reward_class=first_reward,
        other_rewards={
            reward_name: globals()[reward_name + "Reward"]
            for reward_name in rewards_list
        },
    )
    
    g2op_env.seed(seed=seed)
    g2op_env.reset()

    # Use custom Gym environment
    gym_env = CustomGymEnv(g2op_env, safe_max_rho=max_rho)

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

    # add do nothing action
    do_nothing_action = g2op_env.action_space({})
    all_actions.insert(0, do_nothing_action)

    gym_env.action_space = DiscreteActSpace(
        g2op_env.action_space, action_list=all_actions
    )

    # Calculate reward dimension
    reward_dim = len(rewards_list) + 1


    return gym_env, gym_env.observation_space.shape, action_space, reward_dim, g2op_env
