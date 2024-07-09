import logging
from typing import List, Tuple

import grid2op
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace, GymEnv
from grid2op.Reward import (
    L2RPNReward,
)
from lightsim2grid import LightSimBackend

from topgrid_morl.envs.CustomGymEnv import (  # Import your custom environment if necessary
    CustomGymEnv,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_environment(
    env_name: str = "l2rpn_case14_sandbox",
    test: bool = False,
    action_space: int = 219,
    seed: int = 0,
    frist_reward: L2RPNReward = L2RPNReward,
    rewards_list: List[str] = ["LinesCapacity", "TopoAction"],
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

    # Create environment
    env = grid2op.make(
        env_name,
        test=test,
        backend=LightSimBackend(),
        reward_class=frist_reward,
        other_rewards={
            reward_name: globals()[reward_name + "Reward"]
            for reward_name in rewards_list
        },
    )

    env.seed(seed=seed)
    env.reset()

    # Use custom Gym environment
    gym_env = CustomGymEnv(env)

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
        env.observation_space, attr_to_keep=obs_tennet
    )

    # Action space setup
    gym_env.action_space = DiscreteActSpace(
        env.action_space, attr_to_keep=["set_bus", "set_line_status"]
    )

    # Calculate reward dimension
    reward_dim = len(rewards_list) + 1

    logger.info(gym_env.action_space)
    logger.info(f"Environment setup completed for {env_name} with Gym compatibility.")

    return gym_env, gym_env.observation_space.shape, action_space, reward_dim
