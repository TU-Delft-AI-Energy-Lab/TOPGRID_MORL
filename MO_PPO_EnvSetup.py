import grid2op
from grid2op.gym_compat import BoxGymObsSpace, BoxGymActSpace, GymEnv
from grid2op.gym_compat import DiscreteActSpace
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNReward, LinesCapacityReward, DistanceReward
from MOGrid2Op import TopoActionReward
from CustomGymEnv import CustomGymEnv  # Import your custom environment if necessary

from MOGrid2Op import TopoActionReward, MaxDistanceReward

def setup_environment(env_name="l2rpn_case14_sandbox", frist_reward = L2RPNReward, rewards_list=["LinesCapacity", "TopoAction"]):
    """
    Sets up the Grid2Op environment with the specified rewards and returns the Gym-compatible environment and reward dimension.

    Parameters:
    - env_name (str): Name of the Grid2Op environment.
    - rewards_list (list): List of reward names to be included in the environment.
    - custom_env (object, optional): Custom Gym environment object if needed.

    Returns:
    - gym_env (GymEnv): Gym-compatible environment instance.
    - reward_dim (int): Dimension of the reward vector (number of rewards + 1).
    """

    # Create environment
    env = grid2op.make(env_name, backend=LightSimBackend(), reward_class=frist_reward,
                       other_rewards={reward_name: globals()[reward_name + 'Reward'] for reward_name in rewards_list})

    # Use custom Gym environment if provided
    gym_env = CustomGymEnv(env)

    # Set rewards in Gym Environment
    gym_env.set_rewards(rewards_list=rewards_list)

    # Modify observation space if needed
    obs_attributes_to_keep = ["rho", "topo_vect", "gen_p", "gen_q", "gen_v", "gen_theta",
                              "load_p", "load_q", "load_v", "load_theta", "p_or", "q_or", "v_or", "a_or", "theta_or"]
    gym_env.observation_space = BoxGymObsSpace(env.observation_space, attr_to_keep=obs_attributes_to_keep)

    # Action space setup
    #gym_env.action_space = BoxGymActSpace(env.action_space, attr_to_keep=["set_bus", "set_line_status"])
    action_dim = 219
    gym_env.action_space = DiscreteActSpace(env.action_space,
                                        attr_to_keep=["set_bus" , "set_line_status"])
    # Calculate reward dimension
    reward_dim = len(rewards_list) + 1

    print(f"Environment setup completed for {env_name} with Gym compatibility.")

    return gym_env, gym_env.observation_space.shape, action_dim, reward_dim

def get_observation_dimension(self):
        """Get the observation dimension of the environment."""
        return self.observation_space.shape

def get_action_dimension(self):
    """Get the action dimension of the environment."""
    return self.action_space.shape[0]