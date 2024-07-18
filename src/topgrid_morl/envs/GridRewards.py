from typing import Optional
import os
import numpy as np
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
from grid2op.Reward import BaseReward
from grid2op.dtypes import dt_float

class ScaledEpisodeDurationReward(BaseReward):
    """
    This reward will always be 0., unless at the end of an episode where it will return the number
    of steps made by the agent divided by the total number of steps possible in the episode.

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:: python

        import grid2op
        from grid2op.Reward import EpisodeDurationReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=EpisodeDurationReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the EpisodeDurationReward class

    Notes
    -----
    In case of an environment being "fast forward" (see :func:`grid2op.Environment.BaseEnv.fast_forward_chronics`)
    the time "during" the fast forward are counted "as if" they were successful.

    This means that if you "fast forward" up until the end of an episode, you are likely to receive a reward of 1.0


    """

    def __init__(self, per_timestep=1, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.per_timestep = dt_float(per_timestep)
        self.total_time_steps = dt_float(0.0)
        self.reward_nr = 0
        self.reward_min, self.reward_max, self.reward_mean, self.reward_std = dt_float(get_mean_std_rewards(self.reward_nr))

    def initialize(self, env):
        self.reset(env)

    def reset(self, env):
        if env.chronics_handler.max_timestep() > 0:
            self.total_time_steps = env.max_episode_duration() * self.per_timestep
        else:
            self.total_time_steps = np.inf
            self.reward_max = np.inf

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done:
            res = env.nb_time_step
            if np.isfinite(self.total_time_steps):
                res /= self.total_time_steps
        else:
            res = self.reward_min
            
        norm_reward = (res - self.reward_min) / (self.reward_max -self.reward_min)
        return norm_reward

class TopoActionReward(BaseReward):
    def __init__(self, penalty_factor=10, logger=None):
        self.penalty_factor = penalty_factor
        super().__init__(logger)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return -1  # Penalize for illegal or erroneous actions
        """
        Compute the reward for the given action in the environment.

        Parameters:
        - action (BaseAction): The action taken by the agent.
        - env (BaseEnv): The environment object.
        - kwargs: Additional arguments if needed.

        Returns:
        - reward (float): The computed reward value.
        """
        reward =0

        action_dict = action.as_dict()

        if action_dict == {}:
            return reward #no topo action
        else:
            if list(action_dict.keys())[0] == 'set_bus_vect':
                #Modification of Topology
                nb_mod_objects = action.as_dict()['set_bus_vect']['nb_modif_objects']
                #print("nb_mod_objects")
                reward = - (self.penalty_factor * nb_mod_objects)
            else: 
                #line switching
                reward = - (1 * self.penalty_factor )      
            return reward

class ScaledTopoActionReward(BaseReward):
    def __init__(self, penalty_factor=10, logger=None):
        self.penalty_factor = penalty_factor
        self.rewardNr = 2
        self.reward_min, self.reward_max, self.reward_mean, self.reward_std = dt_float(get_mean_std_rewards(self.rewardNr))
        super().__init__(logger)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return -1  # Penalize for illegal or erroneous actions
        """
        Compute the reward for the given action in the environment.

        Parameters:
        - action (BaseAction): The action taken by the agent.
        - env (BaseEnv): The environment object.
        - kwargs: Additional arguments if needed.

        Returns:
        - reward (float): The computed reward value.
        """
        reward =0

        action_dict = action.as_dict()

        if action_dict == {}:
            return reward #no topo action
        else:
            if list(action_dict.keys())[0] == 'set_bus_vect':
                #Modification of Topology
                nb_mod_objects = action.as_dict()['set_bus_vect']['nb_modif_objects']
                #print("nb_mod_objects")
                reward = - (self.penalty_factor * nb_mod_objects)
            else: 
                #line switching
                reward = - (1 * self.penalty_factor )  
        # Check for zero division
        if self.reward_max != self.reward_min:
            norm_reward = (reward - self.reward_min) / (self.reward_max - self.reward_min)
        else:
            norm_reward = 0.0  # or handle it in another appropriate way

        return norm_reward   
            
class MaxDistanceReward(BaseReward):
    """
    Reward based on the maximum topological deviation from the initial state where everything is connected to bus 1.
    This reward encourages the agent to maintain the original topology as much as possible.
    """

    def __init__(self, logger: Optional[object] = None) -> None:
        """
        Initialize the MaxDistanceReward.

        Args:
            logger (Optional[object]): Logger for debugging purposes.
        """
        super().__init__(logger)
        self.reward_min = 0.0
        self.reward_max = 1.0
        self.max_deviation = 0.0  # Initialize the maximum deviation to zero

    def __call__(
        self,
        action: BaseAction,
        env: BaseEnv,
        has_error: bool,
        is_done: bool,
        is_illegal: bool,
        is_ambiguous: bool,
    ) -> float:
        """
        Compute the reward based on the maximum topological deviation.

        Args:
            action (BaseAction): The action taken by the agent.
            env (BaseEnv): The environment object.
            has_error (bool): Whether the action resulted in an error.
            is_done (bool): Whether the episode is done.
            is_illegal (bool): Whether the action was illegal.
            is_ambiguous (bool): Whether the action was ambiguous.

        Returns:
            float: The computed reward value.
        """
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        # Get topology vector from environment observation
        obs = env.get_obs(_do_copy=False)
        topo = obs.topo_vect

        idx = 0
        diff = 0.0

        # Iterate over substation information in the observation
        for n_elems_on_sub in obs.sub_info:
            sub_start = idx
            sub_end = idx + n_elems_on_sub
            current_sub_topo = topo[sub_start:sub_end]

            # Count the number of elements not connected to bus 1
            diff += 1.0 * np.count_nonzero(current_sub_topo != 1)

            # Move index to the start of the next substation
            idx += n_elems_on_sub

        # Update the maximum deviation
        if diff > self.max_deviation:
            self.max_deviation = diff

        # Compute the reward based on the maximum deviation recorded
        r = float(
            np.interp(
                self.max_deviation,
                [0.0, len(topo) * 1.0],
                [self.reward_max, self.reward_min],
            )
        )

        return r

    def reset(self, env: BaseEnv) -> None:
        """
        Reset the maximum deviation to zero. Called by the environment each time it is reset.

        Args:
            env (BaseEnv): The environment object.
        """
        self.max_deviation = 0.0

class LinesCapacityReward(BaseReward):
    """
    Reward based on lines capacity usage
    Returns max reward if no current is flowing in the lines
    Returns min reward if all lines are used at max capacity

    Compared to `:class:L2RPNReward`:
    This reward is linear (instead of quadratic) and only
    considers connected lines capacities

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:: python

        import grid2op
        from grid2op.Reward import ScaledLinesCapacityReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT, reward_class=ScaledLinesCapacityReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the ScaledLinesCapacityReward class
    """

    def __init__(self, logger=None):
        # Initialize the base class
        BaseReward.__init__(self, logger=logger)
        # Define the minimum and maximum reward values
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        # If there's an error, the action is illegal, or ambiguous, return the minimum reward
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        # Get the observation
        obs = env.get_obs(_do_copy=False)
        # Calculate the number of connected lines
        n_connected = dt_float(obs.line_status.sum())
        # Calculate the total usage of connected lines
        usage = obs.rho[obs.line_status].sum()
        # Ensure the usage is within valid range
        usage = np.clip(usage, 0.0, float(n_connected))
        # Calculate the reward: if no usage, reward is max; if full usage, reward is min
        reward = (n_connected - usage) / n_connected if n_connected > 0 else self.reward_min
        # Scale the reward between self.reward_min and self.reward_max
        # Return the calculated reward
        return reward
    
class ScaledLinesCapacityReward(BaseReward):
    """
    Reward based on lines capacity usage
    Returns max reward if no current is flowing in the lines
    Returns min reward if all lines are used at max capacity

    Compared to `:class:L2RPNReward`:
    This reward is linear (instead of quadratic) and only
    considers connected lines capacities

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:: python

        import grid2op
        from grid2op.Reward import ScaledLinesCapacityReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT, reward_class=ScaledLinesCapacityReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the ScaledLinesCapacityReward class
    """

    def __init__(self, logger=None):
        # Initialize the base class
        BaseReward.__init__(self, logger=logger)
        # Define the minimum and maximum reward values
        self.rewardNr = 1
        self.reward_min, self.reward_max, self.reward_mean, self.reward_std = dt_float(get_mean_std_rewards(self.rewardNr))
      
        

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        # If there's an error, the action is illegal, or ambiguous, return the minimum reward
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        # Get the observation
        obs = env.get_obs(_do_copy=False)
        # Calculate the number of connected lines
        n_connected = dt_float(obs.line_status.sum())
        # Calculate the total usage of connected lines
        usage = obs.rho[obs.line_status].sum()
        # Ensure the usage is within valid range
        usage = np.clip(usage, 0.0, float(n_connected))
        # Calculate the reward: if no usage, reward is max; if full usage, reward is min
        reward = (n_connected - usage) / n_connected if n_connected > 0 else self.reward_min
        # Scale the reward between self.reward_min and self.reward_max
        # Return the calculated reward
        #std scale: 
        norm_reward = (reward - self.reward_min) / (self.reward_max -self.reward_min)
        return norm_reward
    
def get_mean_std_rewards(rewardNr: int):
    script_dir = os.getcwd()
    rewards_dir = os.path.join(script_dir, "data", "rewards", "5bus_maxgymsteps_1024")
    if rewardNr==0: 
        training_rewards_path = os.path.join(rewards_dir, "generate_training_rewards_weights_1_0_0.npy")
    elif rewardNr==1: 
        training_rewards_path = os.path.join(rewards_dir, "generate_training_rewards_weights_1_0_0.npy")
    elif rewardNr==2:
        training_rewards_path = os.path.join(rewards_dir, "generate_training_rewards_weights_1_0_0.npy")
    
    training_rewards = np.load(training_rewards_path)
    mean = np.mean(training_rewards, axis=0)[rewardNr]
    std = np.std(training_rewards, axis=0)[rewardNr]
    min_r = np.min(training_rewards, axis=0)[rewardNr]
    max_r = np.max(training_rewards, axis=0)[rewardNr]
    return(min_r, max_r, mean, std)


    
    