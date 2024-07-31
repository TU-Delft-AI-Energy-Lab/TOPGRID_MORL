from typing import Optional
import os
import numpy as np
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
from grid2op.Reward import BaseReward
from grid2op.dtypes import dt_float
from grid2op.Action._backendAction import _BackendAction


class N1Reward(BaseReward):
    """
    This class implements a reward that is inspired
    by the "n-1" criterion widely used in power system.
    
    More specifically it returns the maximum flows (on all the powerlines) after a given (as input) a powerline
    has been disconnected.

    Examples
    --------

    This can be used as:

    .. code-block:: python

        import grid2op
        from grid2op.Reward import N1Reward
        L_ID = 0
        env = grid2op.make("l2rpn_case14_sandbox",
                           reward_class=N1Reward(l_id=L_ID)
                          )
        obs = env.reset()
        obs, reward, *_ = env.step(env.action_space())
        print(f"reward: {reward:.3f}")
        print("We can check that it is exactly like 'simulate' on the current step the disconnection of the same powerline")
        obs_n1, *_ = obs.simulate(env.action_space({"set_line_status": [(L_ID, -1)]}), time_step=0)
        print(f"\tmax flow after disconnection of line {L_ID}: {obs_n1.rho.max():.3f}")

    Notes
    -----
    It is also possible to use the `other_rewards` argument to simulate multiple powerline disconnections, for example:

    .. code-block:: python

        import grid2op
        from grid2op.Reward import N1Reward
        L_ID = 0
        env = grid2op.make("l2rpn_case14_sandbox",
                           other_rewards={f"line_{l_id}": N1Reward(l_id=l_id)  for l_id in [0, 1]}
                           )
        obs = env.reset()
        obs, reward, *_ = env.step(env.action_space())
        print(f"reward: {reward:.3f}")
        print("We can check that it is exactly like 'simulate' on the current step the disconnection of the same powerline")
        obs_n1, *_ = obs.simulate(env.action_space({"set_line_status": [(L_ID, -1)]}), time_step=0)
        print(f"\tmax flow after disconnection of line {L_ID}: {obs_n1.rho.max():.3f}")

    """

    def __init__(self, l_id=0, logger=None):
        BaseReward.__init__(self, logger=logger)
        self._backend = None
        self._backend_action = None
        self.l_id = l_id


    def initialize(self, env):
        self._backend = env.backend.copy()
        bk_act_cls = _BackendAction.init_grid(env.backend)
        self._backend_action = bk_act_cls()


    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done:
            return self.reward_min
        self._backend_action.reset()
        act = env.backend.get_action_to_set()
        th_lim = env.get_thermal_limit()
        th_lim[th_lim <= 1] = 1  # assign 1 for the thermal limit
        this_n1 = copy.deepcopy(act)
        self._backend_action += this_n1
            
        self._backend.apply_action(self._backend_action)
        self._backend._disconnect_line(self.l_id)
        div_exc_ = None
        try:
            # TODO there is a bug in lightsimbackend that make it crash instead of diverging
            conv, div_exc_ = self._backend.runpf()
        except Exception as exc_:
            conv = False
            div_exc_ = exc_

        if conv:
            flow = self._backend.get_line_flow()
            res = (flow / th_lim).max()
        else:
            self.logger.info(f"Divergence of the backend at step {env.nb_time_step} for N1Reward with error `{div_exc_}`")
            res = -1
        return res


    def close(self):
        self._backend.close()
        del self._backend
        self._backend = None



class CloseToOverflowReward(BaseReward):
    """
    This reward finds all lines close to overflowing.
    Returns max reward when there is no overflow, min reward if more than one line is close to overflow
    and the mean between max and min reward if one line is close to overflow

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:: python

        import grid2op
        from grid2op.Reward import CloseToOverflowReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=CloseToOverflowReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with this class (computing the penalty based on the number of overflow)

    """

    def __init__(self, max_lines=5, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)
        self.max_overflowed = dt_float(max_lines)


    def initialize(self, env):
        pass


    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        thermal_limits = env.backend.get_thermal_limit()
        lineflow_ratio = env.current_obs.rho

        close_to_overflow = dt_float(0.0)
        for ratio, limit in zip(lineflow_ratio, thermal_limits):
            # Seperate big line and small line
            if (limit < 400.00 and ratio >= 0.95) or ratio >= 0.975:
                close_to_overflow += dt_float(1.0)

        close_to_overflow = np.clip(
            close_to_overflow, dt_float(0.0), self.max_overflowed
        )
        reward = np.interp(
            close_to_overflow,
            [dt_float(0.0), self.max_overflowed],
            [self.reward_max, self.reward_min],
        )
        return reward



class EpisodeDurationReward(BaseReward):
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
        return res    
        

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
            reward= 0 #no topo action
        else:
            reward =-1 
            
        return reward

class ScaledTopoActionReward(BaseReward):
    def __init__(self, penalty_factor=10, logger=None):
        self.penalty_factor = penalty_factor
        super().__init__(logger)
        self.reward_max = 1
        self.reward_min = 0

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
            reward= 0 #no topo action
        else:
            reward =-1 
        
        norm_rew = reward
        return norm_rew

            
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


    
    