import copy
import os
from typing import Optional

import numpy as np
from grid2op.Action import BaseAction
from grid2op.Action._backendAction import _BackendAction
from grid2op.dtypes import dt_float
from grid2op.Environment import BaseEnv
from grid2op.Reward import BaseReward


class TopoDepthReward(BaseReward):
    """
    Reward class that penalizes based on the depth of topology changes.

    Attributes:
        last_reward (float): The last computed reward.
        reward_min (float): Minimum reward value.
        reward_max (float): Maximum reward value.
        penalize (int): Penalty factor for topology depth.
    """

    def __init__(self, logger=None):
        """
        Initializes the TopoDepthReward class.

        Args:
            logger (Logger, optional): Logger for the reward class. Defaults to None.
        """
        BaseReward.__init__(self, logger=logger)
        self.last_reward = 0
        self.reward_min = 0.0
        self.reward_max = 1.0
        self.penalize = 1

    def __call__(
        self,
        action: BaseAction,
        env,
        has_error: bool,
        is_done: bool,
        is_illegal: bool,
        is_ambiguous: bool,
    ) -> float:
        """
        Computes the reward based on the depth of topology changes.

        Args:
            action (Action): The action taken by the agent.
            env: The environment object.
            has_error (bool): Whether there was an error.
            is_done (bool): Whether the episode is done.
            is_illegal (bool): Whether the action is illegal.
            is_ambiguous (bool): Whether the action is ambiguous.

        Returns:
            float: The computed reward.
        """
        
        if has_error or is_done:
            return self.reward_min
        
        idx = 0
        topo_dist = 0
        obs = env.get_obs(_do_copy=False)
        for n_elems_on_sub in obs.sub_info:
            #print(n_elems_on_sub)
            # Find this substation elements range in topology vect
            sub_start = idx
            #print(sub_start)
            sub_end = idx + n_elems_on_sub
            current_sub_topo = obs.topo_vect[sub_start:sub_end]
            #print(current_sub_topo)
            # Count number of elements not on bus 1
            # Because at the initial state, all elements are on bus 1
            if np.any(current_sub_topo == 2):
                topo_dist -=1
            #print(topo_dist)
            idx += n_elems_on_sub
            #print(topo_dist)
        
        if topo_dist == 0 : 
            reward = 1
        elif topo_dist <=5:
            reward = -topo_dist**2
        else: 
            reward = -50
        
        norm_reward = reward/2000     
        
        return norm_reward

class ScaledTopoDepthReward(BaseReward):
    """
    Reward class that penalizes based on the depth of topology changes and scales the reward.
    
    Attributes:
        last_reward (float): The last computed reward.
        reward_min (float): Minimum reward value.
        reward_max (float): Maximum reward value.
        penalize (int): Penalty factor for topology depth.
    """

    def __init__(self, logger=None):
        """
        Initializes the ScaledTopoDepthReward class.

        Args:
            logger (Logger, optional): Logger for the reward class. Defaults to None.
        """
        BaseReward.__init__(self, logger=logger)
        self.last_reward = 0
        self.reward_min = 0.0
        self.reward_max = 1.0
        self.penalize = 1

    def __call__(
        self,
        action: BaseAction,
        env,
        has_error: bool,
        is_done: bool,
        is_illegal: bool,
        is_ambiguous: bool,
    ) -> float:
        """
        Computes the scaled reward based on the depth of topology changes.

        Args:
            action (Action): The action taken by the agent.
            env: The environment object.
            has_error (bool): Whether there was an error.
            is_done (bool): Whether the episode is done.
            is_illegal (bool): Whether the action is illegal.
            is_ambiguous (bool): Whether the action is ambiguous.

        Returns:
            float: The computed and scaled reward between 0 and 1.
        """
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        # Get topology vector from the environment observation
        obs = env.get_obs(_do_copy=False)
        topo = obs.topo_vect

        # Count the number of elements connected to busbar 2
        busbar2 = np.sum(topo == 2)

        # Calculate the reward, penalized by the number of elements connected to busbar 2
        r = busbar2 / len(topo) * -self.penalize

        # Scale the reward, initially between -1 and 0, then shift to 0-1 range
        norm_r = r / 500

        return norm_r


class MaxTopoDepthReward(BaseReward):
    """
    Reward class that tracks and penalizes the maximum depth of topology changes.

    Attributes:
        max_depth (int): The maximum depth recorded.
        reward_min (float): Minimum reward value.
        reward_max (float): Maximum reward value.
        penalize (int): Penalty factor for topology depth.
    """

    def __init__(self, logger=None):
        """
        Initializes the MaxTopoDepthReward class.

        Args:
            logger (Logger, optional): Logger for the reward class. Defaults to None.
        """
        BaseReward.__init__(self, logger=logger)
        self.max_depth = 0
        self.reward_min = 0.0
        self.reward_max = 1.0
        self.penalize = 1

    def __call__(
        self,
        action: BaseAction,
        env,
        has_error: bool,
        is_done: bool,
        is_illegal: bool,
        is_ambiguous: bool,
    ) -> float:
        """
        Computes the reward based on the maximum depth of topology changes.

        Args:
            action (Action): The action taken by the agent.
            env: The environment object.
            has_error (bool): Whether there was an error.
            is_done (bool): Whether the episode is done.
            is_illegal (bool): Whether the action is illegal.
            is_ambiguous (bool): Whether the action is ambiguous.

        Returns:
            float: The computed reward.
        """
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        # Get topology vector from the environment observation
        obs: Observation = env.get_obs(_do_copy=False)
        topo = obs.topo_vect

        # Count the number of elements connected to busbar 2
        busbar2 = np.sum(topo == 2)

        # Update the maximum depth if the current depth is greater
        if busbar2 > self.max_depth:
            self.max_depth = busbar2

        # Calculate the reward, penalized by the maximum depth
        r = self.max_depth / len(topo) * -self.penalize

        return r


class ScaledMaxTopoDepthReward(BaseReward):
    """
    Reward class that tracks and penalizes the maximum depth of topology changes and scales the reward.

    Attributes:
        max_depth (int): The maximum depth recorded.
        reward_min (float): Minimum reward value.
        reward_max (float): Maximum reward value.
        penalize (int): Penalty factor for topology depth.
    """

    def __init__(self, logger=None):
        """
        Initializes the ScaledMaxTopoDepthReward class.

        Args:
            logger (Logger, optional): Logger for the reward class. Defaults to None.
        """
        BaseReward.__init__(self, logger=logger)
        self.max_depth = 0
        self.reward_min = 0.0
        self.reward_max = 1.0
        self.penalize = 1

    def __call__(
        self,
        action: BaseAction,
        env,
        has_error: bool,
        is_done: bool,
        is_illegal: bool,
        is_ambiguous: bool,
    ) -> float:
        """
        Computes the scaled reward based on the maximum depth of topology changes.

        Args:
            action (Action): The action taken by the agent.
            env: The environment object.
            has_error (bool): Whether there was an error.
            is_done (bool): Whether the episode is done.
            is_illegal (bool): Whether the action is illegal.
            is_ambiguous (bool): Whether the action is ambiguous.

        Returns:
            float: The computed and scaled reward.
        """
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        # Get topology vector from the environment observation
        obs: Observation = env.get_obs(_do_copy=False)
        topo = obs.topo_vect

        # Count the number of elements connected to busbar 2
        busbar2 = np.sum(topo == 2)

        # Update the maximum depth if the current depth is greater
        if busbar2 > self.max_depth:
            self.max_depth = busbar2

        # Calculate the reward, penalized by the maximum depth
        r = self.max_depth / len(topo) * -self.penalize

        # Scale the reward
        norm_r = r / 1200

        return norm_r


class SubstationSwitchingReward(BaseReward):
    """
    Reward class that penalizes switching actions on substations.

    Attributes:
        last_sub (int or None): The last substation that was switched.
        sub_station_switchings (np.ndarray): Array to track the number of switchings for each substation.
        calls (int): The number of calls to the reward function.
        reward_min (float): Minimum reward value.
        reward_max (float): Maximum reward value.
        penalize (int): Penalty factor for substation switching.
    """

    def __init__(self, logger=None):
        """
        Initializes the SubstationSwitchingReward class.

        Args:
            logger (Logger, optional): Logger for the reward class. Defaults to None.
        """
        BaseReward.__init__(self, logger=logger)
        self.last_sub = None
        self.sub_station_switchings = np.zeros((5,), dtype=int)
        self.calls = 0
        self.reward_min = 0.0
        self.reward_max = 1.0
        self.penalize = 1

    def __call__(
        self,
        action: BaseAction,
        env,
        has_error: bool,
        is_done: bool,
        is_illegal: bool,
        is_ambiguous: bool,
    ) -> float:
        """
        Computes the reward based on substation switching actions.

        Args:
            action (Action): The action taken by the agent.
            env: The environment object.
            has_error (bool): Whether there was an error.
            is_done (bool): Whether the episode is done.
            is_illegal (bool): Whether the action is illegal.
            is_ambiguous (bool): Whether the action is ambiguous.

        Returns:
            float: The computed reward.
        """
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min
        if is_done:
            self.calls = 0
        self.calls += 1
        r = 0
        if self.calls >= 2016:
            self.sub_station_switchings = np.zeros((5,), dtype=int)

        # Check if the action involves setting the bus
        if action.as_dict():
            # Extract the substation ID from the action
            sub_id = int(action.as_dict()["set_bus_vect"]["modif_subs_id"][0])

            # Increment the count of switchings for this substation
            self.sub_station_switchings[sub_id] += 1

            # Penalize if the action is on a different substation from the last one
            if sub_id != self.last_sub:
                # Calculate penalty factor
                pen_factor = 1 / (self.sub_station_switchings[sub_id])
                r = -self.penalize * pen_factor
                self.last_sub = sub_id

        return r  # Penalize the reward because it increases with substation switching


class DistanceReward(BaseReward):
    """
    This reward computes a penalty based on the distance of the current grid to the grid at time 0 where
    everything is connected to bus 1.

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:: python

        import grid2op
        from grid2op.Reward import DistanceReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=DistanceReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the DistanceReward class

    """

    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        # Get topo from env
        obs = env.get_obs(_do_copy=False)
        topo = obs.topo_vect

        idx = 0
        diff = dt_float(0.0)
        for n_elems_on_sub in obs.sub_info:
            # Find this substation elements range in topology vect
            sub_start = idx
            sub_end = idx + n_elems_on_sub
            current_sub_topo = topo[sub_start:sub_end]

            # Count number of elements not on bus 1
            # Because at the initial state, all elements are on bus 1
            diff += dt_float(1.0) * np.count_nonzero(current_sub_topo != 1)

            # Set index to next sub station
            idx += n_elems_on_sub

        r = np.interp(
            diff,
            [dt_float(0.0), len(topo) * dt_float(1.0)],
            [self.reward_max, self.reward_min],
        )
        return r


class ScaledDistanceReward(BaseReward):
    """
    This reward computes a penalty based on the distance of the current grid to the grid at time 0 where
    everything is connected to bus 1.

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:: python

        import grid2op
        from grid2op.Reward import DistanceReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=DistanceReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the DistanceReward class

    """

    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)
        self.reward_max_stat = 1800

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        # Get topo from env
        obs = env.get_obs(_do_copy=False)
        topo = obs.topo_vect

        idx = 0
        diff = dt_float(0.0)
        for n_elems_on_sub in obs.sub_info:
            # Find this substation elements range in topology vect
            sub_start = idx
            sub_end = idx + n_elems_on_sub
            current_sub_topo = topo[sub_start:sub_end]

            # Count number of elements not on bus 1
            # Because at the initial state, all elements are on bus 1
            diff += dt_float(1.0) * np.count_nonzero(current_sub_topo != 1)

            # Set index to next sub station
            idx += n_elems_on_sub

        r = np.interp(
            diff,
            [dt_float(0.0), len(topo) * dt_float(1.0)],
            [self.reward_max, self.reward_min],
        )
        normed_r = r / self.reward_max_stat
        return normed_r


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
            self.logger.info(
                f"Divergence of the backend at step {env.nb_time_step} for N1Reward with error `{div_exc_}`"
            )
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
        self.reward_min, self.reward_max, self.reward_mean, self.reward_std = dt_float(
            get_mean_std_rewards(self.reward_nr)
        )

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
        self.reward_min, self.reward_max, self.reward_mean, self.reward_std = dt_float(
            get_mean_std_rewards(self.reward_nr)
        )

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

        norm_reward = (res - self.reward_min) / (self.reward_max - self.reward_min)
        return norm_reward


class TopoActionReward(BaseReward):
    def __init__(self, penalty_factor=10, logger=None):
        self.penalty_factor = penalty_factor
        self.calls = 0
        super().__init__(logger)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        self.calls +=1 
        if has_error or is_done:
            return 0  # Penalize for illegal or erroneous actions
        """
        Compute the reward for the given action in the environment.

        Parameters:
        - action (BaseAction): The action taken by the agent.
        - env (BaseEnv): The environment object.
        - kwargs: Additional arguments if needed.

        Returns:
        - reward (float): The computed reward value.
        """
        
        
        reward = 0

        action_dict = action.as_dict()
        if action_dict == {}:
            reward = 0  # no topo action
        else:
            reward = -1

        return reward


class TopoActionDayReward(BaseReward):
    def __init__(self, penalty_factor=10, logger=None):
        self.penalty_factor = penalty_factor
        self.calls =0
        self.switchings_per_day = 0 
        super().__init__(logger)
        self.reward_max = 1
        self.reward_min = 0

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        self.calls +=1 
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
        
        if self.calls >= 288:
            self.switchings_per_day = 0
            self.calls = 0
        
        reward = 0

        action_dict = action.as_dict()
        if action_dict == {}:
            reward = 0  # no topo action
        else:
            self.switchings_per_day+=1
        
        if self.switchings_per_day<=3: 
            reward =0
        elif self.switchings_per_day <=5:
            reward = -0.5
        elif self.switchings_per_day >5:
            reward = -1

        
        reward = reward #penalize if there is more switching actions per day, as it is not desired. 
        
        return reward
    
class TopoActionHourReward(BaseReward): #for 5bus system the switching per hour makes more sense.
    def __init__(self, penalty_factor=10, logger=None):
        self.penalty_factor = penalty_factor
        self.calls =0
        self.switchings_per_hour = 0 
        super().__init__(logger)
        self.reward_max = 1
        self.reward_min = 0

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        self.calls +=1 
        
        if has_error or is_done:
            self.calls = 0
            return self.reward_min  # Penalize for illegal or erroneous actions

        """
        Compute the reward for the given action in the environment.

        Parameters:
        - action (BaseAction): The action taken by the agent.
        - env (BaseEnv): The environment object.
        - kwargs: Additional arguments if needed.

        Returns:
        - reward (float): The computed reward value.
        """
        #print(self.calls)
        if self.calls >= 12:
            #self.__init__()
            self.switchings_per_hour = 0
            self.calls = 0
            #print('one hour passed')
        

        action_dict = action.as_dict()
        if action_dict == {}:
            reward = 0.1  # no topo action
        else:
            self.switchings_per_hour+=1
            #print(f'switching per hour {self.switchings_per_hour}')
            
        if self.switchings_per_hour ==0:
            reward = 0.1
        elif self.switchings_per_hour==1: 
            reward = -1
        elif self.switchings_per_hour==2:
            reward = -5
        else:
            reward = -10


            
        #penalize if there is more switching actions per hour, as it is not desired. 
        norm_reward = reward/50
        return norm_reward

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
        reward = 0

        action_dict = action.as_dict()
        if action_dict == {}:
            reward = 0  # no topo action
        else:
            reward = -1

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
        from grid2op.Reward import LinesCapacityReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=LinesCapacityReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the LinesCapacityReward class

    """

    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)


    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        obs = env.get_obs(_do_copy=False)
        n_connected = dt_float(obs.line_status.sum())
        usage = obs.rho[obs.line_status == True].sum()
        usage = np.clip(usage, 0.0, float(n_connected))
        reward = np.interp(
            n_connected - usage,
            [dt_float(0.0), float(n_connected)],
            [self.reward_min, self.reward_max],
        )
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
        from grid2op.Reward import LinesCapacityReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=LinesCapacityReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the LinesCapacityReward class

    """

    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)


    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_done:
            return self.reward_min

        obs = env.get_obs(_do_copy=False)
        n_connected = dt_float(obs.line_status.sum())
        usage = obs.rho[obs.line_status == True].sum()
        usage = np.clip(usage, 0.0, float(n_connected))
        reward = np.interp(
            n_connected - usage,
            [dt_float(0.0), float(n_connected)],
            [self.reward_min, self.reward_max],
        )
        norm_reward = reward / 1500
        return reward #return unscaled

    
class L2RPNReward(BaseReward):
    """
    This is the historical :class:`BaseReward` used for the Learning To Run a Power Network competition on WCCI 2019

    See `L2RPN <https://l2rpn.chalearn.org/>`_ for more information.

    This rewards makes the sum of the "squared margin" on each powerline.

    The margin is defined, for each powerline as:
    `margin of a powerline = (thermal limit - flow in amps) / thermal limit`
    (if flow in amps <= thermal limit) else `margin of a powerline  = 0.`

    This rewards is then: `sum (margin of this powerline) ^ 2`, for each powerline.


    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:: python

        import grid2op
        from grid2op.Reward import L2RPNReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=L2RPNReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the L2RPNReward class

    """

    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)


    def initialize(self, env):
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(env.backend.n_line)


    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_done:
            return self.reward_min
        else:
            line_cap = self.__get_lines_capacity_usage(env)
            res = line_cap.sum()
            # print(f"\t env.backend.get_line_flow(): {env.backend.get_line_flow()}")
            res = res/1000
            return res


    @staticmethod
    def __get_lines_capacity_usage(env):
        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        thermal_limits += 1e-1  # for numerical stability
        relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)

        x = np.minimum(relative_flow, dt_float(1.0))
        lines_capacity_usage_score = np.maximum(
            dt_float(1.0) - x**2, np.zeros(x.shape, dtype=dt_float)
        )
        return lines_capacity_usage_score

class ScaledL2RPNReward(BaseReward):
    """
    This is the historical :class:`BaseReward` used for the Learning To Run a Power Network competition on WCCI 2019

    See `L2RPN <https://l2rpn.chalearn.org/>`_ for more information.

    This rewards makes the sum of the "squared margin" on each powerline.

    The margin is defined, for each powerline as:
    `margin of a powerline = (thermal limit - flow in amps) / thermal limit`
    (if flow in amps <= thermal limit) else `margin of a powerline  = 0.`

    This rewards is then: `sum (margin of this powerline) ^ 2`, for each powerline.


    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:: python

        import grid2op
        from grid2op.Reward import L2RPNReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=L2RPNReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the L2RPNReward class

    """

    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)


    def initialize(self, env):
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(env.backend.n_line)


    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not is_done and not has_error:
            line_cap = self.__get_lines_capacity_usage(env)
            res = line_cap.sum()
        else:
            # no more data to consider, no powerflow has been run, reward is what it is
            res = self.reward_min
        # print(f"\t env.backend.get_line_flow(): {env.backend.get_line_flow()}")
        
        norm_res = res/15000
        return norm_res


    @staticmethod
    def __get_lines_capacity_usage(env):
        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        thermal_limits += 1e-1  # for numerical stability
        relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)

        x = np.minimum(relative_flow, dt_float(1.0))
        lines_capacity_usage_score = np.maximum(
            dt_float(1.0) - x**2, np.zeros(x.shape, dtype=dt_float)
        )
        return lines_capacity_usage_score
    
def get_mean_std_rewards(rewardNr: int):
    script_dir = os.getcwd()
    rewards_dir = os.path.join(script_dir, "data", "rewards", "5bus_maxgymsteps_1024")
    if rewardNr == 0:
        training_rewards_path = os.path.join(
            rewards_dir, "generate_training_rewards_weights_1_0_0.npy"
        )
    elif rewardNr == 1:
        training_rewards_path = os.path.join(
            rewards_dir, "generate_training_rewards_weights_1_0_0.npy"
        )
    elif rewardNr == 2:
        training_rewards_path = os.path.join(
            rewards_dir, "generate_training_rewards_weights_1_0_0.npy"
        )

    training_rewards = np.load(training_rewards_path)
    mean = np.mean(training_rewards, axis=0)[rewardNr]
    std = np.std(training_rewards, axis=0)[rewardNr]
    min_r = np.min(training_rewards, axis=0)[rewardNr]
    max_r = np.max(training_rewards, axis=0)[rewardNr]
    return (min_r, max_r, mean, std)
