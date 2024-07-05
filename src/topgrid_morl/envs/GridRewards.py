from typing import Optional

import numpy as np
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
from grid2op.Reward import BaseReward


class TopoActionReward(BaseReward):
    """
    Reward for taking a topological action in the grid environment.
    Penalizes the agent for taking any action to encourage minimal intervention.
    """

    def __init__(
        self, penalty_factor: float = 0.1, logger: Optional[object] = None
    ) -> None:
        """
        Initialize the TopoActionReward.

        Args:
            penalty_factor (float): The penalty factor for taking an action.
            logger (Optional[object]): Logger for debugging purposes.
        """
        self.penalty_factor = penalty_factor
        super().__init__(logger)

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
        Compute the reward for the given action in the environment.

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
            return -1.0  # Penalize for illegal or erroneous actions

        # Penalize the reward if action is taken (negative reward)
        if action is not None:
            return -self.penalty_factor

        return 0.0


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
        r = np.interp(
            self.max_deviation,
            [0.0, len(topo) * 1.0],
            [self.reward_max, self.reward_min],
        )

        return r

    def reset(self, env: BaseEnv) -> None:
        """
        Reset the maximum deviation to zero. Called by the environment each time it is reset.

        Args:
            env (BaseEnv): The environment object.
        """
        self.max_deviation = 0.0
