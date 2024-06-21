#import grid2op
from grid2op.Reward import BaseReward
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
import numpy as np


class TopoActionReward(BaseReward):
    def __init__(self, penalty_factor=0.1, logger=None):
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
        reward =1

        action_dict = action.as_dict()

        if action_dict == {}:
            print("The dictionary is empty")
            reward = 1
        else:
            if list(action_dict.keys())[0] == 'set_bus_vect':
                #Modification of Topology
                nb_mod_objects = action.as_dict()['set_bus_vect']['nb_modif_objects']
                #print("nb_mod_objects")
                reward = reward - self.penalty_factor * nb_mod_objects
            else: 
                #line switching
                reward = reward - 1 * self.penalty_factor        
        return reward
    
class MaxDistanceReward(BaseReward):
    """
    This reward computes a penalty based on the maximum topological deviation from the original state
    where everything is connected to bus 1, encountered during the episode.
    """

    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = 0.0
        self.reward_max = 1.0
        self.max_deviation = 0.0  # Initialize the maximum deviation to zero

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        # Get topology vector from environment observation
        obs = env.get_obs(_do_copy=False)
        topo = obs.topo_vect

        idx = 0
        diff = 0.0

        # Iterate over substation information in the observation
        for n_elems_on_sub in obs.sub_info:
            # Determine the range of elements belonging to the current substation in the topology vector
            sub_start = idx
            sub_end = idx + n_elems_on_sub
            current_sub_topo = topo[sub_start:sub_end]

            # Count the number of elements not connected to bus 1
            # In the initial state, all elements are connected to bus 1
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

    def reset(self, env: BaseEnv):
        """
        Called by the environment each time it is "reset".
        Resets the maximum deviation to zero.
        """
        self.max_deviation = 0.0
