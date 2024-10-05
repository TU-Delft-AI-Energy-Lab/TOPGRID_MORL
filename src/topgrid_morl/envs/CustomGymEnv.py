from typing import Any, Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from grid2op.Environment import BaseEnv
from grid2op.gym_compat import GymEnv

from grid2op.Opponent import (
    BaseActionBudget,
    RandomLineOpponent,
)

class CustomGymEnv(GymEnv):
    """Implements a grid2op environment in gym."""

    def __init__(self, env: BaseEnv, safe_max_rho: float = 0.9, eval=False) -> None:
        """
        Initialize the CustomGymEnv.

        Args:
            env (BaseEnv): The base grid2op environment.
            safe_max_rho (float): Safety threshold for the line loadings.
        """
        super().__init__(env)
        self.idx: int = 0
        self.reconnect_line: List[Any] = []
        self.rho_threshold: float = safe_max_rho
        self.steps: int = 0
        self.eval = eval

    def set_rewards(self, rewards_list: List[str]) -> None:
        """
        Set the list of rewards to be used in the environment.

        Args:
            rewards_list (List[str]): List of reward names.
        """
        self.rewards = rewards_list
        self.reward_dim = len(self.rewards) + 1

    def reset(
        self,
        *,
        seed: Union[int, None] = None,
        options: Union[Dict[str, Any], None] = None
    ) -> npt.NDArray[np.float64]:
        """
        Reset the environment.

        Args:
            seed (Union[int, None]): Seed for the environment reset.
            options (Union[dict, None]): Additional options for reset.

        Returns:
            npt.NDArray[np.float64]: The initial observation of the environment.
        """
        g2op_obs = self.init_env.reset()
        #max_iter = 28*288
        #self.init_env.set_max_iter(max_iter)
        self.steps = 0
        self.reconnect_line=[] #reset self.reconnect line in case of env.reset -> avoid cross episode contimination
        self.terminated_gym = False
    
        #print('in customGymEnv Reset')
        return self.observation_space.to_gym(g2op_obs)

    def step(
        self, action: int
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action (int): The action to take in the environment.

        Returns:
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], bool, Dict[str, Any]]:
            Observation, reward, done flag, and additional info.
        """
        #print('in Step')

        
        tmp_steps = 0 
        g2op_act = self.action_space.from_gym(action)
        # Reconnect lines if necessary      
        
        #cum_reward = np.zeros(self.reward_dim)  # Initialize cumulative reward
        #print(g2op_act)
        if self.reconnect_line:
            for line in self.reconnect_line:
                g2op_act += line
            self.reconnect_line = []
            #print(action)
            print(g2op_act)
            
        g2op_obs, reward1, done, info = self.init_env.step(action=g2op_act)
        reward = np.array(
                [reward1] + [info["rewards"].get(reward, 0) for reward in self.rewards],
                 dtype=np.float64,
        )
        print(reward)
        print(info['rewards'])
        print(g2op_obs.rho)
        print(done)
                #print(reward)
        self.steps += 1
        tmp_steps +=1 
        #cum_reward += tmp_reward   #line reco doesnt influence the rewards okay
        #g2op_obs, reward1, done, info = self.init_env.step(g2op_act)
        if done: 
            self.terminated_gym = True # Terminated is true if episode ends after Gym step and not in grid2op loop. I tbhought this would be good due to the SMDP property

        do_nothing_action = self.action_space.from_gym(0)  
        # Handle line loadings and ensure safety threshold is maintained
        while (max(g2op_obs.rho) < self.rho_threshold) and (not done):
            
            g2op_obs, reward1, done, info = self.init_env.step(action=do_nothing_action)
            tmp_reward = np.array(
                [reward1] + [info["rewards"].get(reward, 0) for reward in self.rewards],
                dtype=np.float64,
            )
            self.steps += 1
            tmp_steps +=1 
            reward += tmp_reward
            
            
            
            

        #reward += cum_reward  # Accumulate the rewards
        info["steps"] = tmp_steps
        g2op_obs_log = g2op_obs
        gym_obs = self.observation_space.to_gym(g2op_obs)

        
        line_stat_s = g2op_obs.line_status
        cooldown = g2op_obs.time_before_cooldown_line
        can_be_reco = ~line_stat_s & (cooldown == 0)
        
        if can_be_reco.any():
            self.reconnect_line = [
                self.init_env.action_space({"set_line_status": [(id_, +1)]})
                for id_ in (can_be_reco).nonzero()[0]
            ]
            #print(self.reconnect_line)
        #reconnect lines
        #to_reco = info["disc_lines"]
        ##edit 
        #if not done, in order to prevent cross episode contaminiation# 
        #if np.any(to_reco == 0): #if not done
        # Get the indices of elements that are 0
        #    reco_id = np.where(to_reco == 0)[0]
        #    
        #    for line_id in reco_id:
        #        line_act = self.init_env.action_space(
        #             {"set_line_status": [(line_id, +1)]}
        #        )
        #            
        #        self.reconnect_line.append(line_act)
        
        if self.eval==True:
            return gym_obs, reward, done, info, g2op_obs_log, self.terminated_gym
        else: 
            return gym_obs, reward, done, info, self.terminated_gym
        
        
        
        """
        if info.get("opponent_attack_duration", 0) == 1:
            line_id_attacked = np.argwhere(info["opponent_attack_line"]).flatten()[0]
            g2op_act = self.init_env.action_space(
                {"set_line_status": [(line_id_attacked, 1)]}
            )
            self.reconnect_line.append(g2op_act)
        """