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

    def __init__(self, env: BaseEnv, safe_max_rho: float = 0.9, eval=False, debug=False) -> None:
        """
        Initialize the CustomGymEnv.

        Args:
            env (BaseEnv): The base grid2op environment.
            safe_max_rho (float): Safety threshold for the line loadings.
            eval (bool): Evaluation mode flag.
            debug (bool): Debug mode flag for toggling logs.
        """
        super().__init__(env)
        self.idx: int = 0
        self.reconnect_line: List[Any] = []
        self.rho_threshold: float = safe_max_rho
        self.steps: int = 0
        self.eval = eval
        self.debug = debug  # Add debug flag

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
        self.steps = 0
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
        if self.debug:
            print('In CustomGymEnv')
            print(f"Action received: {action}")
        
        tmp_steps = 0 
        g2op_act = self.action_space.from_gym(action)
        
        cum_reward = np.zeros(self.reward_dim)  # Initialize cumulative reward

        # Step in the environment
        g2op_obs, reward1, done, info = self.init_env.step(g2op_act)
        
        gym_obs = self.observation_space.to_gym(g2op_obs)
        tmp_steps += 1 
        self.steps += 1

        # Create reward array
        gym_reward = np.array(
            [reward1] + [info["rewards"].get(reward, 0) for reward in self.rewards],
            dtype=np.float64,
        )
        
        if self.debug:
            print(f"Initial step: Observation: {g2op_obs}, Done: {done}")
            print(f"Initial reward array: {gym_reward}")
        
        cum_reward += gym_reward

        # Reconnect lines
        to_reco = info["disc_lines"]
        self.reconnect_line = []
        if np.any(to_reco == 0):
            reco_id = np.where(to_reco == 0)[0]

            for line_id in reco_id:
                g2op_act = self.init_env.action_space(
                    {"set_line_status": [(line_id, +1)]}
                )
                self.reconnect_line.append(g2op_act)
        
        if self.reconnect_line:
            line_act = self.action_space.from_gym(0)
            for line in self.reconnect_line:
                line_act += line
            if not done: 
                if self.debug:
                    print(f"Reconnecting lines. Updated g2op_act: {line_act}")
                    
                g2op_obs, reward1, done, info = self.init_env.step(action=line_act)
                
                if self.debug:
                    print(f"Step after reconnection: Observation: {g2op_obs}, Reward: {reward1}, Done: {done}, Info: {info}")
                
                line_reward = np.array(
                    [reward1] + [info["rewards"].get(reward, 0) for reward in self.rewards],
                    dtype=np.float64,
                )   
                self.steps += 1
                tmp_steps += 1 
                cum_reward += line_reward
                
                if self.debug:
                    print(f"Updated cumulative reward after reconnection: {cum_reward}")
                
                self.reconnect_line = []

        safe_state_reward = 0 
        
        # Handle line loadings and ensure safety threshold is maintained
        while (max(g2op_obs.rho) < self.rho_threshold) and (not done):
            do_nothing = 0                    
            action = self.action_space.from_gym(do_nothing) 
            g2op_obs, reward1, done, info = self.init_env.step(action=action)
            tmp_reward = np.array(
                [reward1] + [info["rewards"].get(reward, 0) for reward in self.rewards],
                dtype=np.float64,
            )
            self.steps += 1
            tmp_steps += 1 
            safe_state_reward += tmp_reward
            
        if self.debug:
            print(f'SafeState Reward: {safe_state_reward}')
        
        cum_reward += safe_state_reward
        
        info["steps"] = tmp_steps
        
        if self.debug:
            print(f'Cumulative Reward after activation threshold loop: {cum_reward}')
            print(f"Steps taken in CustomGymEnv step: {tmp_steps}")

        # Return based on evaluation mode
        if self.eval:
            if self.debug:
                print(f"Returning in eval mode: Observation: {gym_obs}, Reward: {cum_reward}, Done: {done}, Info: {info}")
            return gym_obs, cum_reward, done, info, g2op_obs
        else: 
            if self.debug:
                print(f"Returning: Observation: {gym_obs}, Reward: {cum_reward}, Done: {done}, Info: {info}")
            return gym_obs, cum_reward, done, info
