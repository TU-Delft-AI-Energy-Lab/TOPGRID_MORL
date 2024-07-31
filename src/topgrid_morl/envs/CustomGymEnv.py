from typing import Any, Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from grid2op.Environment import BaseEnv
from grid2op.gym_compat import GymEnv


class CustomGymEnv(GymEnv):
    """Implements a grid2op environment in gym."""

    def __init__(self, env: BaseEnv, safe_max_rho: float = 0.9) -> None:
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
        tmp_steps = 0 
        g2op_act = self.action_space.from_gym(action)
        # Reconnect lines if necessary
        if self.reconnect_line:
            for line in self.reconnect_line:
                g2op_act += line
            self.reconnect_line = []

        cum_reward = np.zeros(self.reward_dim)  # Initialize cumulative reward

        g2op_obs, reward1, done, info = self.init_env.step(g2op_act)
        
        to_reco = ~g2op_obs.line_status
        self.reconnect_line = []
        if np.any(to_reco):
            reco_id = np.where(to_reco)[0]
            for line_id in reco_id:
                g2op_act = self.init_env.action_space(
                    {"set_line_status": [(line_id, +1)]}
                )

        #         self.reconnect_line.append(g2op_act)
        tmp_steps +=1 
        self.steps += 1

        # Create reward array
        reward = np.array(
            [reward1] + [info["rewards"].get(reward, 0) for reward in self.rewards],
            dtype=np.float64,
        )

        # Handle line loadings and ensure safety threshold is maintained
        while (max(g2op_obs.rho) < self.rho_threshold) and (not done):
            action = 0
            do_nothing = self.action_space.from_gym(action)
            g2op_obs, reward1, done, info = self.init_env.step(do_nothing)
            reward = np.array(
                [reward1] + [info["rewards"].get(reward, 0) for reward in self.rewards],
                dtype=np.float64,
            )
            self.steps += 1
            tmp_steps +=1 
            cum_reward += reward

            if done:
                break  # Exit the loop if done is True

        reward += cum_reward  # Accumulate the rewards
        info["steps"] = tmp_steps

        # Handle opponent attack
        if info.get("opponent_attack_duration", 0) == 1:
            line_id_attacked = np.argwhere(info["opponent_attack_line"]).flatten()[0]
            g2op_act = self.init_env.action_space(
                {"set_line_status": [(line_id_attacked, 1)]}
            )
            self.reconnect_line.append(g2op_act)

        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs, reward, done, info
