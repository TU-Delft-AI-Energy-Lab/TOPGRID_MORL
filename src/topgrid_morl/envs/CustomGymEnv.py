from grid2op.gym_compat import GymEnv
from grid2op.Environment import BaseEnv
import numpy as np

class CustomGymEnv(GymEnv):
    """Implements a grid2op env in gym."""

    def __init__(self, env: BaseEnv):
        super().__init__(env)
        self.idx = 0
        self.reconnect_line = []
    
    def set_rewards(self, rewards_list):
        """Set the list of rewards to be used."""
        self.rewards = rewards_list

    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        g2op_obs = self.init_env.reset()
        return self.observation_space.to_gym(g2op_obs)

    def step(self, action):
        """Take a step in the environment."""
        g2op_act = self.action_space.from_gym(action)

        if self.reconnect_line:
            for line in self.reconnect_line:
                g2op_act = g2op_act + line

            self.reconnect_line = []

        g2op_obs, reward1, done, info = self.init_env.step(g2op_act)
        
        # Create reward array
        reward = np.array([reward1] + [info['rewards'].get(reward, 0) for reward in self.rewards])

        # Handle opponent attack
        if info["opponent_attack_duration"] == 1:
            line_id_attacked = np.argwhere(info["opponent_attack_line"]).flatten()[0]
            g2op_act = self.init_env.action_space({"set_line_status": [(line_id_attacked, 1)]})
            self.reconnect_line.append(g2op_act)

        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs, reward, done, info
