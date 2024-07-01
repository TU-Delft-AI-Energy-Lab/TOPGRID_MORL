from grid2op.gym_compat import GymEnv
from grid2op.Environment import BaseEnv
import numpy as np

class CustomGymEnv(GymEnv):
    """Implements a grid2op env in gym."""

    def __init__(self, env: BaseEnv, safe_max_rho=0.9):
        super().__init__(env)
        self.idx = 0
        self.reconnect_line = []
        self.rho_threshold = safe_max_rho
        self.steps = 0
    
    def set_rewards(self, rewards_list):
        """Set the list of rewards to be used."""
        self.rewards = rewards_list
        self.reward_dim = len(self.rewards)+ 1

    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        g2op_obs = self.init_env.reset()
        return self.observation_space.to_gym(g2op_obs)
        self.steps = 0

    def step(self, action):

        """Take a step in the environment."""
        g2op_act = self.action_space.from_gym(action)

        if self.reconnect_line:
            for line in self.reconnect_line:
                g2op_act = g2op_act + line

            self.reconnect_line = []

        cum_reward = np.zeros(self.reward_dim) #initialize initial reward defined as array of size 1,reward_dim
        
        g2op_obs, reward1, done, info = self.init_env.step(g2op_act)
        self.steps +=1
        
        # Create reward array
        reward = np.array([reward1] + [info['rewards'].get(reward, 0) for reward in self.rewards])
        
        while (max(g2op_obs.rho) < self.rho_threshold) and (not done):
              action = 0
              do_nothing = self.action_space.from_gym(action)
              g2op_obs, reward, done, info = self.init_env.step(do_nothing)
              reward = np.array([reward1] + [info['rewards'].get(reward, 0) for reward in self.rewards])
              self.steps += 1
              cum_reward += reward
              
        reward += cum_reward     
         
        info["steps"] = self.steps
        # Handle opponent attack
        if info["opponent_attack_duration"] == 1:
            line_id_attacked = np.argwhere(info["opponent_attack_line"]).flatten()[0]
            g2op_act = self.init_env.action_space({"set_line_status": [(line_id_attacked, 1)]})
            self.reconnect_line.append(g2op_act)

        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs, reward, done, info
