from grid2op.gym_compat import GymEnv
from grid2op.Environment import BaseEnv
#from CustomGymEnv_l2rpn import GymEnvWithRecoWithDN
import numpy as np

class CustomGymEnv(GymEnv):
    """Implements a grid2op env in gym."""

    def __init__(self, env: BaseEnv, safe_max_rho=0.9):
        super().__init__(env)
        self.idx = 0
        self.reconnect_line = []
        self.rho_threshold = safe_max_rho
    
    def set_rewards(self, rewards_list):
        """Set the list of rewards to be used."""
        self.rewards = rewards_list
        self.reward_dim = len(self.rewards)+1

    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        g2op_obs = self.init_env.reset()
        self.steps = 0
        #return g2op_obs #returning grid2op_obs
        return self.observation_space.to_gym(g2op_obs)

    def step(self, action):
        """Take a step in the environment."""
        g2op_act = self.action_space.from_gym(action)
        self.steps += 1
        if self.reconnect_line:
            for line in self.reconnect_line:
                g2op_act = g2op_act + line

            self.reconnect_line = []

        cum_reward = np.zeros(self.reward_dim) #initialize initial reward defined as array of size 1,reward_dim

        g2op_obs, reward1, done, info = self.init_env.step(g2op_act)
        # Create reward array
        reward = np.array([reward1] + [info['rewards'].get(reward, 0) for reward in self.rewards])
        
        while (max(g2op_obs.rho) < self.rho_threshold) and (not done):
              action = 0
              do_nothing = self.action_space.from_gym(action)
              g2op_obs, reward, done, info = self.init_env.step(do_nothing)
              self.steps += 1
              reward = np.array([reward1] + [info['rewards'].get(reward, 0) for reward in self.rewards])
              #self.steps += 1
              cum_reward += reward


        reward += cum_reward
        #print(g2op_obs)
        # Handle opponent attack
        if info["opponent_attack_duration"] == 1:
            line_id_attacked = np.argwhere(info["opponent_attack_line"]).flatten()[0]
            g2op_act = self.init_env.action_space({"set_line_status": [(line_id_attacked, 1)]})
            self.reconnect_line.append(g2op_act)

        info["steps"] = self.steps

        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs, reward, done, info
    

"""
    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        
        cum_reward: float = 0.0
        obs, reward, done, info = self.env_gym.step(action)
        self.steps += 1
        cum_reward += reward
        while (max(obs["rho"]) < self.rho_threshold) and (not done):
            obs, reward, done,  = self.env_gym.step(0)
            self.steps += 1
            cum_reward += reward

        if done:
            info["steps"] = self.steps
        return obs, cum_reward, done, False, info
"""