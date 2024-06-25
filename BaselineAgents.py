from grid2op.Agent.baseAgent import BaseAgent
import numpy as np
import torch as th
from morl_baselines.common.evaluation import log_episode_info
from typing import List, Optional, Union



class DoNothingAgent(BaseAgent):
    """
    This is the most basic BaseAgent. It is purely passive, and does absolutely nothing.

    As opposed to most reinforcement learning environments, in grid2op, doing nothing is often
    the best solution.

    """

    def __init__(self, action_space, gymenv, device: Union[th.device, str] = "cpu"):
        BaseAgent.__init__(self, 
                           action_space)
        self.env = gymenv
        self.device = device

    def act(self, observation, reward, done=False):
        """
        As better explained in the document of :func:`grid2op.BaseAction.update` or
        :func:`grid2op.BaseAction.ActionSpace.__call__`.

        The preferred way to make an object of type action is to call :func:`grid2op.BaseAction.ActionSpace.__call__`
        with the dictionary representing the action. In this case, the action is "do nothing" and it is represented by
        the empty dictionary.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The current observation of the :class:`grid2op.Environment.Environment`

        reward: ``float``
            The current reward. This is the reward obtained by the previous action

        done: ``bool``
            Whether the episode has ended or not. Used to maintain gym compatibility

        Returns
        -------
        res: :class:`grid2op.Action.Action`
            The action chosen by the bot / controller / agent.

        """
        res = self.action_space({})
        return res
    
    def train(self, num_episodes: int, max_ep_steps: int, reward_dim: int, print_every: int = 100, print_flag: bool = True) -> np.ndarray:
        """
        Trains the policy for a specified number of episodes.
        
        Args:
            num_episodes (int): Number of episodes to train.
            max_ep_steps (int): Maximum steps per episode.
            reward_dim (int): Dimension of the reward space.
            print_every (int): Frequency of printing training progress.
            print_flag (bool): Whether to print training progress.
        
        Returns:
            np.ndarray: Matrix of rewards for each episode.
        """
        reward_matrix = np.zeros((num_episodes, reward_dim))
        total_steps= []
        for i_episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = th.zeros(reward_dim).to(self.device)
            done = False
            cum_reward = 0
            gym_steps = 0
            grid2op_steps = 0
            while (done == False) and gym_steps < max_ep_steps:
                next_obs, reward, done, info = self.env.step(0)
                episode_reward += reward
                
                grid2op_steps = info['steps']

                if "episode" in info.keys():
                    log_episode_info(
                        info["episode"],
                        scalarization=np.dot,
                        weights=self.weights, 
                        global_timestep=self.global_step,
                        id=self.id,
                    )
                
                gym_steps += 1
        
            
            
            reward_matrix[i_episode] = episode_reward.cpu().numpy()
            total_steps.append(grid2op_steps)

        print('Training complete')
        print(total_steps)
        return reward_matrix, total_steps