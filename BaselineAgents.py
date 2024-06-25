from grid2op.Agent.baseAgent import BaseAgent
import numpy as np
import torch as th
from morl_baselines.common.evaluation import log_episode_info


class DoNothingAgent(BaseAgent):
    """
    This is the most basic BaseAgent. It is purely passive, and does absolutely nothing.

    As opposed to most reinforcement learning environments, in grid2op, doing nothing is often
    the best solution.

    """

    def __init__(self, action_space):
        BaseAgent.__init__(self, action_space)

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
        actions = []
        total_steps = []
        for i_episode in range(num_episodes):
            state = self.env.reset()
            next_obs = th.Tensor(state).to(self.device)
            episode_reward = th.zeros(reward_dim).to(self.device)
            action_list_episode = []     
            done = False
            cum_reward = 0
            action_list = []
            gym_steps = 0
            grid2op_steps = 0
            while (done == False) and gym_steps < max_ep_steps:
                self.global_step += 1

                with th.no_grad():
                    action, logprob, _, value = self.networks.get_action_and_value(obs)
                    value = value.view(self.networks.reward_dim)

                next_obs, reward, next_done, info = self.env.step(action.cpu().item())
                reward = th.tensor(reward).to(self.device).view(self.networks.reward_dim)
                cum_reward += reward
                self.batch.add(obs, action, logprob, reward, done, value)
                action_list.append(action)
                steps_in_gymSteps = info['steps']
                obs, done = th.Tensor(next_obs).to(self.device), th.tensor(next_done).float().to(self.device)

                if "episode" in info.keys():
                    log_episode_info(
                        info["episode"],
                        scalarization=np.dot,
                        weights=self.weights,
                        global_timestep=self.global_step,
                        id=self.id,
                    )
                
                gym_steps += 1
                grid2op_steps += steps_in_gymSteps
            
            
            actions.append([action.cpu().numpy() for action in action_list_episode])
            reward_matrix[i_episode] = episode_reward.cpu().numpy()
            total_steps.append(grid2op_steps)


            if print_flag and (i_episode + 1) % print_every == 0:
                print(f"Episode {i_episode + 1}/{num_episodes}")
                print(f"  Episode Reward Sum: {episode_reward.sum().item()}")
                for j in range(reward_dim):
                    print(f"  Episode Reward {j}: {episode_reward[j].item()}")
                    print(f"  Actions: {actions[-1]}")

        print('Training complete')
        print(total_steps)
        return reward_matrix, actions, total_steps