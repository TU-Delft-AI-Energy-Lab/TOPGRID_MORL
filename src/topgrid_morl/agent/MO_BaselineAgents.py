from typing import List, Tuple, Union

import grid2op
import numpy as np
import numpy.typing as npt
import torch as th
from grid2op.Agent.baseAgent import BaseAgent
from morl_baselines.common.evaluation import log_episode_info


class DoNothingAgent(BaseAgent):
    """
    This is the most basic BaseAgent. It is purely passive, and does absolutely nothing.

    As opposed to most reinforcement learning environments, in grid2op, doing nothing is often
    the best solution.
    """

    def __init__(
        self,
        action_space: grid2op.Action.ActionSpace,
        gymenv: grid2op.Environment,
        device: Union[th.device, str] = "cpu",
    ) -> None:
        """
        Initialize the DoNothingAgent.

        Args:
            action_space (grid2op.Action.ActionSpace): The action space of the environment.
            gymenv (grid2op.Environment): The gym environment.
            device (Union[th.device, str]): The device to run the agent on (CPU or GPU).
        """
        super().__init__(action_space)
        self.env = gymenv
        self.device = device

    def act(
        self,
        observation: grid2op.Observation.Observation,
        reward: float,
        done: bool = False,
    ) -> grid2op.Action.Action:
        """
        The preferred way to make an object of type action is to call grid2op.BaseAction.ActionSpace.__call__
        with the dictionary representing the action. In this case, the action is "do nothing" and it is represented by
        the empty dictionary.

        Args:
            observation (grid2op.Observation.Observation): The current observation of the environment.
            reward (float): The current reward obtained by the previous action.
            done (bool): Whether the episode has ended or not. Used to maintain gym compatibility.

        Returns:
            grid2op.Action.Action: The action chosen by the agent.
        """
        res = self.action_space({})
        return res

    def train(
        self,
        num_episodes: int,
        max_ep_steps: int,
        reward_dim: int,
        print_every: int = 100,
        print_flag: bool = True,
    ) -> Tuple[npt.NDArray[np.float64], List[int]]:
        """
        Trains the policy for a specified number of episodes.

        Args:
            num_episodes (int): Number of episodes to train.
            max_ep_steps (int): Maximum steps per episode.
            reward_dim (int): Dimension of the reward space.
            print_every (int): Frequency of printing training progress.
            print_flag (bool): Whether to print training progress.

        Returns:
            Tuple[npt.NDArray[np.float64], List[int]]: Matrix of rewards for each episode and total steps per episode.
        """
        reward_matrix: npt.NDArray[np.float64] = np.zeros(
            (num_episodes, reward_dim), dtype=np.float64
        )
        total_steps: List[int] = []

        for i_episode in range(num_episodes):
            self.env.reset()
            episode_reward = th.zeros(reward_dim).to(self.device)
            done = False
            gym_steps = 0
            grid2op_steps = 0

            while not done and gym_steps < max_ep_steps:
                next_obs, reward, done, info = self.env.step(0)
                episode_reward += reward
                grid2op_steps = info["steps"]

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

        """
        if print_flag:
            print("Training complete")
            print(total_steps)
        """
        return reward_matrix, total_steps
