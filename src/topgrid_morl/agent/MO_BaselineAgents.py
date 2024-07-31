import numpy as np
import torch as th
import wandb
from typing import Tuple, Union, Optional
from grid2op.Reward import EpisodeDurationReward
from topgrid_morl.envs.CustomGymEnv import CustomGymEnv
from topgrid_morl.envs.EnvSetup import setup_environment
from mo_gymnasium import MORecordEpisodeStatistics
import gymnasium as gym
import numpy as np


class DoNothingAgent:
    """Baseline agent that always performs action 0."""

    def __init__(
        self,
        env: CustomGymEnv,
        env_val: CustomGymEnv,
        log: bool = False,
        device: Union[th.device, str] = "cpu",
    ) -> None:
        """
        Initialize the baseline agent.

        Args:
            env (CustomGymEnv): Custom gym environment for training.
            env_val (CustomGymEnv): Custom gym environment for validation.
            log (bool): Whether to log the training process.
            device (Union[th.device, str]): Device to use.
        """
        self.env = env
        self.env_val = env_val
        self.device = device
        self.log = log
        self.global_step = 0
        self.training_rewards = []
        self.evaluation_rewards = []

    def eval(self, obs: np.ndarray) -> int:
        """
        Evaluate the policy for a given observation (always returns action 0).

        Args:
            obs (np.ndarray): Observation.

        Returns:
            int: Action (always 0).
        """
        return 0

    def train(self, max_gym_steps: int, reward_dim: int) -> None:
        """
        Train the agent.

        Args:
            max_gym_steps (int): Total gym steps.
            reward_dim (int): Dimension of the reward.
        """
        state = self.env.reset(options={"max step": 7 * 288})
        done = False
        grid2op_steps = 0
        episode_rewards = np.zeros(reward_dim)
        while self.global_step < max_gym_steps:
            if done:
                state = self.env.reset(options={"max step": 7 * 288})
                grid2op_steps = 0
                episode_rewards = np.zeros(reward_dim)

            action = self.eval(state)
            next_state, reward, done, info = self.env.step(action)

            reward = np.array(reward)
            episode_rewards += reward
            steps_in_episode = info["steps"]
            grid2op_steps += steps_in_episode
            state = next_state

            # Save training rewards
            self.training_rewards.append(reward)

            # Log the training reward for each step
            log_data = {
                f"DoNothing/train/reward_{i}": reward[i] for i in range(reward_dim)
            }
            log_data["DoNothing/train/grid2opsteps"] = grid2op_steps
            if self.log:
                wandb.log(log_data, step=self.global_step)

            self.global_step += 1

        # Save training rewards to file
        np.save("training_rewards.npy", np.array(self.training_rewards))

        # Evaluate and log evaluation rewards
        eval_rewards = []
        eval_done = False
        eval_state = self.env_val.reset()
        while not eval_done:
            eval_action = self.eval(eval_state)
            eval_state, eval_reward, eval_done, _ = self.env_val.step(eval_action)
            eval_reward = np.array(eval_reward)
            eval_rewards.append(eval_reward)

        eval_rewards = np.mean(eval_rewards, axis=0)
        self.evaluation_rewards = eval_rewards

        # Save evaluation rewards to file
        np.save("evaluation_rewards.npy", np.array(self.evaluation_rewards))

        if self.log:
            log_val_data = {
                f"DoNothing/eval/reward_{i}": eval_rewards[i]
                for i in range(reward_dim)
            }
            wandb.log(log_val_data, step=self.global_step)