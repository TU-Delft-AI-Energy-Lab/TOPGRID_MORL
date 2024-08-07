from tqdm import tqdm
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import torch as th
import wandb
from topgrid_morl.utils.MO_PPO_train_utils import initialize_agent
from topgrid_morl.utils.Grid2op_eval import evaluate_agent
from topgrid_morl.agent.MO_BaselineAgents import DoNothingAgent
from topgrid_morl.agent.MO_PPO import MOPPO, MOPPONet


def sample_weights(self) -> np.ndarray:
        return self.np_random.uniform(
            self.weight_range[0], self.weight_range[1], (self.num_samples, self.reward_dim)
        )

def run(self, reward_list: List[str]) -> None:
    weight_vectors = self.sample_weights()
    for i, weights in tqdm(enumerate(weight_vectors), total=self.num_samples):
        print(f"Running MOPPO for weight vector {i + 1}/{self.num_samples}")
        self.run_single(weights, reward_list)
        
def train_agent(
    weights: np.array,
    max_gym_steps: int,
    results_dir: str,
    seed: int,
    env: Any,
    env_val: Any,
    g2op_env: Any, 
    g2op_env_val: Any, 
    obs_dim: Tuple[int],
    action_dim: int,
    reward_dim: int,
    run_name: str,
    project_name: str = "TOPGrid_MORL_5",
    net_arch: List[int] = [64, 64],
    generate_reward: bool = False,
    reward_list: List = ["ScaledEpisodeDuration", "ScaledTopoAction"],
    **agent_params: Any,
) -> None:
    """
    Train the agent using MO-PPO.

    Args:
        weight_vectors (List[npt.NDArray[np.float64]]): List of weight vectors.
        num_episodes (int): Number of episodes.
        max_ep_steps (int): Maximum steps per episode.
        results_dir (str): Directory to save results.
        seed (int): Random seed.
        env: The environment object.
        obs_dim (Tuple[int]): Observation dimension.
        action_dim (int): Action dimension.
        reward_dim (int): Reward dimension.
        run_name (str): Name of the run.
        agent_params: Additional parameters for the agent.
    """
    os.makedirs(results_dir, exist_ok=True)
    weights_str = "_".join(map(str, weights))
    agent = initialize_agent(
        env,env_val, g2op_env, g2op_env_val, weights, obs_dim, action_dim, reward_dim, net_arch, seed, generate_reward, **agent_params
    )
    agent.weights = th.tensor(weights).cpu().to(agent.device)
    run = wandb.init(
        project=project_name,
        name=f"{run_name}_{reward_list[0]}_{reward_list[1]}_weights_{weights_str}_seed_{seed}",
        group=f"{reward_list[0]}_{reward_list[1]}",
        tags=[run_name]
    )
    agent.train(max_gym_steps=max_gym_steps, reward_dim=reward_dim, reward_list=reward_list)
    run.finish()
    eval_data_dict = {}
    weights = th.tensor(weights).cpu().to(agent.device)
    chronics = g2op_env_val.chronics_handler.available_chronics()
    for idx, chronic in enumerate(chronics):
        key = f'eval_data_{idx}'
        eval_data_dict[key] = evaluate_agent(
                agent=agent,
                env=env_val,
                g2op_env=g2op_env_val,
                g2op_env_val=g2op_env_val,
                weights=weights,
                eval_steps=7 * 288,
                chronic=chronic,
                idx=idx,
                reward_list=reward_list,
                seed=seed,)
    return eval_data_dict