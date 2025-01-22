import numpy as np
import os
import torch as th
import wandb
from typing import Any, List, Tuple
from tqdm import tqdm
from topgrid_morl.utils.MO_PPO_train_utils import initialize_agent
from topgrid_morl.utils.Grid2op_eval import evaluate_agent

def sum_rewards(rewards):
    rewards_np = np.array(rewards)
    summed_rewards = rewards_np.sum(axis=0)
    return summed_rewards.tolist()

class MCMOPPOTrainer:
    def __init__(self, 
                 iterations: int,
                 max_gym_steps: int,
                 results_dir: str,
                 seed: int,
                 env: Any,
                 env_val: Any,
                 env_test: Any, 
                 g2op_env: Any, 
                 g2op_env_val: Any, 
                 g2op_env_test: Any,
                 obs_dim: Tuple[int],
                 action_dim: int,
                 reward_dim: int,
                 run_name: str,
                 project_name: str = "TOPGrid_MORL_5_mc",
                 net_arch: List[int] = [64, 64],
                 generate_reward: bool = False,
                 reward_list: List[str] = ["ScaledEpisodeDuration", "ScaledTopoAction"],
                 **agent_params: Any):
        
        self.iterations = iterations
        self.max_gym_steps = max_gym_steps
        self.results_dir = results_dir
        self.seed = seed
        self.env = env
        self.env_val = env_val
        self.env_test = env_test
        self.g2op_env = g2op_env
        self.g2op_env_val = g2op_env_val
        self.g2op_env_test = g2op_env_test
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.run_name = run_name
        self.project_name = project_name
        self.net_arch = net_arch
        self.generate_reward = generate_reward
        self.reward_list = reward_list
        self.agent_params = agent_params
        self.np_random = np.random.RandomState(seed)
    
    def sample_weights(self, num_samples: int) -> np.ndarray:
        """Samples random weight vectors from a uniform distribution."""
        weight_vectors = []
        for _ in range(num_samples):
            weights = np.random.uniform(0, 1, self.reward_dim)
            normalized_weights = weights / np.sum(weights)
            weight_vectors.append(normalized_weights)
        return np.array(weight_vectors)

    def run(self, num_samples: int = 5):
        """Runs MOPPO for randomly sampled weight vectors."""
        weight_vectors = self.sample_weights(num_samples)
        for i, weights in tqdm(enumerate(weight_vectors), total=num_samples):
            print(f"Running MOPPO for weight vector {i + 1}/{num_samples}")
            self.run_single(weights)

    def run_single(self, weights: np.array) -> None:
        """Trains and evaluates the agent with the given weight vector."""
        os.makedirs(self.results_dir, exist_ok=True)
        weights_str = "_".join(map(str, weights))
        agent = initialize_agent(
            self.env,
            self.env_val,
            self.g2op_env,
            self.g2op_env_val,
            weights,
            self.obs_dim,
            self.action_dim,
            self.reward_dim,
            self.net_arch,
            self.seed,
            self.generate_reward,
            **self.agent_params
        )
        agent.weights = th.tensor(weights).cpu().to(agent.device)
        
        if self.agent_params['log']: 
            run = wandb.init(
                project=self.project_name,
                name=f"MonteCarlo_{self.run_name}_{self.reward_list[0]}_{self.reward_list[1]}_weights_{weights_str}_seed_{self.seed}",
                group=f"{self.reward_list[0]}_{self.reward_list[1]}",
                tags=[self.run_name]
            )
        agent.train(max_gym_steps=self.max_gym_steps, reward_dim=self.reward_dim, reward_list=self.reward_list)
        if self.agent_params['log']:
            run.finish()
        
        eval_data_dict = {}
        chronics = self.g2op_env_val.chronics_handler.available_chronics()
        for idx, chronic in enumerate(chronics):
            key = f'eval_data_{idx}'
            eval_data_dict[key] = evaluate_agent(
                agent=agent,
                env=self.env_val,
                g2op_env=self.g2op_env_val,
                g2op_env_val=self.g2op_env_val,
                weights=agent.weights,
                eval_steps=7 * 288,
                chronic=chronic,
                idx=idx,
                reward_list=self.reward_list,
                seed=self.seed
            )
            
        test_data_dict = {}
        
        chronics = self.g2op_env_test.chronics_handler.available_chronics()
        
        for idx, chronic in enumerate(chronics):
            key = f'test_data_{idx}'
            test_data_dict[key] = evaluate_agent(
                agent=agent,
                env=self.env_test,
                g2op_env=self.g2op_env_val,
                g2op_env_val=self.g2op_env_test,
                weights=agent.weights,
                eval_steps=7 * 288,
                chronic=chronic,
                idx=idx,
                reward_list=self.reward_list,
                seed=self.seed,
                eval=False
            )
        return eval_data_dict, test_data_dict, agent 
