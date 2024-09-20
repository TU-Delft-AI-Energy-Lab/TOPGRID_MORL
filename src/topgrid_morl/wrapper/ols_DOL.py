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


class MOPPOTrainer:
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
                 reuse: str = "none",  # Added reuse type parameter (none, partial, full)
                 project_name: str = "TOPGrid_MORL_14bus",
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
        self.reuse = reuse  # Store the reuse type
        self.project_name = project_name
        self.net_arch = net_arch
        self.generate_reward = generate_reward
        self.reward_list = reward_list
        self.agent_params = agent_params
        self.models = {}  # Dictionary to store trained models for reuse
        self.np_random = np.random.RandomState(seed)
        self.weight_range = [0, 1]  # Default range for weights
        self.num_samples = 5      # Default number of samples for weights
    
    def sample_weights(self):
        weights = np.random.rand(self.reward_dim)
        normalized_weights = weights / np.sum(weights)
        rounded_weights = np.round(normalized_weights, 2)
        return rounded_weights

    def run(self):
        weight_vectors = self.sample_weights()
        for i, weights in tqdm(enumerate(weight_vectors), total=self.num_samples):
            print(f"Running MOPPO for weight vector {i + 1}/{self.num_samples}")
            self.run_single(weights)

    def initialize_or_reuse_model(self, weights):
        """
        Reuse model parameters based on the reuse strategy specified.
        """
        weight_key = tuple(weights)
        
        if self.reuse == "none" or weight_key not in self.models:
            # No reuse, or no existing model for the given weights
            model = initialize_agent(
                self.env, self.env_val, self.g2op_env, self.g2op_env_val, 
                weights, self.obs_dim, self.action_dim, self.reward_dim, 
                self.net_arch, self.seed, self.generate_reward, **self.agent_params
            )
        else:
            # Reuse nearest model
            model = self.models[weight_key]
            if self.reuse == "partial":
                # Reinitialize the last layer
                model.reinitialize_last_layer()
                
        return model

    def run_single(self, weights: np.array):
        # Check if a model with similar weights already exists for reuse
        model = self.initialize_or_reuse_model(weights)

        # Set weights for the model
        model.weights = th.tensor(weights).cpu().to(model.device)
        
        if self.agent_params['log']: 
            weights_str = "_".join(map(str, weights))
            run = wandb.init(
                project=self.project_name,
                name=f"OLS_{self.run_name}_{self.reward_list[0]}_{self.reward_list[1]}_weights_{weights_str}_seed_{self.seed}",
                group=f"{self.reward_list[0]}_{self.reward_list[1]}",
                tags=[self.run_name]
            )

        # Train the agent
        model.train(max_gym_steps=self.max_gym_steps, reward_dim=self.reward_dim, reward_list=self.reward_list)

        if self.agent_params['log']:
            run.finish()

        # Save the trained model for future reuse if applicable
        weight_key = tuple(weights)
        if self.reuse != "none":
            self.models[weight_key] = model

        eval_data_dict, test_data_dict = self.evaluate_model(model, weights)
        
        return eval_data_dict, test_data_dict, model

    def evaluate_model(self, agent, weights):
        eval_data_dict = {}
        test_data_dict = {}

        weights = th.tensor(weights).cpu().to(agent.device)
        chronics = self.g2op_env_val.chronics_handler.available_chronics()
        
        # Validation evaluation
        for idx, chronic in enumerate(chronics):
            key = f'eval_data_{idx}'
            eval_data_dict[key] = evaluate_agent(
                agent=agent,
                env=self.env_val,
                g2op_env=self.g2op_env_val,
                g2op_env_val=self.g2op_env_val,
                weights=weights,
                eval_steps=7 * 288,
                chronic=chronic,
                idx=idx,
                reward_list=self.reward_list,
                seed=self.seed
            )

        # Test evaluation
        chronics = self.g2op_env_test.chronics_handler.available_chronics()
        for idx, chronic in enumerate(chronics):
            key = f'test_data_{idx}'
            test_data_dict[key] = evaluate_agent(
                agent=agent,
                env=self.env_test,
                g2op_env=self.g2op_env_val,
                g2op_env_val=self.g2op_env_test,
                weights=weights,
                eval_steps=7 * 288,
                chronic=chronic,
                idx=idx,
                reward_list=self.reward_list,
                seed=self.seed,
                eval=False
            )

        return eval_data_dict, test_data_dict
