import numpy as np
import os
import torch as th
import wandb
from typing import Any, List, Tuple
from tqdm import tqdm
from topgrid_morl.utils.MO_PPO_train_utils import initialize_agent
from topgrid_morl.utils.Grid2op_eval import evaluate_agent
import logging
logging.basicConfig(level=logging.INFO)

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
                 reuse: str = "none",  # "none", "partial", "full"
                 project_name: str = "TOPGrid_MORL_14bus",
                 net_arch: List[int] = [64, 64],
                 generate_reward: bool = False,
                 reward_list: List[str] = ["ScaledEpisodeDuration", "ScaledTopoAction"],
                 network_params: dict = {}, 
                 env_params: dict = {}, 
                 logging_params: dict = {},
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
        self.env_params = env_params
        self.network_params = network_params
        self.logging_params = logging_params
        self.models = {}  # Dictionary to store trained models for reuse
        self.np_random = np.random.RandomState(seed)
        self.weight_range = [0, 1]  # Default range for weights
        self.num_samples = len(self.agent_params.get('weight_vectors', [[1, 0, 0]]))  # Default number of weight vectors
        self.iterations = 0
        self.mc_samples = 0 

    def run(self):
        weight_vectors = self.agent_params.get('weight_vectors', [self.sample_weights() for _ in range(self.num_samples)])

        # Loop over the weight vectors to train the agent for each set of weights
        for i, weights in tqdm(enumerate(weight_vectors), total=self.num_samples):
            print(f"Running MOPPO for weight vector {i + 1}/{self.num_samples}")
            eval_data, test_data, model = self.run_single(weights)

            # Log the rewards to wandb
            if self.agent_params['log']:
                wandb.log({"mean_rewards": sum_rewards(eval_data)})

    def sample_weights(self):
        # Sample weights and normalize to sum to 1
        """Samples random weight vectors from a uniform distribution."""
        self.mc_samples+=1
        
        if self.mc_samples==1: #non generic -> need to adjust to extrema weights of reward dim
            return([1,0,0])
        elif self.mc_samples ==2: 
            return( [0,1,0])
        elif self.mc_samples ==3: 
            return([0,0,1])
        else: 
            weights = np.random.rand(self.reward_dim)
            normalized_weights = weights / np.sum(weights)
            return np.round(normalized_weights, 2)

    def initialize_or_reuse_model(self, weights):
        """
        Initialize or reuse a model based on the reuse strategy specified.
        """
        print('initializing model')
        eps = 1e-6  # Epsilon to handle rounding errors
        weight_key = tuple(weights)

        if self.reuse == "none" or not self.models:
            # No reuse or no existing model for the given weights
            model = initialize_agent(
                self.env, self.env_val, self.g2op_env, self.g2op_env_val, 
                weights, self.obs_dim, self.action_dim, self.reward_dim, 
                self.net_arch, self.seed, self.generate_reward, **self.agent_params
            )
            print('no reuse')
        else:
            # Find the model with the closest weight vector, excluding extrema weight vectors
            valid_keys = [k for k in self.models.keys() if np.linalg.norm(np.array(k) - np.array([1, 0, 0])) > eps and np.linalg.norm(np.array(k) - np.array([0, 1, 0])) > eps and np.linalg.norm(np.array(k) - np.array([0, 0, 1])) > eps]
            if not valid_keys:
                # If no valid models exist, initialize a new agent
                model = initialize_agent(
                    self.env, self.env_val, self.g2op_env, self.g2op_env_val, 
                    weights, self.obs_dim, self.action_dim, self.reward_dim, 
                    self.net_arch, self.seed, self.generate_reward, **self.agent_params
                )
                print('no valid weight vectors to copy model')
            else:
                closest_weight_key = min(
                    valid_keys,
                    key=lambda k: np.linalg.norm(np.array(k) - np.array(weights))
                )
                closest_model = self.models[closest_weight_key]
                model = initialize_agent(
                    self.env, self.env_val, self.g2op_env, self.g2op_env_val, 
                    weights, self.obs_dim, self.action_dim, self.reward_dim, 
                    self.net_arch, self.seed, self.generate_reward, **self.agent_params
                )
                # Copy the parameters from the closest model's network
                model.networks.load_state_dict(closest_model.networks.state_dict())
                print(f"For {weights} copied model parameters of {closest_weight_key}")
                
                if self.reuse == "partial":
                    model.networks.reinitialize_last_layer()  # Reinitialize only the last layer if partial reuse
                    print('last layer initialized randomly')

        return model


    def run_single(self, weights: np.array):
        """
        Run training for a single set of weights.
        """
        rounded_weights = np.round(weights, 2)
        
        # Initialize or reuse a model for the current weight vector
        model = self.initialize_or_reuse_model(rounded_weights)

        # Set the model weights
        model.weights = th.tensor(rounded_weights).cpu().to(model.device)

        if self.agent_params['log']:
            weights_str = "_".join(map(str, rounded_weights))
            runname = f'OLS_lr{self.agent_params['learning_rate']}_vf{self.agent_params['vf_coef']}_ent{self.agent_params['ent_coef']}_w{weights_str}_s{self.seed}'
            # Initialize wandb run
            run = wandb.init(
                project=self.project_name,
                name=runname,
                group=f"{self.reward_list[0]}_{self.reward_list[1]}",
                tags=[self.run_name],
                config=self.agent_params
            )
            print(self.agent_params)
            # Log agent params and weights to wandb config
            wandb.config.update({"rounded_weights": rounded_weights.tolist()})
            wandb.config.update(self.network_params)
            wandb.config.update(self.env_params)
            wandb.config.update(self.logging_params)
            
            

        # Train the agent
        model.train(max_gym_steps=self.max_gym_steps, reward_dim=self.reward_dim, reward_list=self.reward_list)

        if self.agent_params['log']:
            run.finish()  # Finish the wandb run

        # Save the trained model for future reuse if applicable
        weight_key = tuple(rounded_weights)
        self.models[weight_key] = model
        print(f'saved models: {self.models}')

        eval_data_dict, test_data_dict = self.evaluate_model(model, rounded_weights)
        
        self.iterations +=1
        #print(test_data_dict)
        return eval_data_dict, test_data_dict, model

    def evaluate_model(self, agent, weights):
        """
        Evaluate the agent using the validation and test environments.
        """
        eval_data_dict = {}
        test_data_dict = {}

        # Convert weights to tensor and move to the correct device
        weights = th.tensor(weights).cpu().to(agent.device)

        # Validation evaluation
        chronics = self.g2op_env_val.chronics_handler.available_chronics()
        for idx, chronic in enumerate(chronics):
            key = f'eval_data_{idx}'
            eval_data_dict[key] = evaluate_agent(
                agent=agent,
                env=self.env_val,
                g2op_env=self.g2op_env_val,
                g2op_env_val=self.g2op_env_val,
                weights=weights,
                eval_steps=28 * 288,  # Custom step count for evaluations
                chronic=chronic,
                idx=idx,
                reward_list=self.reward_list,
                seed=self.seed, 
                final=True
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
                eval_steps=28 * 288,
                chronic=chronic,
                idx=idx,
                reward_list=self.reward_list,
                seed=self.seed,
                eval=False
            )

        return eval_data_dict, test_data_dict
