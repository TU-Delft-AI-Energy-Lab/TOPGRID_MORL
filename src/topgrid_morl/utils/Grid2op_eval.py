import json
import os
import torch as th
import numpy as np
from typing import List, Dict, Any, Tuple

def format_weights(weights: np.ndarray) -> str:
    weights_np = weights.cpu().numpy()
    if weights_np.dtype.kind in 'iu':  # Integer or unsigned integer
        return "_".join(map(str, weights_np.astype(int)))
    else:
        return "_".join(map(lambda x: f"{x:.2f}", weights_np))

def evaluate_agent(agent, weights, env, eval_steps: int, eval_counter: int) -> None:
    """
    Evaluate the agent on a single chronic and save the results.

    Args:
        agent (MOPPO): The agent to evaluate.
        weights (npt.NDArray[np.float64]): Weights for the objectives.
        env (CustomGymEnv): The evaluation environment.
        eval_steps (int): The maximum number of steps for evaluation.
        eval_counter (int): The current evaluation counter.
    """
    eval_rewards = []
    eval_actions = []
    eval_states = []
    
    eval_done = False
    eval_state = env.reset(options={"max step": eval_steps})
    eval_steps = 0 
    while not eval_done: 
        eval_action = agent.eval(eval_state, agent.weights.cpu().numpy())
        eval_state, eval_reward, eval_done, eval_info = env.step(eval_action)
        eval_steps += eval_info['steps']
        
        eval_reward = th.tensor(eval_reward).to(agent.device).view(agent.networks.reward_dim)
        
        eval_rewards.append(eval_reward.cpu().numpy().tolist())
        eval_actions.append(eval_action.tolist() if isinstance(eval_action, (list, np.ndarray)) else eval_action)
        eval_states.append(eval_state.tolist() if isinstance(eval_state, (list, np.ndarray)) else eval_state)
    
    eval_data = {
        "eval_rewards": eval_rewards,
        "eval_actions": eval_actions,
        "eval_states": eval_states,
        "eval_steps": eval_steps
    }
    
    # Create directories for environment name and weights
    env_name = env.init_env.name if hasattr(env.init_env, 'name') else 'default_env'
    weights_str = format_weights(weights)
    
    dir_path = os.path.join("eval_logs", env_name, f"weights_{weights_str}")
    os.makedirs(dir_path, exist_ok=True)
    
    # Generate a unique filename using the counter
    filename = f"eval_data_weights_{weights_str}_counter_{eval_counter}.json"
    
    # Save the evaluation data to a JSON file
    filepath = os.path.join(dir_path, filename)
    with open(filepath, "w") as json_file:
        json.dump(eval_data, json_file, indent=4)
    
  