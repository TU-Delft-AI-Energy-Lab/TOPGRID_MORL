import json
import torch as th
import numpy as np
from typing import List, Dict, Any, Tuple

def evaluate_agent(agent, env, eval_steps: int, eval_counter: int) -> None:
    """
    Evaluate the agent on a single chronic and save the results.

    Args:
        agent (MOPPO): The agent to evaluate.
        env (CustomGymEnv): The evaluation environment.
        eval_steps (int): The maximum number of steps for evaluation.
        eval_counter (int): The current evaluation counter.
    """
    eval_rewards = []
    eval_actions = []
    eval_states = []
    
    eval_done = False
    eval_state = env.reset(options={"max step": eval_steps})
    
    while not eval_done: 
        eval_action = agent.eval(eval_state, agent.weights.cpu().numpy())
        eval_state, eval_reward, eval_done, eval_info = env.step(eval_action)
        
        eval_reward = th.tensor(eval_reward).to(agent.device).view(agent.networks.reward_dim)
        
        eval_rewards.append(eval_reward.cpu().numpy().tolist())
        eval_actions.append(eval_action.tolist() if isinstance(eval_action, (list, np.ndarray)) else eval_action)
        eval_states.append(eval_state.tolist() if isinstance(eval_state, (list, np.ndarray)) else eval_state)
    
    eval_data = {
        "eval_rewards": eval_rewards,
        "eval_actions": eval_actions,
        "eval_states": eval_states
    }
    
    # Generate a unique filename using the counter
    filename = f"eval_data_{eval_counter}.json"
    
    # Save the evaluation data to a JSON file
    with open(filename, "w") as json_file:
        json.dump(eval_data, json_file, indent=4)
    
