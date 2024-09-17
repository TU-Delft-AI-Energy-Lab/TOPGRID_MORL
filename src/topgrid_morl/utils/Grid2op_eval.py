import json
import os
from datetime import datetime
from typing import Any, Dict, List

import grid2op
import numpy as np
import torch as th
import wandb
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace
from lightsim2grid import LightSimBackend

from topgrid_morl.envs.CustomGymEnv import CustomGymEnv
from topgrid_morl.envs.GridRewards import (
    ScaledLinesCapacityReward,
    ScaledTopoActionReward,
)


def format_weights(weights: np.ndarray) -> str:
    weights_np = weights.cpu().numpy()
    if weights_np.dtype.kind in "iu":  # Integer or unsigned integer
        return "_".join(map(str, weights_np.astype(int)))
    else:
        return "_".join(map(lambda x: f"{x:.2f}", weights_np))


def load_action_space(env_name: str, g2op_env) -> DiscreteActSpace:
    current_dir = os.getcwd()
    path = os.path.join(current_dir, "action_spaces", env_name, "filtered_actions.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Action file not found: {path}")

    with open(path, "rt", encoding="utf-8") as action_set_file:
        all_actions = [
            g2op_env.action_space(action_dict)
            for action_dict in json.load(action_set_file)
        ]

    do_nothing_action = g2op_env.action_space({})
    all_actions.insert(0, do_nothing_action)

    return DiscreteActSpace(g2op_env.action_space, action_list=all_actions)


def setup_gym_env(
    g2op_env_val, rewards_list: List[str], obs_tennet: List[str]
) -> CustomGymEnv:
    gym_env = CustomGymEnv(g2op_env_val, safe_max_rho=0.95)
    gym_env.set_rewards(rewards_list=rewards_list)
    gym_env.observation_space = BoxGymObsSpace(
        g2op_env_val.observation_space, attr_to_keep=obs_tennet
    )
    return gym_env


def log_evaluation_data(
    env_name: str,
    weights_str: str,
    eval_counter: int,
    idx: int,
    eval_data: Dict[str, Any],
    rewards_list,
    eval:bool = True,
    seed=42,
) -> None:
    current_date = datetime.now().strftime("%Y-%m-%d")
    if eval: 
        dir_path = os.path.join(
            "eval_logs",
            env_name,
            f"{current_date}",
            f"{rewards_list}",
            f"weights_{weights_str}",
            f"seed_{seed}",
        )
        os.makedirs(dir_path, exist_ok=True)

        filename = f"eval_data_weights_{weights_str}_counter_{eval_counter}_{idx}.json"
        filepath = os.path.join(dir_path, filename)

        eval_data_serializable = {
            "eval_chronic": eval_data["eval_chronic"],
            "eval_rewards": [reward.tolist() for reward in eval_data["eval_rewards"]],
            "eval_actions": eval_data["eval_actions"],
            "eval_states": eval_data["eval_states"],
            "eval_steps": eval_data["eval_steps"],
        }

        with open(filepath, "w") as json_file:
            json.dump(eval_data_serializable, json_file, indent=4)
            
    else: 
        dir_path = os.path.join(
            "test_logs",
            env_name,
            f"{current_date}",
            f"{rewards_list}",
            f"weights_{weights_str}",
            f"seed_{seed}",
        )
        os.makedirs(dir_path, exist_ok=True)

        filename = f"test_data_weights_{weights_str}_{idx}.json"
        filepath = os.path.join(dir_path, filename)

        eval_data_serializable = {
            "test_chronic": eval_data["eval_chronic"],
            "test_rewards": [reward.tolist() for reward in eval_data["eval_rewards"]],
            "test_actions": eval_data["eval_actions"],
            "test_states": eval_data["eval_states"],
            "test_steps": eval_data["eval_steps"],
        }

        with open(filepath, "w") as json_file:
            json.dump(eval_data_serializable, json_file, indent=4)


def evaluate_agent(
    agent,
    weights,
    env,
    g2op_env,
    g2op_env_val,
    eval_steps: int,
    chronic,
    idx,
    reward_list,
    seed,
    eval_counter: int = 1,
    eval=True
) -> Dict[str, Any]:
    g2op_env_val.set_id(chronic)
    rewards_list = reward_list
    obs_tennet = [
        "rho",
        "gen_p",
        "load_p",
        "topo_vect",
        "p_or",
        "p_ex",
        "timestep_overflow",
    ]
    gym_env = setup_gym_env(g2op_env_val, rewards_list, obs_tennet)

    env_name = "l2rpn_case14_sandbox"
    
    if env_name == "rte_case5_example":
        results_dir = "training_results_5bus_4094"
        action_dim = 53
        actions_file = "filtered_actions.json"
    elif env_name == "l2rpn_case14_sandbox":
        results_dir = "training_results_14bus_4096"
        action_dim = 134
        actions_file = "medha_actions.json"
        
    gym_env.action_space = load_action_space(env_name, g2op_env)
    
    eval_rewards, eval_actions, eval_states = [], [], []
    eval_done = False
    eval_state = gym_env.reset(options={"max step": eval_steps})
    total_eval_steps = 0
    discount_factor = 0.995

    while not eval_done:
        eval_action = agent.eval(eval_state, agent.weights.cpu().numpy())
        eval_state, eval_reward, eval_done, eval_info = gym_env.step(eval_action)
        total_eval_steps += eval_info["steps"]

        eval_reward = (
            th.tensor(eval_reward).to(agent.device).view(agent.networks.reward_dim)
        )
        eval_rewards.append(eval_reward)
        eval_actions.append(
            eval_action.tolist()
            if isinstance(eval_action, (list, np.ndarray))
            else eval_action
        )
        eval_states.append(
            eval_state.tolist()
            if isinstance(eval_state, (list, np.ndarray))
            else eval_state
        )

    # Calculate the discounted reward
    discounted_rewards = []
    running_add = th.zeros_like(eval_rewards[0])
    for reward in reversed(eval_rewards):
        running_add = reward + discount_factor * running_add
        discounted_rewards.insert(0, running_add)

    eval_data = {
        "eval_chronic": chronic,
        "eval_rewards": discounted_rewards,  # Storing PyTorch tensors
        "eval_actions": eval_actions,
        "eval_states": eval_states,
        "eval_steps": total_eval_steps,
    }
    
    env_name = (
        gym_env.init_env.name if hasattr(gym_env.init_env, "name") else "default_env"
    )
    weights_str = format_weights(weights)
    log_evaluation_data(
        env_name,
        weights_str,
        eval_counter,
        idx,
        eval_data,
        rewards_list=reward_list,
        seed=seed,
        eval=eval
    )

    return eval_data


