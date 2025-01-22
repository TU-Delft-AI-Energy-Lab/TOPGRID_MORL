import json
import os
from datetime import datetime
from typing import Any, Dict, List
import re
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


def load_action_space(env_name: str, g2op_env, actions_file) -> DiscreteActSpace:
    current_dir = os.getcwd()
    path = os.path.join(current_dir, "action_spaces", env_name, actions_file)
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
    g2op_env_val, rewards_list: List[str], obs_tennet: List[str], actions_file, env_name
) -> CustomGymEnv:
    gym_env = CustomGymEnv(g2op_env_val, safe_max_rho=0.95, eval=True)
    gym_env.set_rewards(rewards_list=rewards_list)
    gym_env.observation_space = BoxGymObsSpace(
        g2op_env_val.observation_space, attr_to_keep=obs_tennet
    )
    
    # Action space setup
    current_dir = os.getcwd()
    path = os.path.join(current_dir, "action_spaces", env_name, actions_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Action file not found: {path}")
    
    with open(path, "rt", encoding="utf-8") as action_set_file:
        all_actions = list(
            (
                g2op_env_val.action_space(action_dict)
                for action_dict in json.load(action_set_file)
            )
        )

    # Add do nothing action
    do_nothing_action = g2op_env_val.action_space({})
    all_actions.insert(0, do_nothing_action)

    gym_env.action_space = DiscreteActSpace(
        g2op_env_val.action_space, action_list=all_actions
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
    final=True
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
        # Extract chronic number from eval_data's eval_chronic using regex
        chronic_match = re.search(r'\\chronics\\(\d+)', eval_data.get('eval_chronic', ''))
        chronic_string = chronic_match.group(1) if chronic_match else "unknown"
        filename = f"eval_chronic{chronic_string}_counter_{eval_counter}_{idx}.json"
        filepath = os.path.join(dir_path, filename)

        eval_data_serializable = {
            "eval_chronic": eval_data["eval_chronic"],
            "eval_rewards": [reward.tolist() for reward in eval_data["eval_rewards"]],
            "eval_actions": eval_data["eval_actions"],
            "eval_sub_ids": eval_data['sub_ids'], 
            'eval_action_timesamps': eval_data['eval_action_timestamp'], 
            "eval_topo_distance": eval_data['eval_topo_distance'],
            "eval_rho": eval_data["eval_rho"],
            "eval_topo_vect": eval_data['eval_topo_vect'], 
            "eval_steps": eval_data["eval_steps"],
        }

        with open(filepath, "w") as json_file:
            json.dump(eval_data_serializable, json_file, indent=4)
    if final: 
        dir_path = os.path.join(
        "final_logs",
        env_name,
        f"{current_date}",
        f"{rewards_list}",
        f"weights_{weights_str}",
        f"seed_{seed}",
        )
        os.makedirs(dir_path, exist_ok=True)
        # Extract chronic number from eval_data's eval_chronic using regex
        chronic_match = re.search(r'\\chronics\\(\d+)', eval_data.get('eval_chronic', ''))
        chronic_string = chronic_match.group(1) if chronic_match else "unknown"
        filename = f"eval_chronic{chronic_string}_counter_{eval_counter}_{idx}.json"
        filepath = os.path.join(dir_path, filename)

        eval_data_serializable = {
            "eval_chronic": eval_data["eval_chronic"],
            "eval_rewards": [reward.tolist() for reward in eval_data["eval_rewards"]],
            "eval_actions": eval_data["eval_actions"],
            "eval_sub_ids": eval_data['sub_ids'], 
            'eval_action_timesamps': eval_data['eval_action_timestamp'], 
            "eval_topo_distance": eval_data['eval_topo_distance'],
            "eval_rho": eval_data["eval_rho"],
            "eval_topo_vect": eval_data['eval_topo_vect'], 
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
        chronic_match = re.search(r'\\chronics\\(\d+)', eval_data.get('eval_chronic', ''))
        chronic_string = chronic_match.group(1) if chronic_match else "unknown"
        filename = f"test_chronic{chronic_string}_counter_{eval_counter}_{idx}.json"
        filepath = os.path.join(dir_path, filename)

        eval_data_serializable = {
            "test_chronic": eval_data["eval_chronic"],
            "test_rewards": [reward.tolist() for reward in eval_data["eval_rewards"]],
            "test_actions": eval_data["eval_actions"],
            "test_sub_ids": eval_data['sub_ids'], 
            "test_action_timestamps": eval_data["eval_action_timestamp"], 
            "test_topo_distance": eval_data['eval_topo_distance'], 
            "test_rho": eval_data["eval_rho"],
            "test_topo_vect": eval_data['eval_topo_vect'], 
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
    eval=True,
    final=False
) -> Dict[str, Any]:
    g2op_env_val.set_id(chronic)
    rewards_list = reward_list
    obs_tennet = [
        "topo_vect",
        "rho",
        "gen_p",
        "load_p",
        "p_or",
        "p_ex",
        "timestep_overflow",
    ]
    env_name = "rte_case5_example"
    if env_name == "rte_case5_example":
        results_dir = "training_results_5bus_4094"
        action_dim = 53
        actions_file = "filtered_actions.json"
    elif env_name == "l2rpn_case14_sandbox":
        results_dir = "training_results_14bus_4096"
        action_dim = 73
        actions_file = "tennet_actions.json"
        
    gym_env = setup_gym_env(g2op_env_val, rewards_list, obs_tennet, actions_file=actions_file, env_name=env_name)

    gym_env.action_space = load_action_space(env_name, g2op_env, actions_file)
    
    eval_rewards, eval_actions, eval_rho, eval_topo_vect, eval_action_timestamp, sub_ids, eval_topo_distance = [], [], [], [], [], [], []
    eval_done = False
    eval_state = gym_env.reset(options={"max step": eval_steps})
    total_eval_steps = 0
    cum_eval_steps = 0 
    discount_factor = 0.995
    while not eval_done and total_eval_steps<= eval_steps:
        eval_action = agent.eval(eval_state, agent.weights.cpu().numpy())
        eval_state, eval_reward, eval_done, eval_info, eval_g2op_obs, terminated_gym = gym_env.step(eval_action)
        cum_eval_steps += eval_info['steps']
        eval_action_timestamp.append(cum_eval_steps)
        total_eval_steps += eval_info["steps"]
        #print(eval_g2op_obs.rho)
        eval_reward = (
            th.tensor(eval_reward).to(agent.device).view(agent.networks.reward_dim)
        )
        eval_rewards.append(eval_reward)
        eval_actions.append(
            eval_action.tolist()
            if isinstance(eval_action, (list, np.ndarray))
            else eval_action
        )
        g2op_act = gym_env.action_space.from_gym(eval_action)
        sub_ids.append(g2op_act.as_dict().get('set_bus_vect', {}).get('modif_subs_id', [None]))
        eval_rho.append(eval_g2op_obs.rho.tolist())
        eval_topo_vect.append(eval_g2op_obs.topo_vect.tolist())
        topo_dist = 0
        idx=0
        #print(chronic)
        #print(eval_g2op_obs.topo_vect)
        for n_elems_on_sub in eval_g2op_obs.sub_info:
            #print(n_elems_on_sub)
            # Find this substation elements range in topology vect
            sub_start = idx
            #print(sub_start)
            sub_end = idx + n_elems_on_sub
            current_sub_topo = eval_g2op_obs.topo_vect[sub_start:sub_end]
            #print(current_sub_topo)
            # Count number of elements not on bus 1
            # Because at the initial state, all elements are on bus 1
            if np.any(current_sub_topo == 2):
                topo_dist += 1
            #print(topo_dist)
            idx += n_elems_on_sub
            #print(topo_dist)
            
        eval_topo_distance.append(topo_dist)
        

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
        "sub_ids": sub_ids, 
        "eval_action_timestamp": eval_action_timestamp,
        "eval_topo_distance": eval_topo_distance,
        "eval_rho": eval_rho,  # Convert NumPy array to list
        "eval_topo_vect": eval_topo_vect,  
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
        eval=eval, 
        final=final
    )

    return eval_data


