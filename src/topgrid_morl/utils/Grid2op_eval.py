import json
import os
import torch as th
import numpy as np
from typing import List, Dict, Any, Tuple
import grid2op
from grid2op.Action import BaseAction
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace, GymEnv
from grid2op.Reward import EpisodeDurationReward, LinesCapacityReward
from gymnasium.spaces import Discrete
from lightsim2grid import LightSimBackend
import wandb
from topgrid_morl.envs.CustomGymEnv import CustomGymEnv
from topgrid_morl.envs.GridRewards import TopoActionReward, ScaledTopoActionReward, ScaledLinesCapacityReward, ScaledEpisodeDurationReward


def format_weights(weights: np.ndarray) -> str:
    weights_np = weights.cpu().numpy()
    if weights_np.dtype.kind in 'iu':  # Integer or unsigned integer
        return "_".join(map(str, weights_np.astype(int)))
    else:
        return "_".join(map(lambda x: f"{x:.2f}", weights_np))

def evaluate_agent(agent, weights, env, g2op_env, g2op_env_val, eval_steps: int, eval_counter: int, global_step, reward_dim, chronic, idx) -> Dict[str, Any]:
    #wandb.init(project="your_project_name", name=f"eval_{eval_counter}")
    #wandb.define_metric("reward_sum", step_metric="global_step")

    # Set the chronic to the environment
    g2op_env.set_id(chronic)
    gym_env = CustomGymEnv(g2op_env)

    # Set rewards in Gym Environment
    rewards_list = ["ScaledLinesCapacity", "ScaledTopoAction"]
    gym_env.set_rewards(rewards_list=rewards_list)

    # Modify observation space
    obs_tennet = [
        "rho",
        "gen_p",
        "load_p",
        "topo_vect",
        "p_or",
        "p_ex",
        "timestep_overflow",
    ]
    gym_env.observation_space = BoxGymObsSpace(
        g2op_env.observation_space, attr_to_keep=obs_tennet
    )
    env_name = "rte_case5_example"
    # Action space setup
    current_dir = os.getcwd()
    path = os.path.join(current_dir, "action_spaces", env_name, "filtered_actions.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Action file not found: {path}")

    with open(path, "rt", encoding="utf-8") as action_set_file:
        all_actions = list(
            (
                g2op_env.action_space(action_dict)
                for action_dict in json.load(action_set_file)
            )
        )

    # add do nothing action
    do_nothing_action = g2op_env.action_space({})
    all_actions.insert(0, do_nothing_action)

    gym_env.action_space = DiscreteActSpace(
        g2op_env.action_space, action_list=all_actions
    )

    # Print the chronic being processed
    print(f"Processing chronic {idx}: {chronic}")
    eval_chronic = chronic
    eval_rewards = []
    eval_actions = []
    eval_states = []
    eval_done = False
    eval_state = gym_env.reset(options={"max step": eval_steps})
    total_eval_steps = 0
    while not eval_done:
        eval_action = agent.eval(eval_state, agent.weights.cpu().numpy())
        eval_state, eval_reward, eval_done, eval_info = gym_env.step(eval_action)
        total_eval_steps += eval_info['steps']

        eval_reward = th.tensor(eval_reward).to(agent.device).view(agent.networks.reward_dim)

        eval_rewards.append(eval_reward.cpu())  # Keep as tensor
        eval_actions.append(eval_action.tolist() if isinstance(eval_action, (list, np.ndarray)) else eval_action)
        eval_states.append(eval_state.tolist() if isinstance(eval_state, (list, np.ndarray)) else eval_state)

        # Log to wandb with custom step
        #log_data = {"eval_steps": total_eval_steps}
        #for i, reward in enumerate(eval_rewards[-1]):
        #    log_data[f"reward_{i}"] = reward.item()
#
        #wandb.log(log_data, step=global_step)
        #global_step += 1

    eval_data = {
        "eval_chronic": eval_chronic,
        "eval_rewards": eval_rewards,  # Keep as tensor for computation
        "eval_actions": eval_actions,
        "eval_states": eval_states,
        "eval_steps": total_eval_steps
    }

    # Create directories for environment name and weights
    env_name = gym_env.init_env.name if hasattr(env.init_env, 'name') else 'default_env'
    weights_str = format_weights(weights)

    dir_path = os.path.join("eval_logs", env_name, f"weights_{weights_str}")
    os.makedirs(dir_path, exist_ok=True)

    # Generate a unique filename using the counter
    filename = f"eval_data_weights_{weights_str}_counter_{eval_counter}_{idx}.json"

    # Convert tensors to lists before saving to JSON
    eval_data_serializable = {
        "eval_chronic": eval_data["eval_chronic"],
        "eval_rewards": [reward.tolist() for reward in eval_data["eval_rewards"]],
        "eval_actions": eval_data["eval_actions"],
        "eval_states": eval_data["eval_states"],
        "eval_steps": eval_data["eval_steps"]
    }

    # Save the evaluation data to a JSON file
    filepath = os.path.join(dir_path, filename)
    with open(filepath, "w") as json_file:
        json.dump(eval_data_serializable, json_file, indent=4)

    #wandb.finish()
    return eval_data
