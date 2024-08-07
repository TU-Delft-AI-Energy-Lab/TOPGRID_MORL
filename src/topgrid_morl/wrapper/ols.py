def train_agent(
    weight_vectors: List[npt.NDArray[np.float64]],
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

    for weights in weight_vectors:
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
        """
        run = wandb.init(
            project="TOPGrid_MORL_5bus",
            name=generate_variable_name(
                base_name=run_name,
                max_gym_steps=max_gym_steps,
                weights=weights,
                seed=seed,
            )+'DoNothing',
            group=weights_str
            
        )
        do_nothing_agent = DoNothingAgent(env=env, env_val=env_val, log=agent_params["log"], device=agent_params["device"])
        do_nothing_agent.train(max_gym_steps=max_gym_steps, reward_dim=reward_dim)
        run.finish()
        """