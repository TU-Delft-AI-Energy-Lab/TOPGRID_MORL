import itertools

import matplotlib.pyplot as plt
import numpy as np

from topgrid_morl.utils.MO_PPO_train_utils import load_saved_data
from topgrid_morl.utils.MORL_analysis_utils import (
    generate_variable_name,
    normalize_reward_matrix,
    plot_3d_mean_std,
    plot_mean_std_rewards,
    plot_mean_std_total_steps,
)


def main() -> None:
    """
    Main function to load saved data, normalize rewards, and plot the results.
    """
    # Set Parameters
    num_episodes = 5
    num_episodes_list = [num_episodes]
    weights_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    results_dir = "training_results_5bus"
    num_seeds = 10
    seeds = np.arange(0, num_seeds)
    reward_dim = 3
    # Load Data
    loaded_data = {}
    do_nothing = True

    # Loop through all combinations of num_episodes, weights, and seeds
    for num_episodes, weights, seed in itertools.product(
        num_episodes_list, weights_list, seeds
    ):
        # Load the reward matrix and actions
        (
            reward_matrix,
            actions,
            total_steps,
            params,
            donothing_reward_matrix,
            donothing_total_steps,
        ) = load_saved_data(
            weights=weights,
            results_dir=results_dir,
            seed=seed,
            num_episodes=num_episodes,
        )

        # Generate variable names based on the specifications
        reward_var_name = generate_variable_name(
            "reward_matrix", num_episodes, weights, seed=seed
        )
        actions_var_name = generate_variable_name(
            "actions", num_episodes, weights, seed=seed
        )
        total_steps_var_name = generate_variable_name(
            "total_steps", num_episodes, weights, seed=seed
        )

        # Store the data in the dictionary
        loaded_data[reward_var_name] = reward_matrix
        loaded_data[actions_var_name] = actions
        loaded_data[total_steps_var_name] = total_steps

        # Optionally store DoNothing data
        if do_nothing:
            donothing_reward_var_name = generate_variable_name(
                "donothing_reward_matrix", num_episodes, weights, seed=seed
            )
            donothing_total_steps_var_name = generate_variable_name(
                "donothing_total_steps", num_episodes, weights, seed=seed
            )
            loaded_data[donothing_reward_var_name] = donothing_reward_matrix
            loaded_data[donothing_total_steps_var_name] = donothing_total_steps

    # Check keys in loaded_data
    # print("Loaded data keys:", loaded_data.keys())

    # Extract specific reward matrices and total steps for each weight setting and DoNothing
    reward_matrix_1_0_0 = []
    reward_matrix_0_1_0 = []
    reward_matrix_0_0_1 = []
    reward_matrix_do_nothing = []
    total_steps_1_0_0 = []
    total_steps_0_1_0 = []
    total_steps_0_0_1 = []
    total_steps_do_nothing = []

    for seed in seeds:
        reward_matrix_1_0_0.append(
            loaded_data[
                f"reward_matrix_episodes_{num_episodes}_weights_1_0_0_seed_{seed}"
            ]
        )
        reward_matrix_0_1_0.append(
            loaded_data[
                f"reward_matrix_episodes_{num_episodes}_weights_0_1_0_seed_{seed}"
            ]
        )
        reward_matrix_0_0_1.append(
            loaded_data[
                f"reward_matrix_episodes_{num_episodes}_weights_0_0_1_seed_{seed}"
            ]
        )
        reward_matrix_do_nothing.append(
            loaded_data[
                f"donothing_reward_matrix_episodes_{num_episodes}_weights_1_0_0_seed_{seed}"
            ]
        )

        total_steps_1_0_0.append(
            loaded_data[
                f"total_steps_episodes_{num_episodes}_weights_1_0_0_seed_{seed}"
            ]
        )
        total_steps_0_1_0.append(
            loaded_data[
                f"total_steps_episodes_{num_episodes}_weights_0_1_0_seed_{seed}"
            ]
        )
        total_steps_0_0_1.append(
            loaded_data[
                f"total_steps_episodes_{num_episodes}_weights_0_0_1_seed_{seed}"
            ]
        )
        total_steps_do_nothing.append(
            loaded_data[
                f"donothing_total_steps_episodes_{num_episodes}_weights_1_0_0_seed_{seed}"
            ]
        )

    # Create dictionaries for plotting functions
    rewards_dict = {
        "weights_1_0_0": reward_matrix_1_0_0,
        "weights_0_1_0": reward_matrix_0_1_0,
        "weights_0_0_1": reward_matrix_0_0_1,
        "do_nothing": reward_matrix_do_nothing,
    }

    normalized_rewards_dict = {
        "weights_1_0_0": normalize_reward_matrix(
            reward_matrix_1_0_0, total_steps_1_0_0, num_seeds=num_seeds
        ),
        "weights_0_1_0": normalize_reward_matrix(
            reward_matrix_0_1_0, total_steps_0_1_0, num_seeds=num_seeds
        ),
        "weights_0_0_1": normalize_reward_matrix(
            reward_matrix_0_0_1, total_steps_0_0_1, num_seeds=num_seeds
        ),
    }

    total_steps_dict = {
        "weights_1_0_0": total_steps_1_0_0,
        "weights_0_1_0": total_steps_0_1_0,
        "weights_0_0_1": total_steps_0_0_1,
    }
    # Plot total steps with mean and std deviation
    plot_mean_std_total_steps(
        total_steps_dict=total_steps_dict, num_episodes=num_episodes
    )

    # Plot 3D scatter plot with mean and std deviation for rewards
    plot_3d_mean_std(returns_dict=rewards_dict)

    # Plot mean and std deviation for normalized rewards
    plot_mean_std_rewards(returns_dict=normalized_rewards_dict, reward_dim=reward_dim)

    plt.show()


if __name__ == "__main__":
    main()
