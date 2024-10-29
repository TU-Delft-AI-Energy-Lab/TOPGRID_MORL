import ast
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting
from scipy.spatial import ConvexHull
from itertools import cycle
from collections import OrderedDict


class ExperimentAnalysis:
    def __init__(self, name, scenario, base_json_path):
        self.scenario = scenario
        # Dictionary of parameter names and values  # List of reward names, e.g., ['TopoDepth', 'TopoActionHour']

        self.base_json_path = base_json_path
        self.seed_paths = []
        self.name = name
        self.mc_seed_path = None
        self.output_dir = None
        # Generate paths for saving CSVs and accessing JSONs
        self.generate_paths()

    def generate_paths(self):
        # Build the directory path based on the parameters
        self.output_dir = os.path.join(
            self.base_json_path, "OLS", self.scenario, self.name
        )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory: {self.output_dir}")
        # For OLS seeds (10 seeds)
        self.seed_paths = []
        for seed in range(20):
            seed_file = f"morl_logs_seed_{seed}.json"
            seed_path = os.path.join(self.output_dir, seed_file)
            self.seed_paths.append(seed_path)
        # For MC seed (1 seed)
        self.rs_seed_paths = []
        rs_seed_dir = os.path.join(self.base_json_path, "RS", self.scenario, self.name)
        for seed in range(5):
            rs_seed_file = f"morl_logs_seed_{seed}.json"
            rs_seed_path = os.path.join(rs_seed_dir, rs_seed_file)
            self.rs_seed_paths.append(rs_seed_path)
        mc_seed_file = "morl_logs_seed_0.json"
        mc_seed_dir = os.path.join(self.base_json_path, "RS", self.scenario, self.name)
        mc_ex_seed_dir = os.path.join(self.base_json_path, "RS_ex", self.scenario, self.name)
        self.mc_seed_path = os.path.join(mc_seed_dir, mc_seed_file)
        self.mc_ex_seed_path = os.path.join(mc_ex_seed_dir, mc_seed_file)
        
        self.iteration_paths = []
        for iteration in [5, 10, 20]:  # List the iteration counts directly
            iterations_file = f"{iteration}_iteration_morl_logs_seed_{seed}.json"
            iteration_path = os.path.join(self.output_dir, iterations_file)
            self.iteration_paths.append(iteration_path)
            
        self.mc_iteration_paths = []
        for iteration in [5, 10, 20]:  # List the iteration counts directly
            iterations_file = f"{iteration}_iteration_morl_logs_seed_{seed}.json"
            iteration_path = os.path.join(mc_seed_dir, iterations_file)
            self.mc_iteration_paths.append(iteration_path)
            
        print(self.seed_paths)
        print(self.iteration_paths)

    def load_data(self):
        # Load the data from the JSON files if needed
        pass

    def calculate_metrics(self, iterations=False):
        # Calculate multi-objective metrics like hypervolumes, max/min rewards, sparsities
        print("Calculating metrics...")
        if iterations:
            # Existing code for iterations
            df_all_metrics_ols = calculate_all_metrics(
                self.iteration_paths, "ols", rs_seed_paths=self.mc_iteration_paths, iterations=True
            )
            # Save the DataFrame to CSV
            df_all_metrics_ols.to_csv(
                os.path.join(self.output_dir, "ols_iterations_metrics.csv"), index=False
            )
        else:
            # Process both OLS and RS seeds
            df_all_metrics, df_mean_std = calculate_all_metrics(
                self.seed_paths, "ols", rs_seed_paths=self.rs_seed_paths, iterations=False
            )
            # Save the DataFrames to CSV
            df_all_metrics.to_csv(
                os.path.join(self.output_dir, "ols_rs_all_metrics.csv"), index=False
            )
            df_mean_std.to_csv(
                os.path.join(self.output_dir, "ols_rs_metrics_mean_std.csv"), index=False
            )
            # Store metrics
            self.df_all_metrics = df_all_metrics
            self.df_mean_std = df_mean_std
            print(self.df_all_metrics)
            print(self.df_mean_std)
        print("Metrics calculation completed.")

    def plot_pareto_frontiers(self, rewards, iterations):
        # Generate the 2D Pareto frontier plots
        print("Plotting Pareto frontiers...")
        # For OLS seeds
        if os.path.exists(self.mc_seed_path):
            plot_2d_projections_matplotlib(
                self.mc_seed_path, self.mc_seed_path, None, None, "mc", save_dir=self.output_dir, rewards=rewards
            )
        else:
            print(f"MC seed path not found: {self.mc_seed_path}")
        if iterations: 
              plot_2d_projections_matplotlib(
            self.seed_paths, self.mc_seed_path, self.iteration_paths, self.mc_iteration_paths,"ols", save_dir=self.output_dir, rewards=rewards, iterations=iterations, benchmark=False
        )
        else: 
            plot_2d_projections_matplotlib(
                self.seed_paths, self.mc_seed_path, None, None,"ols", save_dir=self.output_dir, rewards=rewards, mc_ex_path=self.mc_ex_seed_path
            )
        
        plot_super_pareto_frontier_2d(seed_paths = self.seed_paths, save_dir=None, rewards=["L2RPN", "TopoDepth", "TopoActionHour"])
        # For MC seed
        
        print("Pareto frontier plotting completed.")

    def in_depth_analysis(self, seed):
        # For the default run, perform in-depth analysis for the trajectory on the real-world test data
        print(f"Performing in-depth analysis for seed {seed}...")
        seed_path = self.seed_paths[seed]
        # Load the matching data
        process_data([seed_path], "ols", self.output_dir)
        csv_path = os.path.join(self.output_dir, "ccs_matching_data.csv")
        # Perform the topo depth process and plot
        topo_depth_process_and_plot(csv_path)
        # Also perform sub_id_process_and_plot if needed
        sub_id_process_and_plot(csv_path)
        print("In-depth analysis completed.")

    def analyse_pareto_values_and_plot(self):
        # For the default run, generate the 2D projection of the real-world data
        print("Analysing Pareto frontier values and plotting projections...")
        csv_path = os.path.join(self.output_dir, "ccs_matching_data.csv")
        if not os.path.exists(csv_path):
            # Need to process data to generate the CSV
            process_data(self.seed_paths, "ols", self.output_dir)
        # Now perform the analysis and plotting
        analyse_pf_values_and_plot_projections(csv_path)
        print("Analysis and plotting of Pareto projections completed.")

    """
    def compare_policies(self):
        # For the opponent, time, and max rho scenarios, determine if there are better RL policies
        print("Comparing policies...")
        csv_path = os.path.join(self.output_dir, 'ccs_matching_data.csv')
        if not os.path.exists(csv_path):
            # Process data to generate the CSV
            df_ccs_matching = process_data(self.seed_paths, 'ols', self.output_dir)
        else:
            df_ccs_matching = pd.read_csv(csv_path)
        # Compare policies based on 'test_steps' metric
        # Extract policies with weight vectors other than [1,0,0]
        df_non_extreme = df_ccs_matching[df_ccs_matching['Weights'].apply(lambda w: not is_extreme_weight(ast.literal_eval(w)))]
        # Extract the [1,0,0] policy
        df_extreme = df_ccs_matching[df_ccs_matching['Weights'].apply(lambda w: is_extreme_weight(ast.literal_eval(w)))]
        # Compare 'test_steps' between extreme and non-extreme policies
        for idx, row in df_non_extreme.iterrows():
            non_extreme_steps = row['test_chronic_0']['test_steps']
            # Find corresponding extreme policy in the same seed
            seed = row['seed']
            extreme_row = df_extreme[df_extreme['seed'] == seed]
            if not extreme_row.empty:
                extreme_steps = extreme_row.iloc[0]['test_chronic_0']['test_steps']
                if non_extreme_steps > extreme_steps:
                    print(f"Seed {seed}: Non-extreme policy {row['Weights']} performed better (Steps: {non_extreme_steps}) than extreme policy (Steps: {extreme_steps})")
                    # Additional comparison can be done here
        print("Policy comparison completed.")
    """


# Helper functions


def load_json_data(json_path):
    """Loads JSON data from a file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def extract_coordinates(ccs_list):
    """Extracts x, y, z coordinates from ccs_list."""
    x_all = [item[0] for item in ccs_list]
    y_all = [item[1] for item in ccs_list]
    z_all = [item[2] for item in ccs_list]
    return x_all, y_all, z_all


def is_pareto_efficient(costs):
    """Find the pareto-efficient points."""
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] <= c, axis=1)
            is_efficient[i] = True
    return is_efficient


def pareto_frontier_2d(x_values, y_values):
    """Computes the Pareto frontier for 2D points considering maximization."""
    points = np.column_stack((x_values, y_values))
    is_efficient = is_pareto_efficient(-points)  # Negative for maximization
    x_pareto = np.array(x_values)[is_efficient]
    y_pareto = np.array(y_values)[is_efficient]
    pareto_indices = np.where(is_efficient)[0]

    sorted_order = np.argsort(x_pareto)
    x_pareto = x_pareto[sorted_order]
    y_pareto = y_pareto[sorted_order]
    pareto_indices = pareto_indices[sorted_order]

    return x_pareto, y_pareto, pareto_indices


def find_matching_weights_and_agent(ccs_list, ccs_data):
    """Finds matching weights and agent information from CCS list and data."""
    matching_entries = []
    for ccs_entry in ccs_list:
        found_match = False
        for data_entry in ccs_data:
            ccs_entry_array = np.array(ccs_entry)
            returns_array = np.array(data_entry["returns"])
            if np.allclose(ccs_entry_array, returns_array, atol=1e-3):
                matching_entries.append(
                    {
                        "weights": data_entry["weights"],
                        "returns": ccs_entry,
                        "agent_file": data_entry.get("agent_file", None),
                        "test_chronic_0": data_entry.get("test_chronic_0", {}),
                        "test_chronic_1": data_entry.get("test_chronic_1", {}),
                        "eval_chronic_0": data_entry.get("eval_chronic_0", {}),
                        "eval_chronic_1": data_entry.get("eval_chronic_1", {}),
                        "seed": data_entry.get("seed", None),
                    }
                )
                found_match = True
                break  # Stop once a match is found
        if not found_match:
            print(f"No match found for CCS entry: {ccs_entry}")
    return matching_entries


def is_extreme_weight(weight, tol=1e-2):
    """
    Check if the weight vector is approximately an extreme weight vector.
    """
    weight = np.array(weight)
    indices = np.where(np.abs(weight - 1.0) < tol)[0]
    if len(indices) == 1:
        if np.all(np.abs(np.delete(weight, indices[0])) < tol):
            return True
    return False


def weight_label(weight):
    """
    Return a string label for the weight vector.
    """
    weight = np.array(weight)
    labels = [
        str(int(round(w))) if abs(w - round(w)) < 1e-2 else "{0:.2f}".format(w)
        for w in weight
    ]
    return "(" + ",".join(labels) + ")"


def calculate_hypervolume(points, reference_point):
    """Calculates the hypervolume for 2D points."""
    # Sort points by the first dimension
    sorted_points = sorted(points, key=lambda x: x[0])
    hv = 0.0
    for i in range(len(sorted_points)):
        width = abs(sorted_points[i][0] - reference_point[0])
        if i == 0:
            height = abs(sorted_points[i][1] - reference_point[1])
        else:
            height = abs(sorted_points[i][1] - sorted_points[i - 1][1])
        hv += width * height
    return hv


def calculate_sparsity(points):
    """Calculates the sparsity metric for 2D points."""
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distances.append(np.linalg.norm(np.array(points[i]) - np.array(points[j])))
    if distances:
        return np.mean(distances)
    else:
        return 0.0


def calculate_3d_hypervolume(points, reference_point):
    """Calculates the hypervolume for 3D points manually without using ConvexHull."""
    # Convert points and reference point to numpy arrays for easier manipulation
    points = np.array(points)
    reference_point = np.array(reference_point)

    # Sort points by the first dimension (x-axis)
    sorted_points = points[np.argsort(points[:, 0])]

    hv = 0.0
    for i in range(len(sorted_points)):
        # Calculate width (difference in x-axis between point and reference point)
        width = abs(sorted_points[i][0] - reference_point[0])

        if i == 0:
            # For the first point, use the reference point's height and depth
            height = abs(sorted_points[i][1] - reference_point[1])
            depth = abs(sorted_points[i][2] - reference_point[2])
        else:
            # For subsequent points, use the difference in height and depth between consecutive points
            height = abs(sorted_points[i][1] - sorted_points[i - 1][1])
            depth = abs(sorted_points[i][2] - sorted_points[i - 1][2])

        # Calculate the volume of the current cuboid and add to the total hypervolume
        hv += width * height * depth

    return hv

# Optional 2D Hypervolume Calculation (if applicable)
def calculate_2d_hypervolume(points_2d, reference_point_2d):
    """Calculates the 2D hypervolume (area) using a convex hull."""
    # Shift points to the reference point
    shifted_points_2d = points_2d - reference_point_2d

    # Ensure all points are in the positive orthant
    if np.any(shifted_points_2d < 0):
        print("Error: 2D points have negative coordinates after shifting to reference point.")
        return 0.0

    try:
        hull = ConvexHull(shifted_points_2d)
        area = hull.volume  # In 2D, the volume attribute of ConvexHull is the area
    except Exception as e:
        print(f"Error calculating 2D hypervolume: {e}")
        area = 0.0

    return area

def calculate_3d_sparsity(points):
    """Calculates the sparsity metric for 3D points."""
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distances.append(np.linalg.norm(np.array(points[i]) - np.array(points[j])))
    if distances:
        return np.mean(distances)
    else:
        return 0.0




def calculate_hypervolumes_and_sparsities(seed_paths, wrapper, mc_seed_path=None):
    """Calculates hypervolumes and sparsities for each seed and aggregates the results."""
    results = [] 
    all_x, all_y, all_z = [], [], []

    hv_xy_list, hv_xz_list, hv_yz_list = [], [], []
    hv_3d_list = []
    sparsity_3d_list = []
    min_return_x_list, max_return_x_list = [], []
    min_return_y_list, max_return_y_list = [], []
    min_return_z_list, max_return_z_list = [], []
    pareto_points_count_list = []

    for i, seed_path in enumerate(seed_paths):
        data = load_json_data(seed_path)
        ccs_list = data["ccs_list"][-1]
        x_all, y_all, z_all = extract_coordinates(ccs_list)

        all_x.extend(x_all)
        all_y.extend(y_all)
        all_z.extend(z_all)

        # Calculate min and max returns for each seed
        min_return_x = min(x_all)
        max_return_x = max(x_all)
        min_return_y = min(y_all)
        max_return_y = max(y_all)
        min_return_z = min(z_all)
        max_return_z = max(z_all)

        min_return_x_list.append(min_return_x)
        max_return_x_list.append(max_return_x)
        min_return_y_list.append(min_return_y)
        max_return_y_list.append(max_return_y)
        min_return_z_list.append(min_return_z)
        max_return_z_list.append(max_return_z)

        # Calculate 2D hypervolumes
        reference_point_xy = (min(x_all), min(y_all))
        reference_point_xz = (min(x_all), min(z_all))
        reference_point_yz = (min(y_all), min(z_all))

        x_pareto_xy, y_pareto_xy, pareto_xy_indices = pareto_frontier_2d(x_all, y_all)
        x_pareto_xz, z_pareto_xz, pareto_xz_indices = pareto_frontier_2d(x_all, z_all)
        y_pareto_yz, z_pareto_yz, pareto_yz_indices = pareto_frontier_2d(y_all, z_all)

        # Number of Pareto points
        pareto_points_count = len(pareto_xy_indices)
        pareto_points_count_list.append(pareto_points_count)

        # Calculate hypervolumes
        hv_xy = calculate_hypervolume(
            list(zip(x_pareto_xy, y_pareto_xy)), reference_point_xy
        )
        hv_xz = calculate_hypervolume(
            list(zip(x_pareto_xz, z_pareto_xz)), reference_point_xz
        )
        hv_yz = calculate_hypervolume(
            list(zip(y_pareto_yz, z_pareto_yz)), reference_point_yz
        )

        hv_xy_list.append(hv_xy)
        hv_xz_list.append(hv_xz)
        hv_yz_list.append(hv_yz)

        # Calculate 3D hypervolume and sparsity
        pareto_points_3d = np.column_stack((x_all, y_all, z_all))
        reference_point_3d = (min(x_all), min(y_all), min(z_all))
        hv_3d = calculate_3d_hypervolume(pareto_points_3d, reference_point_3d)
        sparsity_3d = calculate_3d_sparsity(pareto_points_3d)

        hv_3d_list.append(hv_3d)
        sparsity_3d_list.append(sparsity_3d)

        # Append results for each seed, rounding values
        results.append(
            {
                "Seed": f"Seed {i+1}",
                "Hypervolume XY": round(hv_xy, 2),
                "Hypervolume XZ": round(hv_xz, 2),
                "Hypervolume YZ": round(hv_yz, 2),
                "Hypervolume 3D": round(hv_3d, 2),
                "Sparsity 3D": round(sparsity_3d, 2),
                "Min Return X": round(min_return_x, 2),
                "Max Return X": round(max_return_x, 2),
                "Min Return Y": round(min_return_y, 2),
                "Max Return Y": round(max_return_y, 2),
                "Min Return Z": round(min_return_z, 2),
                "Max Return Z": round(max_return_z, 2),
                "Pareto Points Count": pareto_points_count,
            }
        )

    # Calculate mean and std dev for metrics
    mean_hv_xy, std_hv_xy = np.mean(hv_xy_list), np.std(hv_xy_list)
    mean_hv_xz, std_hv_xz = np.mean(hv_xz_list), np.std(hv_xz_list)
    mean_hv_yz, std_hv_yz = np.mean(hv_yz_list), np.std(hv_yz_list)
    mean_hv_3d, std_hv_3d = np.mean(hv_3d_list), np.std(hv_3d_list)

    mean_sparsity_3d, std_sparsity_3d = np.mean(sparsity_3d_list), np.std(
        sparsity_3d_list
    )

    mean_min_return_x, std_min_return_x = np.mean(min_return_x_list), np.std(
        min_return_x_list
    )
    mean_max_return_x, std_max_return_x = np.mean(max_return_x_list), np.std(
        max_return_x_list
    )
    mean_min_return_y, std_min_return_y = np.mean(min_return_y_list), np.std(
        min_return_y_list
    )
    mean_max_return_y, std_max_return_y = np.mean(max_return_y_list), np.std(
        max_return_y_list
    )
    mean_min_return_z, std_min_return_z = np.mean(min_return_z_list), np.std(
        min_return_z_list
    )
    mean_max_return_z, std_max_return_z = np.mean(max_return_z_list), np.std(
        max_return_z_list
    )

    mean_pareto_points_count, std_pareto_points_count = np.mean(
        pareto_points_count_list
    ), np.std(pareto_points_count_list)

    # Calculate for the superseed set (OLS data only)
    superseed_results = calculate_hypervolume_and_sparsity_superseed(
        all_x, all_y, all_z
    )

    pareto_points_superseed_3d = np.column_stack((all_x, all_y, all_z))
    reference_point_superseed_3d = (min(all_x), min(all_y), min(all_z))
    hv_superseed_3d = calculate_3d_hypervolume(
        pareto_points_superseed_3d, reference_point_superseed_3d
    )
    sparsity_superseed_3d = calculate_3d_sparsity(pareto_points_superseed_3d)

    # Append results for the superseed set, rounding values
    results.append(
        {
            "Seed": "Superseed",
            "Hypervolume XY": round(superseed_results["Hypervolume XY"], 2),
            "Hypervolume XZ": round(superseed_results["Hypervolume XZ"], 2),
            "Hypervolume YZ": round(superseed_results["Hypervolume YZ"], 2),
            "Hypervolume 3D": round(hv_superseed_3d, 2),
            "Sparsity 3D": round(sparsity_superseed_3d, 2),
            "Min Return X": round(min(all_x), 2),
            "Max Return X": round(max(all_x), 2),
            "Min Return Y": round(min(all_y), 2),
            "Max Return Y": round(max(all_y), 2),
            "Min Return Z": round(min(all_z), 2),
            "Max Return Z": round(max(all_z), 2),
            "Pareto Points Count": len(pareto_points_superseed_3d),
        }
    )

    # Append mean and std as final rows, rounding values
    results.append(
        {
            "Seed": "Mean",
            "Hypervolume XY": round(mean_hv_xy, 2),
            "Hypervolume XZ": round(mean_hv_xz, 2),
            "Hypervolume YZ": round(mean_hv_yz, 2),
            "Hypervolume 3D": round(mean_hv_3d, 2),
            "Sparsity 3D": round(mean_sparsity_3d, 2),
            "Min Return X": round(mean_min_return_x, 2),
            "Max Return X": round(mean_max_return_x, 2),
            "Min Return Y": round(mean_min_return_y, 2),
            "Max Return Y": round(mean_max_return_y, 2),
            "Min Return Z": round(mean_min_return_z, 2),
            "Max Return Z": round(mean_max_return_z, 2),
            "Pareto Points Count": round(mean_pareto_points_count, 2),
        }
    )

    results.append(
        {
            "Seed": "Std Dev",
            "Hypervolume XY": round(std_hv_xy, 2),
            "Hypervolume XZ": round(std_hv_xz, 2),
            "Hypervolume YZ": round(std_hv_yz, 2),
            "Hypervolume 3D": round(std_hv_3d, 2),
            "Sparsity 3D": round(std_sparsity_3d, 2),
            "Min Return X": round(std_min_return_x, 2),
            "Max Return X": round(std_max_return_x, 2),
            "Min Return Y": round(std_min_return_y, 2),
            "Max Return Y": round(std_max_return_y, 2),
            "Min Return Z": round(std_min_return_z, 2),
            "Max Return Z": round(std_max_return_z, 2),
            "Pareto Points Count": round(std_pareto_points_count, 2),
        }
    )

    # Convert the results to a DataFrame
    df_results = pd.DataFrame(results)

    return df_results


def calculate_hypervolume_and_sparsity_superseed(all_x, all_y, all_z):
    """Calculates the hypervolume and sparsity for the combined superseed set."""
    # Reference points for hypervolume calculation (minimums from all points)
    reference_point_xy = (min(all_x), min(all_y))
    reference_point_xz = (min(all_x), min(all_z))
    reference_point_yz = (min(all_y), min(all_z))

    # Pareto frontiers for the superseed set
    superseed_pareto_xy, superseed_pareto_yy, _ = pareto_frontier_2d(all_x, all_y)
    superseed_pareto_xz, superseed_pareto_zz, _ = pareto_frontier_2d(all_x, all_z)
    superseed_pareto_yz, superseed_pareto_zz2, _ = pareto_frontier_2d(all_y, all_z)

    # Calculate hypervolumes for superseed set
    hypervolume_xy = calculate_hypervolume(
        list(zip(superseed_pareto_xy, superseed_pareto_yy)), reference_point_xy
    )
    hypervolume_xz = calculate_hypervolume(
        list(zip(superseed_pareto_xz, superseed_pareto_zz)), reference_point_xz
    )
    hypervolume_yz = calculate_hypervolume(
        list(zip(superseed_pareto_yz, superseed_pareto_zz2)), reference_point_yz
    )

    return {
        "Hypervolume XY": hypervolume_xy,
        "Hypervolume XZ": hypervolume_xz,
        "Hypervolume YZ": hypervolume_yz,
    }


def calculate_3d_metrics_only_for_mc(seed_path):
    """
    Calculates 3D Hypervolume and Sparsity for a single seed path.
    """
    # Load seed data
    data = load_json_data(seed_path)
    ccs_list = data["ccs_list"][-1]
    x_all, y_all, z_all = extract_coordinates(ccs_list)

    # Calculate 3D hypervolume and sparsity for the seed
    pareto_points_3d = np.column_stack((x_all, y_all, z_all))
    reference_point_3d = (min(x_all), min(y_all), min(z_all))
    hv_3d = calculate_3d_hypervolume(pareto_points_3d, reference_point_3d)
    sparsity_3d = calculate_3d_sparsity(pareto_points_3d)

    hv_3d = round(hv_3d, 2)
    sparsity_3d = round(sparsity_3d, 2)

    return hv_3d, sparsity_3d


def process_data(seed_paths, wrapper, output_dir):
    """Processes the data for all seeds and generates the 3D and 2D plots."""
    all_data = []
    if wrapper == "mc":
        seed_paths = [seed_paths]
    seed_paths

    for seed, seed_path in enumerate(seed_paths):
        if not os.path.exists(seed_path):
            print(f"File not found: {seed_path}")
            continue
        
        data = load_json_data(seed_path)
        ccs_list = data["ccs_list"][-1]
        x_all, y_all, z_all = extract_coordinates(ccs_list)

        # Get matching weights for each point
        matching_entries = find_matching_weights_and_agent(ccs_list, data["ccs_data"])

        # Collect data for DataFrame
        print(matching_entries)
        for entry in matching_entries:
            all_data.append(
                {
                    "seed": seed,
                    "Returns": entry["returns"],
                    "Weights": entry["weights"],
                    "test_chronic_0": entry["test_chronic_0"],
                    "test_chronic_1": entry["test_chronic_1"],
                    "eval_chronic_0": entry['eval_chronic_0'],
                    "eval_chronic_1": entry['eval_chronic_1']
                }
            )

    df_ccs_matching = pd.DataFrame(all_data) if all_data else pd.DataFrame()

    if not df_ccs_matching.empty:
        # Save the DataFrame to the constructed path
        csv_file_path = os.path.join(output_dir, "ccs_matching_data.csv")  
        if wrapper == "MC":
            csv_file_path = os.path.join(output_dir, "mc_ccs_matching_data.csv")
        df_ccs_matching.to_csv(csv_file_path, index=False)
        print(f"Saved ccs_matching_data.csv to {csv_file_path}")
    else:
        print("No matching entries found.")
    return df_ccs_matching


def plot_2d_projections_matplotlib(
    seed_paths,  mc_path, iteration_paths,  mc_iteration_paths, wrapper, save_dir=None, rewards=["L2RPN", "TopoDepth", "TopoActionHour"], iterations=False, mc_ex_path='\\C', benchmark=True
):
    """
    Plots X vs Y, X vs Z, and Y vs Z using matplotlib, highlighting Pareto frontier points.
    Connects the points of each Pareto frontier with lines to make it easier to see the different Pareto frontiers.
    Annotates the extrema points corresponding to extreme weight vectors like (1,0,0), (0,1,0), (0,0,1).
    """
    
    # Set up matplotlib parameters for a more scientific look
    plt.rcParams.update(
        {
            "font.size": 14,
            "figure.figsize": (20, 6),
            "axes.grid": True,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "font.family": "serif",
        }
    )

    fig, axs = plt.subplots(1, 3)

    if wrapper=='mc':
        return None

    else:
        # Handle OLS paths
        colors = plt.cm.tab10.colors  # Use a colormap for different seeds
        if iterations:
            seed_paths = iteration_paths
            iter = [5,10,20]
        print(seed_paths)
        for i, seed_path in enumerate(seed_paths[:10]):
            data = load_json_data(seed_path)
            ccs_list = data["ccs_list"][-1]
            x_all, y_all, z_all = extract_coordinates(ccs_list)
            print(seed_path)
            print(x_all)
            # Get matching weights for each point
            matching_entries = find_matching_weights_and_agent(
                ccs_list, data["ccs_data"]
            )

            # Create a mapping from coordinates to weights
            coord_to_weight = {}
            for entry in matching_entries:
                x, y, z = entry["returns"]
                weight = entry["weights"]
                coord_to_weight[(x, y, z)] = weight

            # Convert coordinates to tuples for matching
            coords_all = list(zip(x_all, y_all, z_all))

            # Create an array of weights corresponding to each point
            weights_all = [coord_to_weight.get(coord, None) for coord in coords_all]

            # Calculate Pareto frontiers
            x_pareto_xy, y_pareto_xy, pareto_indices_xy = pareto_frontier_2d(
                x_all, y_all
            )
            x_pareto_xz, z_pareto_xz, pareto_indices_xz = pareto_frontier_2d(
                x_all, z_all
            )
            y_pareto_yz, z_pareto_yz, pareto_indices_yz = pareto_frontier_2d(
                y_all, z_all
            )

            # Sort the Pareto frontier points for plotting lines
            sorted_indices_xy = np.argsort(x_pareto_xy)
            x_pareto_xy_sorted = np.array(x_pareto_xy)[sorted_indices_xy]
            y_pareto_xy_sorted = np.array(y_pareto_xy)[sorted_indices_xy]

            sorted_indices_xz = np.argsort(x_pareto_xz)
            x_pareto_xz_sorted = np.array(x_pareto_xz)[sorted_indices_xz]
            z_pareto_xz_sorted = np.array(z_pareto_xz)[sorted_indices_xz]

            sorted_indices_yz = np.argsort(y_pareto_yz)
            y_pareto_yz_sorted = np.array(y_pareto_yz)[sorted_indices_yz]
            z_pareto_yz_sorted = np.array(z_pareto_yz)[sorted_indices_yz]

            # Plot Pareto frontiers with lines
            # X vs Y
            if iterations: 
                label = f"iterations {iter[i]}"
            else:
                label = f"Seed {i+1}"
                
           
            axs[0].scatter(
                x_pareto_xy,
                y_pareto_xy,
                color=colors[i % len(colors)],
                edgecolors="black",
                marker="o",
                s=100,
                label=label,
            )
                        
            axs[0].plot(
                x_pareto_xy_sorted,
                y_pareto_xy_sorted,
                color=colors[i % len(colors)],
                linestyle="-",
                linewidth=1,
            )
            
            axs[0].set_xlabel(rewards[0])
            axs[0].set_ylabel(rewards[1])

            # X vs Z
            axs[1].scatter(
                x_pareto_xz,
                z_pareto_xz,
                color=colors[i % len(colors)],
                edgecolors="black",
                marker="o",
                s=100,
                label=label,
            )
            axs[1].plot(
                x_pareto_xz_sorted,
                z_pareto_xz_sorted,
                color=colors[i % len(colors)],
                linestyle="-",
                linewidth=1,
            )
            axs[1].set_xlabel(rewards[0])
            axs[1].set_ylabel(rewards[2])

            # Y vs Z
            axs[2].scatter(
                y_pareto_yz,
                z_pareto_yz,
                color=colors[i % len(colors)],
                edgecolors="black",
                marker="o",
                s=100,
                label=label,
            )
            axs[2].plot(
                y_pareto_yz_sorted,
                z_pareto_yz_sorted,
                color=colors[i % len(colors)],
                linestyle="-",
                linewidth=1,
            )
            axs[2].set_xlabel(rewards[1])
            axs[2].set_ylabel(rewards[2])

        #processing RS data
        # Load data
        if os.path.exists(mc_path) and benchmark:
            if iterations: 
                seed_paths = mc_iteration_paths
            else: 
                seed_paths = [mc_path]
            for i, seed_path in enumerate(seed_paths):
                data = load_json_data(seed_path)
                ccs_list = data["ccs_list"][-1]
                x_all, y_all, z_all = extract_coordinates(ccs_list)
                print(seed_path)
                print(x_all)
                # Get matching weights for each point
                matching_entries = find_matching_weights_and_agent(
                    ccs_list, data["ccs_data"]
                )

                # Create a mapping from coordinates to weights
                coord_to_weight = {}
                for entry in matching_entries:
                    x, y, z = entry["returns"]
                    weight = entry["weights"]
                    coord_to_weight[(x, y, z)] = weight

                # Convert coordinates to tuples for matching
                coords_all = list(zip(x_all, y_all, z_all))

                # Create an array of weights corresponding to each point
                weights_all = [coord_to_weight.get(coord, None) for coord in coords_all]

                # Calculate Pareto frontiers
                x_pareto_xy, y_pareto_xy, pareto_indices_xy = pareto_frontier_2d(
                    x_all, y_all
                )
                x_pareto_xz, z_pareto_xz, pareto_indices_xz = pareto_frontier_2d(
                    x_all, z_all
                )
                y_pareto_yz, z_pareto_yz, pareto_indices_yz = pareto_frontier_2d(
                    y_all, z_all
                )

                # Sort the Pareto frontier points for plotting lines
                sorted_indices_xy = np.argsort(x_pareto_xy)
                x_pareto_xy_sorted = np.array(x_pareto_xy)[sorted_indices_xy]
                y_pareto_xy_sorted = np.array(y_pareto_xy)[sorted_indices_xy]

                sorted_indices_xz = np.argsort(x_pareto_xz)
                x_pareto_xz_sorted = np.array(x_pareto_xz)[sorted_indices_xz]
                z_pareto_xz_sorted = np.array(z_pareto_xz)[sorted_indices_xz]

                sorted_indices_yz = np.argsort(y_pareto_yz)
                y_pareto_yz_sorted = np.array(y_pareto_yz)[sorted_indices_yz]
                z_pareto_yz_sorted = np.array(z_pareto_yz)[sorted_indices_yz]

                # Plot Pareto frontiers with lines
                # X vs Y
                if iterations: 
                    label = f"RS Benchmark iter {iter[i]}"
                else:
                    label = f"RS Benchmark {i+1}"
                    
                colors = ["lightgray", 'gray', "black"]
                axs[0].scatter(
                    x_pareto_xy,
                    y_pareto_xy,
                    color=colors[i % len(colors)],
                    edgecolors="black",
                    marker="o",
                    s=100,
                    label=label,
                )
                            
                axs[0].plot(
                    x_pareto_xy_sorted,
                    y_pareto_xy_sorted,
                    color=colors[i % len(colors)],
                    linestyle="-",
                    linewidth=1,
                )
                
                axs[0].set_xlabel(rewards[0])
                axs[0].set_ylabel(rewards[1])

                # X vs Z
                axs[1].scatter(
                    x_pareto_xz,
                    z_pareto_xz,
                    color=colors[i % len(colors)],
                    edgecolors="black",
                    marker="o",
                    s=100,
                    label=label,
                )
                axs[1].plot(
                    x_pareto_xz_sorted,
                    z_pareto_xz_sorted,
                    color=colors[i % len(colors)],
                    linestyle="-",
                    linewidth=1,
                )
                axs[1].set_xlabel(rewards[0])
                axs[1].set_ylabel(rewards[2])

                # Y vs Z
                axs[2].scatter(
                    y_pareto_yz,
                    z_pareto_yz,
                    color=colors[i % len(colors)],
                    edgecolors="black",
                    marker="o",
                    s=100,
                    label=label,
                )
                axs[2].plot(
                    y_pareto_yz_sorted,
                    z_pareto_yz_sorted,
                    color=colors[i % len(colors)],
                    linestyle="-",
                    linewidth=1,
                )
                axs[2].set_xlabel(rewards[1])
                axs[2].set_ylabel(rewards[2])



        for ax in axs:
            ax.legend()
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        plt.suptitle("Projections of Pareto Frontier in Return Domain", fontsize=20)
        plt.subplots_adjust(top=0.88)  # Adjust the top to make room for suptitle
        if save_dir:
            plt.savefig(os.path.join(save_dir, "ols_pareto_frontiers.png"))
        plt.show()
def plot_ccs_points_only(
    ols_seed_paths,
    rs_seed_paths,
    save_dir=None,
    rewards=["L2RPN", "TopoDepth", "TopoActionHour"]
):
    """
    Plots only the points that form the CCS over all seeds from both OLS and RS.
    Points are plotted in full color, indicating whether they came from OLS or RS.

    Parameters:
    - ols_seed_paths: List of file paths to OLS JSON data.
    - rs_seed_paths: List of file paths to RS JSON data.
    - save_dir: Directory to save the plot image (optional).
    - rewards: List of reward names for labeling axes.
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    # --- Plotting starts here ---
    # Set up matplotlib parameters for a more scientific look
    plt.rcParams.update(
        {
            "font.size": 14,
            "figure.figsize": (20, 6),
            "axes.grid": True,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "font.family": "serif",
        }
    )

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Initialize lists to collect all data points and labels
    all_coords = []
    labels_list = []  # 'OLS' or 'RS' for each point

    # Function to load data from seed paths
    def load_data(seed_paths, label):
        coords_list = []
        for seed_path in seed_paths:
            if not os.path.exists(seed_path):
                print(f"File not found: {seed_path}")
                continue

            data = load_json_data(seed_path)
            ccs_list = data["ccs_list"][-1]
            x_all, y_all, z_all = extract_coordinates(ccs_list)
            coords = np.column_stack((x_all, y_all, z_all))
            coords_list.append(coords)
            labels_list.extend([label] * len(coords))
        if coords_list:
            return np.vstack(coords_list)
        else:
            return np.array([])

    # Load OLS data
    ols_coords = load_data(ols_seed_paths, 'OLS')

    # Load RS data
    rs_coords = load_data(rs_seed_paths, 'RS')

    # Combine all data
    if ols_coords.size > 0 and rs_coords.size > 0:
        all_coords = np.vstack((ols_coords, rs_coords))
    elif ols_coords.size > 0:
        all_coords = ols_coords
    elif rs_coords.size > 0:
        all_coords = rs_coords
    else:
        print("No data available to plot.")
        return

    labels = np.array(labels_list)

    # Compute the 3D Convex Coverage Set (CCS) over all points
    ccs_mask = get_pareto_front(all_coords)
    ccs_indices = np.where(ccs_mask)[0]
    ccs_coords = all_coords[ccs_indices]
    ccs_labels = labels[ccs_indices]

    # Separate CCS points by their source (OLS or RS)
    ols_ccs_coords = ccs_coords[ccs_labels == 'OLS']
    rs_ccs_coords = ccs_coords[ccs_labels == 'RS']

    # Plot only the CCS points, in full color
    # OLS CCS points
    if ols_ccs_coords.size > 0:
        axs[0].scatter(
            ols_ccs_coords[:, 0],
            ols_ccs_coords[:, 1],
            color='lightcoral',
            marker='o',
            s=100,
            label='OLS CCS Points'
        )
        axs[1].scatter(
            ols_ccs_coords[:, 0],
            ols_ccs_coords[:, 2],
            color='lightcoral',
            marker='o',
            s=100,
            label='OLS CCS Points'
        )
        axs[2].scatter(
            ols_ccs_coords[:, 1],
            ols_ccs_coords[:, 2],
            color='lightcoral',
            marker='o',
            s=100,
            label='OLS CCS Points'
        )

    # RS CCS points
    if rs_ccs_coords.size > 0:
        axs[0].scatter(
            rs_ccs_coords[:, 0],
            rs_ccs_coords[:, 1],
            color='darkgrey',
            marker='o',
            s=100,
            label='RS CCS Points'
        )
        axs[1].scatter(
            rs_ccs_coords[:, 0],
            rs_ccs_coords[:, 2],
            color='darkgrey',
            marker='o',
            s=100,
            label='RS CCS Points'
        )
        axs[2].scatter(
            rs_ccs_coords[:, 1],
            rs_ccs_coords[:, 2],
            color='darkgrey',
            marker='o',
            s=100,
            label='RS CCS Points'
        )

    # Set labels
    axs[0].set_xlabel(rewards[0])
    axs[0].set_ylabel(rewards[1])
    axs[1].set_xlabel(rewards[0])
    axs[1].set_ylabel(rewards[2])
    axs[2].set_xlabel(rewards[1])
    axs[2].set_ylabel(rewards[2])

    # Remove duplicate legends and adjust
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.suptitle("CCS over All Seeds and Methods", fontsize=20)
    plt.subplots_adjust(top=0.88)  # Adjust the top to make room for suptitle
    if save_dir:
        plt.savefig(os.path.join(save_dir, "ccs_points_only.png"))
    plt.show()
def plot_super_pareto_frontier_2d_ols_vs_rs_ccs(
    ols_seed_paths,
    rs_seed_paths,
    save_dir=None,
    rewards=["L2RPN", "TopoDepth", "TopoActionHour"]
):
    """
    Plots the 2D projections of the 3D convex coverage set (CCS) over all seeds from both OLS and RS.
    Highlights the CCS points from OLS in lightcoral and from RS in dark grey.
    Additionally, plots all other points (non-CCS) with respective colours and transparency.

    Parameters:
    - ols_seed_paths: List of file paths to OLS JSON data.
    - rs_seed_paths: List of file paths to RS JSON data.
    - save_dir: Directory to save the plot image (optional).
    - rewards: List of reward names for labeling axes.
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    # --- Plotting starts here ---
    # Set up matplotlib parameters for a more scientific look
    plt.rcParams.update(
        {
            "font.size": 14,
            "figure.figsize": (20, 6),
            "axes.grid": True,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "font.family": "serif",
        }
    )

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Initialize lists to collect all data points and labels
    all_coords = []
    labels_list = []  # 'OLS' or 'RS' for each point

    # Function to load data from seed paths
    def load_data(seed_paths, label):
        coords_list = []
        for seed_path in seed_paths:
            if not os.path.exists(seed_path):
                print(f"File not found: {seed_path}")
                continue

            data = load_json_data(seed_path)
            ccs_list = data["ccs_list"][-1]
            x_all, y_all, z_all = extract_coordinates(ccs_list)
            coords = np.column_stack((x_all, y_all, z_all))
            coords_list.append(coords)
            labels_list.extend([label] * len(coords))
        if coords_list:
            return np.vstack(coords_list)
        else:
            return np.array([])

    # Load OLS data
    ols_coords = load_data(ols_seed_paths, 'OLS')

    # Load RS data
    rs_coords = load_data(rs_seed_paths, 'RS')

    # Combine all data
    if ols_coords.size > 0 and rs_coords.size > 0:
        all_coords = np.vstack((ols_coords, rs_coords))
    elif ols_coords.size > 0:
        all_coords = ols_coords
    elif rs_coords.size > 0:
        all_coords = rs_coords
    else:
        print("No data available to plot.")
        return

    labels = np.array(labels_list)

    # Compute the 3D Convex Coverage Set (CCS) over all points
    ccs_mask = get_pareto_front(all_coords)
    ccs_indices = np.where(ccs_mask)[0]
    ccs_coords = all_coords[ccs_indices]
    ccs_labels = labels[ccs_indices]

    # Identify which CCS points come from OLS and which from RS
    ols_ccs_indices = ccs_indices[ccs_labels == 'OLS']
    rs_ccs_indices = ccs_indices[ccs_labels == 'RS']
    ols_ccs_coords = all_coords[ols_ccs_indices]
    rs_ccs_coords = all_coords[rs_ccs_indices]

    # Now plot all the points with respective colors and transparency
    # All OLS points in lightcoral with transparency
    if ols_coords.size > 0:
        axs[0].scatter(
            ols_coords[:, 0],
            ols_coords[:, 1],
            color='red',
            alpha=0.3,
            marker='o',
            label='OLS Points'
        )
        axs[1].scatter(
            ols_coords[:, 0],
            ols_coords[:, 2],
            color='red',
            alpha=0.3,
            marker='o',
            label='OLS Points'
        )
        axs[2].scatter(
            ols_coords[:, 1],
            ols_coords[:, 2],
            color='red',
            alpha=0.3,
            marker='o',
            label='OLS Points'
        )

    # All RS points in dark grey with transparency
    if rs_coords.size > 0:
        axs[0].scatter(
            rs_coords[:, 0],
            rs_coords[:, 1],
            color='darkgrey',
            alpha=0.3,
            marker='o',
            label='RS Points'
        )
        axs[1].scatter(
            rs_coords[:, 0],
            rs_coords[:, 2],
            color='darkgrey',
            alpha=0.3,
            marker='o',
            label='RS Points'
        )
        axs[2].scatter(
            rs_coords[:, 1],
            rs_coords[:, 2],
            color='darkgrey',
            alpha=0.3,
            marker='o',
            label='RS Points'
        )

    # Now plot the CCS points with solid colors and larger markers
    # OLS CCS points
    if ols_ccs_coords.size > 0:
        axs[0].scatter(
            ols_ccs_coords[:, 0],
            ols_ccs_coords[:, 1],
            color='red',
            edgecolors='black',
            marker='o',
            s=100,
            label='OLS CCS Points'
        )
        axs[1].scatter(
            ols_ccs_coords[:, 0],
            ols_ccs_coords[:, 2],
            color='red',
            edgecolors='black',
            marker='o',
            s=100,
            label='OLS CCS Points'
        )
        axs[2].scatter(
            ols_ccs_coords[:, 1],
            ols_ccs_coords[:, 2],
            color='red',
            edgecolors='black',
            marker='o',
            s=100,
            label='OLS CCS Points'
        )

    # RS CCS points
    if rs_ccs_coords.size > 0:
        axs[0].scatter(
            rs_ccs_coords[:, 0],
            rs_ccs_coords[:, 1],
            color='darkgrey',
            edgecolors='black',
            marker='o',
            s=80,
            label='RS CCS Points'
        )
        axs[1].scatter(
            rs_ccs_coords[:, 0],
            rs_ccs_coords[:, 2],
            color='darkgrey',
            edgecolors='black',
            marker='o',
            s=80,
            label='RS CCS Points'
        )
        axs[2].scatter(
            rs_ccs_coords[:, 1],
            rs_ccs_coords[:, 2],
            color='darkgrey',
            edgecolors='black',
            marker='o',
            s=80,
            label='RS CCS Points'
        )

    # Set labels
    axs[0].set_xlabel(rewards[0])
    axs[0].set_ylabel(rewards[1])
    axs[1].set_xlabel(rewards[0])
    axs[1].set_ylabel(rewards[2])
    axs[2].set_xlabel(rewards[1])
    axs[2].set_ylabel(rewards[2])

    # Remove duplicate legends and adjust
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.suptitle("3D CCS Projection: OLS and RS Super CCS", fontsize=20)
    plt.subplots_adjust(top=0.88)  # Adjust the top to make room for suptitle
    if save_dir:
        plt.savefig(os.path.join(save_dir, "super_ccs_ols_vs_rs.png"))
    plt.show()






def plot_super_pareto_frontier_2d(seed_paths, save_dir=None, rewards=["L2RPN", "TopoDepth", "TopoActionHour"]):
    """
    Plots the super Pareto frontier across all seeds on the 2D projections (X vs Y, X vs Z, Y vs Z) using matplotlib.
    """
    import matplotlib.pyplot as plt
    import os

    # Set up matplotlib parameters for a more scientific look
    plt.rcParams.update(
        {
            "font.size": 14,
            "figure.figsize": (20, 6),
            "axes.grid": True,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "font.family": "serif",
        }
    )

    fig, axs = plt.subplots(1, 3)

    # Initialize lists to collect all data points
    x_all_seeds = []
    y_all_seeds = []
    z_all_seeds = []

    # Initialize lists to collect weights, if needed
    coords_all = []
    weights_all = []

    for seed_path in seed_paths:
        data = load_json_data(seed_path)
        ccs_list = data["ccs_list"][-1]
        x_all, y_all, z_all = extract_coordinates(ccs_list)
        x_all_seeds.extend(x_all)
        y_all_seeds.extend(y_all)
        z_all_seeds.extend(z_all)
        # Collect weights for annotations
        matching_entries = find_matching_weights_and_agent(
            ccs_list, data["ccs_data"]
        )
        # Create a mapping from coordinates to weights
        coord_to_weight = {}
        for entry in matching_entries:
            x, y, z = entry["returns"]
            weight = entry["weights"]
            coord_to_weight[(x, y, z)] = weight
        # Convert coordinates to tuples for matching
        coords = list(zip(x_all, y_all, z_all))
        coords_all.extend(coords)
        # Create an array of weights corresponding to each point
        weights = [coord_to_weight.get(coord, None) for coord in coords]
        weights_all.extend(weights)

    # Now, compute the super Pareto frontiers in 2D
    x_pareto_xy, y_pareto_xy, pareto_indices_xy = pareto_frontier_2d(x_all_seeds, y_all_seeds)
    x_pareto_xz, z_pareto_xz, pareto_indices_xz = pareto_frontier_2d(x_all_seeds, z_all_seeds)
    y_pareto_yz, z_pareto_yz, pareto_indices_yz = pareto_frontier_2d(y_all_seeds, z_all_seeds)

    # Plot all data points in gray
    gray_color = 'gray'
    # X vs Y
    axs[0].scatter(
        x_all_seeds,
        y_all_seeds,
        color=gray_color,
        alpha=0.5,
        label='All Data Points',
    )
    # X vs Z
    axs[1].scatter(
        x_all_seeds,
        z_all_seeds,
        color=gray_color,
        alpha=0.5,
        label='All Data Points',
    )
    # Y vs Z
    axs[2].scatter(
        y_all_seeds,
        z_all_seeds,
        color=gray_color,
        alpha=0.5,
        label='All Data Points',
    )

    # Plot the super Pareto frontiers
    pareto_color = 'red'
    # X vs Y
    axs[0].scatter(
        x_pareto_xy,
        y_pareto_xy,
        color=pareto_color,
        edgecolors='black',
        marker='o',
        s=100,
        label='Super Pareto Frontier',
    )
    axs[0].set_xlabel(rewards[0])
    axs[0].set_ylabel(rewards[1])

    # Annotate extrema points for X vs Y
    for idx in pareto_indices_xy:
        weight = weights_all[idx]
        if weight is not None:
            if is_extreme_weight(weight):
                x = x_all_seeds[idx]
                y = y_all_seeds[idx]
                label = weight_label(weight)
                axs[0].annotate(
                    label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=12,
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                )

    # X vs Z
    axs[1].scatter(
        x_pareto_xz,
        z_pareto_xz,
        color=pareto_color,
        edgecolors='black',
        marker='o',
        s=100,
        label='Super Pareto Frontier',
    )
    axs[1].set_xlabel(rewards[0])
    axs[1].set_ylabel(rewards[2])

    # Annotate extrema points for X vs Z
    for idx in pareto_indices_xz:
        weight = weights_all[idx]
        if weight is not None:
            if is_extreme_weight(weight):
                x = x_all_seeds[idx]
                z = z_all_seeds[idx]
                label = weight_label(weight)
                axs[1].annotate(
                    label,
                    (x, z),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=12,
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                )

    # Y vs Z
    axs[2].scatter(
        y_pareto_yz,
        z_pareto_yz,
        color=pareto_color,
        edgecolors='black',
        marker='o',
        s=100,
        label='Super Pareto Frontier',
    )
    axs[2].set_xlabel(rewards[1])
    axs[2].set_ylabel(rewards[2])

    # Annotate extrema points for Y vs Z
    for idx in pareto_indices_yz:
        weight = weights_all[idx]
        if weight is not None:
            if is_extreme_weight(weight):
                y = y_all_seeds[idx]
                z = z_all_seeds[idx]
                label = weight_label(weight)
                axs[2].annotate(
                    label,
                    (y, z),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=12,
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                )

    for ax in axs:
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.suptitle("Super Pareto Frontier Projections in Return Domain", fontsize=20)
    plt.subplots_adjust(top=0.88)  # Adjust the top to make room for suptitle
    if save_dir:
        plt.savefig(os.path.join(save_dir, "super_pareto_frontiers.png"))
    plt.show()

def plot_super_pareto_frontier_2d_multiple_settings(base_path, scenario, settings, save_dir=None, rewards=["L2RPN", "TopoDepth", "TopoActionHour"]):
    """
    Plots the super Pareto frontier across different settings on the 2D projections (X vs Y, X vs Z, Y vs Z) using matplotlib.
    Each setting is plotted with a different color and label.
    
    Parameters:
    - base_path: The base directory where the JSON log files are stored.
    - scenario: The scenario name (e.g., "Reuse", "Opponent").
    - settings: A list of setting names (e.g., ["Baseline", "Full", "Partial"]).
    - save_dir: Directory to save the plot image (optional).
    - rewards: List of reward names for labeling axes.
    """
    import matplotlib.pyplot as plt
    import os

    # --- Generate paths within the function ---
    if scenario == 'Reuse':
        settings_paths = {
            "Baseline": os.path.join(base_path, "Baseline"),
            "Full": os.path.join(base_path, "Full_Reuse"),
            "Partial": os.path.join(base_path, "Partial_Reuse"),
            # Add more settings if needed
        }
    elif scenario == 'Opponent':
        settings_paths = {
            "Baseline": os.path.join(base_path, "Baseline"),
            "Opponent": os.path.join(base_path, "Opponent")
        }
    # Add additional scenarios as needed

    # Prepare the settings_paths dictionary
    settings_seed_paths = {}
    for setting in settings:
        path = settings_paths.get(setting)
        if not path:
            print(f"Path for setting '{setting}' not found.")
            continue

        seed_paths = []
        for seed in range(5):  # Adjust the range based on your seeds
            seed_file = f"morl_logs_seed_{seed}.json"
            seed_path = os.path.join(path, seed_file)
            if os.path.exists(seed_path):
                seed_paths.append(seed_path)
            else:
                print(f"Seed path not found: {seed_path}")
        settings_seed_paths[setting] = seed_paths

    # --- Plotting starts here ---
    # Set up matplotlib parameters for a more scientific look
    plt.rcParams.update(
        {
            "font.size": 14,
            "figure.figsize": (20, 6),
            "axes.grid": True,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "font.family": "serif",
        }
    )

    fig, axs = plt.subplots(1, 3)

    # Get a list of colors to assign to settings
    colors = plt.cm.tab10.colors  # You can choose other colormaps

    for idx, (setting_name, seed_paths) in enumerate(settings_seed_paths.items()):
        # Initialize lists to collect all data points for this setting
        x_all_seeds = []
        y_all_seeds = []
        z_all_seeds = []

        # Initialize lists to collect weights, if needed
        coords_all = []
        weights_all = []

        for seed_path in seed_paths:
            if not os.path.exists(seed_path):
                print(f"File not found: {seed_path}")
                continue

            data = load_json_data(seed_path)
            ccs_list = data["ccs_list"][-1]
            x_all, y_all, z_all = extract_coordinates(ccs_list)
            x_all_seeds.extend(x_all)
            y_all_seeds.extend(y_all)
            z_all_seeds.extend(z_all)
            # Collect weights for annotations
            matching_entries = find_matching_weights_and_agent(
                ccs_list, data["ccs_data"]
            )
            # Create a mapping from coordinates to weights
            coord_to_weight = {}
            for entry in matching_entries:
                x, y, z = entry["returns"]
                weight = entry["weights"]
                coord_to_weight[(x, y, z)] = weight
            # Convert coordinates to tuples for matching
            coords = list(zip(x_all, y_all, z_all))
            coords_all.extend(coords)
            # Create an array of weights corresponding to each point
            weights = [coord_to_weight.get(coord, None) for coord in coords]
            weights_all.extend(weights)

        if not x_all_seeds:
            print(f"No data for setting {setting_name}")
            continue

        # Now, compute the super Pareto frontiers in 2D for this setting
        x_pareto_xy, y_pareto_xy, pareto_indices_xy = pareto_frontier_2d(x_all_seeds, y_all_seeds)
        x_pareto_xz, z_pareto_xz, pareto_indices_xz = pareto_frontier_2d(x_all_seeds, z_all_seeds)
        y_pareto_yz, z_pareto_yz, pareto_indices_yz = pareto_frontier_2d(y_all_seeds, z_all_seeds)

        # Assign a color to this setting
        color = colors[idx % len(colors)]

        # Plot the super Pareto frontiers
        # X vs Y
        axs[0].scatter(
            x_pareto_xy,
            y_pareto_xy,
            color=color,
            edgecolors='black',
            marker='o',
            s=100,
            label=f'{setting_name}',
        )
        axs[0].set_xlabel(rewards[0])
        axs[0].set_ylabel(rewards[1])

        # Annotate extrema points for X vs Y
        for idx_pareto in pareto_indices_xy:
            weight = weights_all[idx_pareto]
            if weight is not None:
                if is_extreme_weight(weight):
                    x = x_all_seeds[idx_pareto]
                    y = y_all_seeds[idx_pareto]
                    label = weight_label(weight)
                    axs[0].annotate(
                        label,
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=12,
                        color=color,
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                    )

        # X vs Z
        axs[1].scatter(
            x_pareto_xz,
            z_pareto_xz,
            color=color,
            edgecolors='black',
            marker='o',
            s=100,
            label=f'{setting_name}',
        )
        axs[1].set_xlabel(rewards[0])
        axs[1].set_ylabel(rewards[2])

        # Annotate extrema points for X vs Z
        for idx_pareto in pareto_indices_xz:
            weight = weights_all[idx_pareto]
            if weight is not None:
                if is_extreme_weight(weight):
                    x = x_all_seeds[idx_pareto]
                    z = z_all_seeds[idx_pareto]
                    label = weight_label(weight)
                    axs[1].annotate(
                        label,
                        (x, z),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=12,
                        color=color,
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                    )

        # Y vs Z
        axs[2].scatter(
            y_pareto_yz,
            z_pareto_yz,
            color=color,
            edgecolors='black',
            marker='o',
            s=100,
            label=f'{setting_name}',
        )
        axs[2].set_xlabel(rewards[1])
        axs[2].set_ylabel(rewards[2])

        # Annotate extrema points for Y vs Z
        for idx_pareto in pareto_indices_yz:
            weight = weights_all[idx_pareto]
            if weight is not None:
                if is_extreme_weight(weight):
                    y = y_all_seeds[idx_pareto]
                    z = z_all_seeds[idx_pareto]
                    label = weight_label(weight)
                    axs[2].annotate(
                        label,
                        (y, z),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=12,
                        color=color,
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                    )

    for ax in axs:
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.suptitle(f"Super Pareto Frontier Projections ({scenario} Scenario)", fontsize=20)
    plt.subplots_adjust(top=0.88)  # Adjust the top to make room for suptitle
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"super_pareto_frontiers_{scenario}.png"))
    plt.show()

def topo_depth_process_and_plot(csv_path):
    """
    Processes the CSV data and plots the topological depth trajectories for Pareto frontier points.
    Adjusts the design and layout for a more scientific appearance.
    """
    
    # Set up matplotlib parameters for a more scientific look
    plt.rcParams.update(
        {
            "font.size": 14,
            "figure.figsize": (12, 8),
            "axes.grid": True,
            "axes.labelsize": 16,
            "axes.titlesize": 20,
            "legend.fontsize": 12,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "font.family": "serif",
        }
    )

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Extract information from the test_chronic columns
    df["test_chronic_0"] = df["test_chronic_0"].apply(ast.literal_eval)

    # Filter rows where 'test_steps' in 'test_chronic_0' is 2016
    df = df[df["test_chronic_0"].apply(lambda x: x["test_steps"] == 2016)]

    # Limit to the first 5 points that reach 2016 steps
    df = df.head(4)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set a title for the plot
    ax.set_title("Topological Trajectory for PF Points", fontsize=20)

    # Initialize index for alternatives
    alternative_idx = 1

    # Define label to color mapping
    label_to_color = {
        "S-O": "black",
        "M-O alternative [1]": "red",
        "M-O alternative [2]": "blue",
        "M-O alternative [3]": "green",
    }

    # Iterate through each row in the dataframe
    for idx, row in df.iterrows():
        weights = [round(float(w), 2) for w in ast.literal_eval(row["Weights"])]

        # Determine the label based on weights
        if np.allclose(weights, [1.0, 0.0, 0.0], atol=1e-2):
            label = "S-O"
        else:
            label = f"M-O alternative [{alternative_idx}]"
            alternative_idx += 1

        # Get color based on label
        color = label_to_color.get(label, "black")  # Default to black if label not found

        # Extract the timestamp and topological depth information from test_chronic_0
        chronic = "test_chronic_0"
        timestamps = [0.0] + list(map(float, row[chronic]["test_action_timestamp"]))
        topo_depths = [0.0] + [
            0.0 if t is None else t for t in row[chronic]["test_topo_distance"]
        ]
        substations = row[chronic]["test_sub_ids"]

        # Plot each action as rectangular steps and fill the area underneath
        for i in range(len(timestamps) - 1):
            if topo_depths[i] is not None and topo_depths[i + 1] is not None:
                # Horizontal line: from timestamps[i] to timestamps[i+1] at topo_depths[i]
                ax.plot(
                    [timestamps[i], timestamps[i + 1]],
                    [topo_depths[i], topo_depths[i]],
                    color=color,
                    linestyle='-',
                    linewidth=3,  # Increased line width
                    label=label if i == 0 else "",
                )
                # Vertical line: from topo_depths[i] to topo_depths[i+1] at timestamps[i+1]
                ax.plot(
                    [timestamps[i + 1], timestamps[i + 1]],
                    [topo_depths[i], topo_depths[i + 1]],
                    color=color,
                    linestyle='-',
                    linewidth=3,  # Increased line width
                )
                # Fill the area underneath the horizontal line
                ax.fill_between(
                    [timestamps[i], timestamps[i + 1]],
                    0,
                    topo_depths[i],
                    color=color,
                    alpha=0.1,
                )

        # Plot markers at action points
        ax.scatter(
            timestamps,
            topo_depths,
            color=color,
            marker='o',
            s=75  # Increased marker size
        )

        # Annotate switching actions directly next to the points
        for timestamp, topo_depth, substation in zip(timestamps, topo_depths, substations):
            if topo_depth is not None and timestamp != 0.0:
                ax.annotate(
                    f"Sub {substation[0]}",
                    (timestamp, topo_depth),
                    textcoords="offset points",
                    xytext=(5, 0),
                    ha="left",
                    va="center",
                    fontsize=10,
                    color=color
                )

    # Formatting the plot
    ax.set_xlabel("Timestamp", fontsize=14)
    ax.set_ylabel("Topological Depth", fontsize=14)
    ax.set_yticks([0, 1, 2])
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(title="Trajectories")

    plt.tight_layout()
    plt.show()

def sub_id_process_and_plot(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Extract information from the test_chronic columns
    df["test_chronic_0"] = df["test_chronic_0"].apply(ast.literal_eval)
    df["test_chronic_1"] = df["test_chronic_1"].apply(ast.literal_eval)

    # Initialize the plot
    fig, ax = plt.subplots()

    # Generate colors for each Pareto point
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))

    # Iterate through each row in the dataframe
    for idx, row in df.iterrows():
        color = colors[idx % len(colors)]
        weights = row["Weights"]
        label = f"Pareto Point {idx+1}: Weights {weights}"

        # For each row, extract the timestamp and substation information from test_chronic_0 and test_chronic_1
        for chronic in ["test_chronic_0", "test_chronic_1"]:
            row[chronic]["test_steps"]
            actions = row[chronic]["test_actions"]
            timestamps = list(map(float, row[chronic]["test_action_timestamp"]))
            sub_ids = row[chronic]["test_sub_ids"]

            # Different marker for each chronic
            marker = "o" if chronic == "test_chronic_0" else "s"
            chronic_label = f"{label} ({chronic})"

            # Plot each action on the graph
            for timestamp, sub_id_list, action in zip(timestamps, sub_ids, actions):
                for sub in sub_id_list:
                    if sub is not None:
                        ax.plot(
                            timestamp,
                            int(sub),
                            marker,
                            color=color,
                            label=chronic_label,
                        )
                        chronic_label = ""  # Avoid repeated labels in legend

    # Formatting the plot
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Substation ID Affected by Switching")
    ax.set_title("Substation Modifications at Different Pareto Points")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        fontsize="small",
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def analyse_pf_values_and_plot_projections(csv_path):
    """
    Analyzes Pareto frontier values and plots 2D projections in the power system domain.
    Adjusts the design to have the same color for all points, transparent points for those not reaching 2016 steps,
    highlights the best result per graph in red, and the single objective point in black.
    Ensures that red and black points are plotted over blue ones in case of overlapping points.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import ast
    import numpy as np

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Convert the string representation of dictionaries in 'test_chronic_0' to actual dictionaries
    df["test_chronic_0"] = df["test_chronic_0"].apply(ast.literal_eval)

    results = []

    # Iterate through each row in the dataframe
    for idx, row in df.iterrows():
        # Extract information from 'test_chronic_0'
        chronic = "test_chronic_0"
        seed = row["seed"]
        test_data = row[chronic]
        steps = test_data["test_steps"]  # Total steps in the test
        actions = test_data["test_actions"]  # List of actions taken
        topo_vects = test_data['test_topo_vect']
        topo_depths = test_data.get("test_topo_distance")
        timestamps = test_data.get("test_action_timestamp")

        if timestamps is None or topo_depths is None:
            print(
                f"Error: 'Timestamps' or 'Topo Depths' not found in test_chronic_0 for Pareto Point {idx + 1}"
            )
            continue  # Skip this row if data is missing

        # Ensure that timestamps and topo_depths have the same length
        if len(timestamps) != len(topo_depths):
            print(
                f"Error: Length of 'Timestamps' and 'Topo Depths' do not match for Pareto Point {idx + 1}"
            )
            continue

        # Initialize cumulative weighted depth time
        cumulative_weighted_depth_time = 0

        # Start from the initial time and depth
        previous_time = 0
        previous_depth = 0  # Default topology depth at start is 0

        # Iterate over the timestamps and topo_depths
        for current_time, current_depth in zip(timestamps, topo_depths):
            # Calculate delta_time
            delta_time = current_time - previous_time

            # Calculate weighted depth time for this interval
            weighted_depth_time = previous_depth * delta_time

            # Add to cumulative sum
            cumulative_weighted_depth_time += weighted_depth_time

            # Update previous time and depth for next iteration
            previous_time = current_time
            previous_depth = current_depth

        # Handle the final interval (from last timestamp to end of test)
        # Assuming the total test time is equal to the total steps
        total_test_time = steps  # If time is measured in steps
        delta_time = total_test_time - previous_time
        weighted_depth_time = previous_depth * delta_time
        cumulative_weighted_depth_time += weighted_depth_time

        # Finally, divide cumulative weighted depth time by number of steps
        weighted_depth_metric = (
            cumulative_weighted_depth_time / steps if steps > 0 else 0
        )

        # Calculate average steps per chronic (assuming steps are total steps over all chronics)
        num_chronics = test_data.get(
            "Num Chronics", 1
        )  # Adjust if 'Num Chronics' is available
        avg_steps = steps / num_chronics if num_chronics > 0 else steps

        # Total number of switching actions (average over chronics)
        total_switching_actions = (
            len(actions) / num_chronics if num_chronics > 0 else len(actions)
        )

        # Extract weights
        weights = ast.literal_eval(row["Weights"])  # Should be list of floats

        # Store the results for the current Pareto point
        results.append(
            {
                "seed": seed,
                "Pareto Point": idx + 1,
                "Average Steps": avg_steps,
                "Total Switching Actions": total_switching_actions,
                "Weighted Depth Metric": weighted_depth_metric,
                "Weights": weights,
                "Steps": steps,
            }
        )

    if not results:
        print("No valid data to plot.")
        return

    # Convert results to DataFrame for easier processing
    results_df = pd.DataFrame(results)

    # Prepare data for plotting
    avg_steps_list = results_df["Average Steps"].values
    weighted_depth_list = results_df["Weighted Depth Metric"].values
    total_actions_list = results_df["Total Switching Actions"].values
    weights_list = results_df["Weights"].values
    steps_list = results_df["Steps"].values

    # Determine which points reached 2016 steps
    is_2016_steps = steps_list >= 2016

    # Alpha values: 1 for points reaching 2016 steps, 0.2 for others
    alpha_values = np.where(is_2016_steps, 1, 0.2)

    # Determine indices of single objective points
    is_single_objective = [np.allclose(w, [1, 0, 0], atol=1e-2) for w in weights_list]
    idx_single_objective = np.where(is_single_objective)[0]

    # For plotting best points per graph, find indices
    # For x (Average Steps), best is max x
    idx_best_x = np.argmax(np.where(is_2016_steps, avg_steps_list, -np.inf))
    # For y (Weighted Depth Metric), best is min y
    idx_best_y = np.argmin(np.where(is_2016_steps, weighted_depth_list, np.inf))
    # For z (Total Switching Actions), best is min z
    idx_best_z = np.argmin(np.where(is_2016_steps, total_actions_list, np.inf))

    # --- Set up matplotlib parameters for a more scientific look ---
    plt.rcParams.update(
        {
            "font.size": 14,
            "figure.figsize": (20, 6),
            "axes.grid": True,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "font.family": "serif",
        }
    )

    # --- 2D Projections ---
    fig2, axs = plt.subplots(1, 3, figsize=(20, 6))
    fig2.suptitle("2D Projections of Pareto Frontier in Power System Domain")

    # General plotting color
    general_color = 'blue'

    # First subplot: x vs y (Average Steps vs Weighted Depth Metric)
    # Plot blue points first
    axs[0].scatter(
        avg_steps_list,
        weighted_depth_list,
        color=general_color,
        alpha=alpha_values,
        label="Pareto Points from Return Domain",
    )

    # Plot single objective points in black over blue points
    axs[0].scatter(
        avg_steps_list[idx_single_objective],
        weighted_depth_list[idx_single_objective],
        color='black',
        edgecolors='black',
        marker='o',
        s=100,
        label='S-O Point',
    )

    # Plot best point in red over others
    axs[0].scatter(
        avg_steps_list[idx_best_x],
        weighted_depth_list[idx_best_y],
        color='red',
        edgecolors='black',
        marker='o',
        s=100,
        label='Dominating M-O Case',
    )

    axs[0].set_xlabel("Average Steps")
    axs[0].set_ylabel("Weighted Depth Metric")
    axs[0].invert_yaxis()

    # Second subplot: x vs z (Average Steps vs Total Switching Actions)
    # Plot blue points first
    axs[1].scatter(
        avg_steps_list,
        total_actions_list,
        color=general_color,
        alpha=alpha_values,
        label="Pareto Points from Return Domain",
    )

    # Plot single objective points in black over blue points
    axs[1].scatter(
        avg_steps_list[idx_single_objective],
        total_actions_list[idx_single_objective],
        color='black',
        edgecolors='black',
        marker='o',
        s=100,
        label='S-O Case',
    )

    # Plot best point in red over others
    axs[1].scatter(
        avg_steps_list[idx_best_z],
        total_actions_list[idx_best_z],
        color='red',
        edgecolors='black',
        marker='o',
        s=100,
        label='Dominating M-O Case',
    )

    axs[1].set_xlabel("Average Steps")
    axs[1].set_ylabel("Total Switching Actions")
    axs[1].invert_yaxis()

    # Third subplot: y vs z (Weighted Depth Metric vs Total Switching Actions)
    # Plot blue points first
    axs[2].scatter(
        weighted_depth_list,
        total_actions_list,
        color=general_color,
        alpha=alpha_values,
        label="Pareto Points from Return Domain",
    )

    # Plot single objective points in black over blue points
    axs[2].scatter(
        weighted_depth_list[idx_single_objective],
        total_actions_list[idx_single_objective],
        color='black',
        edgecolors='black',
        marker='o',
        s=100,
        label='S-O Case',
    )

    # Plot best point in red over others
    axs[2].scatter(
        weighted_depth_list[idx_best_y],
        total_actions_list[idx_best_y],
        color='red',
        edgecolors='black',
        marker='o',
        s=100,
        label='Dominating M-O Case',
    )

    axs[2].set_xlabel("Weighted Depth Metric")
    axs[2].set_ylabel("Total Switching Actions")
    axs[2].invert_xaxis()
    axs[2].invert_yaxis()

    for ax in axs:
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.show()


def calculate_hypervolume_and_sparsity(json_data):
    """
    Given JSON data containing CCS lists and coordinates, calculates 2D and 3D Hypervolume
    and Sparsity using the pre-existing utility functions.
    """
    # Extract the CCS list (assuming last element holds relevant data)

    ccs_list = json_data["ccs_list"][-1]
    if not ccs_list:
        return None, None  # Handle cases where CCS list is missing or empty

    # Extract coordinates
    x_all, y_all, z_all = extract_coordinates(ccs_list)

    # Calculate 2D hypervolumes (XY plane), sparsity, and 3D hypervolume
    reference_point_3d = (min(x_all), min(y_all), min(z_all))
    pareto_points_3d = np.column_stack((x_all, y_all, z_all))

    # Calculate 3D Hypervolume and Sparsity
    hv_3d = calculate_3d_hypervolume(pareto_points_3d, reference_point_3d)
    sparsity_3d = calculate_3d_sparsity(pareto_points_3d)

    return hv_3d, sparsity_3d


def compare_hv_with_combined_boxplots(base_path, scenario):
    """
    Compares the HV (Hypervolume), Sparsity, and Min/Max Returns metrics for the settings:
    - Baseline
    - Full Reuse
    - Partial Reuse
    Generates two separate boxplots:
    1. Boxplot showing Hypervolume and Sparsity side-by-side for each setting.
    2. Boxplot showing Min/Max Returns for X, Y, Z coordinates side-by-side for each setting.
    3. A plot showing the average delta (range) of returns for each return dimension and setting.
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Your existing imports and utility functions

    if scenario == 'Reuse':
        settings = ["Baseline", "Full", "Partial", "Baseline_reduced", "Full_reduced", "Partial_reduced", "Baseline_min", "Full_min", "Partial_min"]
    elif scenario == 'Opponent':
        settings = ['Baseline', 'Opponent']

    reuse_paths = {
        "Baseline": os.path.join(base_path, "Baseline"),
        "Full": os.path.join(base_path, "Full_Reuse"),
        "Partial": os.path.join(base_path, "Partial_Reuse"),
        "Baseline_reduced": os.path.join(base_path, "med_learning_none"),
        "Full_reduced": os.path.join(base_path, "med_learning_full"),
        "Partial_reduced": os.path.join(base_path, "med_learning_partial"),
        "Baseline_min": os.path.join(base_path, "min_learning_none"),
        "Full_min": os.path.join(base_path, "min_learning_full"),
        "Partial_min": os.path.join(base_path, "min_learning_partial"),
    }
    opponent_paths = {
        "Baseline": os.path.join(base_path, "Baseline"),
        "Opponent": os.path.join(base_path, "Opponen")
    }

    # Initialize dictionary to store results
    hv_metrics = {"Seed": [], "Setting": [], "Metric": [], "Value": []}
    return_metrics = {"Seed": [], "Setting": [], "Metric": [], "Value": []}
    # Loop through the settings and load corresponding JSON files
    for setting in settings:
        if scenario == 'Reuse':
            path = reuse_paths[setting]
        elif scenario == 'Opponent':
            path = opponent_paths[setting]

        # Load the JSON log files for this setting
        json_files = [f for f in os.listdir(path) if f.startswith("morl_logs")]
        seed = 0
        for json_file in json_files:
            file_path = os.path.join(path, json_file)

            # Load the JSON data
            data = load_json_data(json_path=file_path)
            print(f"Processing file: {file_path}")

            # Extract coordinates and calculate hypervolume, sparsity, and returns
            hv_3d, sparsity_3d = calculate_hypervolume_and_sparsity(data)
            x_all, y_all, z_all = extract_coordinates(data['ccs_list'][-1])
            
            # Min/Max Returns
            min_return_x, max_return_x = min(x_all), max(x_all)
            min_return_y, max_return_y = min(y_all), max(y_all)
            min_return_z, max_return_z = min(z_all), max(z_all)

            if hv_3d is not None and sparsity_3d is not None:
                # Store Hypervolume and Sparsity in the hv_metrics dictionary
                hv_metrics["Seed"].append(seed)
                hv_metrics["Setting"].append(setting)
                hv_metrics["Metric"].append("Hypervolume")
                hv_metrics["Value"].append(hv_3d)
                
                hv_metrics["Seed"].append(seed)
                hv_metrics["Setting"].append(setting)
                hv_metrics["Metric"].append("Sparsity")
                hv_metrics["Value"].append(sparsity_3d)

                # Store Min/Max Returns for X, Y, Z coordinates in return_metrics dictionary
                # Include seed information
                return_metrics["Seed"].append(seed)
                return_metrics["Setting"].append(setting)
                return_metrics["Metric"].append("Max Return X")
                return_metrics["Value"].append(max_return_x)
                
                return_metrics["Seed"].append(seed)
                return_metrics["Setting"].append(setting)
                return_metrics["Metric"].append("Min Return X")
                return_metrics["Value"].append(min_return_x)

                return_metrics["Seed"].append(seed)
                return_metrics["Setting"].append(setting)
                return_metrics["Metric"].append("Min Return Y")
                return_metrics["Value"].append(min_return_y)
                
                return_metrics["Seed"].append(seed)
                return_metrics["Setting"].append(setting)
                return_metrics["Metric"].append("Max Return Y")
                return_metrics["Value"].append(max_return_y)

                return_metrics["Seed"].append(seed)
                return_metrics["Setting"].append(setting)
                return_metrics["Metric"].append("Min Return Z")
                return_metrics["Value"].append(min_return_z)
                
                return_metrics["Seed"].append(seed)
                return_metrics["Setting"].append(setting)
                return_metrics["Metric"].append("Max Return Z")
                return_metrics["Value"].append(max_return_z)
                
            seed += 1

    # Convert the dictionaries to DataFrames for easier comparison and visualization
    # Convert the dictionaries to DataFrames for easier comparison and visualization
    df_hv_metrics = pd.DataFrame(hv_metrics)
    df_return_metrics = pd.DataFrame(return_metrics)

    # Add 'Method' and 'Learning' columns based on 'Setting'
    setting_to_method_and_learning = {
        'Baseline': ('Baseline', 'full training'),
        'Baseline_reduced': ('Baseline', '75% training'),
        'Baseline_min': ('Baseline', '50% training'),
        'Full': ('Full Reuse', 'full training'),
        'Full_reduced': ('Full Reuse', '75% training'),
        'Full_min': ('Full Reuse', '50% training'),
        'Partial': ('Partial Reuse', 'full training'),
        'Partial_reduced': ('Partial Reuse', '75% training'),
        'Partial_min': ('Partial Reuse', '50% training'),
    }

    # For df_hv_metrics
    df_hv_metrics['Method'] = df_hv_metrics['Setting'].map(lambda x: setting_to_method_and_learning[x][0])
    df_hv_metrics['Learning'] = df_hv_metrics['Setting'].map(lambda x: setting_to_method_and_learning[x][1])

    # **Add 'Method' and 'Learning' columns to df_return_metrics**
    df_return_metrics['Method'] = df_return_metrics['Setting'].map(lambda x: setting_to_method_and_learning[x][0])
    df_return_metrics['Learning'] = df_return_metrics['Setting'].map(lambda x: setting_to_method_and_learning[x][1])


    # Compute mean and std for hv_metrics
    hv_stats = df_hv_metrics.groupby(['Setting', 'Metric'])['Value'].agg(['mean', 'std']).reset_index()
    # Compute mean and std for return_metrics
    return_stats = df_return_metrics.groupby(['Setting', 'Metric'])['Value'].agg(['mean', 'std']).reset_index()

    # Print the results
    print("\nHypervolume and Sparsity Metrics (Mean and Std):")
    print(hv_stats.to_string(index=False))
    print("\nMin/Max Returns (Mean and Std):")
    print(return_stats.to_string(index=False))

    # --- Adjusted Coloring Starts Here ---

    # Define custom colors for Methods to match the desired color scheme
    method_palette = {
        'Baseline': 'grey',
        'Full Reuse': 'red',  # light red
        'Partial Reuse': 'lightcoral',
    }

    # Boxplot for Hypervolume and Sparsity split into 3 subfigures
    learning_settings = ['full training', '75% training', '50% training']
    num_learning_settings = len(learning_settings)

    fig, axes = plt.subplots(1, num_learning_settings, figsize=(18, 6), sharey=True)

    for i, learning_setting in enumerate(learning_settings):
        ax = axes[i]
        data_subset = df_hv_metrics[df_hv_metrics['Learning'] == learning_setting]

        sns.boxplot(
            ax=ax,
            x='Metric',
            y='Value',
            hue='Method',
            data=data_subset,
            palette=method_palette,
            medianprops={'color': 'black'},
            whiskerprops={'color': 'black'},
            capprops={'color': 'black'},
            flierprops={'color': 'black', 'markeredgecolor': 'black'},
            showcaps=True
        )
        ax.set_title(f"Learning: {learning_setting}")
        ax.set_xlabel('Metric')

        # Limit the y-axis to 20
        ax.set_ylim(0, 20)

        # Adjust x-axis labels
        ax.set_xticklabels(data_subset['Metric'].unique(), rotation=0)

        if i == 0:
            ax.set_ylabel('Metric Value')
        else:
            ax.set_ylabel('')

        if i == num_learning_settings - 1:
            ax.legend(title='Method', loc='upper right', bbox_to_anchor=(1.15, 1))
        else:
            ax.legend_.remove()
    plt.tight_layout()
    plt.show()

    # Boxplot for Min/Max Returns with adjusted coloring
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="Metric",
        y="Value",
        hue="Method",
        data=df_return_metrics,
        palette=method_palette,
        medianprops={'color': 'black'},
        whiskerprops={'color': 'black'},
        capprops={'color': 'black'},
        flierprops={'color': 'black', 'markeredgecolor': 'black'},
        showcaps=True
    )

    plt.xticks(rotation=45)
    plt.title("Boxplot of Min/Max Returns (X, Y, Z) across Different Settings")
    plt.ylabel("Return Value")
    plt.xlabel("Metric")

    # Adjust legend location to be inside the plot
    plt.legend(title="Method", loc='upper right', bbox_to_anchor=(0.95, 0.95))
    plt.tight_layout()
    plt.show()

    # The rest of the function remains unchanged
    # ...

    # Return the DataFrames for further analysis
    return df_hv_metrics, df_return_metrics


def compare_hv_and_sparsity_with_separate_boxplots(base_path, scenario):
    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Paths for different scenarios (from your provided code)
    reuse_paths = {
        "Baseline": os.path.join(base_path, "Baseline"),
        "Full": os.path.join(base_path, "Full_Reuse"),
        "Partial": os.path.join(base_path, "Partial_Reuse"),
        "Baseline_reduced": os.path.join(base_path, "med_learning_none"),
        "Full_reduced": os.path.join(base_path, "med_learning_full"),
        "Partial_reduced": os.path.join(base_path, "med_learning_partial"),
        "Baseline_min": os.path.join(base_path, "min_learning_none"),
        "Full_min": os.path.join(base_path, "min_learning_full"),
        "Partial_min": os.path.join(base_path, "min_learning_partial"),
    }

    settings = ["Baseline", "Full", "Partial",
                "Baseline_reduced", "Full_reduced", "Partial_reduced",
                "Baseline_min", "Full_min", "Partial_min"]

    # Initialize dictionaries to store results (from your code)
    hv_metrics = {"Seed": [], "Setting": [], "Metric": [], "Value": []}
    return_metrics = {"Seed": [], "Setting": [], "Metric": [], "Value": []}

    # Loop through the settings and load corresponding JSON files
    for setting in settings:
        if scenario == 'Reuse':
            path = reuse_paths[setting]
        else:
            print(f"Scenario '{scenario}' not recognized.")
            continue

        # Load the JSON log files for this setting
        json_files = [f for f in os.listdir(path) if f.startswith("morl_logs")]
        seed = 0
        for json_file in json_files:
            file_path = os.path.join(path, json_file)

            # Load the JSON data
            data = load_json_data(json_path=file_path)
            print(f"Processing file: {file_path}")

            # Extract coordinates and calculate hypervolume and sparsity
            hv_3d, sparsity_3d = calculate_hypervolume_and_sparsity(data)
            x_all, y_all, z_all = extract_coordinates(data['ccs_list'][-1])

            # Min/Max Returns
            min_return_x, max_return_x = min(x_all), max(x_all)
            min_return_y, max_return_y = min(y_all), max(y_all)
            min_return_z, max_return_z = min(z_all), max(z_all)

            if hv_3d is not None and sparsity_3d is not None:
                # Store Hypervolume and Sparsity in the hv_metrics dictionary
                hv_metrics["Seed"].append(seed)
                hv_metrics["Setting"].append(setting)
                hv_metrics["Metric"].append("Hypervolume")
                hv_metrics["Value"].append(hv_3d)

                hv_metrics["Seed"].append(seed)
                hv_metrics["Setting"].append(setting)
                hv_metrics["Metric"].append("Sparsity")
                hv_metrics["Value"].append(sparsity_3d)

                # Store Min/Max Returns for X, Y, Z coordinates in return_metrics dictionary
                # Include seed information
                return_metrics["Seed"].append(seed)
                return_metrics["Setting"].append(setting)
                return_metrics["Metric"].append("Max Return X")
                return_metrics["Value"].append(max_return_x)

                return_metrics["Seed"].append(seed)
                return_metrics["Setting"].append(setting)
                return_metrics["Metric"].append("Min Return X")
                return_metrics["Value"].append(min_return_x)

                return_metrics["Seed"].append(seed)
                return_metrics["Setting"].append(setting)
                return_metrics["Metric"].append("Max Return Y")
                return_metrics["Value"].append(max_return_y)

                return_metrics["Seed"].append(seed)
                return_metrics["Setting"].append(setting)
                return_metrics["Metric"].append("Min Return Y")
                return_metrics["Value"].append(min_return_y)

                return_metrics["Seed"].append(seed)
                return_metrics["Setting"].append(setting)
                return_metrics["Metric"].append("Max Return Z")
                return_metrics["Value"].append(max_return_z)

                return_metrics["Seed"].append(seed)
                return_metrics["Setting"].append(setting)
                return_metrics["Metric"].append("Min Return Z")
                return_metrics["Value"].append(min_return_z)

            seed += 1

    # Convert the dictionaries to DataFrames for easier comparison and visualization
    df_hv_metrics = pd.DataFrame(hv_metrics)
    df_return_metrics = pd.DataFrame(return_metrics)

    # Add 'Method' and 'Learning' columns based on 'Setting' (from your code)
    setting_to_method_and_learning = {
        'Baseline': ('Baseline', 'full training'),
        'Baseline_reduced': ('Baseline', '75% training'),
        'Baseline_min': ('Baseline', '50% training'),
        'Full': ('Full Reuse', 'full training'),
        'Full_reduced': ('Full Reuse', '75% training'),
        'Full_min': ('Full Reuse', '50% training'),
        'Partial': ('Partial Reuse', 'full training'),
        'Partial_reduced': ('Partial Reuse', '75% training'),
        'Partial_min': ('Partial Reuse', '50% training'),
    }

    # For df_hv_metrics
    df_hv_metrics['Method'] = df_hv_metrics['Setting'].map(lambda x: setting_to_method_and_learning[x][0])
    df_hv_metrics['Learning'] = df_hv_metrics['Setting'].map(lambda x: setting_to_method_and_learning[x][1])

    # Add 'Method' and 'Learning' columns to df_return_metrics
    df_return_metrics['Method'] = df_return_metrics['Setting'].map(lambda x: setting_to_method_and_learning[x][0])
    df_return_metrics['Learning'] = df_return_metrics['Setting'].map(lambda x: setting_to_method_and_learning[x][1])

    # --- Adjusted Coloring Starts Here ---

    # Define custom colors for Methods to match the desired color scheme (from your code)
    method_palette = {
        'Baseline': 'grey',
        'Full Reuse': 'red',
        'Partial Reuse': 'lightcoral',
    }

    # Separate DataFrames for Hypervolume and Sparsity
    df_hv = df_hv_metrics[df_hv_metrics['Metric'] == 'Hypervolume'].copy()
    df_sparsity = df_hv_metrics[df_hv_metrics['Metric'] == 'Sparsity'].copy()

    # Set up matplotlib parameters for a consistent look
    plt.rcParams.update({
        'font.size': 14,
        'figure.figsize': (12, 8),
        'axes.grid': True,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'font.family': 'serif',
    })

    sns.set_style("whitegrid")

    # Create separate boxplots for Hypervolume and Sparsity

    # 1. Hypervolume Boxplot
    plt.figure(figsize=(12, 8))
    ax_hv = sns.boxplot(
        data=df_hv,
        x='Learning',
        y='Value',
        hue='Method',
        palette=method_palette
    )
    ax_hv.set_ylim(top=100, bottom=0)
    plt.title('Hypervolume Comparison')
    plt.xlabel('Learning Setting')
    plt.ylabel('Hypervolume')
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # 2. Sparsity Boxplot
    plt.figure(figsize=(12, 8))
    ax_sparsity = sns.boxplot(
        data=df_sparsity,
        x='Learning',
        y='Value',
        hue='Method',
        palette=method_palette
    )
    ax_sparsity.set_ylim(top=25)
    plt.title('Sparsity Comparison')
    plt.xlabel('Learning Setting')
    plt.ylabel('Sparsity')
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Optionally, return the DataFrames
    return df_hv, df_sparsity



def compare_policies_weights_all_seeds(base_path, scenario):
    """
    Compares policies across all seeds for different scenarios and plots the results.
    Modifications:
    - Titles and setting descriptions are made dependent on the scenario.
    - For scenario 'Opponent', title is '... under contingencies', settings are 'Baseline' and 'moderate contingencies'.
    - For scenario 'Time', title is '... under time constraints', settings are 'Baseline' and 'moderate time constraints'.
    - For scenario 'max_rho', title is '... for unknown max line loading', settings are 'Baseline' and 'rho 90%'.
    """
    

    # Paths for different scenarios
    opponent_paths = {
        "Baseline": os.path.join(base_path, "Baseline"),
        "moderate contingencies": os.path.join(base_path, "op_normal"),
        "high contingencies": os.path.join(base_path, "op_hard")
    }

    time_paths = {
        "Baseline": os.path.join(base_path, "Baseline"),
        "moderate time constraints": os.path.join(base_path, "med_learning_none"),
        "high time constraints": os.path.join(base_path, "min_time_none")
    }

    rho_paths = {
        "Baseline": os.path.join(base_path, "Baseline"),
        "rho 70%": os.path.join(base_path, 'Rho70'),
        "rho 50%": os.path.join(base_path, 'Rho50'),
        "rho 00%": os.path.join(base_path, 'Rho00'),
        # Add more rho settings here if needed
    }
    
    # Determine settings and titles based on scenario
    if scenario == 'Opponent':
        settings = ['Baseline', 'moderate contingencies',"high contingencies" ]
        paths = opponent_paths
        title = 'Single- and Multi-Objective Solutions under contingencies'
    elif scenario == "Time":
        settings = ['Baseline', 'moderate time constraints', "high time constraints"]
        paths = time_paths
        title = 'Single- and Multi-Objective Solutions under time constraints'
    elif scenario == "Max_rho":
        settings = ['Baseline', 'rho 70%', 'rho 50%','rho 00%']
        paths = rho_paths
        title = 'Single- and Multi-Objective Solutions for unknown max line loading'
    else:
        raise ValueError("Invalid scenario provided.")

    results = []
    print('settings')
    print(settings)
    # Loop over each setting
    for setting in settings:
        path = paths[setting]
        seed_paths = []
        print(path)
        for seed in range(20):  # Adjust the range as needed
            seed_file = f"morl_logs_seed_{seed}.json"
            seed_path = os.path.join(path, seed_file)
            seed_paths.append(seed_path)

        print(f"Processing for setting: {setting} with seed paths. {seed_paths}")

        # Process data for this scenario and setting
        df_ccs_matching_seeds = process_data(
            seed_paths=seed_paths, wrapper='ols', output_dir=path)

        # For each seed, process the data
        for seed, group in df_ccs_matching_seeds.groupby('seed'):
            # S-O Case [1,0,0]
            default_runs = group[group['Weights'].apply(
                lambda w: w == [1.0, 0.0, 0.0])]
            default_run = default_runs.loc[
                default_runs['test_chronic_0'].apply(
                    lambda x: x['test_steps']).idxmax()
            ]

            # M-O Case (exclude [1.0, 0.0, 0.0])
            non_default_runs = group[group['Weights'].apply(
                lambda w: w != [1.0, 0.0, 0.0])]

            # Extract 'test_steps' and 'test_actions' for both chronic_0 and chronic_1
            non_default_runs['avg_test_steps'] = non_default_runs.apply(
                lambda row: (row['test_chronic_0']['test_steps'] +
                             row['test_chronic_1']['test_steps']) / 2, axis=1
            )

            non_default_runs['test_actions'] = non_default_runs.apply(
                lambda row: len(row['test_chronic_0']['test_actions']) +
                len(row['test_chronic_1']['test_actions']), axis=1
            )

            # Identify the top run with the highest average test_steps,
            # using fewer actions to break ties
            best_run = non_default_runs.sort_values(
                by=['avg_test_steps', 'test_actions'],
                ascending=[False, True]
            ).iloc[0]

            # Best run data
            best_run_actions = best_run['test_actions']
            best_run_steps = best_run['avg_test_steps']

            # Save the results for the M-O Case
            results.append({
                'Seed': seed,
                'Setting': setting,
                'Run Type': 'M-O Case',
                'Switching Actions': best_run_actions,
                'Steps': best_run_steps
            })

            # S-O Case data
            default_run_actions = (
                len(default_run['test_chronic_0']['test_actions']) +
                len(default_run['test_chronic_1']['test_actions'])
            )
            default_run_steps = (
                default_run['test_chronic_0']['test_steps'] +
                default_run['test_chronic_1']['test_steps']
            ) / 2

            results.append({
                'Seed': seed,
                'Setting': setting,
                'Run Type': 'S-O Case',
                'Switching Actions': default_run_actions,
                'Steps': default_run_steps
            })

    # Convert the results into a DataFrame for plotting
    df_results = pd.DataFrame(results)

    # Set up matplotlib parameters for a more scientific look
    plt.rcParams.update({
        'font.size': 14,
        'figure.figsize': (14, 8),
        'axes.grid': True,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 12,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'font.family': 'serif',
    })

    # Plotting
    sns.set_style("whitegrid")

    # Define custom colors
    box_colors = {'S-O Case': 'grey', 'M-O Case': 'lightcoral'}  # light red
    dot_colors = {'S-O Case': 'black', 'M-O Case': 'red'}

    # Define hue order to ensure consistent ordering
    hue_order = ['S-O Case', 'M-O Case']

    # Boxplot of Switching Actions
    plt.figure(figsize=(14, 8))
    ax1 = sns.boxplot(
        x='Setting', y='Switching Actions', hue='Run Type',
        data=df_results, hue_order=hue_order, palette=box_colors, width=0.6,
        medianprops={'color': 'black'},
        whiskerprops={'color': 'black'},
        capprops={'color': 'black'},
        flierprops={'color': 'black', 'markeredgecolor': 'black'},
        showcaps=True
    )

    # Swarmplot with edge colors
    sns.swarmplot(
        x='Setting', y='Switching Actions', hue='Run Type',
        data=df_results, hue_order=hue_order, dodge=True, palette=dot_colors, size=6, alpha=0.7,
        edgecolor='black', linewidth=0.5, ax=ax1
    )

    plt.title(title)
    plt.ylabel('Number of Switching Actions')
    plt.xlabel('Setting')

    # Adjust legend to prevent duplicates
    handles, labels = ax1.get_legend_handles_labels()
    n = len(hue_order)
    ax1.legend(handles[:n], labels[:n], title='Run Type',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    # Boxplot of Steps
    plt.figure(figsize=(14, 8))
    ax2 = sns.boxplot(
        x='Setting', y='Steps', hue='Run Type',
        data=df_results, hue_order=hue_order, palette=box_colors, width=0.6,
        medianprops={'color': 'black'},
        whiskerprops={'color': 'black'},
        capprops={'color': 'black'},
        flierprops={'color': 'black', 'markeredgecolor': 'black'},
        showcaps=True
    )

    # Swarmplot with edge colors
    sns.swarmplot(
        x='Setting', y='Steps', hue='Run Type',
        data=df_results, hue_order=hue_order, dodge=True, palette=dot_colors, size=6, alpha=0.7,
        edgecolor='black', linewidth=0.5, ax=ax2
    )

    plt.title(title)
    plt.ylabel('Average Number of Steps')
    plt.xlabel('Setting')

    # Adjust legend to prevent duplicates
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[:n], labels[:n], title='Run Type',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    return df_results



def visualize_successful_weights(base_path, scenario, plot_option='combined'):
    """
    Visualizes the weight distributions of multi-objective policies that successfully
    complete the episode (steps=2016) for a given scenario.

    Display Options:
    - 'combined': For each weight, plot KDE and strip plots across all settings in one figure.
    - 'separate': For each setting, plot the KDE distributions of all weights in one figure.

    Parameters:
    - base_path: The base directory path containing the scenario data.
    - scenario: The scenario name ('Opponent', 'Time', or 'Max_rho').
    - plot_option: 'combined' or 'separate' (default is 'combined').

    The function processes the data for each setting in the scenario, extracts
    the weight vectors of successful M-O policies, and plots them using matplotlib
    and seaborn.
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import OrderedDict

    # Paths for different scenarios
    opponent_paths = {
        "Baseline": os.path.join(base_path, "Baseline"),
        "moderate contingencies": os.path.join(base_path, "op_normal"),
        "high contingencies": os.path.join(base_path, "op_hard")
    }

    time_paths = {
        "Baseline": os.path.join(base_path, "Baseline"),
        "moderate time constraints": os.path.join(base_path, "med_learning_none"),
        "high time constraints": os.path.join(base_path, "min_time_none")
    }

    rho_paths = {
        "Baseline": os.path.join(base_path, "Baseline"),
        "rho 70%": os.path.join(base_path, 'Rho70'),
        "rho 50%": os.path.join(base_path, 'Rho50'),
        "rho 00%": os.path.join(base_path, 'Rho00'),
        # Add more rho settings here if needed
    }

    # Determine settings and titles based on scenario
    if scenario == 'Opponent':
        settings = ["Baseline", "moderate contingencies", "high contingencies"]
        paths = opponent_paths
        scenario_title = 'Contingencies'
    elif scenario == "Time":
        settings = ["Baseline", "moderate learning constraints", "high learning constraints"]
        paths = time_paths
        scenario_title = 'Learning Constraints'
    elif scenario == "Max_rho":
        settings = ["Baseline", "rho 50%", "rho 00%"]
        paths = rho_paths
        scenario_title = 'Unknown Max Line Loading'
    else:
        raise ValueError("Invalid scenario provided.")

    results = []

    # Loop over each setting
    for setting in settings:
        path = paths[setting]
        seed_paths = []
        for seed in range(5):  # Adjust the range as needed
            seed_file = f"morl_logs_seed_{seed}.json"
            seed_path = os.path.join(path, seed_file)
            if os.path.exists(seed_path):
                seed_paths.append(seed_path)
            else:
                print(f"Seed file not found: {seed_path}")

        # Process data for this scenario and setting
        for seed_path in seed_paths:
            print(f"Processing for setting: {setting} with seed path: {seed_path}")
            # Load the data (implement your own load function or adjust accordingly)
            df_ccs_matching_seeds = process_data(
                seed_paths=[seed_path], wrapper='ols', output_dir=path)

            # For each run in the data
            for index, row in df_ccs_matching_seeds.iterrows():
                weights = row['Weights']
                # Check if it's a multi-objective policy (weights not equal to [1.0, 0.0, 0.0])
                if weights != [1.0, 0.0, 0.0]:
                    # Check if the policy completes the episode (steps=2016) for both chronic_0 and chronic_1
                    steps_chronic_0 = row['test_chronic_0']['test_steps']
                    steps_chronic_1 = row['test_chronic_1']['test_steps']
                    if steps_chronic_0 == 2016 and steps_chronic_1 == 2016:
                        avg_steps = (steps_chronic_0 + steps_chronic_1) / 2
                        results.append({
                            'Seed': row['seed'],
                            'Setting': setting,
                            'Weights': weights,
                            'Average Steps': avg_steps
                        })

    # Convert the results into a DataFrame
    df_results = pd.DataFrame(results)

    if df_results.empty:
        print("No successful M-O policies found for the given scenario.")
        return

    # Convert the list of weights to columns for easier plotting
    weights_array = np.array(df_results['Weights'].tolist())
    df_results['Weight 1'] = weights_array[:, 0]
    df_results['Weight 2'] = weights_array[:, 1]
    df_results['Weight 3'] = weights_array[:, 2]

    # Ensure weights are within the range 0-1
    df_results['Weight 1'] = df_results['Weight 1'].clip(0, 1)
    df_results['Weight 2'] = df_results['Weight 2'].clip(0, 1)
    df_results['Weight 3'] = df_results['Weight 3'].clip(0, 1)

    # Set up matplotlib parameters for a consistent look
    plt.rcParams.update({
        'font.size': 12,
        'axes.grid': True,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'font.family': 'serif',
    })

    sns.set_style("whitegrid")
    
    
    results = []
    unsuccessful_results = []

    # Loop over each setting
    for setting in settings:
        path = paths[setting]
        seed_paths = []
        for seed in range(5):  # Adjust the range as needed
            seed_file = f"morl_logs_seed_{seed}.json"
            seed_path = os.path.join(path, seed_file)
            if os.path.exists(seed_path):
                seed_paths.append(seed_path)
            else:
                print(f"Seed file not found: {seed_path}")

        # Process data for this scenario and setting
        for seed_path in seed_paths:
            print(f"Processing for setting: {setting} with seed path: {seed_path}")
            # Load the data (implement your own load function or adjust accordingly)
            df_ccs_matching_seeds = process_data(
                seed_paths=[seed_path], wrapper='ols', output_dir=path)

            # For each run in the data
            for index, row in df_ccs_matching_seeds.iterrows():
                weights = row['Weights']
                steps_chronic_0 = row['test_chronic_0']['test_steps']
                steps_chronic_1 = row['test_chronic_1']['test_steps']
                avg_steps = (steps_chronic_0 + steps_chronic_1) / 2

                if weights != [1.0, 0.0, 0.0]:
                    if steps_chronic_0 == 2016 and steps_chronic_1 == 2016:
                        results.append({
                            'Seed': row['seed'],
                            'Setting': setting,
                            'Weights': weights,
                            'Average Steps': avg_steps
                        })
                    else:
                        unsuccessful_results.append({
                            'Seed': row['seed'],
                            'Setting': setting,
                            'Weights': weights,
                            'Average Steps': avg_steps,
                            'Steps Chronic 0': steps_chronic_0,
                            'Steps Chronic 1': steps_chronic_1
                        })

    # Convert the results into DataFrames
    df_results = pd.DataFrame(results)
    df_unsuccessful = pd.DataFrame(unsuccessful_results)

    if df_results.empty:
        print("No successful M-O policies found for the given scenario.")
        return

    if df_unsuccessful.empty:
        print("No unsuccessful M-O policies found for the given scenario.")
    else:
        print("Unsuccessful M-O policies found for the given scenario.")

    # Convert the list of weights to columns for easier plotting in successful policies
    weights_array = np.array(df_results['Weights'].tolist())
    df_results['Weight 1'] = weights_array[:, 0]
    df_results['Weight 2'] = weights_array[:, 1]
    df_results['Weight 3'] = weights_array[:, 2]

    # Convert the list of weights to columns for easier plotting in unsuccessful policies
    if not df_unsuccessful.empty:
        weights_array_unsuccessful = np.array(df_unsuccessful['Weights'].tolist())
        df_unsuccessful['Weight 1'] = weights_array_unsuccessful[:, 0]
        df_unsuccessful['Weight 2'] = weights_array_unsuccessful[:, 1]
        df_unsuccessful['Weight 3'] = weights_array_unsuccessful[:, 2]
        

    if plot_option == 'combined':
        # Option 1: Combined KDE and Strip Plots by Weight
        # Create a color palette for the settings
        unique_settings = df_results['Setting'].unique()
        palette = sns.color_palette("tab10", len(unique_settings))
        setting_palette = dict(zip(unique_settings, palette))

        # Figure 1: KDE Plots for Each Weight
        weights = ['Weight 1', 'Weight 2', 'Weight 3']
        weight_titles = ['L2RPN', 'Actions', 'Depth']

        fig_kde, axes_kde = plt.subplots(1, 3, figsize=(18, 6))

        # List to store maximum y-values
        max_y_values = []

        for i, weight in enumerate(weights):
            ax = axes_kde[i]
            # Plot KDE
            sns.kdeplot(
                data=df_results,
                x=weight,
                hue='Setting',
                palette=setting_palette,
                ax=ax,
                common_norm=False,
                fill=True,
                clip=(0, 1),
                legend=False  # We'll add a single legend later
            )

            ax.set_title(f'Distribution of {weight_titles[i]}')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Density')

            # Collect the maximum y-value for this subplot
            y_min, y_max = ax.get_ylim()
            max_y_values.append(y_max)

        # Set the same y-limit for all subplots
        common_y_max = max(max_y_values)
        for ax in axes_kde:
            ax.set_ylim(0, common_y_max)

        # Create a common legend
        handles, labels = axes_kde[-1].get_legend_handles_labels()
        # Remove duplicates while preserving order
        legend_data = OrderedDict()
        for handle, label in zip(handles, labels):
            if label not in legend_data:
                legend_data[label] = handle

        fig_kde.legend(
            legend_data.values(),
            legend_data.keys(),
            title='Setting',
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )

        fig_kde.suptitle(f'KDE Plots by Weight - {scenario_title}', fontsize=16)
        fig_kde.tight_layout(rect=[0, 0, 0.85, 0.95])  # Adjust layout to make room for the legend and title
        plt.show()

        # Figure 2: Stripplots for Each Weight
        fig_strip, axes_strip = plt.subplots(1, 3, figsize=(18, 6))

        for i, weight in enumerate(weights):
            ax = axes_strip[i]
            sns.stripplot(
                data=df_results,
                x=weight,
                y='Setting',  # Plot horizontally grouped by setting
                hue='Setting',
                palette=setting_palette,
                ax=ax,
                dodge=False,
                size=8,  # More expressive points
                alpha=0.7,
                jitter=True,
                legend=False  # We'll add a single legend later
            )

            ax.set_title(f'Weight Values of {weight_titles[i]}')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 2)
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Setting')

        # Create a common legend for the stripplots
        handles, labels = axes_strip[-1].get_legend_handles_labels()
        # Remove duplicates while preserving order
        legend_data_strip = OrderedDict()
        for handle, label in zip(handles, labels):
            if label not in legend_data_strip:
                legend_data_strip[label] = handle

        fig_strip.legend(
            legend_data_strip.values(),
            legend_data_strip.keys(),
            title='Setting',
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )

        fig_strip.suptitle(f'Strip Plots by Weight - {scenario_title}', fontsize=16)
        fig_strip.tight_layout(rect=[0, 0, 0.85, 0.95])  # Adjust layout to make room for the legend and title
        plt.show()

    elif plot_option == 'separate':
        # Option 2: Separate KDE Plots by Setting - Successful Cases
        num_settings = len(settings)

        # Plot for Successful Cases
        fig_success_kde, axes_success_kde = plt.subplots(1, num_settings, figsize=(6 * num_settings, 6), sharex=True, sharey=True)
        if num_settings == 1:
            axes_success_kde = [axes_success_kde]  # Ensure axes_success_kde is iterable when there's only one setting

        for i, setting in enumerate(settings):
            df_setting_successful = df_results[df_results['Setting'] == setting]

            if df_setting_successful.empty:
                print(f"No successful data available for setting: {setting}")
                continue

            print(setting)
            weights = ['Weight 1', 'Weight 2', 'Weight 3']
            weight_titles = ['L2RPN', 'Actions', 'Depth']

            # Plot successful KDE
            for weight, weight_title in zip(weights, weight_titles):
                sns.kdeplot(
                    data=df_setting_successful,
                    x=weight,
                    label=f'Successful - {weight_title}',
                    fill=True,
                    clip=(0, 1),
                    common_norm=False,
                    alpha=0.5,
                    ax=axes_success_kde[i]
                )

            axes_success_kde[i].set_title(f'{setting} - Successful')
            axes_success_kde[i].set_xlabel('Weight Value')
            axes_success_kde[i].set_ylabel('Density')
            axes_success_kde[i].set_xlim(0, 1)
            axes_success_kde[i].set_ylim(0, 2)
            axes_success_kde[i].legend(title='Rewards')

        plt.tight_layout()
        plt.show()

        # Plot for Unsuccessful Cases
        fig_unsuccess_kde, axes_unsuccess_kde = plt.subplots(1, num_settings, figsize=(6 * num_settings, 6), sharex=True, sharey=True)
        if num_settings == 1:
            axes_unsuccess_kde = [axes_unsuccess_kde]  # Ensure axes_unsuccess_kde is iterable when there's only one setting

        for i, setting in enumerate(settings):
            df_setting_unsuccessful = df_unsuccessful[df_unsuccessful['Setting'] == setting]

            if df_setting_unsuccessful.empty:
                print(f"No unsuccessful data available for setting: {setting}")
                continue

            weights = ['Weight 1', 'Weight 2', 'Weight 3']
            weight_titles = ['L2RPN', 'Actions', 'Depth']

            # Plot unsuccessful KDE
            for weight, weight_title in zip(weights, weight_titles):
                sns.kdeplot(
                    data=df_setting_unsuccessful,
                    x=weight,
                    label=f'Unsuccessful - {weight_title}',
                    fill=True,
                    clip=(0, 1),
                    common_norm=False,
                    alpha=0.5,
                    ax=axes_unsuccess_kde[i]
                )

            axes_unsuccess_kde[i].set_title(f'{setting} - Unsuccessful')
            axes_unsuccess_kde[i].set_xlabel('Weight Value')
            axes_unsuccess_kde[i].set_ylabel('Density')
            axes_unsuccess_kde[i].set_xlim(0, 1)
            axes_unsuccess_kde[i].set_ylim(0, 2)
            axes_unsuccess_kde[i].legend(title='Rewards')

        plt.tight_layout()
        plt.show()

        # Separate Histogram Plots by Setting with Parallel Bins for Each Reward
        fig_hist, axes_hist = plt.subplots(2, num_settings, figsize=(6 * num_settings, 12), sharex=True, sharey=True)
        if num_settings == 1:
            axes_hist = [axes_hist]  # Ensure axes_hist is iterable when there's only one setting

        for i, setting in enumerate(settings):
            df_setting_successful = df_results[df_results['Setting'] == setting]
            df_setting_unsuccessful = df_unsuccessful[df_unsuccessful['Setting'] == setting]

            if not df_setting_successful.empty:
                # Plot successful histograms with parallel bins
                for weight, weight_title in zip(weights, weight_titles):
                    sns.histplot(
                        data=df_setting_successful.melt(value_vars=weights, var_name="Weight Type", value_name="Weight Value"),
                        x="Weight Value",
                        hue="Weight Type",
                        multiple="dodge",
                        bins=5,
                        palette="Blues",
                        alpha=0.5,
                        ax=axes_hist[0][i]
                    )

                axes_hist[0][i].set_title(f'{setting} - Successful')
                axes_hist[0][i].set_xlabel('Weight Value')
                axes_hist[0][i].set_ylabel('Frequency')
                axes_hist[0][i].set_xlim(0, 1)
                axes_hist[0][i].legend(title='Weight Types')

            if not df_setting_unsuccessful.empty:
                # Plot unsuccessful histograms with parallel bins
                for weight, weight_title in zip(weights, weight_titles):
                    sns.histplot(
                        data=df_setting_unsuccessful.melt(value_vars=weights, var_name="Weight Type", value_name="Weight Value"),
                        x="Weight Value",
                        hue="Weight Type",
                        multiple="dodge",
                        bins=5,
                        palette="Reds",
                        alpha=0.5,
                        ax=axes_hist[1][i]
                    )

                axes_hist[1][i].set_title(f'{setting} - Unsuccessful')
                axes_hist[1][i].set_xlabel('Weight Value')
                axes_hist[1][i].set_ylabel('Frequency')
                axes_hist[1][i].set_xlim(0, 1)
                axes_hist[1][i].legend(title='Weight Types')

        plt.tight_layout()
        plt.show()

        # Existing scatter plot logic remains unchanged
        unique_weights_all_settings = []  # To store unique weights for line plot

        for i, setting in enumerate(settings):
            df_setting_successful = df_results[df_results['Setting'] == setting]
            df_setting_unsuccessful = df_unsuccessful[df_unsuccessful['Setting'] == setting]

            if df_setting_successful.empty:
                print(f"No successful data available for setting: {setting}")
            else:
                print(setting)
                unique_weights = df_setting_successful[['Weight 1', 'Weight 2', 'Weight 3']].drop_duplicates()
                print("Unique successful weight vectors:")
                print(np.round(unique_weights, decimals=2))

                # Store unique weights with setting information for line plot
                for _, row in unique_weights.iterrows():
                    unique_weights_all_settings.append({
                        'Setting': setting,
                        'Weight 1': row['Weight 1'],
                        'Weight 2': row['Weight 2'],
                        'Weight 3': row['Weight 3']
                    })
                    
        # Create a DataFrame from unique weights for line plot
        df_unique_weights = pd.DataFrame(unique_weights_all_settings)

        # Define colors: black for baseline, use a color palette for the others
        palette = sns.color_palette("husl", len(df_unique_weights['Setting'].unique()))  # Generate a color palette
        setting_palette = dict(zip(df_unique_weights['Setting'].unique(), palette))      # Map settings to colors
        setting_palette['Baseline'] = 'black'  # Set baseline to black

        # Plotting unique successful weight vectors as scatter plots
        plt.figure(figsize=(12, 6))

        # Plot each weight dimension as scatter points, coloring according to the setting
        for index, row in df_unique_weights.iterrows():
            plt.scatter(['Weight 1', 'Weight 2', 'Weight 3'],
                        [row['Weight 1'], row['Weight 2'], row['Weight 3']],
                        color=setting_palette[row['Setting']],
                        label=row['Setting'] if row['Setting'] not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.title('Scatter Plot of Weight Vector Values Across Different Settings')
        plt.xlabel('Weight Dimension')
        plt.ylabel('Weight Value')
        plt.ylim(0, 1)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Create a legend and specify unique entries
        plt.legend(title='Settings', loc='upper right')

        plt.xticks(['Weight 1', 'Weight 2', 'Weight 3'])
        plt.tight_layout()
        plt.show()
        # Call the function with the DataFrames
        #plot_weight_distributions(df_results, df_unsuccessful)
        plot_steps_vs_weights_by_scenario(df_results, df_unsuccessful)




def plot_steps_vs_weights_by_scenario(df_results, df_unsuccessful):
    """
    Creates 3x3 scatter plots to show the relationship between the average number of steps and each weight dimension
    (Weight 1, Weight 2, and Weight 3) across different scenarios (Baseline, moderate contingencies, high contingencies).
    Each weight dimension corresponds to a reward: L2RPN, Actions, and Depth.

    Parameters:
    - df_results: A DataFrame containing successful policies, with columns for each weight dimension
                  and the average number of steps.
    - df_unsuccessful: A DataFrame containing unsuccessful policies, with columns for each weight dimension
                       and the average number of steps.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Define weight columns and their corresponding reward names
    weights = ['Weight 1', 'Weight 2', 'Weight 3']
    reward_titles = ['L2RPN', 'Actions', 'Depth']

    # Define scenarios
    scenarios = ['Baseline', 'moderate contingencies', 'high contingencies']

    # Set up a figure with a 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 18), sharex=True, sharey=True)
    
    for row, scenario in enumerate(scenarios):
        # Filter successful and unsuccessful data for each scenario
        df_results_scenario = df_results[df_results['Setting'] == scenario]
        df_unsuccessful_scenario = df_unsuccessful[df_unsuccessful['Setting'] == scenario]

        for col, (weight, reward_title) in enumerate(zip(weights, reward_titles)):
            # Plot successful policies
            sns.scatterplot(
                data=df_results_scenario,
                x=weight,
                y='Average Steps',
                hue='Setting',
                palette='viridis',
                ax=axes[row, col],
                marker='o',
                edgecolor='black',
                s=100,  # Size of successful markers
                alpha=0.7
            )

            # Plot unsuccessful policies
            if not df_unsuccessful_scenario.empty:
                sns.scatterplot(
                    data=df_unsuccessful_scenario,
                    x=weight,
                    y='Average Steps',
                    hue='Setting',
                    palette='coolwarm',
                    ax=axes[row, col],
                    marker='X',
                    edgecolor='red',
                    s=60,  # Size of unsuccessful markers
                    alpha=0.7
                )

            # Titles and labels for each subplot
            if row == 0:
                axes[row, col].set_title(f'{reward_title} Weight vs. Average Steps')
            if col == 0:
                axes[row, col].set_ylabel(f'{scenario}\nAverage Steps')
            axes[row, col].set_xlabel(f'{reward_title} Weight')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()





def compare_policies_weights(base_path, scenario):
    """
    Compares policies for the given scenario and plots the results using barplots.
    Modifications:
    - Adjusted the plotting code to make the number of bars per setting generic depending on the scenario.
    - Switched the 'x' and 'hue' parameters in the barplot to accommodate variable numbers of bars per setting.
    - Used the same coloring scheme as previously discussed.
    - Assigned different colors to M-O cases for different settings.
    - Included the weights of the best M-O alternative in the labels.
    """
    # Paths for different scenarios
    opponent_paths = {
        "Baseline": os.path.join(base_path, "Baseline"),
        "moderate contingencies": os.path.join(base_path, "op_normal"),
        "high contingencies": os.path.join(base_path, "op_hard")
    }

    time_paths = {
        "Baseline": os.path.join(base_path, "Baseline"),
        'moderate learning constraints': os.path.join(base_path, "med_time_none"),
        'high learning constraints': os.path.join(base_path, "min_time")
    }

    rho_paths = {
        "Baseline": os.path.join(base_path, "Baseline"),
        'rho 90%': os.path.join(base_path, 'rho90'),
        'rho 80%': os.path.join(base_path, 'rho80'),
        'rho 70%': os.path.join(base_path, 'rho70')
    }

    # Determine settings and titles based on scenario
    if scenario == 'Opponent':
        settings = ['Baseline', 'moderate contingencies', "high contingencies"]
        paths = opponent_paths
        title = 'Single- and Multi-Objective Solutions under contingencies'
    elif scenario == "Time":
        settings = ['Baseline', 'moderate learning constraints', 'high learning constraints']
        paths = time_paths
        title = 'Single- and Multi-Objective Solutions under learning constraints'
    elif scenario == "Max_rho":
        settings = ['Baseline', 'rho 90%', 'rho 80%', 'rho 70%']
        paths = rho_paths
        title = 'Single- and Multi-Objective Solutions for unknown max line loading'
    else:
        raise ValueError("Invalid scenario provided.")

    results = []

    # Loop over each setting
    for setting in settings:
        path = paths[setting]
        seed_paths = []
        for seed in range(5):  # Adjust the range as needed
            seed_file = f"morl_logs_seed_{seed}.json"
            seed_path = os.path.join(path, seed_file)
            seed_paths.append(seed_path)

        seed_paths = [seed_paths[0]]  # Only using seed 0
        print(f"Processing for setting: {setting} with seed path: {seed_paths[0]}")

        # Process data for this scenario and setting
        df_ccs_matching_seeds = process_data(
            seed_paths=seed_paths, wrapper='ols', output_dir=path)

        # For seed=0, process the data
        for seed, group in df_ccs_matching_seeds.groupby('seed'):
            # S-O Case [1,0,0]
            default_runs = group[group['Weights'].apply(
                lambda w: w == [1.0, 0.0, 0.0])]
            default_run = default_runs.loc[
                default_runs['test_chronic_0'].apply(
                    lambda x: x['test_steps']).idxmax()
            ]

            # M-O Case (exclude [1.0, 0.0, 0.0])
            non_default_runs = group[group['Weights'].apply(
                lambda w: w != [1.0, 0.0, 0.0])]

            # Extract 'test_steps' and 'test_actions' for both chronic_0 and chronic_1
            non_default_runs['avg_test_steps'] = non_default_runs.apply(
                lambda row: (row['test_chronic_0']['test_steps'] +
                             row['test_chronic_1']['test_steps']) / 2, axis=1
            )

            non_default_runs['test_actions'] = non_default_runs.apply(
                lambda row: len(row['test_chronic_0']['test_actions']) +
                len(row['test_chronic_1']['test_actions']), axis=1
            )

            # Identify the top run with the highest average test_steps,
            # using fewer actions to break ties
            best_run = non_default_runs.sort_values(
                by=['avg_test_steps', 'test_actions'],
                ascending=[False, True]
            ).iloc[0]

            # Best run data
            best_run_actions = best_run['test_actions']
            best_run_steps = best_run['avg_test_steps']
            best_run_weights = np.round(best_run['Weights'], 1)

            # Save the results for the M-O Case, including weights in the label
            results.append({
                'Seed': seed,
                'Setting': setting,
                'Run Type': f'M-O Case {best_run_weights}',
                'Switching Actions': best_run_actions,
                'Steps': best_run_steps
            })

            # S-O Case data
            default_run_actions = (
                len(default_run['test_chronic_0']['test_actions']) +
                len(default_run['test_chronic_1']['test_actions'])
            )
            default_run_steps = (
                default_run['test_chronic_0']['test_steps'] +
                default_run['test_chronic_1']['test_steps']
            ) / 2

            results.append({
                'Seed': seed,
                'Setting': setting,
                'Run Type': 'S-O Case [1.0, 0.0, 0.0]',
                'Switching Actions': default_run_actions,
                'Steps': default_run_steps
            })

    # Convert the results into a DataFrame for plotting
    df_results = pd.DataFrame(results)

    # Set up matplotlib parameters for a consistent look
    plt.rcParams.update({
        'font.size': 14,
        'figure.figsize': (14, 8),
        'axes.grid': True,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 12,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'font.family': 'serif',
    })

    # Plotting
    sns.set_style("whitegrid")

    # Create a list of unique 'Setting' labels
    unique_settings = df_results['Setting'].unique()

    # Generate a list of colors for Settings
    setting_colors = ['lightcoral', 'skyblue', 'lightgreen', 'gold', 'violet', 'orange']  # Add more colors as needed

    # Create iterators for colors
    setting_color_cycle = cycle(setting_colors)

    palette = {}
    for setting in unique_settings:
        palette[setting] = next(setting_color_cycle)

    # Define hue order to ensure consistent ordering
    hue_order = unique_settings.tolist()

    # Barplot of Switching Actions
    plt.figure(figsize=(14, 8))
    ax1 = sns.barplot(
        x='Run Type', y='Switching Actions', hue='Setting',
        data=df_results, hue_order=hue_order, palette=palette, ci=None
    )

    plt.title(title)
    plt.ylabel('Number of Switching Actions')
    plt.xlabel('Run Type')

    # Adjust legend to prevent duplicates
    handles, labels = ax1.get_legend_handles_labels()
    new_labels = OrderedDict()
    for handle, label in zip(handles, labels):
        if label not in new_labels:
            new_labels[label] = handle
    ax1.legend(new_labels.values(), new_labels.keys(), title='Setting',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    # Barplot of Steps
    plt.figure(figsize=(14, 8))
    ax2 = sns.barplot(
        x='Run Type', y='Steps', hue='Setting',
        data=df_results, hue_order=hue_order, palette=palette, ci=None
    )

    plt.title(title)
    plt.ylabel('Average Number of Steps')
    plt.xlabel('Run Type')

    # Adjust legend to prevent duplicates
    handles, labels = ax2.get_legend_handles_labels()
    new_labels = OrderedDict()
    for handle, label in zip(handles, labels):
        if label not in new_labels:
            new_labels[label] = handle
    ax2.legend(new_labels.values(), new_labels.keys(), title='Setting',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    return df_results

def plot_2d_projections_all_points(
    seed_paths, mc_path, iteration_paths, mc_iteration_paths, wrapper, save_dir=None, rewards=["L2RPN", "TopoDepth", "TopoActionHour"], iterations=False, mc_ex_path=None
):
    """
    Plots X vs Y, X vs Z, and Y vs Z using matplotlib, plotting all CCS points even if they are not Pareto-optimal in 2D projections.
    Optionally highlights Pareto frontier points.
    """
    
    # Set up matplotlib parameters for a more scientific look
    plt.rcParams.update(
        {
            "font.size": 14,
            "figure.figsize": (20, 6),
            "axes.grid": True,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "font.family": "serif",
        }
    )

    fig, axs = plt.subplots(1, 3)

    if wrapper == 'mc':
        return None
    else:
        # Handle OLS paths
        colors = plt.cm.tab10.colors  # Use a colormap for different seeds
        if iterations:
            seed_paths = iteration_paths
            iter = [5,10,20]

        for i, seed_path in enumerate(seed_paths[:3]):
            data = load_json_data(seed_path)
            ccs_list = data["ccs_list"][-1]
            x_all, y_all, z_all = extract_coordinates(ccs_list)
            print(seed_path)
            print(x_all)
            # Get matching weights for each point
            matching_entries = find_matching_weights_and_agent(
                ccs_list, data["ccs_data"]
            )

            # Create a mapping from coordinates to weights
            coord_to_weight = {}
            for entry in matching_entries:
                x, y, z = entry["returns"]
                weight = entry["weights"]
                coord_to_weight[(x, y, z)] = weight

            # Convert coordinates to tuples for matching
            coords_all = list(zip(x_all, y_all, z_all))

            # Create an array of weights corresponding to each point
            weights_all = [coord_to_weight.get(coord, None) for coord in coords_all]

            # Plot all CCS points
            if iterations:
                label = f"iterations {iter[i]}"
            else:
                label = f"Seed {i+1}"

            # X vs Y
            axs[0].scatter(
                x_all,
                y_all,
                color=colors[i % len(colors)],
                edgecolors="black",
                marker="o",
                s=100,
                label=label,
            )
            axs[0].set_xlabel(rewards[0])
            axs[0].set_ylabel(rewards[1])

            # X vs Z
            axs[1].scatter(
                x_all,
                z_all,
                color=colors[i % len(colors)],
                edgecolors="black",
                marker="o",
                s=100,
                label=label,
            )
            axs[1].set_xlabel(rewards[0])
            axs[1].set_ylabel(rewards[2])

            # Y vs Z
            axs[2].scatter(
                y_all,
                z_all,
                color=colors[i % len(colors)],
                edgecolors="black",
                marker="o",
                s=100,
                label=label,
            )
            axs[2].set_xlabel(rewards[1])
            axs[2].set_ylabel(rewards[2])

        # Processing RS data
        # Load data
        if os.path.exists(mc_path):
            if iterations: 
                seed_paths = mc_iteration_paths
            else: 
                seed_paths = [mc_path]
            for i, seed_path in enumerate(seed_paths):
                data = load_json_data(seed_path)
                ccs_list = data["ccs_list"][-1]
                x_all, y_all, z_all = extract_coordinates(ccs_list)
                print(seed_path)
                print(x_all)
                # Get matching weights for each point
                matching_entries = find_matching_weights_and_agent(
                    ccs_list, data["ccs_data"]
                )

                # Create a mapping from coordinates to weights
                coord_to_weight = {}
                for entry in matching_entries:
                    x, y, z = entry["returns"]
                    weight = entry["weights"]
                    coord_to_weight[(x, y, z)] = weight

                # Convert coordinates to tuples for matching
                coords_all = list(zip(x_all, y_all, z_all))

                # Create an array of weights corresponding to each point
                weights_all = [coord_to_weight.get(coord, None) for coord in coords_all]

                # Plot all CCS points
                if iterations: 
                    label = f"RS Benchmark iter {iter[i]}"
                else:
                    label = f"RS Benchmark {i+1}"

                colors = ["lightgray", 'gray', "black"]
                # X vs Y
                axs[0].scatter(
                    x_all,
                    y_all,
                    color=colors[i % len(colors)],
                    edgecolors="black",
                    marker="o",
                    s=100,
                    label=label,
                )
                axs[0].set_xlabel(rewards[0])
                axs[0].set_ylabel(rewards[1])

                # X vs Z
                axs[1].scatter(
                    x_all,
                    z_all,
                    color=colors[i % len(colors)],
                    edgecolors="black",
                    marker="o",
                    s=100,
                    label=label,
                )
                axs[1].set_xlabel(rewards[0])
                axs[1].set_ylabel(rewards[2])

                # Y vs Z
                axs[2].scatter(
                    y_all,
                    z_all,
                    color=colors[i % len(colors)],
                    edgecolors="black",
                    marker="o",
                    s=100,
                    label=label,
                )
                axs[2].set_xlabel(rewards[1])
                axs[2].set_ylabel(rewards[2])

        for ax in axs:
            ax.legend()
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        plt.suptitle("Projections of CCS Points in Return Domain", fontsize=20)
        plt.subplots_adjust(top=0.88)  # Adjust the top to make room for suptitle
        if save_dir:
            plt.savefig(os.path.join(save_dir, "ols_ccs_projections.png"))
        plt.show()
        
def plot_super_pareto_frontier_2d_multiple_settings_with_3d_pf(base_path, scenario, settings, save_dir=None, rewards=["L2RPN", "TopoDepth", "TopoActionHour"]):
    """
    Plots the super Pareto frontier across different settings on the 2D projections (X vs Y, X vs Z, Y vs Z) using matplotlib.
    Each setting is plotted with a different color and label.
    Includes all CCS points in the plot.
    Highlights the points that are on the Pareto frontier in the 3D space.
    
    Parameters:
    - base_path: The base directory where the JSON log files are stored.
    - scenario: The scenario name (e.g., "Reuse").
    - settings: A list of setting names (e.g., ["Baseline", "Partial", "Full"]).
    - save_dir: Directory to save the plot image (optional).
    - rewards: List of reward names for labeling axes.
    """
    import matplotlib.pyplot as plt
    import os

    # --- Generate paths within the function ---
    if scenario == 'Reuse':
        settings_paths = {
            "Baseline": os.path.join(base_path, "Baseline"),
            "Partial": os.path.join(base_path, "Partial_Reuse"),
            "Full": os.path.join(base_path, "Full_Reuse"),
            # Add more settings if needed
        }
    else:
        print("Scenario not supported.")
        return

    # Prepare the settings_paths dictionary
    settings_seed_paths = {}
    for setting in settings:
        path = settings_paths.get(setting)
        if not path:
            print(f"Path for setting '{setting}' not found.")
            continue

        seed_paths = []
        for seed in range(5):  # Adjust the range based on your seeds
            seed_file = f"morl_logs_seed_{seed}.json"
            seed_path = os.path.join(path, seed_file)
            if os.path.exists(seed_path):
                seed_paths.append(seed_path)
            else:
                print(f"Seed path not found: {seed_path}")
        settings_seed_paths[setting] = seed_paths

    # --- Plotting starts here ---
    # Set up matplotlib parameters for a more scientific look
    plt.rcParams.update(
        {
            "font.size": 14,
            "figure.figsize": (20, 6),
            "axes.grid": True,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "font.family": "serif",
        }
    )

    fig, axs = plt.subplots(1, 3)

    # Get a list of colors to assign to settings
    colors = plt.cm.tab10.colors  # You can choose other colormaps

    for idx, (setting_name, seed_paths) in enumerate(settings_seed_paths.items()):
        # Initialize lists to collect all data points for this setting
        x_all_seeds = []
        y_all_seeds = []
        z_all_seeds = []
        # List to keep track of which points are on the 3D Pareto frontier
        is_pareto = []

        # Initialize lists to collect weights, if needed
        coords_all = []
        weights_all = []

        for seed_path in seed_paths:
            if not os.path.exists(seed_path):
                print(f"File not found: {seed_path}")
                continue

            data = load_json_data(seed_path)
            ccs_list = data["ccs_list"][-1]
            x_all, y_all, z_all = extract_coordinates(ccs_list)
            # Stack the coordinates
            coords = np.column_stack((x_all, y_all, z_all))
            coords_all.extend(coords.tolist())

            # Collect weights for annotations
            matching_entries = find_matching_weights_and_agent(
                ccs_list, data["ccs_data"]
            )
            # Create a mapping from coordinates to weights
            coord_to_weight = {}
            for entry in matching_entries:
                x, y, z = entry["returns"]
                weight = entry["weights"]
                coord_to_weight[(x, y, z)] = weight
            # Create an array of weights corresponding to each point
            weights = [coord_to_weight.get(tuple(coord), None) for coord in coords]
            weights_all.extend(weights)

        if not coords_all:
            print(f"No data for setting {setting_name}")
            continue

        # Convert lists to numpy arrays
        coords_all = np.array(coords_all)
        x_all_seeds = coords_all[:, 0]
        y_all_seeds = coords_all[:, 1]
        z_all_seeds = coords_all[:, 2]

        # Compute the 3D Pareto frontier for this setting
        pareto_mask = get_pareto_front(coords_all)
        pareto_indices = np.where(pareto_mask)[0]

        # Assign a color to this setting
        color = colors[idx % len(colors)]

        # Plot all CCS points for this setting
        # X vs Y
        axs[0].scatter(
            x_all_seeds,
            y_all_seeds,
            color=color,
            alpha=0.3,
            label=f'{setting_name} All Points' if idx == 0 else None,
        )
        # X vs Z
        axs[1].scatter(
            x_all_seeds,
            z_all_seeds,
            color=color,
            alpha=0.3,
            label=f'{setting_name} All Points' if idx == 0 else None,
        )
        # Y vs Z
        axs[2].scatter(
            y_all_seeds,
            z_all_seeds,
            color=color,
            alpha=0.3,
            label=f'{setting_name} All Points' if idx == 0 else None,
        )

        # Highlight the points on the 3D Pareto frontier
        pareto_coords = coords_all[pareto_indices]
        x_pareto = pareto_coords[:, 0]
        y_pareto = pareto_coords[:, 1]
        z_pareto = pareto_coords[:, 2]

        # Use a distinct marker or edgecolor for Pareto points
        # X vs Y
        axs[0].scatter(
            x_pareto,
            y_pareto,
            color=color,
            edgecolors='black',
            marker='o',
            s=100,
            label=f'{setting_name} 3D Pareto Frontier',
        )
        axs[0].set_xlabel(rewards[0])
        axs[0].set_ylabel(rewards[1])

        # X vs Z
        axs[1].scatter(
            x_pareto,
            z_pareto,
            color=color,
            edgecolors='black',
            marker='o',
            s=100,
            label=f'{setting_name} 3D Pareto Frontier',
        )
        axs[1].set_xlabel(rewards[0])
        axs[1].set_ylabel(rewards[2])

        # Y vs Z
        axs[2].scatter(
            y_pareto,
            z_pareto,
            color=color,
            edgecolors='black',
            marker='o',
            s=100,
            label=f'{setting_name} 3D Pareto Frontier',
        )
        axs[2].set_xlabel(rewards[1])
        axs[2].set_ylabel(rewards[2])

    for ax in axs:
        # Remove duplicate legends
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.suptitle(f"Super Pareto Frontier Projections ({scenario} Scenario)", fontsize=20)
    plt.subplots_adjust(top=0.88)  # Adjust the top to make room for suptitle
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"super_pareto_frontiers_{scenario}_3d_pf.png"))
    plt.show()
    
def get_pareto_front(points):
    """
    Identify the Pareto-optimal points in a set of points.
    This function computes the convex coverage set (CCS) in multi-objective optimization.
    Parameters:
    - points: numpy array of shape (n_points, n_dimensions)
    Returns:
    - pareto_mask: boolean array indicating whether each point is Pareto-optimal
    """
    import numpy as np

    n_points = points.shape[0]
    pareto_mask = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        for j in range(n_points):
            if all(points[j] >= points[i]) and any(points[j] > points[i]):
                pareto_mask[i] = False
                break
    return pareto_mask

def plot_pareto_frontiers_for_training_settings(base_path, scenario, save_dir=None, rewards=["L2RPN", "TopoDepth", "TopoActionHour"]):
    """
    Plots the super Pareto frontiers for full training and 50% training settings.
    - First figure: Only points of the CCS generated by the full training (Baseline). No other points.
    - Second figure: Points of the CCS generated by the 50% training. Baseline (full training) points are in grey.
      The super Pareto frontier is indicated with colors depending on the method (Baseline, Partial, Full).

    Parameters:
    - base_path: The base directory where the JSON log files are stored.
    - scenario: The scenario name (e.g., "Reuse").
    - save_dir: Directory to save the plot images (optional).
    - rewards: List of reward names for labeling axes.
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    # Reuse paths as provided
    reuse_paths = {
        "Baseline": os.path.join(base_path, "Baseline"),
        "Full": os.path.join(base_path, "Full_Reuse"),
        "Partial": os.path.join(base_path, "Partial_Reuse"),
        "Baseline_min": os.path.join(base_path, "min_learning_none"),
        "Full_min": os.path.join(base_path, "min_learning_full"),
        "Partial_min": os.path.join(base_path, "min_learning_partial"),
    }

    # --- First Figure: Full Training (Baseline) ---

    print("Plotting Pareto Frontiers for Full Training (Baseline)")

    # Initialize the plot
    plt.rcParams.update(
        {
            "font.size": 14,
            "figure.figsize": (20, 6),
            "axes.grid": True,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "font.family": "serif",
        }
    )

    fig, axs = plt.subplots(1, 3)

    # Only process the 'Baseline' method
    method = 'Baseline'
    path = reuse_paths.get(method)
    if not path:
        print(f"Path for method '{method}' not found.")
        return

    # Collect the seed paths
    seed_paths = []
    for seed in range(10):  # Adjust based on your seeds
        seed_file = f"morl_logs_seed_{seed}.json"
        seed_path = os.path.join(path, seed_file)
        if os.path.exists(seed_path):
            seed_paths.append(seed_path)
        else:
            print(f"Seed path not found: {seed_path}")

    if not seed_paths:
        print(f"No seed paths found for method '{method}' in full training")
        return

    # Initialize lists to collect all data points for this method
    coords_all = []

    for seed_path in seed_paths:
        data = load_json_data(seed_path)
        ccs_list = data["ccs_list"][-1]
        x_all, y_all, z_all = extract_coordinates(ccs_list)
        coords = np.column_stack((x_all, y_all, z_all))
        coords_all.extend(coords.tolist())

    if not coords_all:
        print(f"No data for method '{method}' in full training")
        return

    coords_all = np.array(coords_all)
    x_all_seeds = coords_all[:, 0]
    y_all_seeds = coords_all[:, 1]
    z_all_seeds = coords_all[:, 2]

    # Compute the 3D Pareto frontier
    adjusted_coords = coords_all.copy()
    #adjusted_coords[:, 1:] = -adjusted_coords[:, 1:]  # Negate objectives to be minimized
    pareto_mask = get_pareto_front(adjusted_coords)
    pareto_indices = np.where(pareto_mask)[0]

    # Plot the CCS points
    axs[0].scatter(
        x_all_seeds,
        y_all_seeds,
        color='blue',
        alpha=0.3,
        label='CCS Points',
    )
    axs[1].scatter(
        x_all_seeds,
        z_all_seeds,
        color='blue',
        alpha=0.3,
        label='CCS Points',
    )
    axs[2].scatter(
        y_all_seeds,
        z_all_seeds,
        color='blue',
        alpha=0.3,
        label='CCS Points',
    )

    # Highlight the Pareto frontier
    pareto_coords = coords_all[pareto_indices]
    x_pareto = pareto_coords[:, 0]
    y_pareto = pareto_coords[:, 1]
    z_pareto = pareto_coords[:, 2]

    axs[0].scatter(
        x_pareto,
        y_pareto,
        color='red',
        edgecolors='black',
        marker='o',
        s=100,
        label='3D Pareto Frontier',
    )
    axs[1].scatter(
        x_pareto,
        z_pareto,
        color='red',
        edgecolors='black',
        marker='o',
        s=100,
        label='3D Pareto Frontier',
    )
    axs[2].scatter(
        y_pareto,
        z_pareto,
        color='red',
        edgecolors='black',
        marker='o',
        s=100,
        label='3D Pareto Frontier',
    )

    # Adjust axes labels
    axs[0].set_xlabel(rewards[0])
    axs[0].set_ylabel(rewards[1])
    axs[1].set_xlabel(rewards[0])
    axs[1].set_ylabel(rewards[2])
    axs[2].set_xlabel(rewards[1])
    axs[2].set_ylabel(rewards[2])

    for ax in axs:
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.suptitle("Pareto Frontiers - Full Training (Baseline)", fontsize=20)
    plt.subplots_adjust(top=0.88)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"pareto_frontiers_full_training_baseline.png"))
    plt.show()

    # --- Second Figure: 50% Training ---

    print("Plotting Pareto Frontiers for 50% Training")

    # Initialize the plot
    plt.rcParams.update(
        {
            "font.size": 14,
            "figure.figsize": (20, 6),
            "axes.grid": True,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "font.family": "serif",
        }
    )

    fig, axs = plt.subplots(1, 3)

    # Collect baseline points from full training
    baseline_coords_all = []

    baseline_path = reuse_paths.get('Baseline')
    if not baseline_path:
        print("Baseline path not found.")
    else:
        seed_paths = []
        for seed in range(5):  # Adjust based on your seeds
            seed_file = f"morl_logs_seed_{seed}.json"
            seed_path = os.path.join(baseline_path, seed_file)
            if os.path.exists(seed_path):
                seed_paths.append(seed_path)
            else:
                print(f"Seed path not found: {seed_path}")

        for seed_path in seed_paths:
            data = load_json_data(seed_path)
            ccs_list = data["ccs_list"][-1]
            x_all, y_all, z_all = extract_coordinates(ccs_list)
            coords = np.column_stack((x_all, y_all, z_all))
            baseline_coords_all.extend(coords.tolist())

    # Plot baseline points in grey
    if baseline_coords_all:
        baseline_coords_all = np.array(baseline_coords_all)
        x_baseline = baseline_coords_all[:, 0]
        y_baseline = baseline_coords_all[:, 1]
        z_baseline = baseline_coords_all[:, 2]

        axs[0].scatter(
            x_baseline,
            y_baseline,
            color='grey',
            alpha=0.3,
            label='Baseline Full Training'
        )
        axs[1].scatter(
            x_baseline,
            z_baseline,
            color='grey',
            alpha=0.3,
            label='Baseline Full Training'
        )
        axs[2].scatter(
            y_baseline,
            z_baseline,
            color='grey',
            alpha=0.3,
            label='Baseline Full Training'
        )

    # Methods to process
    methods = ['Baseline', 'Partial', 'Full']

    # Colors for methods
    method_colors = {
        'Baseline': 'blue',
        'Partial': 'green',
        'Full': 'red'
    }

    for idx, method in enumerate(methods):
        key = method + '_min'
        path = reuse_paths.get(key)
        if not path:
            print(f"Path for method '{method}' in 50% training not found.")
            continue

        seed_paths = []
        for seed in range(5):  # Adjust based on your seeds
            seed_file = f"morl_logs_seed_{seed}.json"
            seed_path = os.path.join(path, seed_file)
            if os.path.exists(seed_path):
                seed_paths.append(seed_path)
            else:
                print(f"Seed path not found: {seed_path}")

        if not seed_paths:
            print(f"No seed paths found for method '{method}' in 50% training")
            continue

        # Initialize lists to collect all data points for this method
        coords_all = []

        for seed_path in seed_paths:
            data = load_json_data(seed_path)
            ccs_list = data["ccs_list"][-1]
            x_all, y_all, z_all = extract_coordinates(ccs_list)
            coords = np.column_stack((x_all, y_all, z_all))
            coords_all.extend(coords.tolist())

        if not coords_all:
            print(f"No data for method '{method}' in 50% training")
            continue

        coords_all = np.array(coords_all)
        x_all_seeds = coords_all[:, 0]
        y_all_seeds = coords_all[:, 1]
        z_all_seeds = coords_all[:, 2]

        # Compute the 3D Pareto frontier for this method
        adjusted_coords = coords_all.copy()
        adjusted_coords[:, 1:] = -adjusted_coords[:, 1:]  # Negate objectives to be minimized
        pareto_mask = get_pareto_front(adjusted_coords)
        pareto_indices = np.where(pareto_mask)[0]

        # Assign a color to this method
        color = method_colors.get(method, 'black')

        # Plot the CCS points for this method
        axs[0].scatter(
            x_all_seeds,
            y_all_seeds,
            color=color,
            alpha=0.3,
            label=f'{method} CCS Points' if idx == 0 else None,
        )
        axs[1].scatter(
            x_all_seeds,
            z_all_seeds,
            color=color,
            alpha=0.3,
            label=f'{method} CCS Points' if idx == 0 else None,
        )
        axs[2].scatter(
            y_all_seeds,
            z_all_seeds,
            color=color,
            alpha=0.3,
            label=f'{method} CCS Points' if idx == 0 else None,
        )

        # Highlight the points on the 3D Pareto frontier
        pareto_coords = coords_all[pareto_indices]
        x_pareto = pareto_coords[:, 0]
        y_pareto = pareto_coords[:, 1]
        z_pareto = pareto_coords[:, 2]

        # Plot Pareto frontier points
        axs[0].scatter(
            x_pareto,
            y_pareto,
            color=color,
            edgecolors='black',
            marker='o',
            s=100,
            label=f'{method} 3D Pareto Frontier',
        )
        axs[1].scatter(
            x_pareto,
            z_pareto,
            color=color,
            edgecolors='black',
            marker='o',
            s=100,
            label=f'{method} 3D Pareto Frontier',
        )
        axs[2].scatter(
            y_pareto,
            z_pareto,
            color=color,
            edgecolors='black',
            marker='o',
            s=100,
            label=f'{method} 3D Pareto Frontier',
        )

    # Adjust axes labels
    axs[0].set_xlabel(rewards[0])
    axs[0].set_ylabel(rewards[1])
    axs[1].set_xlabel(rewards[0])
    axs[1].set_ylabel(rewards[2])
    axs[2].set_xlabel(rewards[1])
    axs[2].set_ylabel(rewards[2])

    for ax in axs:
        # Remove duplicate legends
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.suptitle("Pareto Frontiers - 50% Training", fontsize=20)
    plt.subplots_adjust(top=0.88)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"pareto_frontiers_50_percent_training.png"))
    plt.show()


def calculate_3d_metrics_with_combined_reference(seed_path, combined_reference_point):
    """
    Calculates 3D Hypervolume and Sparsity for a single seed path with a provided combined reference point.
    """
    # Load seed data
    data = load_json_data(seed_path)
    ccs_list = data["ccs_list"][-1]
    x_all, y_all, z_all = extract_coordinates(ccs_list)

    # Calculate 3D hypervolume and sparsity for the seed
    pareto_points_3d = np.column_stack((x_all, y_all, z_all))
    print(pareto_points_3d)
    hv_3d = calculate_3d_hypervolume(pareto_points_3d, combined_reference_point)
    sparsity_3d = calculate_3d_sparsity(pareto_points_3d)

    hv_3d = round(hv_3d, 2)
    sparsity_3d = round(sparsity_3d, 2)

    return hv_3d, sparsity_3d


def find_combined_reference_point(ols_paths, rs_paths):
    """
    Finds the combined minimum reference point for each seed over OLS and RS paths.
    The function assumes that ols_paths and rs_paths have the same number of seeds and are aligned by index.
    """
    combined_reference_points = []

    # Iterate over seeds based on the index
    for idx in range(len(ols_paths)):
        min_x, min_y, min_z = float('inf'), float('inf'), float('inf')

        # Read OLS JSON for the current seed
        ols_seed_path = ols_paths[idx]
        if os.path.exists(ols_seed_path):
            data_ols = load_json_data(ols_seed_path)
            ccs_list_ols = data_ols["ccs_list"][-1]
            x_ols, y_ols, z_ols = extract_coordinates(ccs_list_ols)
            min_x = min(min_x, *x_ols)
            min_y = min(min_y, *y_ols)
            min_z = min(min_z, *z_ols)
        else:
            print(f"OLS seed path not found: {ols_seed_path}")

        # Read RS JSON for the current seed (if available)
        if rs_paths and idx < len(rs_paths):
            rs_seed_path = rs_paths[idx]
            if os.path.exists(rs_seed_path):
                data_rs = load_json_data(rs_seed_path)
                ccs_list_rs = data_rs["ccs_list"][-1]
                x_rs, y_rs, z_rs = extract_coordinates(ccs_list_rs)
                min_x = min(min_x, *x_rs)
                min_y = min(min_y, *y_rs)
                min_z = min(min_z, *z_rs)
            else:
                print(f"RS seed path not found: {rs_seed_path}")

        # Add the combined minimum point for the current seed
        combined_reference_points.append((min_x, min_y, min_z))

    return combined_reference_points


def calculate_all_metrics(seed_paths_ols, wrapper, rs_seed_paths=None, iterations=False):
    """
    Calculates hypervolume and sparsity metrics for OLS and RS seeds with a combined reference point per seed.
    Returns a DataFrame with per-seed metrics and another DataFrame with mean and std for each method.
    """
    import pandas as pd

    if iterations:
        # Handle cases when iterations are enabled
        combined_reference_points = find_combined_reference_point(wrapper.iteration_paths, wrapper.mc_iteration_paths)
        paths_to_process_ols = wrapper.iteration_paths
        paths_to_process_rs = wrapper.mc_iteration_paths
    else:
        # Find combined reference point for each seed for OLS and RS seeds
        combined_reference_points = find_combined_reference_point(seed_paths_ols, rs_seed_paths)
        paths_to_process_ols = seed_paths_ols
        paths_to_process_rs = rs_seed_paths
    print(combined_reference_points)
    # Process OLS seeds
    ols_metrics_list = []
    for idx, seed_path in enumerate(paths_to_process_ols):
        if not os.path.exists(seed_path):
            print(f"OLS seed path not found: {seed_path}")
            continue
        reference_point = combined_reference_points[idx]
        print(reference_point)
        
        hv_3d_ols, sparsity_3d_ols = calculate_3d_metrics_with_combined_reference(seed_path, reference_point)
        data = {
            'Method': 'OLS',
            'Seed': f'Seed_{idx}',
            'Hypervolume 3D': hv_3d_ols,
            'Sparsity 3D': sparsity_3d_ols
        }
        print(data)
        ols_metrics_list.append(data)
    print(combined_reference_points)
    # Process RS seeds if provided
    rs_metrics_list = []
    if paths_to_process_rs:
        for idx, seed_path in enumerate(paths_to_process_rs):
            if not os.path.exists(seed_path):
                print(f"RS seed path not found: {seed_path}")
                continue
            reference_point = combined_reference_points[idx]
            print(reference_point)
            hv_3d_rs, sparsity_3d_rs = calculate_3d_metrics_with_combined_reference(seed_path, reference_point)
            data = {
                'Method': 'RS',
                'Seed': f'Seed_{idx}',
                'Hypervolume 3D': hv_3d_rs,
                'Sparsity 3D': sparsity_3d_rs
            }
            print(data)
            rs_metrics_list.append(data)

    # Combine the lists
    all_metrics_list = ols_metrics_list + rs_metrics_list

    # Convert to DataFrame
    df_all_metrics = pd.DataFrame(all_metrics_list)

    # Compute mean and std for each method
    df_mean_std = df_all_metrics.groupby('Method').agg(
        {'Hypervolume 3D': ['mean', 'std'], 'Sparsity 3D': ['mean', 'std']}
    ).reset_index()

    # Flatten MultiIndex columns
    df_mean_std.columns = ['Method', 'Hypervolume 3D Mean', 'Hypervolume 3D Std', 'Sparsity 3D Mean', 'Sparsity 3D Std']

    return df_all_metrics, df_mean_std


# ---- Main Function ----
def main():
    base_json_path = "C:\\Users\\thoma\MA\\TOPGRID_MORL\\morl_logs\\4th_trial"  # The base path where the JSON files are stored
    scenarios = ["Baseline", "Max_rho", "Opponent", "Reuse", "Time", "name"]
    names = ["Baseline", "rho095", "rho090", "rho080", "rho070", "Opponent", 'name']

    name = names[0]
    scenario = scenarios[2]
    reward_names = ["L2RPN", "TopoDepth", "TopoActionHour"]

    # Loop through scenarios and parameters
    print(f"Processing scenario: {scenario}")
    # Create an ExperimentAnalysis object
    analysis = ExperimentAnalysis(
        scenario=scenario, name=name, base_json_path=base_json_path
    )
    # Perform the analyses
    if scenario == "name":
        # Perform in-depth analysis on a selected seed
        analysis.calculate_metrics(iterations=True)
        analysis.plot_pareto_frontiers(rewards=reward_names)
        analysis.in_depth_analysis(seed=0)  # For example, seed 0
        analysis.analyse_pareto_values_and_plot()
    # Perform the analyses
    if scenario == "Baseline":
        plot_ccs_points_only(
            ols_seed_paths=analysis.seed_paths,
            rs_seed_paths=analysis.rs_seed_paths,
            save_dir=analysis.output_dir,
            rewards=reward_names
        )
        # Call the updated plotting function
        plot_super_pareto_frontier_2d_ols_vs_rs_ccs(
            ols_seed_paths=analysis.seed_paths,
            rs_seed_paths=analysis.rs_seed_paths,
            save_dir=analysis.output_dir,
            rewards=reward_names
        )
       
        # Perform in-depth analysis on a selected seed
        analysis.calculate_metrics(iterations=False)
        analysis.plot_pareto_frontiers(rewards=reward_names, iterations=False)
        plot_2d_projections_all_points(
            analysis.seed_paths, analysis.mc_seed_path, None, None, "ols", save_dir=analysis.output_dir, rewards=reward_names
        )
        #analysis.in_depth_analysis(seed=0)  # For example, seed 0
        #analysis.analyse_pareto_values_and_plot()
    if scenario == "Reuse":
        compare_hv_and_sparsity_with_separate_boxplots(os.path.join(base_json_path, "OLS", scenario), scenario)
        compare_hv_with_combined_boxplots(os.path.join(base_json_path, "OLS", scenario), scenario=scenario)
        plot_pareto_frontiers_for_training_settings(
            os.path.join(base_json_path, "OLS", scenario),
            scenario=scenario,
            rewards=reward_names
        )
        plot_super_pareto_frontier_2d_multiple_settings_with_3d_pf(
            os.path.join(base_json_path, "OLS", scenario),
            scenario=scenario,
            settings=["Baseline", "Partial", "Full"],  # Ensure the settings match your data
            rewards=reward_names
        )
        #plot_super_pareto_frontier_2d_multiple_settings(os.path.join(base_json_path, "OLS", scenario), scenario=scenario, settings = ["Baseline", "Full", "Partial"] )
    if scenario == 'Opponent':
        compare_policies_weights_all_seeds(os.path.join(base_json_path, 'OLS', scenario), scenario)
        #visualize_successful_weights(os.path.join(base_json_path, 'OLS', scenario), scenario)
        visualize_successful_weights(os.path.join(base_json_path, 'OLS', scenario), scenario, plot_option='separate')
        # Call the function with the results DataFrames
        
    if scenario == 'Time':
        compare_policies_weights_all_seeds(os.path.join(base_json_path, 'OLS', scenario), scenario)
        visualize_successful_weights(os.path.join(base_json_path, 'OLS', scenario), scenario)
        visualize_successful_weights(os.path.join(base_json_path, 'OLS', scenario), scenario, plot_option='separate')
        
    if scenario == "Max_rho": 
        compare_policies_weights_all_seeds(os.path.join(base_json_path, 'OLS', scenario), scenario)
        visualize_successful_weights(os.path.join(base_json_path, 'OLS', scenario), scenario)
        visualize_successful_weights(os.path.join(base_json_path, 'OLS', scenario), scenario, plot_option='separate')

if __name__ == "__main__":
    main()
