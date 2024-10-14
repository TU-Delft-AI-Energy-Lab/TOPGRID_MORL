import os
import ast
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting

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
            self.base_json_path,
            'OLS',
            self.scenario,
            self.name
            
        )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory: {self.output_dir}")
        # For OLS seeds (10 seeds)
        self.seed_paths = []
        for seed in range(5):
            seed_file = f'morl_logs_seed_{seed}.json'
            seed_path = os.path.join(self.output_dir, seed_file)
            self.seed_paths.append(seed_path)
        # For MC seed (1 seed)
        mc_seed_file = 'morl_logs_seed_0.json'
        mc_seed_dir = os.path.join(self.base_json_path, 'MC', self.scenario)
        self.mc_seed_path = os.path.join(mc_seed_dir, mc_seed_file)
        
    def load_data(self):
        # Load the data from the JSON files if needed
        pass
    
    def calculate_metrics(self):
        # Calculate multi-objective metrics like hypervolumes, max/min rewards, sparsities
        print("Calculating metrics...")
        df_all_metrics_ols = calculate_all_metrics(self.seed_paths, 'ols', self.mc_seed_path)
        # Save the DataFrame to CSV
        df_all_metrics_ols.to_csv(os.path.join(self.output_dir, 'ols_all_metrics.csv'), index=False)
        # Also process MC seed if available
        if os.path.exists(self.mc_seed_path):
            df_3d_metrics_mc = calculate_3d_metrics_only_for_mc(self.mc_seed_path)
            df_3d_metrics_mc.to_csv(os.path.join(self.output_dir, 'mc_3d_metrics.csv'), index=False)
            self.df_3d_metrics_mc = df_3d_metrics_mc
        else:
            print(f"MC seed path not found: {self.mc_seed_path}")
        # Store metrics
        self.df_all_metrics_ols = df_all_metrics_ols
        print("Metrics calculation completed.")
        
    def plot_pareto_frontiers(self, rewards):
        # Generate the 2D Pareto frontier plots
        print("Plotting Pareto frontiers...")
        # For OLS seeds
        plot_2d_projections_matplotlib(self.seed_paths, 'ols', save_dir=self.output_dir, rewards=rewards)
        # For MC seed
        if os.path.exists(self.mc_seed_path):
            plot_2d_projections_matplotlib(self.mc_seed_path, 'mc', save_dir=self.output_dir, rewards=rewards)
        else:
            print(f"MC seed path not found: {self.mc_seed_path}")
        print("Pareto frontier plotting completed.")
        
    def in_depth_analysis(self, seed):
        # For the default run, perform in-depth analysis for the trajectory on the real-world test data
        print(f"Performing in-depth analysis for seed {seed}...")
        seed_path = self.seed_paths[seed]
        # Load the matching data
        df_ccs_matching = process_data([seed_path], 'ols', self.output_dir)
        csv_path = os.path.join(self.output_dir, "ccs_matching_data.csv")
        # Perform the topo depth process and plot
        topo_depth_process_and_plot(csv_path)
        # Also perform sub_id_process_and_plot if needed
        sub_id_process_and_plot(csv_path)
        print("In-depth analysis completed.")
        
    def analyse_pareto_values_and_plot(self):
        # For the default run, generate the 2D projection of the real-world data
        print("Analysing Pareto frontier values and plotting projections...")
        csv_path = os.path.join(self.output_dir, 'ccs_matching_data.csv')
        if not os.path.exists(csv_path):
            # Need to process data to generate the CSV
            df_ccs_matching = process_data(self.seed_paths, 'ols', self.output_dir)
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
    with open(json_path, 'r') as f:
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
            is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)
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
            returns_array = np.array(data_entry['returns'])
            if np.allclose(ccs_entry_array, returns_array, atol=1e-3):
                matching_entries.append({
                    "weights": data_entry['weights'],
                    "returns": ccs_entry,
                    "agent_file": data_entry.get('agent_file', None),
                    "test_chronic_0": data_entry.get('test_chronic_0', {}),
                    "test_chronic_1": data_entry.get('test_chronic_1', {}),
                    "seed": data_entry.get('seed', None)
                })
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
    labels = [str(int(round(w))) if abs(w - round(w)) < 1e-2 else "{0:.2f}".format(w) for w in weight]
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
            height = abs(sorted_points[i][1] - sorted_points[i-1][1])
        hv += width * height
    return hv

def calculate_sparsity(points):
    """Calculates the sparsity metric for 2D points."""
    distances = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            distances.append(np.linalg.norm(np.array(points[i]) - np.array(points[j])))
    if distances:
        return np.mean(distances)
    else:
        return 0.0

def calculate_3d_hypervolume(points, reference_point):
    """Calculates the hypervolume for 3D points."""
    # Convert points to a numpy array
    points = np.array(points)
    # Shift points to the reference point
    shifted_points = points - reference_point
    # Ensure all points are in the positive orthant
    if np.any(shifted_points < 0):
        print("Error: Shifted points have negative coordinates after shifting to reference point.")
        return 0.0
    # Calculate the hypervolume using ConvexHull
    try:
        hull = ConvexHull(shifted_points)
        hv = hull.volume
    except Exception as e:
        print(f"Error calculating 3D hypervolume: {e}")
        hv = 0.0
    return hv

def calculate_3d_sparsity(points):
    """Calculates the sparsity metric for 3D points."""
    distances = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            distances.append(np.linalg.norm(np.array(points[i]) - np.array(points[j])))
    if distances:
        return np.mean(distances)
    else:
        return 0.0

def calculate_all_metrics(seed_paths, wrapper, mc_seed_path=None):
    """
    Calculates all metrics for each seed and the superseed set:
    - 2D Hypervolumes (XY, XZ, YZ)
    - 3D Hypervolume
    - 2D Sparsities (XY, XZ, YZ)
    - 3D Sparsity
    - Min/Max Returns for X, Y, Z
    - Pareto Points Count
    - Mean and Std of each metric across all seeds (except superseed)
    """
    return calculate_hypervolumes_and_sparsities(seed_paths, wrapper, mc_seed_path=mc_seed_path)

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
        ccs_list = data['ccs_list'][-1]
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
        hv_xy = calculate_hypervolume(list(zip(x_pareto_xy, y_pareto_xy)), reference_point_xy)
        hv_xz = calculate_hypervolume(list(zip(x_pareto_xz, z_pareto_xz)), reference_point_xz)
        hv_yz = calculate_hypervolume(list(zip(y_pareto_yz, z_pareto_yz)), reference_point_yz)

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
        results.append({
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
            "Pareto Points Count": pareto_points_count
        })

    # Calculate mean and std dev for metrics
    mean_hv_xy, std_hv_xy = np.mean(hv_xy_list), np.std(hv_xy_list)
    mean_hv_xz, std_hv_xz = np.mean(hv_xz_list), np.std(hv_xz_list)
    mean_hv_yz, std_hv_yz = np.mean(hv_yz_list), np.std(hv_yz_list)
    mean_hv_3d, std_hv_3d = np.mean(hv_3d_list), np.std(hv_3d_list)

    mean_sparsity_3d, std_sparsity_3d = np.mean(sparsity_3d_list), np.std(sparsity_3d_list)

    mean_min_return_x, std_min_return_x = np.mean(min_return_x_list), np.std(min_return_x_list)
    mean_max_return_x, std_max_return_x = np.mean(max_return_x_list), np.std(max_return_x_list)
    mean_min_return_y, std_min_return_y = np.mean(min_return_y_list), np.std(min_return_y_list)
    mean_max_return_y, std_max_return_y = np.mean(max_return_y_list), np.std(max_return_y_list)
    mean_min_return_z, std_min_return_z = np.mean(min_return_z_list), np.std(min_return_z_list)
    mean_max_return_z, std_max_return_z = np.mean(max_return_z_list), np.std(max_return_z_list)

    mean_pareto_points_count, std_pareto_points_count = np.mean(pareto_points_count_list), np.std(pareto_points_count_list)

    # Calculate for the superseed set (OLS data only)
    superseed_results = calculate_hypervolume_and_sparsity_superseed(all_x, all_y, all_z)

    pareto_points_superseed_3d = np.column_stack((all_x, all_y, all_z))
    reference_point_superseed_3d = (min(all_x), min(all_y), min(all_z))
    hv_superseed_3d = calculate_3d_hypervolume(pareto_points_superseed_3d, reference_point_superseed_3d)
    sparsity_superseed_3d = calculate_3d_sparsity(pareto_points_superseed_3d)

    # Append results for the superseed set, rounding values
    results.append({
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
        "Pareto Points Count": len(pareto_points_superseed_3d)
    })

    # Append mean and std as final rows, rounding values
    results.append({
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
        "Pareto Points Count": round(mean_pareto_points_count, 2)
    })

    results.append({
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
        "Pareto Points Count": round(std_pareto_points_count, 2)
    })

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
    hypervolume_xy = calculate_hypervolume(list(zip(superseed_pareto_xy, superseed_pareto_yy)), reference_point_xy)
    hypervolume_xz = calculate_hypervolume(list(zip(superseed_pareto_xz, superseed_pareto_zz)), reference_point_xz)
    hypervolume_yz = calculate_hypervolume(list(zip(superseed_pareto_yz, superseed_pareto_zz2)), reference_point_yz)

    return {
        "Hypervolume XY": hypervolume_xy,
        "Hypervolume XZ": hypervolume_xz,
        "Hypervolume YZ": hypervolume_yz,
    }

def calculate_3d_metrics_only_for_mc(mc_seed_path):
    """
    Calculates 3D Hypervolume and Sparsity for the MC seed (single seed case).
    This is for the MC dataset, which consists of only one seed.
    """
    # Load MC seed data
    data = load_json_data(mc_seed_path)
    ccs_list = data['ccs_list'][-1]
    x_all, y_all, z_all = extract_coordinates(ccs_list)

    # Calculate 3D hypervolume and sparsity for the MC seed
    pareto_points_3d = np.column_stack((x_all, y_all, z_all))
    reference_point_3d = (min(x_all), min(y_all), min(z_all))
    hv_3d = calculate_3d_hypervolume(pareto_points_3d, reference_point_3d)
    sparsity_3d = calculate_3d_sparsity(pareto_points_3d)

    # Store results for MC
    results_mc = {
        "Seed": "MC",
        "Hypervolume 3D": hv_3d,
        "Sparsity 3D": sparsity_3d
    }

    # Convert to DataFrame
    df_results_mc = pd.DataFrame([results_mc])

    return df_results_mc

def process_data(seed_paths, wrapper, output_dir):
    """Processes the data for all seeds and generates the 3D and 2D plots."""
    all_data = []
    if wrapper == 'mc':
        seed_paths = [seed_paths]
    
    for seed, seed_path in enumerate(seed_paths):
            
            
        if not os.path.exists(seed_path):
            print(f"File not found: {seed_path}")
            continue
        
        data = load_json_data(seed_path)
        ccs_list = data['ccs_list'][-1]
        x_all, y_all, z_all = extract_coordinates(ccs_list)

        # Get matching weights for each point
        matching_entries = find_matching_weights_and_agent(ccs_list, data['ccs_data'])
        
        # Collect data for DataFrame
        print(matching_entries)
        for entry in matching_entries:
            all_data.append({
                "seed": seed, 
                "Returns": entry['returns'],
                "Weights": entry['weights'],
                'test_chronic_0': entry['test_chronic_0'],
                'test_chronic_1': entry['test_chronic_1']
            })
        

    df_ccs_matching = pd.DataFrame(all_data) if all_data else pd.DataFrame()
    
    if not df_ccs_matching.empty:
        # Save the DataFrame to the constructed path
        csv_file_path = os.path.join(output_dir, "ccs_matching_data.csv")
        if wrapper == 'MC':
            csv_file_path = os.path.join(output_dir,  "mc_ccs_matching_data.csv")
        df_ccs_matching.to_csv(csv_file_path, index=False)
        print(f"Saved ccs_matching_data.csv to {csv_file_path}")
    else:
        print("No matching entries found.")
    return df_ccs_matching

def plot_2d_projections_matplotlib(seed_paths, wrapper, save_dir=None, rewards = ['L2RPN', 'TopoDepth', 'TopoActionHour']):
    """
    Plots X vs Y, X vs Z, and Y vs Z using matplotlib, highlighting Pareto frontier points.
    Annotates the extrema points corresponding to extreme weight vectors like (1,0,0), (0,1,0), (0,0,1).
    """
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np

    # Set up matplotlib parameters for a more scientific look
    plt.rcParams.update({
        'font.size': 14,
        'figure.figsize': (20, 6),
        'axes.grid': True,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 12,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'font.family': 'serif',
    })

    fig, axs = plt.subplots(1, 3)

    if wrapper == "mc":
        # Handle random sampling paths (RS-Benchmark)
        # Assuming seed_paths is a single path or a list with one path
        if isinstance(seed_paths, list):
            seed_path = seed_paths[0]
        else:
            seed_path = seed_paths

        # Load data
        data = load_json_data(seed_path)
        ccs_list = data['ccs_list'][-1]
        x_all, y_all, z_all = extract_coordinates(ccs_list)

        # Get matching weights for each point
        matching_entries = find_matching_weights_and_agent(ccs_list, data['ccs_data'])

        # Create a mapping from coordinates to weights
        coord_to_weight = {}
        for entry in matching_entries:
            x, y, z = entry['returns']
            weight = entry['weights']
            coord_to_weight[(x, y, z)] = weight

        # Convert coordinates to tuples for matching
        coords_all = list(zip(x_all, y_all, z_all))

        # Create an array of weights corresponding to each point
        weights_all = [coord_to_weight.get(coord, None) for coord in coords_all]

        # Pareto frontiers
        x_pareto_xy, y_pareto_xy, pareto_indices_xy = pareto_frontier_2d(x_all, y_all)
        x_pareto_xz, z_pareto_xz, pareto_indices_xz = pareto_frontier_2d(x_all, z_all)
        y_pareto_yz, z_pareto_yz, pareto_indices_yz = pareto_frontier_2d(y_all, z_all)

        # Plot color is gray
        gray_color = 'gray'

        # Plot full dataset and Pareto frontiers for each projection
        # X vs Y
        axs[0].scatter(x_all, y_all, color=gray_color, alpha=0.5,
                       label='RS-Benchmark Data')
        axs[0].scatter(x_pareto_xy, y_pareto_xy, color=gray_color,
                       edgecolors='black', marker='o', s=100, label='RS-Benchmark Pareto')

        # Annotate extrema points
        for idx in pareto_indices_xy:
            weight = weights_all[idx]
            if weight is not None:
                if is_extreme_weight(weight):
                    x = x_all[idx]
                    y = y_all[idx]
                    label = weight_label(weight)
                    axs[0].annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12,
                                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))

        axs[0].set_xlabel(rewards[0])
        axs[0].set_ylabel(rewards[1])
        

        # X vs Z
        axs[1].scatter(x_all, z_all, color=gray_color, alpha=0.5,
                       label='RS-Benchmark Data')
        axs[1].scatter(x_pareto_xz, z_pareto_xz, color=gray_color,
                       edgecolors='black', marker='o', s=100, label='RS-Benchmark Pareto')

        # Annotate extrema points
        for idx in pareto_indices_xz:
            weight = weights_all[idx]
            if weight is not None:
                if is_extreme_weight(weight):
                    x = x_all[idx]
                    z = z_all[idx]
                    label = weight_label(weight)
                    axs[1].annotate(label, (x, z), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12,
                                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))

        axs[1].set_xlabel(rewards[1])
        axs[1].set_ylabel(rewards[2])
        

        # Y vs Z
        axs[2].scatter(y_all, z_all, color=gray_color, alpha=0.5,
                       label='RS-Benchmark Data')
        axs[2].scatter(y_pareto_yz, z_pareto_yz, color=gray_color,
                       edgecolors='black', marker='o', s=100, label='RS-Benchmark Pareto')

        # Annotate extrema points
        for idx in pareto_indices_yz:
            weight = weights_all[idx]
            if weight is not None:
                if is_extreme_weight(weight):
                    y = y_all[idx]
                    z = z_all[idx]
                    label = weight_label(weight)
                    axs[2].annotate(label, (y, z), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12,
                                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))

        axs[2].set_xlabel(rewards[1])
        axs[2].set_ylabel(rewards[2])
        

        for ax in axs:
            ax.legend()
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        plt.suptitle('RS-Benchmark', fontsize=20)
        plt.subplots_adjust(top=0.88)  # Adjust the top to make room for suptitle
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'mc_pareto_frontiers.png'))
        plt.show()

    else:
        # Handle OLS paths
        colors = plt.cm.tab10.colors  # Use a colormap for different seeds

        for i, seed_path in enumerate(seed_paths):
            data = load_json_data(seed_path)
            ccs_list = data['ccs_list'][-1]
            x_all, y_all, z_all = extract_coordinates(ccs_list)

            # Get matching weights for each point
            matching_entries = find_matching_weights_and_agent(ccs_list, data['ccs_data'])

            # Create a mapping from coordinates to weights
            coord_to_weight = {}
            for entry in matching_entries:
                x, y, z = entry['returns']
                weight = entry['weights']
                coord_to_weight[(x, y, z)] = weight

            # Convert coordinates to tuples for matching
            coords_all = list(zip(x_all, y_all, z_all))

            # Create an array of weights corresponding to each point
            weights_all = [coord_to_weight.get(coord, None) for coord in coords_all]

            # Calculate Pareto frontiers
            x_pareto_xy, y_pareto_xy, pareto_indices_xy = pareto_frontier_2d(x_all, y_all)
            x_pareto_xz, z_pareto_xz, pareto_indices_xz = pareto_frontier_2d(x_all, z_all)
            y_pareto_yz, z_pareto_yz, pareto_indices_yz = pareto_frontier_2d(y_all, z_all)

            # Plot full dataset and Pareto frontiers for each projection
            # X vs Y
            axs[0].scatter(x_all, y_all, color=colors[i % len(colors)], alpha=0.5,
                           label=f'Seed {i+1} Data')
            axs[0].scatter(x_pareto_xy, y_pareto_xy, color=colors[i % len(colors)],
                           edgecolors='black', marker='o', s=100, label=f'Seed {i+1} Pareto')

            # Annotate extrema points
            for idx in pareto_indices_xy:
                weight = weights_all[idx]
                if weight is not None:
                    if is_extreme_weight(weight):
                        x = x_all[idx]
                        y = y_all[idx]
                        label = weight_label(weight)
                        axs[0].annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12,
                                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))

            axs[0].set_xlabel(rewards[0])
            axs[0].set_ylabel(rewards[1])

            # X vs Z
            axs[1].scatter(x_all, z_all, color=colors[i % len(colors)], alpha=0.5,
                           label=f'Seed {i+1} Data')
            axs[1].scatter(x_pareto_xz, z_pareto_xz, color=colors[i % len(colors)],
                           edgecolors='black', marker='o', s=100, label=f'Seed {i+1} Pareto')

            # Annotate extrema points
            for idx in pareto_indices_xz:
                weight = weights_all[idx]
                if weight is not None:
                    if is_extreme_weight(weight):
                        x = x_all[idx]
                        z = z_all[idx]
                        label = weight_label(weight)
                        axs[1].annotate(label, (x, z), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12,
                                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))

            axs[1].set_xlabel(rewards[0])
            axs[1].set_ylabel(rewards[2])

            # Y vs Z
            axs[2].scatter(y_all, z_all, color=colors[i % len(colors)], alpha=0.5,
                           label=f'Seed {i+1} Data')
            axs[2].scatter(y_pareto_yz, z_pareto_yz, color=colors[i % len(colors)],
                           edgecolors='black', marker='o', s=100, label=f'Seed {i+1} Pareto')

            # Annotate extrema points
            for idx in pareto_indices_yz:
                weight = weights_all[idx]
                if weight is not None:
                    if is_extreme_weight(weight):
                        y = y_all[idx]
                        z = z_all[idx]
                        label = weight_label(weight)
                        axs[2].annotate(label, (y, z), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12,
                                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))

            axs[2].set_xlabel(rewards[1])
            axs[2].set_ylabel(rewards[2])

        for ax in axs:
            ax.legend()
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        plt.suptitle('Projections of Pareo Frontier in Return Domain', fontsize=20)
        plt.subplots_adjust(top=0.88)  # Adjust the top to make room for suptitle
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'ols_pareto_frontiers.png'))
        plt.show()

def topo_depth_process_and_plot(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract information from the test_chronic columns
    df['test_chronic_0'] = df['test_chronic_0'].apply(ast.literal_eval)
    
    # Filter rows where 'test_steps' in 'test_chronic_0' is 2016
    df = df[df['test_chronic_0'].apply(lambda x: x['test_steps'] == 2016)]
    
    # Limit to the first 5 points that reach 2016 steps
    df = df.head(5)
    
    # Initialize the plot with subplots for each Pareto point
    fig, axes = plt.subplots(len(df), 1, figsize=(12, 6 * len(df)), sharex=True)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.4)
    
    # Set a global title for the plot
    fig.suptitle("Topological Trajectory for PF Points", fontsize=20, weight='bold')

    # Generate colors for each Pareto point
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))

    # Iterate through each row in the dataframe
    for idx, (ax, row) in enumerate(zip(axes, df.iterrows())):
        _, row = row
        color = colors[idx % len(colors)]
        weights = [round(float(w), 2) for w in ast.literal_eval(row['Weights'])]
        
        # Set the title for the subplot (using larger font for better visibility)
        ax.set_title(f"Weights: {weights}", fontsize=16, weight='bold')

        # Extract the timestamp and topological depth information from test_chronic_0
        chronic = 'test_chronic_0'
        steps = row[chronic]['test_steps']
        actions = row[chronic]['test_actions']
        timestamps = [0.0] + list(map(float, row[chronic]['test_action_timestamp']))
        topo_depths = [0.0] + [0.0 if t is None else t for t in row[chronic]['test_topo_distance']]
        substations = row[chronic]['test_sub_ids']

        # Different marker for chronic_0
        marker = 'o'

        # Plot each action on the graph and fill the area underneath
        for i in range(len(timestamps) - 1):
            if topo_depths[i] is not None and topo_depths[i + 1] is not None:
                # Draw rectangular lines connecting the points starting from (0,0)
                ax.plot([timestamps[i], timestamps[i + 1]], [topo_depths[i], topo_depths[i]],
                        color=color, linestyle='-', linewidth=1)
                ax.plot([timestamps[i + 1], timestamps[i + 1]], [topo_depths[i], topo_depths[i + 1]],
                        color=color, linestyle='-', linewidth=1)
                # Fill the area underneath the rectangular lines
                ax.fill_between([timestamps[i], timestamps[i + 1]], 0, topo_depths[i], color=color, alpha=0.3)

        # Plot each action on the graph with markers and annotate switching actions
        for j, (timestamp, topo_depth, action, substation) in enumerate(zip(timestamps, topo_depths, actions, substations)):
            if topo_depth is not None and timestamp != 0.0:  # Avoid annotating at (0, 0)
                ax.plot(timestamp, topo_depth, marker, color=color, label=f"Weights: {weights}" if j == 0 else "", markersize=8)
                ax.annotate(f"Sub {substation[0]}", (timestamp, topo_depth),
                            textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)

        # Formatting each subplot
        ax.set_ylabel("Topo. Depth", fontsize=12)
        ax.grid(True)

    # Set x-axis label only for the bottom plot
    ax.set_xlabel("Timestamp", fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust to make room for the suptitle
    plt.show()
    
    
def sub_id_process_and_plot(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract information from the test_chronic columns
    df['test_chronic_0'] = df['test_chronic_0'].apply(ast.literal_eval)
    df['test_chronic_1'] = df['test_chronic_1'].apply(ast.literal_eval)
    
    # Initialize the plot
    fig, ax = plt.subplots()
    
    # Generate colors for each Pareto point
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))

    # Iterate through each row in the dataframe
    for idx, row in df.iterrows():
        color = colors[idx % len(colors)]
        weights = row['Weights']
        label = f"Pareto Point {idx+1}: Weights {weights}"
        
        # For each row, extract the timestamp and substation information from test_chronic_0 and test_chronic_1
        for chronic in ['test_chronic_0', 'test_chronic_1']:
            steps = row[chronic]['test_steps']
            actions = row[chronic]['test_actions']
            timestamps = list(map(float, row[chronic]['test_action_timestamp']))
            sub_ids = row[chronic]['test_sub_ids']

            # Different marker for each chronic
            marker = 'o' if chronic == 'test_chronic_0' else 's'
            chronic_label = f"{label} ({chronic})"

            # Plot each action on the graph
            for timestamp, sub_id_list, action in zip(timestamps, sub_ids, actions):
                for sub in sub_id_list:
                    if sub is not None:
                        ax.plot(timestamp, int(sub), marker, color=color, label=chronic_label)
                        chronic_label = ""  # Avoid repeated labels in legend

    # Formatting the plot
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Substation ID Affected by Switching")
    ax.set_title("Substation Modifications at Different Pareto Points")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyse_pf_values_and_plot_projections(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert the string representation of dictionaries in 'test_chronic_0' to actual dictionaries
    df['test_chronic_0'] = df['test_chronic_0'].apply(ast.literal_eval)
    
    results = []
    
    # Iterate through each row in the dataframe
    for idx, row in df.iterrows():
        # Extract information from 'test_chronic_0'
        chronic = 'test_chronic_0'
        seed = row['seed']
        test_data = row[chronic]
        steps = test_data['test_steps']  # Total steps in the test
        actions = test_data['test_actions']  # List of actions taken
        topo_depths = test_data.get('test_topo_distance')
        timestamps = test_data.get('test_action_timestamp')
        
        if timestamps is None or topo_depths is None:
            print(f"Error: 'Timestamps' or 'Topo Depths' not found in test_chronic_0 for Pareto Point {idx + 1}")
            continue  # Skip this row if data is missing
        
        # Ensure that timestamps and topo_depths have the same length
        if len(timestamps) != len(topo_depths):
            print(f"Error: Length of 'Timestamps' and 'Topo Depths' do not match for Pareto Point {idx + 1}")
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
        weighted_depth_metric = cumulative_weighted_depth_time / steps if steps > 0 else 0
        
        # Calculate average steps per chronic (assuming steps are total steps over all chronics)
        num_chronics = test_data.get('Num Chronics', 1)  # Adjust if 'Num Chronics' is available
        avg_steps = steps / num_chronics if num_chronics > 0 else steps
        
        # Total number of switching actions (average over chronics)
        total_switching_actions = len(actions) / num_chronics if num_chronics > 0 else len(actions)
        
        # Extract weights
        weights = ast.literal_eval(row['Weights'])  # Should be list of floats
        
        # Store the results for the current Pareto point
        results.append({
            'seed': seed,
            'Pareto Point': idx + 1,
            'Average Steps': avg_steps,
            'Total Switching Actions': total_switching_actions,
            'Weighted Depth Metric': weighted_depth_metric,
            'Weights': weights,
            'Steps': steps
        })
    
    if not results:
        print("No valid data to plot.")
        return
    
    # Convert results to DataFrame for easier processing
    results_df = pd.DataFrame(results)
    
    # Prepare data for plotting
    seeds = results_df['seed'].unique()
    colors = plt.cm.tab10.colors  # Use a colormap for different seeds
    color_map = {seed: colors[i % len(colors)] for i, seed in enumerate(seeds)}
    
    # --- Set up matplotlib parameters for a more scientific look ---
    plt.rcParams.update({
        'font.size': 14,
        'figure.figsize': (20, 6),
        'axes.grid': True,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 12,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'font.family': 'serif',
    })
    
    # --- 2D Projections ---
    fig2, axs = plt.subplots(1, 3, figsize=(20, 6))
    fig2.suptitle('2D Projections of Pareto Frontier in Power System Domain' )
    
    for seed in seeds:
        seed_data = results_df[results_df['seed'] == seed]
        avg_steps_list = seed_data['Average Steps'].values
        total_actions_list = seed_data['Total Switching Actions'].values
        weighted_depth_list = seed_data['Weighted Depth Metric'].values
        weights_list = seed_data['Weights'].values
        steps_list = seed_data['Steps'].values
        
        color = color_map[seed]
        
        # Darker points for 2016 steps
        darker_color = 'black'
        is_2016_steps = (steps_list >= 2016)
        
        # X vs Y
        axs[0].scatter(avg_steps_list, total_actions_list, color=color, alpha=0.5, label=f'Seed {seed} Data')
        axs[0].scatter(avg_steps_list[is_2016_steps], total_actions_list[is_2016_steps],
                       color=darker_color, edgecolors='black', marker='o', s=100, label=f'Steps=2016 (Seed {seed})')
        
        # Annotate weights for points with 2016 steps
        for idx in np.where(is_2016_steps)[0]:
            weights_str = "(" + ", ".join([f"{w:.1f}" for w in weights_list[idx]]) + ")"
            axs[0].annotate(weights_str, (avg_steps_list[idx], total_actions_list[idx]), textcoords="offset points", xytext=(0, 10), ha='center')

        axs[0].set_xlabel('Average Steps')
        axs[0].set_ylabel('Weighted Depth Metric')
        
        
        # X vs Z
        axs[1].scatter(avg_steps_list, weighted_depth_list, color=color, alpha=0.5, label=f'Seed {seed} Data')
        axs[1].scatter(avg_steps_list[is_2016_steps], weighted_depth_list[is_2016_steps],
                       color=darker_color, edgecolors='black', marker='o', s=100, label=f'Steps=2016 (Seed {seed})')
        
        for idx in np.where(is_2016_steps)[0]:
            weights_str = "(" + ", ".join([f"{w:.1f}" for w in weights_list[idx]]) + ")"
            axs[1].annotate(weights_str, (avg_steps_list[idx], weighted_depth_list[idx]), textcoords="offset points", xytext=(0, 10), ha='center')

        axs[1].set_xlabel('Average Steps')
        axs[1].set_ylabel('Total Switching Actions')
        
        
        # Y vs Z
        axs[2].scatter(total_actions_list, weighted_depth_list, color=color, alpha=0.5, label=f'Seed {seed} Data')
        axs[2].scatter(total_actions_list[is_2016_steps], weighted_depth_list[is_2016_steps],
                       color=darker_color, edgecolors='black', marker='o', s=100, label=f'Steps=2016 (Seed {seed})')
        
        for idx in np.where(is_2016_steps)[0]:
            weights_str = "(" + ", ".join([f"{w:.1f}" for w in weights_list[idx]]) + ")"
            axs[2].annotate(weights_str, (total_actions_list[idx], weighted_depth_list[idx]), textcoords="offset points", xytext=(0, 10), ha='center')

        axs[2].set_xlabel('Weighted Depth Metric')
        axs[2].set_ylabel('Total Switching Actions')
        
    
    # Reverse the y-axis (Total Switching Actions) and z-axis (Weighted Depth Metric) in the 2D projections
    axs[0].invert_yaxis()
    axs[1].invert_yaxis()
    axs[2].invert_yaxis()
    axs[2].invert_xaxis()

    for ax in axs:
        ax.legend()
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
def calculate_hypervolume_and_sparsity(json_data):
    """
    Given JSON data containing CCS lists and coordinates, calculates 2D and 3D Hypervolume
    and Sparsity using the pre-existing utility functions.
    """
    # Extract the CCS list (assuming last element holds relevant data)
    
    ccs_list = json_data['ccs_list'][-1]
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

def compare_hv(base_path):
    """
    Compares the HV (Hypervolume) and Sparsity metrics for the settings:
    - Baseline
    - Full Reuse
    - Partial Reuse
    """
    settings = ["Baseline", "Full", "Partial"]
    reuse_paths = {
        "Baseline": os.path.join(base_path, "Baseline"),
        "Full": os.path.join(base_path,  "re_full"),
        "Partial": os.path.join(base_path, "re_partial")
    }

    # Initialize dictionary to store results
    hv_metrics = {"Setting": [], "Sparsity": [], "Hypervolume": []}

    # Loop through the settings and load corresponding JSON files
    for setting in settings:
        reuse_path = reuse_paths[setting]
        
        # Load the JSON log files for this setting
        json_files = [f for f in os.listdir(reuse_path) if f.startswith('morl_logs')]
        
        for json_file in json_files:
            file_path = os.path.join(reuse_path, json_file)
            
            # Load the JSON data
            data = load_json_data(json_path=file_path)
            print(file_path)
            # Calculate hypervolume and sparsity using existing functions
            hv_3d, sparsity_3d = calculate_hypervolume_and_sparsity(data)
            
            if hv_3d is not None and sparsity_3d is not None:
                # Store the metrics
                hv_metrics["Setting"].append(setting)
                hv_metrics["Sparsity"].append(sparsity_3d)
                hv_metrics["Hypervolume"].append(hv_3d)

    # Convert the dictionary to a DataFrame for easier comparison and visualization
    df_hv_metrics = pd.DataFrame(hv_metrics)
    
    # Print the results
    print(df_hv_metrics)
    
    # Return the DataFrame
    return df_hv_metrics

# ---- Main Function ----
def main():
    base_json_path = 'C:\\Users\\thoma\MA\\TOPGRID_MORL\\morl_logs\\trial'  # The base path where the JSON files are stored
    scenarios = ['Baseline', 'Maxrho', 'Opponent', 'Reuse', 'Time']
    names = ['Baseline', 'rho095', 'rho090', 'rho080', 'rho070', 'Opponent']
    
    
    
    name=names[0]
    scenario = scenarios[3]
    reward_names = ['L2RPN', 'TopoDepth', 'TopoActionHour']

    # Loop through scenarios and parameters
    print(f"Processing scenario: {scenario}")
    # Create an ExperimentAnalysis object
    analysis = ExperimentAnalysis(scenario=scenario, name=name, base_json_path=base_json_path)
    # Perform the analyses
    
    if scenario == 'Baseline':
        # Perform in-depth analysis on a selected seed
        analysis.calculate_metrics()
        analysis.plot_pareto_frontiers(rewards=reward_names)
        analysis.in_depth_analysis(seed=0)  # For example, seed 0
        analysis.analyse_pareto_values_and_plot() 
    if scenario == 'Reuse': 
        compare_hv(os.path.join(base_json_path, 'OLS', scenario))

if __name__ == "__main__":
    main()
