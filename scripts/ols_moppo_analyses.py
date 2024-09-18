import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import plotly.express as px
import dash
from dash import dash_table
from dash.dependencies import Input, Output
from dash import html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import pdist, squareform
from topgrid_morl.utils.MORL_analysis_utils import create_action_to_substation_mapping


# ---- Utility Functions ----
def load_json_data(relative_path):
    """Loads JSON data from a given relative path."""
    absolute_path = os.path.abspath(relative_path)
    with open(absolute_path, 'r') as file:
        data = json.load(file)
    return data


def extract_coordinates(ccs_list):
    """
    Extracts x, y, z coordinates from a list of CCS points.
    Handles both nested list shapes (lists of lists) and flat lists of floats.
    """
    # If ccs_list contains only a flat list of 3 values, treat it as (x, y, z)
    if len(ccs_list) == 3 and all(isinstance(coord, float) for coord in ccs_list):
        return [ccs_list[0]], [ccs_list[1]], [ccs_list[2]]  # Single point case

    x_values = []
    y_values = []
    z_values = []

    for item in ccs_list:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            # If the item is a list or tuple of 3 elements (x, y, z coordinates)
            x_values.append(item[0])  # ScaledLinesCapacity
            y_values.append(item[1])  # ScaledL2RPN
            z_values.append(item[2])  # ScaledTopoDepth
        elif isinstance(item, float):
            raise ValueError("Expected a list of (x, y, z) coordinates but found floats instead.")

    return x_values, y_values, z_values




# ---- Pareto Calculations ----
def is_pareto_efficient(costs):
    """Finds the Pareto-efficient points with maximization in mind."""
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] >= c, axis=1)
            is_efficient[i] = True
    return is_efficient


def pareto_frontier_2d(x_values, y_values):
    """Computes the Pareto frontier for 2D points considering maximization."""
    points = np.column_stack((x_values, y_values))
    is_efficient = is_pareto_efficient(points)
    x_pareto = np.array(x_values)[is_efficient]
    y_pareto = np.array(y_values)[is_efficient]

    sorted_indices = np.argsort(x_pareto)
    return x_pareto[sorted_indices], y_pareto[sorted_indices], is_efficient


def calculate_hypervolume(pareto_points, reference_point):
    """Calculate the hypervolume dominated by the Pareto frontier in 2D."""
    pareto_points = np.array(pareto_points)
    hypervolume = 0.0

    # Ensure the points are sorted by the first objective (x-axis)
    pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]

    # Calculate the hypervolume
    for i in range(len(pareto_points)):
        width = pareto_points[i, 0] - (reference_point[0] if i == 0 else pareto_points[i - 1, 0])
        height = pareto_points[i, 1] - reference_point[1]
        hypervolume += width * height

    return hypervolume

# ---- Sparsity Calculation Function ----

def calculate_sparsity(pareto_points):
    """
    Calculate the sparsity of a set of Pareto points by measuring the spread of the points.

    Parameters:
        pareto_points (List[Tuple[float, float]]): The list of Pareto frontier points (in 2D space).

    Returns:
        float: The sparsity metric, computed as the average pairwise distance between all points.
    """
    if len(pareto_points) <= 1:
        return 0.0  # If only one or no points, sparsity is zero

    # Calculate pairwise distances between all Pareto points
    distances = pdist(pareto_points, metric='euclidean')  # Use Euclidean distance between points

    # Compute the average pairwise distance as the sparsity metric
    sparsity_metric = np.mean(distances)

    return sparsity_metric


def calculate_sparsities_for_all_projections(seed_paths, wrapper):
    """
    Calculate sparsities for X vs Y, X vs Z, and Y vs Z projections for each seed.

    Parameters:
        seed_paths (List[str]): A list of file paths to the seed data.
        wrapper (str): A string identifier for the wrapper type ('mc' or others).

    Returns:
        List[Dict[str, float]]: A list of dictionaries with sparsity metrics for each projection (XY, XZ, YZ).
    """
    sparsities = []
    for seed_path in seed_paths:
        data = load_json_data(seed_path)
        ccs_list = data['ccs_list'][-1]
        x_all, y_all, z_all = extract_coordinates(ccs_list)

        # Calculate Pareto frontiers
        x_pareto_xy, y_pareto_xy, _ = pareto_frontier_2d(x_all, y_all)
        x_pareto_xz, z_pareto_xz, _ = pareto_frontier_2d(x_all, z_all)
        y_pareto_yz, z_pareto_yz, _ = pareto_frontier_2d(y_all, z_all)

        # Calculate sparsity for each Pareto frontier
        sparsity_xy = calculate_sparsity(list(zip(x_pareto_xy, y_pareto_xy)))
        sparsity_xz = calculate_sparsity(list(zip(x_pareto_xz, z_pareto_xz)))
        sparsity_yz = calculate_sparsity(list(zip(y_pareto_yz, z_pareto_yz)))

        sparsities.append({
            "Sparsity XY": sparsity_xy,
            "Sparsity XZ": sparsity_xz,
            "Sparsity YZ": sparsity_yz
        })

    return sparsities

def calculate_hypervolumes_for_all_projections(seed_paths, wrapper):
    """Calculates the hypervolume for X vs Y, X vs Z, and Y vs Z projections for each seed."""
    hypervolumes = []
    for seed_path in seed_paths:
        data = load_json_data(seed_path)
        ccs_list = data['ccs_list'][-1]
        x_all, y_all, z_all = extract_coordinates(ccs_list)

        # Define reference points based on minimum values
        reference_point_xy = (min(x_all), min(y_all))
        reference_point_xz = (min(x_all), min(z_all))
        reference_point_yz = (min(y_all), min(z_all))

        # Calculate Pareto frontiers and hypervolumes
        x_pareto_xy, y_pareto_xy, _ = pareto_frontier_2d(x_all, y_all)
        x_pareto_xz, z_pareto_xz, _ = pareto_frontier_2d(x_all, z_all)
        y_pareto_yz, z_pareto_yz, _ = pareto_frontier_2d(y_all, z_all)

        hv_xy = calculate_hypervolume(list(zip(x_pareto_xy, y_pareto_xy)), reference_point_xy)
        hv_xz = calculate_hypervolume(list(zip(x_pareto_xz, z_pareto_xz)), reference_point_xz)
        hv_yz = calculate_hypervolume(list(zip(y_pareto_yz, z_pareto_yz)), reference_point_yz)

        hypervolumes.append({
            "Hypervolume XY": hv_xy,
            "Hypervolume XZ": hv_xz,
            "Hypervolume YZ": hv_yz
        })

    return hypervolumes


# ---- Visualization Functions ----
def plot_3d_scatter(x_values, y_values, z_values, label, ax=None, color=None):
    """Creates a 3D scatter plot for given x, y, z values with a specific label and color."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_values, y_values, z_values, label=label, color=color)
    ax.set_xlabel('ScaledLinesCapacity')
    ax.set_ylabel('ScaledL2RPN')
    ax.set_zlabel('ScaledTopoDepth')
    return ax


def plot_2d_projections_seeds(seed_paths, wrapper):
    """Plots X vs Y, X vs Z, and Y vs Z in interactive 2D plots, highlighting Pareto frontier points and calculating hypervolumes."""
    colors = px.colors.qualitative.T10  # A built-in colormap
    hypervolumes = calculate_hypervolumes_for_all_projections(seed_paths, wrapper=wrapper)
    sparsities = calculate_sparsities_for_all_projections(seed_paths, wrapper=wrapper)

    fig = make_subplots(rows=1, cols=3, subplot_titles=[
        'ScaledLinesCapacity vs ScaledL2RPN',
        'ScaledLinesCapacity vs ScaledTopoDepth',
        'ScaledL2RPN vs ScaledTopoDepth'
    ])

    table_data = []
    for i, seed_path in enumerate(seed_paths):
        data = load_json_data(seed_path)
        ccs_list = data['ccs_list'][-1]
        x_all, y_all, z_all = extract_coordinates(ccs_list)

        # Pareto frontiers
        x_pareto_xy, y_pareto_xy, _ = pareto_frontier_2d(x_all, y_all)
        x_pareto_xz, z_pareto_xz, _ = pareto_frontier_2d(x_all, z_all)
        y_pareto_yz, z_pareto_yz, _ = pareto_frontier_2d(y_all, z_all)

        # Custom data to match the index with the table data
        row_indices = list(range(len(x_all)))

        # Add traces for each 2D projection
        fig.add_trace(go.Scatter(x=x_all, y=y_all, mode='markers', marker=dict(color=colors[i % len(colors)], opacity=0.3), name=f'Seed {i+1} (Non-Pareto)', customdata=row_indices), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_pareto_xy, y=y_pareto_xy, mode='markers+lines', marker=dict(color=colors[i % len(colors)], size=10, line=dict(width=2)), name=f'Seed {i+1} (Pareto)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_all, y=z_all, mode='markers', marker=dict(color=colors[i % len(colors)], opacity=0.3), name=f'Seed {i+1} (Non-Pareto)', customdata=row_indices), row=1, col=2)
        fig.add_trace(go.Scatter(x=x_pareto_xz, y=z_pareto_xz, mode='markers+lines', marker=dict(color=colors[i % len(colors)], size=10, line=dict(width=2)), name=f'Seed {i+1} (Pareto)'), row=1, col=2)
        fig.add_trace(go.Scatter(x=y_all, y=z_all, mode='markers', marker=dict(color=colors[i % len(colors)], opacity=0.3), name=f'Seed {i+1} (Non-Pareto)', customdata=row_indices), row=1, col=3)
        fig.add_trace(go.Scatter(x=y_pareto_yz, y=z_pareto_yz, mode='markers+lines', marker=dict(color=colors[i % len(colors)], size=10, line=dict(width=2)), name=f'Seed {i+1} (Pareto)'), row=1, col=3)

        # Append row data for the table
        for idx in range(len(x_all)):
            table_data.append({
                "Seed": f"Seed {i+1}",
                "X": x_all[idx],
                "Y": y_all[idx],
                "Z": z_all[idx],
                "Hypervolume XY": hypervolumes[i]["Hypervolume XY"],
                "Hypervolume XZ": hypervolumes[i]["Hypervolume XZ"],
                "Hypervolume YZ": hypervolumes[i]["Hypervolume YZ"],
                "Sparsity XY": sparsities[i]["Sparsity XY"],
                "Sparsity XZ": sparsities[i]["Sparsity XZ"],
                "Sparsity YZ": sparsities[i]["Sparsity YZ"],
            })

    fig.update_layout(height=600, width=1200, title_text="2D Projections of Seeds", template="plotly_white", showlegend=True)
    df = pd.DataFrame(table_data)
    return fig, df


def create_dash_app(fig, df_display):
    """Creates the Dash app for visualizing the data and projections."""
    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Tabs([
            dcc.Tab(label='Graph', children=[
                html.H1("Seed Metrics and Projections", style={'textAlign': 'center', 'color': '#007BFF'}),
                dcc.Graph(id='2d-projections', figure=fig),
                html.Div(id='output-data-click', style={'fontSize': 20, 'marginTop': '20px'}),
            ]),
            dcc.Tab(label='Data Table', children=[
                html.H1("Data Table", style={'textAlign': 'center', 'color': '#007BFF'}),
                dash_table.DataTable(
                    id='data-table',
                    columns=[{"name": i, "id": i} for i in df_display.columns],
                    data=df_display.to_dict('records'),
                    page_size=10,
                    style_table={'overflowX': 'auto', 'marginBottom': '20px'},
                    style_cell={'textAlign': 'center', 'backgroundColor': '#f9f9f9', 'color': 'black', 'border': '1px solid #ddd'},
                    style_header={'backgroundColor': '#007BFF', 'fontWeight': 'bold', 'color': 'white'},
                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f2f2f2'}]
                )
            ])
        ])
    ])

    @app.callback(
    Output('output-data-click', 'children'),
    [Input('2d-projections', 'clickData')]
    )
    def display_click_data(clickData):
        """Callback to display clicked point details, including substation."""
        if clickData:
            # Extract custom data index from the clicked point
            point_index = clickData['points'][0]['customdata']
            selected_row = df_display.iloc[point_index]
            return [
                html.P(f"Seed: {selected_row['Seed']}"),
                html.P(f"X: {selected_row['X']}"),
                html.P(f"Y: {selected_row['Y']}"),
                html.P(f"Z: {selected_row['Z']}"),
                html.P(f"Substation: {selected_row['Substation']}"),  # Display the substation
                html.P(f"Hypervolume XY: {selected_row['Hypervolume XY']}"),
                html.P(f"Hypervolume XZ: {selected_row['Hypervolume XZ']}"),
                html.P(f"Hypervolume YZ: {selected_row['Hypervolume YZ']}")
            ]
        return "Click on a point in the graph to see its details."

    app.run_server(debug=True)


# ---- Data Processing ----
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
                    "agent_file": data_entry['agent_file'],
                    "test_steps": data_entry["test_steps"],
                    "test_actions": data_entry['test_actions']
                })
                found_match = True
                break  # Stop once a match is found
        if not found_match:
            print(f"No match found for CCS entry: {ccs_entry}")
    return matching_entries

def process_data(seed_paths, wrapper):
    """Processes the data for all seeds and generates the 3D and 2D plots."""
    all_data = []
    
    # Create the action-to-substation mapping using the gym environment
    action_to_substation_mapping = create_action_to_substation_mapping()

    for seed_path in seed_paths:
        if not os.path.exists(seed_path):
            print(f"File not found: {seed_path}")
            continue

        data = load_json_data(seed_path)
        ccs_list = data['ccs_list'][-1]
        if wrapper == 'mc':
            ccs_list = data['ccs_list']
        ccs_data = data['ccs_data']
        matching_entries = find_matching_weights_and_agent(ccs_list, ccs_data)
        print(matching_entries)
        # Collect data for DataFrame
        for entry in matching_entries:
            actions = entry['test_actions'] # Assuming test_actions is a list of actions
            substations = [action_to_substation_mapping.get(action, 'Unknown') for action in actions]# Get substation based on action

            all_data.append({
                "Weights": entry['weights'],
                "Returns": entry['returns'],
                "Test Steps": entry['test_steps'],
                "Test Actions": entry['test_actions'],
                "Substation": substations  # Add the substation to the data
            })

    df_ccs_matching = pd.DataFrame(all_data) if all_data else pd.DataFrame()

    if not df_ccs_matching.empty:
        df_ccs_matching.to_csv("ccs_matching_data.csv", index=False)
        print(df_ccs_matching)
    # Call the function to calculate hypervolumes and sparsities and output the DataFrame
    df_metrics = calculate_hypervolumes_and_sparsities(seed_paths, wrapper)
    print(df_metrics)
    
    plot_2d_projections_matplotlib(seed_paths, wrapper)   # Matplotlib-based visualization
    # Call the plotting functions
    #plot_all_seeds(seed_paths, wrapper, df_ccs_matching)  # Dash-based visualization
    



def plot_all_seeds(seed_paths, wrapper, df_ccs_matching):
    """Plots all seeds in 3D and 2D projections, and integrates substation information."""
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    colors = cm.tab10.colors

    for i, seed_path in enumerate(seed_paths):
        data = load_json_data(seed_path)
        ccs_list = data['ccs_list'][-1]
        x_all, y_all, z_all = extract_coordinates(ccs_list)
        plot_3d_scatter(x_all, y_all, z_all, f'Seed {i+1}', ax_3d, color=colors[i % len(colors)])

    ax_3d.legend()
    plt.show()

    fig, df_display = plot_2d_projections_seeds(seed_paths, wrapper=wrapper)

    # Add substation information to the Dash app
    df_display['Substation'] = df_ccs_matching['Substation']
    
    create_dash_app(fig, df_display)

# ---- 2D Plotting with Matplotlib (with Superseed Pareto Markings) ----
def plot_2d_projections_matplotlib(seed_paths, wrapper):
    """Plots X vs Y, X vs Z, and Y vs Z using matplotlib, highlighting Pareto frontier points."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    all_x, all_y, all_z = [], [], []
    hypervolumes = calculate_hypervolumes_for_all_projections(seed_paths, wrapper=wrapper)
    sparsities = calculate_sparsities_for_all_projections(seed_paths, wrapper=wrapper)
    colors = plt.cm.tab10.colors  # Use built-in colormap

    # Aggregate all seed data for superseed calculation
    for i, seed_path in enumerate(seed_paths):
        data = load_json_data(seed_path)
        ccs_list = data['ccs_list'][-1]
        x_all, y_all, z_all = extract_coordinates(ccs_list)

        all_x.extend(x_all)
        all_y.extend(y_all)
        all_z.extend(z_all)

        # Calculate Pareto frontiers
        x_pareto_xy, y_pareto_xy, _ = pareto_frontier_2d(x_all, y_all)
        x_pareto_xz, z_pareto_xz, _ = pareto_frontier_2d(x_all, z_all)
        y_pareto_yz, z_pareto_yz, _ = pareto_frontier_2d(y_all, z_all)

        # Plot full dataset and Pareto frontiers for each projection
        # X vs Y
        axs[0].scatter(x_all, y_all, color=colors[i % len(colors)], alpha=0.3, label=f'Seed {i+1} (Non-Pareto)')
        axs[0].plot(x_pareto_xy, y_pareto_xy, color=colors[i % len(colors)], marker='o', label=f'Seed {i+1} (Pareto)')
        axs[0].set_xlabel('ScaledLinesCapacity')
        axs[0].set_ylabel('ScaledL2RPN')
        axs[0].set_title('ScaledLinesCapacity vs ScaledL2RPN')

        # X vs Z
        axs[1].scatter(x_all, z_all, color=colors[i % len(colors)], alpha=0.3, label=f'Seed {i+1} (Non-Pareto)')
        axs[1].plot(x_pareto_xz, z_pareto_xz, color=colors[i % len(colors)], marker='o', label=f'Seed {i+1} (Pareto)')
        axs[1].set_xlabel('ScaledLinesCapacity')
        axs[1].set_ylabel('ScaledTopoDepth')
        axs[1].set_title('ScaledLinesCapacity vs ScaledTopoDepth')

        # Y vs Z
        axs[2].scatter(y_all, z_all, color=colors[i % len(colors)], alpha=0.3, label=f'Seed {i+1} (Non-Pareto)')
        axs[2].plot(y_pareto_yz, z_pareto_yz, color=colors[i % len(colors)], marker='o', label=f'Seed {i+1} (Pareto)')
        axs[2].set_xlabel('ScaledL2RPN')
        axs[2].set_ylabel('ScaledTopoDepth')
        axs[2].set_title('ScaledL2RPN vs ScaledTopoDepth')

    # Calculate and plot the Pareto frontier for the superseed set
    superseed_pareto_xy, superseed_pareto_yy, _ = pareto_frontier_2d(all_x, all_y)
    superseed_pareto_xz, superseed_pareto_zz, _ = pareto_frontier_2d(all_x, all_z)
    superseed_pareto_yz, superseed_pareto_zz2, _ = pareto_frontier_2d(all_y, all_z)

    axs[0].plot(superseed_pareto_xy, superseed_pareto_yy, color='red', marker='x', markersize=8, linewidth=0.5, label='Superseed Pareto')
    axs[1].plot(superseed_pareto_xz, superseed_pareto_zz, color='red', marker='x', markersize=8, linewidth=0.5, label='Superseed Pareto')
    axs[2].plot(superseed_pareto_yz, superseed_pareto_zz2, color='red', marker='x', markersize=8, linewidth=0.5, label='Superseed Pareto')

    for ax in axs:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
def calculate_3d_hypervolume(pareto_points, reference_point):
    """Calculate the 3D hypervolume dominated by the Pareto frontier."""
    pareto_points = np.array(pareto_points)
    hypervolume = 0.0

    # Ensure the points are sorted by the first objective (x-axis)
    pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]

    # Calculate the 3D hypervolume (the volume in between Pareto points and the reference point)
    for i in range(len(pareto_points)):
        width = pareto_points[i, 0] - (reference_point[0] if i == 0 else pareto_points[i - 1, 0])
        length = pareto_points[i, 1] - reference_point[1]
        height = pareto_points[i, 2] - reference_point[2]
        hypervolume += width * length * height

    return hypervolume

def calculate_3d_sparsity(pareto_points):
    """Calculate the sparsity of a set of Pareto points in 3D space."""
    if len(pareto_points) <= 1:
        return 0.0  # If only one or no points, sparsity is zero

    # Calculate pairwise distances between all Pareto points in 3D
    distances = pdist(pareto_points, metric='euclidean')  # Use Euclidean distance in 3D

    # Compute the average pairwise distance as the sparsity metric
    sparsity_metric = np.mean(distances)

    return sparsity_metric


def calculate_hypervolumes_and_sparsities(seed_paths, wrapper):
    """Calculates the 2D and 3D hypervolumes, sparsities, min/max returns, and Pareto points for each seed and the superseed set.
    Also calculates the mean and std of hypervolumes, sparsities, and returns across seeds.
    """
    all_x, all_y, all_z = [], [], []
    results = []
    
    # Initialize lists to store HV, sparsity, and return statistics for mean/std calculation
    hv_2d_xy_list, hv_2d_xz_list, hv_2d_yz_list = [], [], []
    hv_3d_list, sparsity_3d_list = [], []
    min_return_x_list, max_return_x_list = [], []
    min_return_y_list, max_return_y_list = [], []
    min_return_z_list, max_return_z_list = [], []
    pareto_points_count_list = []

    # Global tracking for the superseed set
    min_return_x_all = float('inf')
    max_return_x_all = float('-inf')
    min_return_y_all = float('inf')
    max_return_y_all = float('-inf')
    min_return_z_all = float('inf')
    max_return_z_all = float('-inf')

    # Calculate hypervolume, sparsity, and min/max returns for each seed
    for i, seed_path in enumerate(seed_paths):
        data = load_json_data(seed_path)
        ccs_list = data['ccs_list'][-1]
        x_all, y_all, z_all = extract_coordinates(ccs_list)

        all_x.extend(x_all)  # Combine all data for the superseed set
        all_y.extend(y_all)
        all_z.extend(z_all)

        # Track min and max returns for each seed across all dimensions
        min_return_x = min(x_all)
        max_return_x = max(x_all)
        min_return_y = min(y_all)
        max_return_y = max(y_all)
        min_return_z = min(z_all)
        max_return_z = max(z_all)

        # Update global (superseed) min/max returns
        min_return_x_all = min(min_return_x_all, min_return_x)
        max_return_x_all = max(max_return_x_all, max_return_x)
        min_return_y_all = min(min_return_y_all, min_return_y)
        max_return_y_all = max(max_return_y_all, max_return_y)
        min_return_z_all = min(min_return_z_all, min_return_z)
        max_return_z_all = max(max_return_z_all, max_return_z)

        # Calculate 2D hypervolume and sparsity as before (XY, XZ, YZ)
        reference_point_xy = (min(x_all), min(y_all))
        reference_point_xz = (min(x_all), min(z_all))
        reference_point_yz = (min(y_all), min(z_all))

        x_pareto_xy, y_pareto_xy, pareto_xy_indices = pareto_frontier_2d(x_all, y_all)
        x_pareto_xz, z_pareto_xz, pareto_xz_indices = pareto_frontier_2d(x_all, z_all)
        y_pareto_yz, z_pareto_yz, pareto_yz_indices = pareto_frontier_2d(y_all, z_all)

        # Number of Pareto points for this seed
        pareto_points_count = len(pareto_xy_indices)
        pareto_points_count_list.append(pareto_points_count)

        # Calculate hypervolumes
        hv_xy = calculate_hypervolume(list(zip(x_pareto_xy, y_pareto_xy)), reference_point_xy)
        hv_xz = calculate_hypervolume(list(zip(x_pareto_xz, z_pareto_xz)), reference_point_xz)
        hv_yz = calculate_hypervolume(list(zip(y_pareto_yz, z_pareto_yz)), reference_point_yz)

        # Append 2D HV values for later mean/std calculation
        hv_2d_xy_list.append(hv_xy)
        hv_2d_xz_list.append(hv_xz)
        hv_2d_yz_list.append(hv_yz)

        # Calculate sparsity for each Pareto frontier
        sparsity_xy = calculate_sparsity(list(zip(x_pareto_xy, y_pareto_xy)))
        sparsity_xz = calculate_sparsity(list(zip(x_pareto_xz, z_pareto_xz)))
        sparsity_yz = calculate_sparsity(list(zip(y_pareto_yz, z_pareto_yz)))

        # Calculate 3D hypervolume and sparsity
        pareto_points_3d = np.column_stack((x_all, y_all, z_all))
        reference_point_3d = (min(x_all), min(y_all), min(z_all))
        hv_3d = calculate_3d_hypervolume(pareto_points_3d, reference_point_3d)
        sparsity_3d = calculate_3d_sparsity(pareto_points_3d)

        # Append 3D HV and sparsity for later mean/std calculation
        hv_3d_list.append(hv_3d)
        sparsity_3d_list.append(sparsity_3d)

        # Append min/max returns for later mean/std calculation
        min_return_x_list.append(min_return_x)
        max_return_x_list.append(max_return_x)
        min_return_y_list.append(min_return_y)
        max_return_y_list.append(max_return_y)
        min_return_z_list.append(min_return_z)
        max_return_z_list.append(max_return_z)

        # Append results for this seed
        results.append({
            "Seed": f"Seed {i+1}",
            "Hypervolume XY": hv_xy,
            "Hypervolume XZ": hv_xz,
            "Hypervolume YZ": hv_yz,
            "Sparsity XY": sparsity_xy,
            "Sparsity XZ": sparsity_xz,
            "Sparsity YZ": sparsity_yz,
            "Hypervolume 3D": hv_3d,
            "Sparsity 3D": sparsity_3d,
            "Min Return X": min_return_x,
            "Max Return X": max_return_x,
            "Min Return Y": min_return_y,
            "Max Return Y": max_return_y,
            "Min Return Z": min_return_z,
            "Max Return Z": max_return_z,
            "Pareto Points Count": pareto_points_count
        })

    # Calculate mean and std for each metric over all seeds
    mean_hv_xy, std_hv_xy = np.mean(hv_2d_xy_list), np.std(hv_2d_xy_list)
    mean_hv_xz, std_hv_xz = np.mean(hv_2d_xz_list), np.std(hv_2d_xz_list)
    mean_hv_yz, std_hv_yz = np.mean(hv_2d_yz_list), np.std(hv_2d_yz_list)
    mean_hv_3d, std_hv_3d = np.mean(hv_3d_list), np.std(hv_3d_list)

    mean_sparsity_3d, std_sparsity_3d = np.mean(sparsity_3d_list), np.std(sparsity_3d_list)

    mean_min_return_x, std_min_return_x = np.mean(min_return_x_list), np.std(min_return_x_list)
    mean_max_return_x, std_max_return_x = np.mean(max_return_x_list), np.std(max_return_x_list)
    mean_min_return_y, std_min_return_y = np.mean(min_return_y_list), np.std(min_return_y_list)
    mean_max_return_y, std_max_return_y = np.mean(max_return_y_list), np.std(max_return_y_list)
    mean_min_return_z, std_min_return_z = np.mean(min_return_z_list), np.std(min_return_z_list)
    mean_max_return_z, std_max_return_z = np.mean(max_return_z_list), np.std(max_return_z_list)

    mean_pareto_points_count, std_pareto_points_count = np.mean(pareto_points_count_list), np.std(pareto_points_count_list)

    # Calculate for the superseed set (same as before)
    superseed_results = calculate_hypervolume_and_sparsity_superseed(all_x, all_y, all_z)

    pareto_points_superseed_3d = np.column_stack((all_x, all_y, all_z))
    reference_point_superseed_3d = (min(all_x), min(all_y), min(all_z))
    hv_superseed_3d = calculate_3d_hypervolume(pareto_points_superseed_3d, reference_point_superseed_3d)
    sparsity_superseed_3d = calculate_3d_sparsity(pareto_points_superseed_3d)

    # Append results for the superseed set
    results.append({
        "Seed": "Superseed",
        "Hypervolume XY": superseed_results["Hypervolume XY"],
        "Hypervolume XZ": superseed_results["Hypervolume XZ"],
        "Hypervolume YZ": superseed_results["Hypervolume YZ"],
        "Sparsity XY": superseed_results["Sparsity XY"],
        "Sparsity XZ": superseed_results["Sparsity XZ"],
        "Sparsity YZ": superseed_results["Sparsity YZ"],
        "Hypervolume 3D": hv_superseed_3d,
        "Sparsity 3D": sparsity_superseed_3d,
        "Min Return X": min_return_x_all,
        "Max Return X": max_return_x_all,
        "Min Return Y": min_return_y_all,
        "Max Return Y": max_return_y_all,
        "Min Return Z": min_return_z_all,
        "Max Return Z": max_return_z_all,
        "Pareto Points Count": len(pareto_points_superseed_3d)
    })

    # Append mean and std as a final row
    results.append({
        "Seed": "Mean",
        "Hypervolume XY": mean_hv_xy,
        "Hypervolume XZ": mean_hv_xz,
        "Hypervolume YZ": mean_hv_yz,
        "Hypervolume 3D": mean_hv_3d,
        "Sparsity 3D": mean_sparsity_3d,
        "Min Return X": mean_min_return_x,
        "Max Return X": mean_max_return_x,
        "Min Return Y": mean_min_return_y,
        "Max Return Y": mean_max_return_y,
        "Min Return Z": mean_min_return_z,
        "Max Return Z": mean_max_return_z,
        "Pareto Points Count": mean_pareto_points_count
    })

    results.append({
        "Seed": "Std Dev",
        "Hypervolume XY": std_hv_xy,
        "Hypervolume XZ": std_hv_xz,
        "Hypervolume YZ": std_hv_yz,
        "Hypervolume 3D": std_hv_3d,
        "Sparsity 3D": std_sparsity_3d,
        "Min Return X": std_min_return_x,
        "Max Return X": std_max_return_x,
        "Min Return Y": std_min_return_y,
        "Max Return Y": std_max_return_y,
        "Min Return Z": std_min_return_z,
        "Max Return Z": std_max_return_z,
        "Pareto Points Count": std_pareto_points_count
    })

    # Convert the results to a DataFrame
    df_results = pd.DataFrame(results)

    return df_results


# ---- Superseed Calculation Functions ----

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

    # Calculate sparsities for superseed set
    sparsity_xy = calculate_sparsity(list(zip(superseed_pareto_xy, superseed_pareto_yy)))
    sparsity_xz = calculate_sparsity(list(zip(superseed_pareto_xz, superseed_pareto_zz)))
    sparsity_yz = calculate_sparsity(list(zip(superseed_pareto_yz, superseed_pareto_zz2)))

    return {
        "Hypervolume XY": hypervolume_xy,
        "Hypervolume XZ": hypervolume_xz,
        "Hypervolume YZ": hypervolume_yz,
        "Sparsity XY": sparsity_xy,
        "Sparsity XZ": sparsity_xz,
        "Sparsity YZ": sparsity_yz,
    }
def calculate_mc_superseed_pareto(mc_seed_paths):
    """Calculates the MC superseed set's Pareto frontier for benchmarking."""
    all_x, all_y, all_z = [], [], []
    
    # Step 1: Load and combine data from all MC seeds
    for seed_path in mc_seed_paths:
        data = load_json_data(seed_path)
        ccs_list = data['ccs_list'][-1]  # Assuming we use the last CCS data entry
        x_all, y_all, z_all = extract_coordinates(ccs_list)

        all_x.extend(x_all)
        all_y.extend(y_all)
        all_z.extend(z_all)

    # Step 2: Calculate the Pareto frontier for the combined superseed set
    pareto_points_3d = np.column_stack((all_x, all_y, all_z))
    reference_point_3d = (min(all_x), min(all_y), min(all_z))  # Minimum reference point for hypervolume

    return pareto_points_3d, reference_point_3d

def benchmark_ols_against_mc(ols_seed_paths, mc_seed_paths):
    """Benchmark OLS seed sets against the MC superseed Pareto frontier."""
    # Step 1: Calculate the MC superseed benchmark
    mc_pareto_points_3d, mc_reference_point_3d = calculate_mc_superseed_pareto(mc_seed_paths)

    # Step 2: Iterate over each OLS seed and benchmark it against the MC superseed set
    results = []
    for seed_path in ols_seed_paths:
        data = load_json_data(seed_path)
        ccs_list = data['ccs_list'][-1]
        x_all, y_all, z_all = extract_coordinates(ccs_list)

        # Combine OLS seed points
        pareto_points_3d = np.column_stack((x_all, y_all, z_all))

        # Step 3: Calculate hypervolume of OLS seed set with respect to the MC benchmark
        ols_hv_vs_mc = calculate_3d_hypervolume(pareto_points_3d, mc_reference_point_3d)

        # Step 4: Calculate sparsity of OLS seed set
        ols_sparsity_vs_mc = calculate_3d_sparsity(pareto_points_3d)

        # Step 5: Save results
        results.append({
            "Seed Path": seed_path,
            "Hypervolume vs MC": ols_hv_vs_mc,
            "Sparsity vs MC": ols_sparsity_vs_mc
        })

    return results

def calculate_all_metrics(seed_paths, wrapper):
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
    return calculate_hypervolumes_and_sparsities(seed_paths, wrapper)

def calculate_3d_metrics_only(seed_paths, wrapper):
    """
    Calculates only 3D Hypervolumes and 3D Sparsities for each seed and the superseed set.
    This is for both OLS and MC seeds.
    """
    all_x, all_y, all_z = [], [], []
    results = []

    # Calculate 3D hypervolume and sparsity for each seed
    for i, seed_path in enumerate(seed_paths):
        data = load_json_data(seed_path)
        ccs_list = data['ccs_list'][-1]
        x_all, y_all, z_all = extract_coordinates(ccs_list)

        all_x.extend(x_all)  # Aggregate data for superseed set
        all_y.extend(y_all)
        all_z.extend(z_all)

        # Calculate 3D hypervolume and sparsity for each seed
        pareto_points_3d = np.column_stack((x_all, y_all, z_all))
        reference_point_3d = (min(x_all), min(y_all), min(z_all))
        hv_3d = calculate_3d_hypervolume(pareto_points_3d, reference_point_3d)
        sparsity_3d = calculate_3d_sparsity(pareto_points_3d)

        # Append results for each seed
        results.append({
            "Seed": f"Seed {i+1}",
            "Hypervolume 3D": hv_3d,
            "Sparsity 3D": sparsity_3d
        })

    # Calculate for the superseed set
    pareto_points_superseed_3d = np.column_stack((all_x, all_y, all_z))
    reference_point_superseed_3d = (min(all_x), min(all_y), min(all_z))
    hv_superseed_3d = calculate_3d_hypervolume(pareto_points_superseed_3d, reference_point_superseed_3d)
    sparsity_superseed_3d = calculate_3d_sparsity(pareto_points_superseed_3d)

    # Append results for the superseed set
    results.append({
        "Seed": "Superseed",
        "Hypervolume 3D": hv_superseed_3d,
        "Sparsity 3D": sparsity_superseed_3d
    })

    # Convert results to DataFrame
    df_results_3d = pd.DataFrame(results)

    return df_results_3d

def calculate_3d_metrics_only_for_mc(mc_seed_path):
    """
    Calculates 3D Hypervolume and Sparsity for the MC seed (single seed case).
    This is for the MC dataset, which consists of only one seed.
    """
    # Load MC seed data
    data = load_json_data(mc_seed_path)
    ccs_list = data['ccs_list']
    x_all, y_all, z_all = extract_coordinates(ccs_git list)

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

def process_data_both(ols_seed_paths, mc_seed_paths):
    """
    Processes the data for OLS and MC seeds, compares OLS against the MC benchmark, and
    returns two DataFrames:
    1. DataFrame with all metrics (2D and 3D hypervolumes, sparsities, min/max returns, etc.).
    2. DataFrame with only 3D hypervolume and sparsity.
    """
    # Step 1: Process the full metrics for OLS seeds
    df_all_metrics_ols = calculate_all_metrics(ols_seed_paths, 'ols')

    # Step 2: Process only 3D metrics for OLS and MC seeds
    df_3d_metrics_ols = calculate_3d_metrics_only(ols_seed_paths, 'ols')
    df_3d_metrics_mc = calculate_3d_metrics_only_for_mc(mc_seed_paths)

    # Save both DataFrames to CSV (optional)
    df_all_metrics_ols.to_csv("ols_all_metrics.csv", index=False)
    df_3d_metrics_ols.to_csv("ols_3d_metrics.csv", index=False)
    df_3d_metrics_mc.to_csv("mc_3d_metrics.csv", index=False)

    print("DataFrames created successfully and saved to CSV.")

    # Return both DataFrames for further analysis or visualization
    return df_all_metrics_ols, df_3d_metrics_ols, df_3d_metrics_mc

def process_data_benchmark(ols_seed_paths, mc_seed_paths):
    """Processes the data for OLS and MC seeds, compares OLS against the MC benchmark."""
    # Step 1: Benchmark OLS seeds against MC superseed set
    benchmark_results = benchmark_ols_against_mc(ols_seed_paths, mc_seed_paths)
    
    # Step 2: Print or save benchmark results
    for result in benchmark_results:
        print(f"Seed: {result['Seed Path']}, Hypervolume vs MC: {result['Hypervolume vs MC']}, Sparsity vs MC: {result['Sparsity vs MC']}")
    
    # You can also save the results to a CSV file if needed:
    df_results = pd.DataFrame(benchmark_results)
    df_results.to_csv("ols_vs_mc_benchmark_results.csv", index=False)

# ---- Main Function ----
def main():
    ols_base_path = r"morl_logs/OLS/rte_case5_example/2024-08-17/['ScaledL2RPN', 'ScaledTopoDepth']"
    mc_base_path = r"morl_logs/MC/rte_case5_example/2024-08-19/['ScaledL2RPN', 'ScaledTopoDepth']"
    seeds = [0,1,2,3,4]
    seed_folder = "seed_0"
    json_filename = "morl_logs_mc0.json"
    ols_seed_paths = [os.path.join(ols_base_path, f'seed_{seed}', f'morl_logs_ols{seed}.json') for seed in seeds]
    mc_seed_paths = os.path.join(mc_base_path, seed_folder, json_filename) 

    # Generate both DataFrames
    df_all_metrics, df_3d_metrics_ols, df_3d_metrics_mc = process_data_both(ols_seed_paths, mc_seed_paths)

    print("All metrics (OLS):")
    print(df_all_metrics)
    print("\n3D Metrics (OLS):")
    print(df_3d_metrics_ols)
    print("\n3D Metrics (MC):")
    print(df_3d_metrics_mc)
    
    print("Processing OLS Data...")
    process_data(ols_seed_paths, 'ols')

    print("Processing MC Data...")
    #process_data(mc_seed_paths, 'mc')


if __name__ == "__main__":
    main()


