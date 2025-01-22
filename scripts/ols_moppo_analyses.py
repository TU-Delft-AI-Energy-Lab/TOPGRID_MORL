import os
import json
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
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
import ast
import matplotlib.cm as cm
import datetime


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
    """
    Plots X vs Y, X vs Z, and Y vs Z in interactive 2D plots, highlighting Pareto frontier points and calculating hypervolumes.
    If wrapper=="mc", the input is only one seed run, the plot color is gray, and the headline is "RS-Benchmark".
    """
    # Initialize the figure
    fig = make_subplots(rows=1, cols=3, subplot_titles=[
        'L2RPN vs Topological Depth',
        'L2RPN vs Topological Actions',
        'Topological Depth vs Topological Actions'
    ])

    table_data = []

    if wrapper == "mc":
        # Assuming seed_paths is a single path or a list with one path
        if isinstance(seed_paths, list):
            seed_path = seed_paths[0]
        else:
            seed_path = seed_paths

        # Load data
        data = load_json_data(seed_path)
        ccs_list = data['ccs_list'][-1]
        x_all, y_all, z_all = extract_coordinates(ccs_list)

        # Pareto frontiers
        x_pareto_xy, y_pareto_xy, _ = pareto_frontier_2d(x_all, y_all)
        x_pareto_xz, z_pareto_xz, _ = pareto_frontier_2d(x_all, z_all)
        y_pareto_yz, z_pareto_yz, _ = pareto_frontier_2d(y_all, z_all)

        # Custom data to match the index with the table data
        row_indices = list(range(len(x_all)))

        # Plot color is gray
        gray_color = 'gray'

        # Add traces for each 2D projection
        # X vs Y
        fig.add_trace(go.Scatter(
            x=x_all, y=y_all, mode='markers',
            marker=dict(color=gray_color, opacity=0.3),
            name='RS-Benchmark (Non-Pareto)',
            customdata=row_indices
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=x_pareto_xy, y=y_pareto_xy, mode='markers+lines',
            marker=dict(color=gray_color, size=10, line=dict(width=2)),
            name='RS-Benchmark (Pareto)'
        ), row=1, col=1)

        # X vs Z
        fig.add_trace(go.Scatter(
            x=x_all, y=z_all, mode='markers',
            marker=dict(color=gray_color, opacity=0.3),
            name='RS-Benchmark (Non-Pareto)',
            customdata=row_indices
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=x_pareto_xz, y=z_pareto_xz, mode='markers+lines',
            marker=dict(color=gray_color, size=10, line=dict(width=2)),
            name='RS-Benchmark (Pareto)'
        ), row=1, col=2)

        # Y vs Z
        fig.add_trace(go.Scatter(
            x=y_all, y=z_all, mode='markers',
            marker=dict(color=gray_color, opacity=0.3),
            name='RS-Benchmark (Non-Pareto)',
            customdata=row_indices
        ), row=1, col=3)

        fig.add_trace(go.Scatter(
            x=y_pareto_yz, y=z_pareto_yz, mode='markers+lines',
            marker=dict(color=gray_color, size=10, line=dict(width=2)),
            name='RS-Benchmark (Pareto)'
        ), row=1, col=3)

        # Update layout
        fig.update_layout(
            height=600, width=1200,
            title_text="RS-Benchmark",
            template="plotly_white",
            showlegend=True
        )

        # Since hypervolumes and sparsities are not calculated, table_data can be empty or None
        df = None  # Or an empty DataFrame

    else:
        # Original code for other wrappers
        colors = px.colors.qualitative.T10  # A built-in colormap
        hypervolumes = calculate_hypervolumes_for_all_projections(seed_paths, wrapper=wrapper)
        sparsities = calculate_sparsities_for_all_projections(seed_paths, wrapper=wrapper)

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
            # X vs Y
            fig.add_trace(go.Scatter(
                x=x_all, y=y_all, mode='markers',
                marker=dict(color=colors[i % len(colors)], opacity=0.3),
                name=f'Seed {i+1} (Non-Pareto)',
                customdata=row_indices
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=x_pareto_xy, y=y_pareto_xy, mode='markers+lines',
                marker=dict(color=colors[i % len(colors)], size=10, line=dict(width=2)),
                name=f'Seed {i+1} (Pareto)'
            ), row=1, col=1)

            # X vs Z
            fig.add_trace(go.Scatter(
                x=x_all, y=z_all, mode='markers',
                marker=dict(color=colors[i % len(colors)], opacity=0.3),
                name=f'Seed {i+1} (Non-Pareto)',
                customdata=row_indices
            ), row=1, col=2)

            fig.add_trace(go.Scatter(
                x=x_pareto_xz, y=z_pareto_xz, mode='markers+lines',
                marker=dict(color=colors[i % len(colors)], size=10, line=dict(width=2)),
                name=f'Seed {i+1} (Pareto)'
            ), row=1, col=2)

            # Y vs Z
            fig.add_trace(go.Scatter(
                x=y_all, y=z_all, mode='markers',
                marker=dict(color=colors[i % len(colors)], opacity=0.3),
                name=f'Seed {i+1} (Non-Pareto)',
                customdata=row_indices
            ), row=1, col=3)

            fig.add_trace(go.Scatter(
                x=y_pareto_yz, y=z_pareto_yz, mode='markers+lines',
                marker=dict(color=colors[i % len(colors)], size=10, line=dict(width=2)),
                name=f'Seed {i+1} (Pareto)'
            ), row=1, col=3)

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

        fig.update_layout(
            height=600, width=1200,
            title_text="2D Projections of Seeds",
            template="plotly_white",
            showlegend=True
        )
        df = pd.DataFrame(table_data)
        fig.show()
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
        """Callback to display clicked point details, including substation, test actions, and test steps."""
        if clickData:
            # Extract custom data index from the clicked point
            point_index = clickData['points'][0]['customdata']
            selected_row = df_display.iloc[point_index]
            return [
                html.P(f"Seed: {selected_row['Seed']}"),
                html.P(f"X: {selected_row['X']}"),
                html.P(f"Y: {selected_row['Y']}"),
                html.P(f"Z: {selected_row['Z']}"),
                html.P(f"Substation: {selected_row['Substation']}"),
                html.P(f"Test Actions: {selected_row['Test Actions']}"),  # Display test actions
                html.P(f"Test Steps: {selected_row['Test Steps']}"),  # Display test steps
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
                    "test_chronic_0": data_entry['test_chronic_0'],
                    "test_chronic_1": data_entry['test_chronic_1']
                })
                found_match = True
                break  # Stop once a match is found
        if not found_match:
            print(f"No match found for CCS entry: {ccs_entry}")
    return matching_entries

def process_data(seed_paths, wrapper):
    """Processes the data for all seeds and generates the 3D and 2D plots."""
    all_data = []
    if wrapper == 'mc':
        seed_paths = [seed_paths]
    config_params = None
    # Create the action-to-substation mapping using the gym environment
    action_to_substation_mapping = create_action_to_substation_mapping()
    seed = 0 
    for seed_path in seed_paths:
        if not os.path.exists(seed_path):
            print(f"File not found: {seed_path}")
            continue

        data = load_json_data(seed_path)
        
        # Extract config parameters from the first seed_path
        if config_params is None:
            config = data.get('config', {})
            config_params = {
                'case_study': config.get('case_study', 'unknown'),
                'config_name': config.get('config_name', 'unknown'),
                'project_name': config.get('project_name', 'unknown'),
                'seed': config.get('seed', 'unknwon'),
                'rewards': [config.get('rewards', {}).get('second', 'unknown'), config.get('rewards', {}).get('third', 'unknown')],
                'reuse': config.get('reuse', 'none')
            }
        ccs_list = data['ccs_list'][-1]
        if wrapper == 'mc':
            ccs_list = data['ccs_list']
        ccs_data = data['ccs_data']
        matching_entries = find_matching_weights_and_agent(ccs_list, ccs_data)
        print(matching_entries)
        # Collect data for DataFrame
        for entry in matching_entries:
            chronic = entry['test_chronic_0']
            
            #actions = entry['test_actions'] # Assuming test_actions is a list of actions
            #substations = [action_to_substation_mapping.get(action, 'Unknown') for action in actions]# Get substation based on action

            all_data.append({
                "seed": seed, 
                "Returns": entry['returns'],
                "Weights": entry['weights'],
                'test_chronic_0':{
                    "Test Steps": entry['test_chronic_0']['test_steps'],
                    "Test Actions": entry['test_chronic_0']['test_actions'],
                    'Test Action Timestamp': entry['test_chronic_0']['test_action_timestamp'],
                    "Test Sub Ids": entry['test_chronic_0']["test_sub_ids"],#
                    "Test Topo Depth": entry['test_chronic_0']["test_topo_distance"]
                    
                    #"Substation": entry['test_chronic0']substations  # Add the substation to the data
                } ,
                'test_chronic_1': {
                    "Test Steps": entry['test_chronic_1']['test_steps'],
                    "Test Actions": entry['test_chronic_1']['test_actions'],
                    'Test Action Timestamp': entry['test_chronic_1']['test_action_timestamp'],
                    "Test Sub Ids": entry['test_chronic_1']["test_sub_ids"],#
                    "Test Topo Depth": entry['test_chronic_1']["test_topo_distance"]
                    #"Substation": entry['test_chronic0']substations  # Add the substation to the data
                }
                
            })
        seed+=1

    df_ccs_matching = pd.DataFrame(all_data) if all_data else pd.DataFrame()
    
    if not df_ccs_matching.empty:
        if wrapper == 'ols':
        # Construct the directory path based on the config parameters
            base_path = os.path.join(
                "morl_logs",
                config_params['case_study'],
                config_params['config_name'],
                config_params['project_name'],
                datetime.date.today().strftime('%Y-%m-%d'),
                str(config_params['rewards']),
                f"re_{config_params['reuse']}"
            )

            # Create the directory if it doesn't exist
            if not os.path.exists(base_path):
                os.makedirs(base_path)
                print(f"Created directory: {base_path}")

            # Save the DataFrame to the constructed path
            csv_file_path = os.path.join(base_path, "ccs_matching_data.csv")
            df_ccs_matching.to_csv(csv_file_path, index=False)
        elif wrapper == 'mc':
            
            # Construct the directory path based on the config parameters
            base_path = os.path.join(
                "MC_logs",
                config_params['case_study'],
                config_params['config_name'],
                config_params['project_name'],
                datetime.date.today().strftime('%Y-%m-%d'),
                str(config_params['rewards']),
                f"re_{config_params['reuse']}"
            )

            # Create the directory if it doesn't exist
            if not os.path.exists(base_path):
                os.makedirs(base_path)
                print(f"Created directory: {base_path}")

            # Save the DataFrame to the constructed path
            csv_file_path = os.path.join(base_path, "ccs_matching_data.csv")
            df_ccs_matching.to_csv(csv_file_path, index=False)

    if not df_ccs_matching.empty:
        df_ccs_matching.to_csv("ccs_matching_data.csv", index=False)
        print(f'machting entries on ccs {df_ccs_matching}')
    # Call the function to calculate hypervolumes and sparsities and output the DataFrame
    df_metrics = calculate_hypervolumes_and_sparsities(seed_paths, wrapper)
    print(df_metrics)
    
    plot_2d_projections_matplotlib(seed_paths, wrapper)   # Matplotlib-based visualization
    # Call the plotting functions
    #plot_all_seeds(seed_paths, wrapper, df_ccs_matching)  # Dash-based visualization
    return df_ccs_matching
    



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
    df_display['Test Actions'] = df_ccs_matching['Test Actions']
    df_display['Test Steps'] = df_ccs_matching['Test Steps']
    
    create_dash_app(fig, df_display)

# ---- 2D Plotting with Matplotlib (with Superseed Pareto Markings) ----
def plot_2d_projections_matplotlib(seed_paths, wrapper):
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

        axs[0].set_xlabel('ScaledLinesCapacity')
        axs[0].set_ylabel('ScaledL2RPN')
        axs[0].set_title('ScaledLinesCapacity vs ScaledL2RPN')

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

        axs[1].set_xlabel('ScaledLinesCapacity')
        axs[1].set_ylabel('ScaledTopoDepth')
        axs[1].set_title('ScaledLinesCapacity vs ScaledTopoDepth')

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

        axs[2].set_xlabel('ScaledL2RPN')
        axs[2].set_ylabel('ScaledTopoDepth')
        axs[2].set_title('ScaledL2RPN vs ScaledTopoDepth')

        for ax in axs:
            ax.legend()
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        plt.suptitle('RS-Benchmark', fontsize=20)
        plt.subplots_adjust(top=0.88)  # Adjust the top to make room for suptitle
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

            axs[0].set_xlabel('ScaledLinesCapacity')
            axs[0].set_ylabel('ScaledL2RPN')
            axs[0].set_title('ScaledLinesCapacity vs ScaledL2RPN')

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

            axs[1].set_xlabel('ScaledLinesCapacity')
            axs[1].set_ylabel('ScaledTopoDepth')
            axs[1].set_title('ScaledLinesCapacity vs ScaledTopoDepth')

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

            axs[2].set_xlabel('ScaledL2RPN')
            axs[2].set_ylabel('ScaledTopoDepth')
            axs[2].set_title('ScaledL2RPN vs ScaledTopoDepth')

        for ax in axs:
            ax.legend()
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        plt.suptitle('2D Projections of Seeds', fontsize=20)
        plt.subplots_adjust(top=0.88)  # Adjust the top to make room for suptitle
        plt.show()

# Helper Functions
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


def calculate_hypervolumes_and_sparsities(seed_paths, wrapper, mc_seed_path=None):
    """
    Calculates the 2D and 3D hypervolumes, sparsities, min/max returns, and Pareto points for each seed and the superseed set.
    Also calculates the mean and std of hypervolumes, sparsities, and returns across seeds.
    If mc_seed_path is provided, includes its results in the DataFrame, but mean, std, and superseed are calculated only on OLS data.
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
    
    # Global tracking for the superseed set (OLS data only)
    min_return_x_all = float('inf')
    max_return_x_all = float('-inf')
    min_return_y_all = float('inf')
    max_return_y_all = float('-inf')
    min_return_z_all = float('inf')
    max_return_z_all = float('-inf')
    
    # Process OLS seeds (seed_paths)
    for i, seed_path in enumerate(seed_paths):
        data = load_json_data(seed_path)
        ccs_list = data['ccs_list'][-1]
        x_all, y_all, z_all = extract_coordinates(ccs_list)
        
        # Add data to the superseed lists
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
        
        # Calculate 2D hypervolume and sparsity (XY, XZ, YZ)
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
            "Hypervolume XY": round(hv_xy, 2),
            "Hypervolume XZ": round(hv_xz, 2),
            "Hypervolume YZ": round(hv_yz, 2),
            "Sparsity XY": round(sparsity_xy, 2),
            "Sparsity XZ": round(sparsity_xz, 2),
            "Sparsity YZ": round(sparsity_yz, 2),
            "Hypervolume 3D": round(hv_3d, 2),
            "Sparsity 3D": round(sparsity_3d, 2),
            "Min Return X": round(min_return_x, 2),
            "Max Return X": round(max_return_x, 2),
            "Min Return Y": round(min_return_y, 2),
            "Max Return Y": round(max_return_y, 2),
            "Min Return Z": round(min_return_z, 2),
            "Max Return Z": round(max_return_z, 2),
            "Pareto Points Count": pareto_points_count  # Integer, no rounding needed
        })
    
    # Calculate mean and std for each metric over all seeds (OLS data only)
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
    
    # Calculate for the superseed set (OLS data only)
    superseed_results = calculate_hypervolume_and_sparsity_superseed(all_x, all_y, all_z)
    
    pareto_points_superseed_3d = np.column_stack((all_x, all_y, all_z))
    reference_point_superseed_3d = (min(all_x), min(all_y), min(all_z))
    hv_superseed_3d = calculate_3d_hypervolume(pareto_points_superseed_3d, reference_point_superseed_3d)
    sparsity_superseed_3d = calculate_3d_sparsity(pareto_points_superseed_3d)
    
    # Append results for the superseed set
    # Append results for the superseed set, rounding values
    results.append({
        "Seed": "Superseed",
        "Hypervolume XY": round(superseed_results["Hypervolume XY"], 2),
        "Hypervolume XZ": round(superseed_results["Hypervolume XZ"], 2),
        "Hypervolume YZ": round(superseed_results["Hypervolume YZ"], 2),
        "Sparsity XY": round(superseed_results["Sparsity XY"], 2),
        "Sparsity XZ": round(superseed_results["Sparsity XZ"], 2),
        "Sparsity YZ": round(superseed_results["Sparsity YZ"], 2),
        "Hypervolume 3D": round(hv_superseed_3d, 2),
        "Sparsity 3D": round(sparsity_superseed_3d, 2),
        "Min Return X": round(min_return_x_all, 2),
        "Max Return X": round(max_return_x_all, 2),
        "Min Return Y": round(min_return_y_all, 2),
        "Max Return Y": round(max_return_y_all, 2),
        "Min Return Z": round(min_return_z_all, 2),
        "Max Return Z": round(max_return_z_all, 2),
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
    
    # Now, if mc_seed_path is provided, process it and append its results to the DataFrame
    if mc_seed_path is not None:
        # Process MC benchmark data
        data = load_json_data(mc_seed_path)
        ccs_list = data['ccs_list'][-1]
        x_all, y_all, z_all = extract_coordinates(ccs_list)
        
        # Calculate min and max returns for MC benchmark
        min_return_x = min(x_all)
        max_return_x = max(x_all)
        min_return_y = min(y_all)
        max_return_y = max(y_all)
        min_return_z = min(z_all)
        max_return_z = max(z_all)
        
        # Calculate 2D hypervolumes
        reference_point_xy = (min(x_all), min(y_all))
        reference_point_xz = (min(x_all), min(z_all))
        reference_point_yz = (min(y_all), min(z_all))
        
        x_pareto_xy, y_pareto_xy, pareto_xy_indices = pareto_frontier_2d(x_all, y_all)
        x_pareto_xz, z_pareto_xz, pareto_xz_indices = pareto_frontier_2d(x_all, z_all)
        y_pareto_yz, z_pareto_yz, pareto_yz_indices = pareto_frontier_2d(y_all, z_all)
        
        # Number of Pareto points
        pareto_points_count = len(pareto_xy_indices)
        
        # Calculate hypervolumes
        hv_xy = calculate_hypervolume(list(zip(x_pareto_xy, y_pareto_xy)), reference_point_xy)
        hv_xz = calculate_hypervolume(list(zip(x_pareto_xz, z_pareto_xz)), reference_point_xz)
        hv_yz = calculate_hypervolume(list(zip(y_pareto_yz, z_pareto_yz)), reference_point_yz)
        
        # Calculate sparsities
        sparsity_xy = calculate_sparsity(list(zip(x_pareto_xy, y_pareto_xy)))
        sparsity_xz = calculate_sparsity(list(zip(x_pareto_xz, z_pareto_xz)))
        sparsity_yz = calculate_sparsity(list(zip(y_pareto_yz, z_pareto_yz)))
        
        # Calculate 3D hypervolume and sparsity
        pareto_points_3d = np.column_stack((x_all, y_all, z_all))
        reference_point_3d = (min(x_all), min(y_all), min(z_all))
        hv_3d = calculate_3d_hypervolume(pareto_points_3d, reference_point_3d)
        sparsity_3d = calculate_3d_sparsity(pareto_points_3d)
        
         # Append results for MC benchmark, rounding values
        results.append({
            "Seed": "MC Benchmark",
            "Hypervolume XY": round(hv_xy, 2),
            "Hypervolume XZ": round(hv_xz, 2),
            "Hypervolume YZ": round(hv_yz, 2),
            "Sparsity XY": round(sparsity_xy, 2),
            "Sparsity XZ": round(sparsity_xz, 2),
            "Sparsity YZ": round(sparsity_yz, 2),
            "Hypervolume 3D": round(hv_3d, 2),
            "Sparsity 3D": round(sparsity_3d, 2),
            "Min Return X": round(min_return_x, 2),
            "Max Return X": round(max_return_x, 2),
            "Min Return Y": round(min_return_y, 2),
            "Max Return Y": round(max_return_y, 2),
            "Min Return Z": round(min_return_z, 2),
            "Max Return Z": round(max_return_z, 2),
            "Pareto Points Count": pareto_points_count  # Integer, no rounding needed
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

def calculate_all_metrics(seed_paths, wrapper, mc_seed_path):
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

def process_data_both(ols_seed_paths, mc_seed_paths):
    """
    Processes the data for OLS and MC seeds, compares OLS against the MC benchmark, and
    returns two DataFrames:
    1. DataFrame with all metrics (2D and 3D hypervolumes, sparsities, min/max returns, etc.).
    2. DataFrame with only 3D hypervolume and sparsity.
    """
    # Step 1: Process the full metrics for OLS seeds
    df_all_metrics_ols = calculate_all_metrics(ols_seed_paths, 'ols', mc_seed_paths)

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
    
def process_single_seed(seed_path):
    # Load data for a single seed path
    data = load_json_data(seed_path)
    ccs_list = data['ccs_list'][-1]  # Assuming you're interested in the last entry of 'ccs_list'
    
    # Extract coordinates from ccs_list
    x_all, y_all, z_all = extract_coordinates(ccs_list)

    # Calculate Pareto frontiers for different projections
    x_pareto_xy, y_pareto_xy, _ = pareto_frontier_2d(x_all, y_all)
    x_pareto_xz, z_pareto_xz, _ = pareto_frontier_2d(x_all, z_all)
    y_pareto_yz, z_pareto_yz, _ = pareto_frontier_2d(y_all, z_all)

    # Set up the plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['blue']  # Using just one color since we are working with one seed

    # Plot for X vs Y
    axs[0].scatter(x_all, y_all, color=colors[0], alpha=0.3, label=f'Seed (Non-Pareto)')
    axs[0].plot(x_pareto_xy, y_pareto_xy, color=colors[0], marker='o', label=f'Seed (Pareto)')
    axs[0].set_xlabel('L2RPN')
    axs[0].set_ylabel('TopoDepth Reward')
    axs[0].set_title('x')
    axs[0].legend()

    # Plot for X vs Z
    axs[1].scatter(x_all, z_all, color=colors[0], alpha=0.3, label=f'Seed (Non-Pareto)')
    axs[1].plot(x_pareto_xz, z_pareto_xz, color=colors[0], marker='o', label=f'Seed (Pareto)')
    axs[1].set_xlabel('L2RPN')
    axs[1].set_ylabel('TopoAction Reward')
    axs[1].set_title('L2RPn vs Topo Action Reward')
    axs[1].legend()

    # Plot for Y vs Z
    axs[2].scatter(y_all, z_all, color=colors[0], alpha=0.3, label=f'Seed (Non-Pareto)')
    axs[2].plot(y_pareto_yz, z_pareto_yz, color=colors[0], marker='o', label=f'Seed (Pareto)')
    axs[2].set_xlabel('TopoDepth Reward')
    axs[2].set_ylabel('TopoAction Reward')
    axs[2].set_title('TopoDepth vs TopoAction')
    axs[2].legend()

    # Display the plots
    plt.tight_layout()
    plt.show()
    
def process_data_mc(mc_seed_path, wrapper):
    all_data = []
    
    # Create the action-to-substation mapping using the gym environment
    action_to_substation_mapping = create_action_to_substation_mapping()
    data = load_json_data(mc_seed_path)
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
    
    plot_2d_projections_seeds(mc_seed_path, wrapper='mc')   # Matplotlib-based visualization
    # Call the plotting functions
    #plot_all_seeds(seed_paths, wrapper, df_ccs_matching)  # Dash-based visualization



# Define the function to process the CSV data
# Define the function to process the CSV data
# Define the function to process the CSV data
# Define the function to process the CSV data
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
            steps = row[chronic]['Test Steps']
            actions = row[chronic]['Test Actions']
            timestamps = list(map(float, row[chronic]['Test Action Timestamp']))
            sub_ids = row[chronic]['Test Sub Ids']

            # Different marker for each chronic
            marker = 'o' if chronic == 'test_chronic_0' else 's'
            chronic_label = f"{label} ({chronic})"

            # Plot each action on the graph
            for timestamp, sub_id, action in zip(timestamps, sub_ids, actions):
                for sub in sub_id:
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


# Define the function to process the CSV data
def topo_depth_process_and_plot(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract information from the test_chronic columns
    df['test_chronic_0'] = df['test_chronic_0'].apply(ast.literal_eval)
    
    # Initialize the plot with subplots for each Pareto point
    fig, axes = plt.subplots(len(df), 1, figsize=(12, 6 * len(df)))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.4)
    if len(df) == 1:
        axes = [axes]
    
    # Generate colors for each Pareto point
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))

    # Iterate through each row in the dataframe
    for idx, (ax, row) in enumerate(zip(axes, df.iterrows())):
        _, row = row
        color = colors[idx % len(colors)]
        weights = [round(float(w), 2) for w in ast.literal_eval(row['Weights'])]
        label = f"Pareto Point {idx+1}: Weights {weights}"
        
        # Extract the timestamp and topological depth information from test_chronic_0
        chronic = 'test_chronic_0'
        steps = row[chronic]['Test Steps']
        actions = row[chronic]['Test Actions']
        timestamps = [0.0] + list(map(float, row[chronic]['Test Action Timestamp']))
        topo_depths = [0.0] + [0.0 if t is None else t for t in row[chronic]['Test Topo Depth']]

        # Different marker for chronic_0
        marker = 'o'
        chronic_label = f"{label} ({chronic})"

        # Plot each action on the graph and fill the area underneath
        for i in range(len(timestamps) - 1):
            if topo_depths[i] is not None and topo_depths[i + 1] is not None:
                # Draw rectangular lines connecting the points starting from (0,0)
                ax.plot([timestamps[i], timestamps[i + 1]],
                        [topo_depths[i], topo_depths[i]],
                        color=color, linestyle='-', linewidth=1)
                ax.plot([timestamps[i + 1], timestamps[i + 1]],
                        [topo_depths[i], topo_depths[i + 1]],
                        color=color, linestyle='-', linewidth=1)
                # Fill the area underneath the rectangular lines
                ax.fill_between([timestamps[i], timestamps[i + 1]], 0, topo_depths[i],
                                 color=color, alpha=0.3)

        # Plot each action on the graph with markers
        for j, (timestamp, topo_depth, action) in enumerate(zip(timestamps, topo_depths, actions)):
            if topo_depth is not None:
                if j == len(timestamps) - 1:
                    # Mark the last point with a distinct edge color
                    ax.plot(timestamp, topo_depth, marker, color=color, markeredgecolor='red', markersize=8, label=chronic_label)
                else:
                    ax.plot(timestamp, topo_depth, marker, color=color, label=chronic_label)
                chronic_label = ""  # Avoid repeated labels in legend

        # Formatting each subplot
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Topological Depth Affected by Switching")
        ax.set_title(f"Topological Depth Modifications for Pareto Point {idx+1}")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize='small')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

# Define the function to analyze Pareto frontier values
def analyse_pf_values(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract information from the test_chronic columns
    df['test_chronic_0'] = df['test_chronic_0'].apply(ast.literal_eval)
    
    results = []
    seed = 0
    # Iterate through each row in the dataframe
    for idx, row in df.iterrows():
        # Extract information from test_chronic_0
        chronic = 'test_chronic_0'
        seed = row['seed']
        steps = row[chronic]['Test Steps']
        actions = row[chronic]['Test Actions']
        sub_ids = row[chronic]['Test Sub Ids']

        # Calculate average steps
        avg_steps = steps / len(actions) if len(actions) > 0 else 0

        # Count the frequency of actions
        action_counts = {action: actions.count(action) for action in set(actions)}

        # Count the frequency of substation modifications
        substation_counts = {}
        for sub_id_list in sub_ids:
            for sub_id in sub_id_list:
                if sub_id is not None:
                    if sub_id not in substation_counts:
                        substation_counts[sub_id] = 0
                    substation_counts[sub_id] += 1

        # Extract and round weights
        weights = [round(float(w), 2) for w in ast.literal_eval(row['Weights'])]

        # Store the results for the current Pareto point
        results.append({
            'seed': seed,
            'Pareto Point': idx + 1,
            'Average Steps': avg_steps,
            'Action Counts': action_counts,
            'Substation Modification Counts': substation_counts,
            'Weights': weights
        })
    
    # Print the results
    for result in results:
        print(f"Seed {seed}: Pareto Point {result['Pareto Point']}:\n Weights: {result['Weights']} \n Average Steps: {result['Average Steps']} \n Action Counts: {result['Action Counts']} \n Substation Modification Counts: {result['Substation Modification Counts']}")

def analyse_pf_values_and_plot(csv_path):
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
        steps = row[chronic]['Test Steps']
        actions = row[chronic]['Test Actions']
        sub_ids = row[chronic]['Test Sub Ids']
    
        # Calculate average steps per chronic (assuming each action corresponds to a chronic)
        num_chronics = len(actions) if len(actions) > 0 else 1
        avg_steps = steps / num_chronics
    
        # Total number of switching actions
        total_switching_actions = len(actions)
    
        # Extract and round weights
        weights = [round(float(w), 2) for w in ast.literal_eval(row['Weights'])]
    
        # Store the results for the current Pareto point
        results.append({
            'seed': seed,
            'Pareto Point': idx + 1,
            'Average Steps': avg_steps,
            'Total Switching Actions': total_switching_actions,
            'Weights': weights
        })
        
    # Prepare data for plotting
    avg_steps_list = [result['Average Steps'] for result in results]
    total_actions_list = [result['Total Switching Actions'] for result in results]
    weights_list = [result['Weights'] for result in results]
    pareto_points = [result['Pareto Point'] for result in results]

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(avg_steps_list, total_actions_list, color='blue')

    # Annotate each point with its weight vector
    for avg_steps, total_actions, weights in zip(avg_steps_list, total_actions_list, weights_list):
        plt.annotate(f"{weights}", (avg_steps, total_actions), textcoords="offset points", xytext=(0,10), ha='center')

    # Set plot labels and title
    plt.xlabel('Average Steps')
    plt.ylabel('Total Number of Switching Actions')
    plt.title('Trade-offs between Average Steps and Total Switching Actions')
    plt.grid(True)
    plt.show()

def analyse_pf_values_and_plot_projections(csv_path):
    import matplotlib.pyplot as plt
    import numpy as np
    import ast
    from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting

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
        steps = test_data['Test Steps']  # Total steps in the test
        actions = test_data['Test Actions']  # List of actions taken
        topo_depths = test_data.get('Test Topo Depth')
        timestamps = test_data.get('Test Action Timestamp')
        
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
            'Weights': weights
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
    
    # --- Helper Functions ---
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
    
    # --- 3D Scatter Plot ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for seed in seeds:
        seed_data = results_df[results_df['seed'] == seed]
        avg_steps_list = seed_data['Average Steps'].values
        total_actions_list = seed_data['Total Switching Actions'].values
        weighted_depth_list = seed_data['Weighted Depth Metric'].values
        weights_list = seed_data['Weights'].values
        
        color = color_map[seed]
        
        # Plot data points for this seed
        ax.scatter(avg_steps_list, total_actions_list, weighted_depth_list,
                   color=color, marker='o', label=f'Seed {seed}', alpha=0.7)
        
        # Alternate annotation positions
        offsets = [(-20, 10), (20, -10), (-20, -10), (20, 10)]
        
        # Annotate extrema weights and points with max steps
        for idx, (avg_steps, total_actions, weighted_depth, weights) in enumerate(zip(avg_steps_list, total_actions_list, weighted_depth_list, weights_list)):
            annotate = False
            label = ''
            if is_extreme_weight(weights):
                annotate = True
                label = weight_label(weights)
            elif avg_steps >= 2016:
                annotate = True
                label = weight_label(weights)
            if annotate:
                # Adjust the annotation position
                xytext = offsets[idx % len(offsets)]
                ax.text(avg_steps, total_actions, weighted_depth, f"{label}", size=9, zorder=1, color='k')
    
    # Set plot labels and title
    ax.set_xlabel('Average Steps (Higher is Better)')
    ax.set_ylabel('Total Switching Actions (Lower is Better)')
    ax.set_zlabel('Weighted Depth Metric (Lower is Better)')
    ax.set_title('Trade-offs among Average Steps, Total Switching Actions, and Weighted Depth Metric')
    
    ax.legend(title='Seeds')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # --- 2D Projections ---
    fig2, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    for seed in seeds:
        seed_data = results_df[results_df['seed'] == seed]
        avg_steps_list = seed_data['Average Steps'].values
        total_actions_list = seed_data['Total Switching Actions'].values
        weighted_depth_list = seed_data['Weighted Depth Metric'].values
        weights_list = seed_data['Weights'].values
        
        color = color_map[seed]
        
        # X vs Y
        axs[0].scatter(avg_steps_list, total_actions_list, color=color, alpha=0.5, label=f'Seed {seed} Data')
        
        # Alternate annotation positions
        offsets = [(-20, 10), (20, -10), (-20, -10), (20, 10)]
        
        for idx, (x, y, weights) in enumerate(zip(avg_steps_list, total_actions_list, weights_list)):
            annotate = False
            label = ''
            #if is_extreme_weight(weights):
            #    annotate = True
            #    label = weight_label(weights)
            #elif x >= 2016:
            #    annotate = True
            #    label = weight_label(weights)
            if annotate:
                xytext = offsets[idx % len(offsets)]
                axs[0].annotate(f"{label}", (x, y), textcoords="offset points", xytext=xytext, ha='center', fontsize=12,
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))
        
        axs[0].set_xlabel('Average Steps ')
        axs[0].set_ylabel('Weighted Depth Metric')
        axs[0].set_title('Average Steps vs Topological Depth')
        
        # X vs Z
        axs[1].scatter(avg_steps_list, weighted_depth_list, color=color, alpha=0.5, label=f'Seed {seed} Data')
        
        for idx, (x, z, weights) in enumerate(zip(avg_steps_list, weighted_depth_list, weights_list)):
            annotate = False
            label = ''
            #if is_extreme_weight(weights):
            #    annotate = True
            #    label = weight_label(weights)
            #elif x >= 2016:
            #    annotate = True
            #    label = weight_label(weights)
            if annotate:
                xytext = offsets[idx % len(offsets)]
                axs[1].annotate(f"{label}", (x, z), textcoords="offset points", xytext=xytext, ha='center', fontsize=12,
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))
        
        axs[1].set_xlabel('Average Steps ')
        axs[1].set_ylabel('Total Switching actions')
        axs[1].set_title('Average Steps vs Total Switching actions')
        
        # Y vs Z
        axs[2].scatter(total_actions_list, weighted_depth_list, color=color, alpha=0.5, label=f'Seed {seed} Data')
        
        for idx, (y, z, weights, x) in enumerate(zip(total_actions_list, weighted_depth_list, weights_list, avg_steps_list)):
            annotate = False
            label = ''
            #if is_extreme_weight(weights):
            #    annotate = True
            #    label = weight_label(weights)
            #elif x >= 2016:
                #annotate = True
                #label = weight_label(weights)
            if annotate:
                xytext = offsets[idx % len(offsets)]
                axs[2].annotate(f"{label}", (y, z), textcoords="offset points", xytext=xytext, ha='center', fontsize=12,
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))
        
        axs[2].set_ylabel('Total Switching Actions (Lower is Better)')
        axs[2].set_xlabel('Weighted Depth Metric (Lower is Better)')
        axs[2].set_title('Weighted Depth Metric vs Total Switching Actions')
    
    for ax in axs:
        ax.legend()
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    plt.show()


# ---- Main Function ----
def main():
    ols_base_path = r"morl_logs/opponent/OLS/rte_case5_example/2024-10-11/['TopoDepth', 'TopoActionHour']/re_none/op_True/rho_0.95"
    mc_base_path = r"morl_logs/trial/MC/rte_case5_example/2024-10-11/['TopoDepth', 'TopoActionHour']/re_partial/op_False/rho_0.95/morl_logs_seed_0.json"
    seeds = [0,1,2]
    ols_seed_paths = [os.path.join(ols_base_path, f'morl_logs_seed_{seed}.json') for seed in seeds]
    mc_seed_paths = mc_base_path
    if not os.path.exists(mc_seed_paths):
        raise FileNotFoundError(f"MC file not found at path: {mc_seed_paths}")
    # Generate both DataFrames
    df_all_metrics, df_3d_metrics_ols, df_3d_metrics_mc = process_data_both(ols_seed_paths, mc_seed_paths)

    print("All metrics (OLS):")
    print(df_all_metrics)
    print("\n3D Metrics (OLS):")
    print(df_3d_metrics_ols)
    print("\n3D Metrics (MC):")
    print(df_3d_metrics_mc)
    #plot_2d_projections_seeds(mc_seed_paths, wrapper='mc')
    #plot_2d_projections_seeds(ols_seed_paths, wrapper='ols')
    print("Processing OLS Data...")
    df_ccs_matching = process_data(ols_seed_paths, 'ols')
    print('Processing MC data')
    process_data(mc_seed_paths, 'mc')
    
    
    analyse_pf_values(csv_path="morl_logs/opponent/TenneT/TOPGRID_MORL_5bus_HPC_trial/2024-10-11/['TopoDepth', 'TopoActionHour']/re_None/ccs_matching_data.csv")
    process_data_mc
    #analyse_pf_values_and_plot(csv_path="morl_logs/trial/base/TOPGRID_MORL_5bus/2024-10-11/['TopoDepth', 'TopoActionHour']/re_partial/ccs_matching_data.csv")
    analyse_pf_values_and_plot_projections(csv_path="morl_logs/opponent/TenneT/TOPGRID_MORL_5bus_HPC_trial/2024-10-11/['TopoDepth', 'TopoActionHour']/re_None/ccs_matching_data.csv")
    #sub_id_process_and_plot(csv_path='ccs_matching_data.csv')
    #topo_depth_process_and_plot(csv_path='ccs_matching_data.csv')
    #print("Processing MC Data...")
    #process_data_mc(mc_seed_paths, 'mc')


if __name__ == "__main__":
    main()


