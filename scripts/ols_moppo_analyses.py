import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the relative path to the JSON file
relative_path = r"morl_logs\OLS\rte_case5_example\2024-08-10\['ScaledL2RPN', 'ScaledTopoDepth']\seed_0\morl_logs_ols.json"

# Construct the absolute path
absolute_path = os.path.abspath(relative_path)

# Load the JSON data
with open(absolute_path, 'r') as file:
    data = json.load(file)

# Extract the ccs_list from the JSON data
ccs_list = data['ccs_list']
print(ccs_list[-1])
last_ccs = ccs_list[-1]
# Extract the last entry from each sublist


# Separate the coordinates for plotting
x_values = [coord[0] for coord in last_ccs]
y_values = [coord[1] for coord in last_ccs]
z_values = [coord[2] for coord in last_ccs]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_values, y_values, z_values)

# Label the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set a title for the plot
ax.set_title('3D Scatter Plot of Last Entries in CCS')

# Show the plot
plt.show()

# Flatten the list of lists into a single list of tuples
flattened_ccs = [tuple(ccs) for sublist in ccs_list for ccs in sublist]

# Separate the coordinates for plotting
x_values = [coord[0] for coord in flattened_ccs]
y_values = [coord[1] for coord in flattened_ccs]
z_values = [coord[2] for coord in flattened_ccs]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_values, y_values, z_values)

# Label the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set a title for the plot
ax.set_title('3D Scatter Plot of CCS')

# Show the plot
plt.show()
