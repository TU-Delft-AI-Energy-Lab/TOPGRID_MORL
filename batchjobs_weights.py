import os
from itertools import product

# Template content for the batch script
template = """#!/bin/bash

#SBATCH --job-name="TOPGRID_MORL_5bus_{weights}"
#SBATCH --time=03:00:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32GB
#SBATCH --account=research-eemcs-ese

# Load modules:
module load 2023r1
module load openmpi
module load miniconda3 
module load py-pip
module load cuda

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
conda activate top

{commands}

conda deactivate
"""

# Directory to store the batch scripts
batch_dir = "batch_scripts"
os.makedirs(batch_dir, exist_ok=True)

# Define the weight vectors (example combinations)
weight_vectors = [ 
    [0.5, 0.25, 0.25], 
    [0.25, 0.25, 0.5], 
    [0.5, 0.5, 0],
    [0.25, 0.25, 0.5],
    [0.75, 0.25, 0],
    [0.25, 0.75, 0], 
    [0.2, 0.6, 0.2]    
]

# Number of seeds
num_seeds = 5  # Change this to the number of seeds you want to use

# Create a batch file for each weight vector and include all seeds
for weights in weight_vectors:
    weights_str = "_".join(map(str, weights))
    commands = ""
    for seed in range(num_seeds):
        command = f"srun python /scratch/trlautenbacher/TOPGRID_MORL/scripts/MORL_execution.py --seed {seed} --weights \"{weights_str}\" > morl_seed_{seed}_weights_{weights_str}.log\n"
        commands += command

    batch_script_content = template.format(weights=weights_str, commands=commands)
    batch_script_path = os.path.join(batch_dir, f"batch_weights_{weights_str}.sh")
    with open(batch_script_path, "w") as f:
        f.write(batch_script_content)

    # Submit the batch script using sbatch
    os.system(f"sbatch {batch_script_path}")