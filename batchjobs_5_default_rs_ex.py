import os

# Template content
template = """#!/bin/bash

#SBATCH --job-name="TOPGRID_MORL_5bus_{seed}_{config_name}"
#SBATCH --time=11:00:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1GB
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

srun python /scratch/trlautenbacher/TOPGRID_MORL/scripts/rs_ex_DOL_exe.py --seed {seed} --config {config_path} > morl_seed_5{seed}_{config_name}.log

conda deactivate
"""

# Directory to store the batch scripts
batch_dir = "batch_scripts"
os.makedirs(batch_dir, exist_ok=True)

# Get the list of all config files with '_5_' in their name
config_dir = "configs"  # Update this path if your config files are in a different directory
config_files = [f for f in os.listdir(config_dir) if '_default' in f and f.endswith('.json')]

# Number of seeds
num_seeds = 1  # Change this to the number of seeds you want to use

# Create a batch file for each seed and config file, then submit
for config_file in config_files:
    config_path =  config_file
    config_name = os.path.splitext(config_file)[0]  # Remove the .json extension
    
    for seed in range(num_seeds):
        batch_script_content = template.format(seed=seed, config_name=config_name, config_path=config_path)
        batch_script_path = os.path.join(batch_dir, f"batch_seed_{seed}_{config_name}.sh")
        
        with open(batch_script_path, "w") as f:
            f.write(batch_script_content)

        # Submit the batch script using sbatch
        os.system(f"sbatch {batch_script_path}")
