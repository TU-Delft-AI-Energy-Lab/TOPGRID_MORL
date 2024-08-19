import os

# Template content
template = """#!/bin/bash

#SBATCH --job-name="TOPGRID_MORL_5bus_{seed}"
#SBATCH --time=03:00:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
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

srun python /scratch/trlautenbacher/TOPGRID_MORL/scripts/mc_MOPPO_exe.py --seed {seed} --config HPC_TenneT_config.json > morl_seed_{seed}.log

conda deactivate
"""

# Directory to store the batch scripts
batch_dir = "batch_scripts"
os.makedirs(batch_dir, exist_ok=True)

# Number of seeds
num_seeds = 5  # Change this to the number of seeds you want to use

# Create a batch file for each seed and submit
for seed in range(num_seeds):
    batch_script_content = template.format(seed=seed)
    batch_script_path = os.path.join(batch_dir, f"batch_seed_{seed}.sh")
    with open(batch_script_path, "w") as f:
        f.write(batch_script_content)

    # Submit the batch script using sbatch
    os.system(f"sbatch {batch_script_path}")