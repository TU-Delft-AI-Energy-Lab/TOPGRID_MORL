import os

# Template content for the batch script
template = """#!/bin/bash

#SBATCH --job-name="TOPGRID_MORL_lr{learning_rate}_vf{vf_coef}_ent{ent_coef}_clip{clip_coef}_epochs{epoch_updates}"
#SBATCH --time=6:00:00
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

srun python /scratch/trlautenbacher/TOPGRID_MORL/scripts/ols_DOL_exe.py --config HPC_TenneT_config_14.json --learning_rate {learning_rate} --vf_coef {vf_coef} --ent_coef {ent_coef} --clip_coef {clip_coef} --epoch_updates {epoch_updates} > morl_lr{learning_rate}_vf{vf_coef}_ent{ent_coef}_clip{clip_coef}_epochs{epoch_updates}.log

conda deactivate
"""

# Directory to store the batch scripts
batch_dir = "batch_scripts"
os.makedirs(batch_dir, exist_ok=True)

# Define the hyperparameters to search over
learning_rates = [5e-4, 5e-5]
vf_coefs = [0.7, 1.0]
ent_coefs = [0.01, 0.02]
clip_coefs = [0.1, 0.3]  # Vary clip coefficient
epoch_updates = [10, 15]  # Vary number of epochs

# Nested loops to generate combinations of hyperparameters
for lr in learning_rates:
    for vf_coef in vf_coefs:
        for ent_coef in ent_coefs:
            for clip_coef in clip_coefs:
                for epoch_update in epoch_updates:
                    # Format the batch script for the current combination of hyperparameters
                    batch_script_content = template.format(
                        learning_rate=lr, 
                        vf_coef=vf_coef, 
                        ent_coef=ent_coef, 
                        clip_coef=clip_coef,
                        epoch_updates=epoch_update
                    )
                    
                    # Create a unique batch script filename
                    batch_script_path = os.path.join(batch_dir, f"batch_lr{lr}_vf{vf_coef}_ent{ent_coef}_clip{clip_coef}_epochs{epoch_update}.sh")
                    
                    # Write the batch script to a file
                    with open(batch_script_path, "w") as f:
                        f.write(batch_script_content)
                    
                    # Submit the batch script using sbatch
                    os.system(f"sbatch {batch_script_path}")