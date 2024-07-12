#!/bin/bash

#SBATCH --job-name="test_code_grid2op"
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

srun python /scratch/trlautenbacher/TOPGRID_MORL/scripts/MORL_execution.py > morl.log

conda deactivate