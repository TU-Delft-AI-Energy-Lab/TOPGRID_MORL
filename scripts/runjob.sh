#!/bin/bash

#SBATCH --job-name="test_code_grid2op"
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-eemcs-ese

# Load modules:
module load 2023r1
module load openmpi
module load miniconda3 
module load py-pip

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
conda activate top

srun python MORL_execution.py > morl.log

conda deactivate