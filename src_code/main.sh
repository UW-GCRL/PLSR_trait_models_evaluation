#!/bin/bash
#SBATCH --job-name=PLSR
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --time=150:00:00
#SBATCH --output=main_output.out

# Activate the conda environment
source /software/fji7/miniconda3/bin/activate /software/fji7/miniconda3/envs/Fujiang_envs
# Run the script
python main.py