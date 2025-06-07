#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:10:00
#SBATCH --job-name=md_sim
#SBATCH --output=md_sim.out

# Load modules if needed
module load cuda/12.2
module load cuda/12.5.1 
module load nvhpc/23.5 
module load gcc/12.1.0
module load cmake/3.21.4
module avail python/3.12-conda

# Run your workflow
bash run_all.sh