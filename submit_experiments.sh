#!/bin/bash
#SBATCH --job-name=sst2_acs
#SBATCH --output=logs/sst2_acs_%j.out
#SBATCH --error=logs/sst2_acs_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Ensure logs directory exists
mkdir -p logs

# Activate environment (User should adjust this if needed)
# source activate acs_env 
# OR
# module load python/3.8 cuda/11.0

echo "Starting SST2 ACS Experiments"
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"

python run_experiments.py

echo "Job Finished"
