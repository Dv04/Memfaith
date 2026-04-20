#!/bin/bash
#SBATCH --job-name=memfaith_ccs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "Initializing Memfaith SLURM Compute Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"

# Load Anaconda from standard HPC module tree
module load anaconda3 || true

# Initialize conda for bash
source $(conda info --base)/etc/profile.d/conda.sh

echo "Running environment setup payload..."
bash ./setup_env.sh

echo "Activating memfaith environment..."
conda activate memfaith

echo "Starting Orchestrator..."
bash ./run_all.sh

echo "Job execution completed!"
