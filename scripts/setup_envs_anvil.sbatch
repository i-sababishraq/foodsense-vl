#!/bin/bash
#SBATCH --job-name=setup_envs
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/setup_envs_%j.out
#SBATCH --error=logs/setup_envs_%j.err

echo "Loading apptainer module..."
module load apptainer 2>/dev/null || true

# Point cache to scratch
export APPTAINER_CACHEDIR="/anvil/scratch/$USER/apptainer_cache"
mkdir -p "$APPTAINER_CACHEDIR"

echo "Running setup script..."
bash scripts/setup_envs.sh all
