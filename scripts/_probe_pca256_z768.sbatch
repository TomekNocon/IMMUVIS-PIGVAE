#!/bin/bash
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=probe_pca256_z768
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
set -e
cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE
export HYDRA_FULL_ERROR=1
# First run also auto-fits the PCA-256 model (one-time); 30 min covers the fit + one step.
uv run python scripts/_probe_pca256_z768.py
