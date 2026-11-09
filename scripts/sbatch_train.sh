#!/bin/bash
#SBATCH --job-name=pigvae_abmil
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%j_abmil_train.log
cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE
uv run python scripts/run_abmil.py
