#!/bin/bash
#SBATCH --job-name=pigvae_abmil
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/abmil_train_%j.log
cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE
uv run python scripts/run_abmil.py
