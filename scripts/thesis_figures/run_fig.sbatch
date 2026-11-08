#!/bin/bash
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=thesis_fig
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
set -e
cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE
.venv/bin/python scripts/thesis_figures/fig_cords_crop.py "$@"
