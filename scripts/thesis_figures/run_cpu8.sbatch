#!/bin/bash
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=empty_shortcut
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
set -e
cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE
.venv/bin/python "$@"
