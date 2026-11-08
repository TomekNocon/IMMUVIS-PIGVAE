#!/bin/bash
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=thesis_gpu
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
set -e
cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE
.venv/bin/python "$@"
