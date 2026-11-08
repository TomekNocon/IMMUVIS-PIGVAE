#!/bin/bash
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=radius_err
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
set -e
cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE
.venv/bin/python scripts/thesis_figures/extract_recon_error_run.py --run /raid_encrypted/immucan/embeddings/tnocon/logs/train/runs/2026-06-06_16-24-41 --tag r1_cuogzab1 --no-final-norm
