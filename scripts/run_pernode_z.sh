#!/bin/bash
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=6:00:00
#SBATCH --job-name=pigvae_pernode_z
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

set -e

PROJECT_DIR=/home/tnocon/master_thesis/IMMUVIS-PIGVAE
cd "$PROJECT_DIR"

mkdir -p logs/slurm

uv run src/train.py \
    experiment=pernode_z_diagnostic \
    logger=wandb \
    trainer=gpu \
    paths=szary
