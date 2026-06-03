#!/bin/bash
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=6:00:00
#SBATCH --job-name=pigvae_ablation
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

set -e

PROJECT_DIR=/home/tnocon/master_thesis/IMMUVIS-PIGVAE
cd "$PROJECT_DIR"

mkdir -p logs/slurm

# Pass experiment name via: --export=ALL,EXPERIMENT=ablation_spatial_A
if [ -z "$EXPERIMENT" ]; then
    echo "ERROR: EXPERIMENT env var not set. Use: sbatch --export=ALL,EXPERIMENT=ablation_spatial_A scripts/ablation_slurm.sh"
    exit 1
fi

uv run src/train.py \
    experiment=${EXPERIMENT} \
    logger=wandb \
    trainer=gpu \
    paths=szary
