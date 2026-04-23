#!/bin/bash
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --job-name=pigvae_train
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

set -e

PROJECT_DIR=/home/tnocon/master_thesis/IMMUVIS-PIGVAE
cd "$PROJECT_DIR"

mkdir -p logs/slurm

# To resume from a checkpoint, pass: --export=ALL,CKPT_PATH=/path/to/last.ckpt
CKPT_ARG=""
if [ -n "$CKPT_PATH" ]; then
    CKPT_ARG="ckpt_path=$CKPT_PATH"
fi

uv run src/train.py \
    logger=wandb \
    trainer=gpu \
    paths=szary \
    $CKPT_ARG
