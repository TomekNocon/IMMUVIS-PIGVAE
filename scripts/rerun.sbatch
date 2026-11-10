#!/bin/bash
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=08:00:00
# Usage: sbatch --job-name=mnist_rerun  --export=ALL,WT=pigvae-mnist,DS=mnist  scripts/rerun.sbatch
#        sbatch --job-name=fmnist_rerun --export=ALL,WT=pigvae-fmnist,DS=fmnist scripts/rerun.sbatch
# (both worktrees share the pigvae-mnist venv; home quota is too tight for two)
set -e
cd /home/tnocon/master_thesis/${WT}
nvidia-smi --query-gpu=name --format=csv,noheader || echo "no GPU (CPU run)"
/home/tnocon/master_thesis/pigvae-mnist/.venv/bin/python src/train.py \
    ${EXP_ARG-experiment=example} \
    logger=wandb \
    trainer=${TRAINER:-gpu} \
    paths.log_dir=/raid_encrypted/immucan/embeddings/tnocon/logs_${DS}_rerun/ \
    logger.wandb.group=${DS}-rerun-2026-11 \
    "tags=[${DS},pigvae,thesis-rerun${TAG:+,$TAG}]" \
    ${EXTRA}
