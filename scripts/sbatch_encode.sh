#!/bin/bash
#SBATCH --job-name=pigvae_encode
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=outputs/logs/encode_full_%j.log
cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE
uv run python scripts/encode_mil_embeddings.py \
  ckpt_path=/raid_encrypted/immucan/embeddings/tnocon/logs/train/runs/2026-06-28_23-11-45/checkpoints/last.ckpt \
  model_experiment=vae16_s0p001
