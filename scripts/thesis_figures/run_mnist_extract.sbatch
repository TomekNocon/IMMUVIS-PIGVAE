#!/bin/bash
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=mnist_extract
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
# Usage: sbatch scripts/thesis_figures/run_mnist_extract.sbatch <code-root worktree> <hydra run dir> <out npz> [ckpt]
set -e
CODE=$1; RUN=$2; OUT=$(realpath -m "$3"); CKPT=${4:-last.ckpt}
ls "$RUN/checkpoints"
cd "$CODE"
/home/tnocon/master_thesis/pigvae-mnist/.venv/bin/python \
  /home/tnocon/master_thesis/IMMUVIS-PIGVAE/scripts/thesis_figures/extract_mnist_recon.py \
  --code-root "$CODE" --run-dir "$RUN" --ckpt "$CKPT" --out "$OUT"
