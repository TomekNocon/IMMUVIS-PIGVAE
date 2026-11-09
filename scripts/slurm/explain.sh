#!/bin/bash
#SBATCH --job-name=pigvae_explain
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
#
# Encoder attribution on a trained PIGVAE checkpoint (CPU only, so it can run next to training).
#
# Usage (from the repo root):
#   sbatch --export=ALL,CKPT=/raid_.../last.ckpt scripts/slurm/explain.sh
#   sbatch --export=ALL,CKPT=/raid_.../last.ckpt,METHOD=rollout scripts/slurm/explain.sh
#
# Environment:
#   CKPT        PIGVAE checkpoint (required)
#   EXPERIMENT  experiment config the checkpoint was trained with (default: pigvae)
#   METHOD      encoder_ig | rollout (default: encoder_ig)
#   TARGET      z_global dimension to attribute (default: 0)
#   NSTEPS      Integrated-Gradients steps (default: 50)
#   NSAMPLES    crops to explain (default: 8)
#   SPLIT       val | test (default: val)
# Outputs go to ${paths.log_dir}/explain/<METHOD> (configs/explain/model.yaml).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f .project-root ]] || { echo "submit from the repo root" >&2; exit 1; }
[[ -f "${CKPT:-}" ]] || { echo "set CKPT to an existing checkpoint" >&2; exit 1; }

export HYDRA_FULL_ERROR=1
uv run python scripts/explain/explain_model.py \
  experiment="${EXPERIMENT:-pigvae}" ckpt_path="$CKPT" \
  method="${METHOD:-encoder_ig}" target="${TARGET:-0}" n_steps="${NSTEPS:-50}" \
  num_samples="${NSAMPLES:-8}" split="${SPLIT:-val}" to_markers=true
