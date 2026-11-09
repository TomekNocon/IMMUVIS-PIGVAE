#!/bin/bash
#SBATCH --job-name=pigvae_train
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
#
# Train PIGVAE (or any experiment config) on one GPU.
#
# Usage (from the repo root):
#   sbatch scripts/slurm/train.sh
#   sbatch --export=ALL,EXPERIMENT=ablation/kl/scale_1e-3 scripts/slurm/train.sh
#   sbatch --export=ALL,CKPT_PATH=/raid_.../last.ckpt scripts/slurm/train.sh   # resume
#
# Environment:
#   EXPERIMENT  experiment config under configs/experiment/ (default: pigvae)
#   CKPT_PATH   checkpoint to resume from (optional)
#   EXTRA       extra Hydra overrides, space separated (optional)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f .project-root ]] || { echo "submit from the repo root" >&2; exit 1; }

EXPERIMENT=${EXPERIMENT:-pigvae}
ARGS=(experiment="$EXPERIMENT" logger=wandb trainer=gpu paths=szary)
[[ -n "${CKPT_PATH:-}" ]] && ARGS+=(ckpt_path="$CKPT_PATH")
read -r -a EXTRA_ARGS <<< "${EXTRA:-}"

echo "=== train: experiment=$EXPERIMENT ${CKPT_PATH:+resume=$CKPT_PATH} ==="
uv run python src/train.py "${ARGS[@]}" "${EXTRA_ARGS[@]}"
