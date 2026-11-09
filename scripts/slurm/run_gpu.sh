#!/bin/bash
#SBATCH --job-name=pigvae_gpu
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
#
# Run any repo Python script on a GPU node (e.g. figure extraction through the model).
#
# Usage (from the repo root):
#   sbatch scripts/slurm/run_gpu.sh scripts/thesis_figures/extract_marker_ablation.py [ARGS...]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f .project-root ]] || { echo "submit from the repo root" >&2; exit 1; }
(( $# > 0 )) || { echo "usage: $0 <script.py> [ARGS...]" >&2; exit 1; }

uv run python "$@"
