#!/bin/bash
#SBATCH --job-name=pigvae_cpu
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
#
# Run any repo Python script on a CPU node that mounts the data (e.g. figure extraction).
#
# Usage (from the repo root):
#   sbatch scripts/slurm/run_cpu.sh scripts/thesis_figures/quantify_empty_shortcut.py [ARGS...]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f .project-root ]] || { echo "submit from the repo root" >&2; exit 1; }
(( $# > 0 )) || { echo "usage: $0 <script.py> [ARGS...]" >&2; exit 1; }

uv run python "$@"
