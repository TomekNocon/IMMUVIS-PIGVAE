#!/bin/bash
#SBATCH --job-name=imc_viz
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
#
# Render a raw IMC image as a per-marker grid and a false-colour composite (CPU, no model).
#
# Usage (from the repo root):
#   sbatch --export=ALL,INPUT=/raid_.../imgs/foo.tiff scripts/slurm/visualize_imc.sh
#
# Environment:
#   INPUT     .tiff image or .npy patch (required)
#   FIGURE    both | grid | composite (default: both)
#   PATCH     patch index (optional, default: whole image)
#   PANEL     panel_config yaml with marker names (optional)
#   DATASET   panel key for marker names (optional)
#   CHANNELS  composite channels, e.g. "[DNA1,CD8a,panCK]" (optional)
# Outputs go to ${paths.log_dir}/explain/imc_viz (configs/explain/visualize_imc.yaml).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f .project-root ]] || { echo "submit from the repo root" >&2; exit 1; }
[[ -n "${INPUT:-}" ]] || { echo "set INPUT to a .tiff or .npy file" >&2; exit 1; }

ARGS=(input_path="'$INPUT'" figure="${FIGURE:-both}")
[[ -n "${PATCH:-}" ]]    && ARGS+=(patch="$PATCH")
[[ -n "${PANEL:-}" ]]    && ARGS+=(panel_config="'$PANEL'")
[[ -n "${DATASET:-}" ]]  && ARGS+=(dataset="$DATASET")
[[ -n "${CHANNELS:-}" ]] && ARGS+=(composite_channels="$CHANNELS")

export HYDRA_FULL_ERROR=1
uv run python scripts/explain/visualize_imc.py "${ARGS[@]}"
