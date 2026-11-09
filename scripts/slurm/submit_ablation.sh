#!/bin/bash
#
# Submit every config of one ablation axis as a separate training job.
#
# Usage (from the repo root, on the login node):
#   bash scripts/slurm/submit_ablation.sh kl              # all of configs/experiment/ablation/kl/
#   bash scripts/slurm/submit_ablation.sh spatial film    # several axes
#   WALLTIME=10:00:00 bash scripts/slurm/submit_ablation.sh kl
#
# Axes: the folder names under configs/experiment/ablation/.
#
# Environment:
#   WALLTIME  per-job time limit (default: 12:00:00)

set -euo pipefail
[[ -f .project-root ]] || { echo "run from the repo root" >&2; exit 1; }
(( $# > 0 )) || { echo "usage: $0 <axis> [<axis> ...]" >&2; ls configs/experiment/ablation >&2; exit 1; }

WALLTIME=${WALLTIME:-12:00:00}
mkdir -p logs/slurm

for axis in "$@"; do
  dir=configs/experiment/ablation/$axis
  [[ -d "$dir" ]] || { echo "unknown axis: $axis" >&2; exit 1; }
  for cfg in "$dir"/*.yaml; do
    exp=${cfg#configs/experiment/}
    exp=${exp%.yaml}
    job=$(sbatch --parsable --time="$WALLTIME" --export=ALL,EXPERIMENT="$exp" scripts/slurm/train.sh)
    echo "$exp -> job $job"
  done
done
