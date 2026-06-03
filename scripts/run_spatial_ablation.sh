#!/bin/bash
# Submit all 4 spatial ablation conditions as separate SLURM jobs.
# Run from project root: bash scripts/run_spatial_ablation.sh

set -e

for COND in A B C D; do
    sbatch --export=ALL,EXPERIMENT=ablation_spatial_${COND} \
        scripts/ablation_slurm.sh
    echo "Submitted condition ${COND}"
done
