#!/bin/bash
# free_bits sweep @ 16x16 — queue all 3 runs on SLURM in one go.
#
# Single-knob sweep (only kld_free_bits varies): 0.2 -> 0.1 -> 0.0, all else identical to
# vae16_lowkl (jtscgh92, fb 0.5 = the existing loose-end anchor). Tests the hypothesis that the
# 16x16 back-half memorisation is a too-loose bottleneck, fixable with prior pressure (not by
# cutting node_z_dim). See research-ml exp-2026-06-24-vae16-overfit-mechanism.
#
# Usage:
#   scripts/run_fb_sweep.sh            # submit all three
#   WALLTIME=10:00:00 scripts/run_fb_sweep.sh   # override walltime
#
# Each run is launched exactly as the other ablations:
#   sbatch --time=$WALLTIME --export=ALL,EXPERIMENT=<name> scripts/ablation_slurm.sh

set -euo pipefail

PROJECT_DIR=/home/tnocon/master_thesis/IMMUVIS-PIGVAE
cd "$PROJECT_DIR"

WALLTIME="${WALLTIME:-12:00:00}"

EXPERIMENTS=(
  vae16_fb0p2   # free_bits 0.2
  vae16_fb0p1   # free_bits 0.1  (primary)
  vae16_fb0p0   # free_bits 0.0  (vanilla VAE)
)

echo "Submitting free_bits sweep (walltime $WALLTIME):"
for exp in "${EXPERIMENTS[@]}"; do
  if [[ ! -f "configs/experiment/${exp}.yaml" ]]; then
    echo "  ! MISSING configs/experiment/${exp}.yaml — skipping" >&2
    continue
  fi
  jobid=$(sbatch --parsable --time="$WALLTIME" \
                 --export=ALL,EXPERIMENT="${exp}" \
                 scripts/ablation_slurm.sh)
  echo "  ${exp}  ->  job ${jobid}"
done

echo "Done. Track with: squeue --me   |   tail -f logs/slurm/<jobid>_pigvae_ablation.out"
