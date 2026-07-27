#!/bin/bash
# kld_loss_scale sweep @ 16x16 on the fb0.0 substrate — queue both runs on SLURM in one go.
#
# Single-knob bracket around the current best (iaktnnux, s0.01 fb0.0): only kld_loss_scale varies,
#   s0.001 (more AE-ward)  <-  [s0.01 = iaktnnux, baseline]  ->  s0.03 (more VAE-ward).
# Tests where on the AE<->VAE spectrum to sit; the deciding metric is downstream transfer, not recon.
# See research-ml kl-regularization / exp-2026-06-24-vae16-overfit-mechanism.
#
# Usage:
#   scripts/run_scale_sweep.sh                    # submit both
#   WALLTIME=10:00:00 scripts/run_scale_sweep.sh  # override walltime

set -euo pipefail

PROJECT_DIR=/home/tnocon/master_thesis/IMMUVIS-PIGVAE
cd "$PROJECT_DIR"

WALLTIME="${WALLTIME:-12:00:00}"

EXPERIMENTS=(
  vae16_s0p03    # scale 0.03  (3x more KL pressure)
  vae16_s0p001   # scale 0.001 (10x less KL pressure)
)

echo "Submitting scale sweep (walltime $WALLTIME):"
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
