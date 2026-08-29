#!/bin/bash
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=explain
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
set -e
# Post-hoc explainability on a trained PIGVAE checkpoint.
# CPU-only (no --gres=gpu) so it runs alongside a training job instead of queueing for the GPU.
#
# Run:
#   sbatch --export=ALL,EXPERIMENT=vae16_fb0p0_film,CKPT=/raid_.../last.ckpt,METHOD=encoder_ig \
#          scripts/explain_slurm.sh
# Optional overrides via --export: TARGET (default 0), NSTEPS (50), NSAMPLES (8), SPLIT (val),
# OUTDIR (default explain_outputs/<METHOD>). METHOD: encoder_ig | rollout.

PROJECT_DIR=/home/tnocon/master_thesis/IMMUVIS-PIGVAE
cd "$PROJECT_DIR"
export HYDRA_FULL_ERROR=1
mkdir -p logs/slurm

if [ -z "$EXPERIMENT" ] || [ -z "$CKPT" ]; then
  echo "ERROR: set EXPERIMENT and CKPT, e.g.:"
  echo "  sbatch --export=ALL,EXPERIMENT=vae16_fb0p0_film,CKPT=/raid_.../last.ckpt,METHOD=encoder_ig scripts/explain_slurm.sh"
  exit 1
fi
METHOD=${METHOD:-encoder_ig}
TARGET=${TARGET:-0}
NSTEPS=${NSTEPS:-50}
NSAMPLES=${NSAMPLES:-8}
SPLIT=${SPLIT:-val}
OUTDIR=${OUTDIR:-$PROJECT_DIR/explain_outputs/$METHOD}

[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
echo "=== explain: experiment=$EXPERIMENT method=$METHOD ckpt=$CKPT -> $OUTDIR ==="

uv run python scripts/explain_model.py \
  experiment="$EXPERIMENT" ckpt_path="$CKPT" \
  method="$METHOD" target="$TARGET" n_steps="$NSTEPS" num_samples="$NSAMPLES" split="$SPLIT" \
  to_markers=true paths=szary \
  out_dir="$OUTDIR" hydra.run.dir=. hydra.output_subdir=null

echo "### DONE -> $OUTDIR ###"
ls -l "$OUTDIR"
