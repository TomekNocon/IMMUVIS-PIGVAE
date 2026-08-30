#!/bin/bash
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=imc_viz
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
set -e
# Professional IMC raw-input visualisation (per-marker grid + false-colour composite).
# CPU-only (no --gres=gpu) -- reads raw pixels only, no model. Needs the raid TIFFs, so it runs
# on szary rather than the login node.
#
# Run (INPUT is required; optional overrides via --export):
#   sbatch --export=ALL,INPUT=/raid_.../imgs/foo.tiff scripts/visualize_imc_slurm.sh
# Optional: FIGURE (both|grid|composite), PATCH (int index), PANEL (panel_config yaml),
#   DATASET (panel key for marker names), CHANNELS ("[DNA1,CD8,panCK]"), OUTDIR.

PROJECT_DIR=/home/tnocon/master_thesis/IMMUVIS-PIGVAE
cd "$PROJECT_DIR"
export HYDRA_FULL_ERROR=1
mkdir -p logs/slurm

if [ -z "$INPUT" ]; then
  echo "ERROR: set INPUT, e.g.:"
  echo "  sbatch --export=ALL,INPUT=/raid_.../imgs/foo.tiff scripts/visualize_imc_slurm.sh"
  exit 1
fi
FIGURE=${FIGURE:-both}
OUTDIR=${OUTDIR:-$PROJECT_DIR/explain_outputs/imc_viz}

ARGS=(input_path="'$INPUT'" figure="$FIGURE" paths=szary out_dir="$OUTDIR"
      hydra.run.dir=. hydra.output_subdir=null)
[ -n "$PATCH" ]    && ARGS+=(patch="$PATCH")
[ -n "$PANEL" ]    && ARGS+=(panel_config="'$PANEL'")
[ -n "$DATASET" ]  && ARGS+=(dataset="$DATASET")
[ -n "$CHANNELS" ] && ARGS+=(composite_channels="$CHANNELS")

echo "=== imc-viz: INPUT=$INPUT figure=$FIGURE -> $OUTDIR ==="
uv run python scripts/visualize_imc.py "${ARGS[@]}"
echo "### DONE -> $OUTDIR ###"
ls -l "$OUTDIR"
