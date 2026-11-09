#!/bin/bash
#SBATCH --job-name=mnist_extract
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
#
# Extract the MNIST / Fashion-MNIST reconstruction arrays for the Sec. 5.2.1 figure.
# Runs the model code of the `mnist` / `fashion-mnist` worktree with that worktree's venv.
#
# Usage (from the repo root):
#   sbatch scripts/slurm/mnist_extract.sh <WORKTREE> <HYDRA_RUN_DIR> <OUT_NPZ> [CKPT]
#   sbatch scripts/slurm/mnist_extract.sh ../pigvae-mnist /raid_.../runs/<run> \
#       figures/_cache/mnist_recon_f1twm4pw.npz

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f .project-root ]] || { echo "submit from the repo root" >&2; exit 1; }
(( $# >= 3 )) || { echo "usage: $0 <WORKTREE> <HYDRA_RUN_DIR> <OUT_NPZ> [CKPT]" >&2; exit 1; }

REPO=$PWD
WORKTREE=$(realpath "$1"); RUN_DIR=$2; OUT=$(realpath -m "$3"); CKPT=${4:-last.ckpt}
[[ -f "$RUN_DIR/checkpoints/$CKPT" ]] || { echo "no checkpoint $RUN_DIR/checkpoints/$CKPT" >&2; exit 1; }

cd "$WORKTREE"
"$WORKTREE/../pigvae-mnist/.venv/bin/python" "$REPO/scripts/thesis_figures/extract_mnist_recon.py" \
  --code-root "$WORKTREE" --run-dir "$RUN_DIR" --ckpt "$CKPT" --out "$OUT"
