#!/bin/bash
#SBATCH --job-name=pigvae_bundle
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
#
# Export the deployable model bundle (ImmuVis, PCA, PIGVAE, ABMIL head + parity check on raw
# TIFFs) and upload it to the Modal Volume the app serves from. Runs on szary: it needs raid, and
# szary has internet for the upload (the Modal login in ~/.modal.toml is shared via /home).
# CPU only, so it can run next to training.
#
# Usage (from the repo root):   make modal-bundle      or
#   sbatch scripts/slurm/deploy_bundle.sh
#
# Environment:
#   LABEL   clinical label of the ABMIL head (default: DX.name; the app serves bundle_DX.name)
#   FOLD    CV fold to export (default: best validation AUROC)
#   OUT     parent directory of the bundle (default: the raid deploy folder)
# Afterwards: `make modal-deploy` (only needed if the app is not deployed yet; running containers
# pick up the new Volume contents when they restart).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f .project-root ]] || { echo "submit from the repo root" >&2; exit 1; }

LABEL="${LABEL:-DX.name}"
OUT="${OUT:-/raid_encrypted/immucan/embeddings/tnocon/deploy}"
BUNDLE="${OUT}/bundle_${LABEL}"
VOLUME="pigvae-imc-bundle"

uv run python scripts/deploy/export_bundle.py --out "${BUNDLE}" --label "${LABEL}" \
    ${FOLD:+--fold "${FOLD}"} --device cpu

uv run modal volume create "${VOLUME}" 2>/dev/null || true   # exists already after the first run
uv run modal volume put --force "${VOLUME}" "${BUNDLE}" /
echo "uploaded ${BUNDLE} to Modal Volume ${VOLUME}:/bundle_${LABEL}"
