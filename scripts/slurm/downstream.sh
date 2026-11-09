#!/bin/bash
#SBATCH --job-name=pigvae_downstream
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
#
# Transfer eval of one frozen checkpoint for one per-crop feature:
# encode -> label join -> 10-fold gated-ABMIL CV.
#
# Usage (from the repo root):
#   sbatch scripts/slurm/downstream.sh <CKPT> <EXPERIMENT> <RUN_TAG> [SOURCE_TAG FEATURE_SOURCE NODE_AGG]
#   sbatch scripts/slurm/downstream.sh $CKPT pigvae pigvae_2026-07-31                  # z_global
#   sbatch scripts/slurm/downstream.sh $CKPT pigvae pigvae_2026-07-31 node_mean node mean
#
# Feature sources: raw | zglobal | node (node_agg: mean | flatten).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f .project-root ]] || { echo "submit from the repo root" >&2; exit 1; }
(( $# >= 3 )) || { echo "usage: $0 <CKPT> <EXPERIMENT> <RUN_TAG> [SOURCE_TAG FEATURE_SOURCE NODE_AGG]" >&2; exit 1; }

CKPT=$1; EXPERIMENT=$2; RUN_TAG=$3
SOURCE_TAG=${4:-zglobal}; FEATURE_SOURCE=${5:-zglobal}; NODE_AGG=${6:-mean}
RAID=/raid_encrypted/immucan/embeddings/tnocon
BUNDLE=$RAID/downstream/$RUN_TAG

echo "=== [$SOURCE_TAG] encode ==="
uv run python scripts/downstream/encode_mil_embeddings.py \
  ckpt_path="$CKPT" model_experiment="$EXPERIMENT" run_tag="$RUN_TAG" \
  feature_source="$FEATURE_SOURCE" node_agg="$NODE_AGG" source_tag="$SOURCE_TAG"

echo "=== [$SOURCE_TAG] label join ==="
uv run python scripts/downstream/build_abmil_meta.py \
  --encode_dir "$BUNDLE/mil_embeddings/$SOURCE_TAG" \
  --melted "$RAID/data/clinical/melted_table_images.csv" \
  --out_dir "$BUNDLE/meta_tables/$SOURCE_TAG" --splits train test

echo "=== [$SOURCE_TAG] 10-fold CV ==="
uv run python scripts/downstream/run_abmil.py run_tag="$RUN_TAG" source_tag="$SOURCE_TAG"
