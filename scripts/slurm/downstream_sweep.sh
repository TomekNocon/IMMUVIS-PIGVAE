#!/bin/bash
#SBATCH --job-name=pigvae_downstream_sweep
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
#
# Transfer eval of one frozen checkpoint for all four per-crop features under one CV protocol,
# then a comparison table:
#   raw           spatial mean-pool of the raw 768-d patch (no encoder, baseline)
#   zglobal       512-d CLS graph embedding
#   node_mean     per-node latent, mean-pooled
#   node_flatten  per-node latent, flattened
#
# Usage (from the repo root):
#   sbatch scripts/slurm/downstream_sweep.sh <CKPT> <EXPERIMENT> <RUN_TAG>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f .project-root ]] || { echo "submit from the repo root" >&2; exit 1; }
(( $# == 3 )) || { echo "usage: $0 <CKPT> <EXPERIMENT> <RUN_TAG>" >&2; exit 1; }

CKPT=$1; EXPERIMENT=$2; RUN_TAG=$3
RAID=/raid_encrypted/immucan/embeddings/tnocon
BUNDLE=$RAID/downstream/$RUN_TAG

run_source() {
  local source_tag=$1 feature_source=$2 node_agg=$3
  echo "=== [$source_tag] encode ==="
  uv run python scripts/downstream/encode_mil_embeddings.py \
    ckpt_path="$CKPT" model_experiment="$EXPERIMENT" run_tag="$RUN_TAG" \
    feature_source="$feature_source" node_agg="$node_agg" source_tag="$source_tag"
  echo "=== [$source_tag] label join ==="
  uv run python scripts/downstream/build_abmil_meta.py \
    --encode_dir "$BUNDLE/mil_embeddings/$source_tag" \
    --melted "$RAID/data/clinical/melted_table_images.csv" \
    --out_dir "$BUNDLE/meta_tables/$source_tag" --splits train test
  echo "=== [$source_tag] 10-fold CV ==="
  uv run python scripts/downstream/run_abmil.py run_tag="$RUN_TAG" source_tag="$source_tag"
}

run_source raw          raw     mean
run_source zglobal      zglobal mean
run_source node_mean    node    mean
run_source node_flatten node    flatten

echo "=== comparison table ==="
for metric in auc macro_f1; do
  uv run python scripts/downstream/collect_downstream_table.py --run_tag "$RUN_TAG" --metric "$metric"
done
