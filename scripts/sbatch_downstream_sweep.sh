#!/bin/bash
#SBATCH --job-name=pigvae_downstream_sweep
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/downstream_sweep_%j.log
set -e
# Feature-source baseline sweep for ONE frozen checkpoint: runs the full
# encode -> label-join -> 10-fold CV pipeline for each per-crop feature, then
# collects a comparison table. All four share an identical bag/CV protocol, so
# the row-wise gap answers: does the trained encoder's z_global (or per-node
# latent) beat the raw-768 mean-pool baseline?
#   raw          : spatial mean-pool of the raw 768-d patch (no encoder)  [baseline]
#   zglobal      : 512-d CLS graph embedding
#   node_mean    : per-node latent mean-pooled (node_z_dim)
#   node_flatten : per-node latent flattened (256 * node_z_dim)
# Usage: sbatch scripts/sbatch_downstream_sweep.sh <CKPT> <EXP> <TAG>
CKPT=$1; EXP=$2; TAG=$3
B=/raid_encrypted/immucan/embeddings/tnocon
cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE

run_one () {
  local SRC=$1 FS=$2 NA=$3
  echo "### [$SRC] ENCODE ###"
  uv run python scripts/encode_mil_embeddings.py \
    ckpt_path=$CKPT model_experiment=$EXP run_tag=$TAG \
    feature_source=$FS node_agg=$NA source_tag=$SRC
  echo "### [$SRC] LABEL-JOIN ###"
  uv run python scripts/build_abmil_meta.py \
    --encode_dir $B/downstream/$TAG/mil_embeddings/$SRC \
    --melted $B/data/clinical/melted_table_images.csv \
    --out_dir $B/downstream/$TAG/meta_tables/$SRC --splits train test
  echo "### [$SRC] 10-FOLD CV ###"
  uv run python scripts/run_abmil.py run_tag=$TAG source_tag=$SRC
}

run_one raw          raw     mean
run_one zglobal      zglobal mean
run_one node_mean    node    mean
run_one node_flatten node    flatten

echo "### COLLECT COMPARISON TABLE ###"
uv run python scripts/collect_downstream_table.py --run_tag $TAG --metric auc
uv run python scripts/collect_downstream_table.py --run_tag $TAG --metric macro_f1
echo "### DONE ###"
