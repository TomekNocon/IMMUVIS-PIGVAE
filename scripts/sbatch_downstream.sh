#!/bin/bash
#SBATCH --job-name=pigvae_downstream
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/downstream_%j.log
set -e
# Single per-crop-feature transfer eval: encode -> label-join -> 10-fold CV.
# Usage: sbatch scripts/sbatch_downstream.sh <CKPT> <EXP> <TAG> [SRC FS NA]
# Defaults probe z_global; override the trailing triple for another source, e.g.
#   sbatch scripts/sbatch_downstream.sh $CKPT $EXP $TAG node_mean node mean
CKPT=$1; EXP=$2; TAG=$3
SRC=${4:-zglobal}; FS=${5:-zglobal}; NA=${6:-mean}
B=/raid_encrypted/immucan/embeddings/tnocon
cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE
echo "### ENCODE ($EXP -> $TAG / $SRC) ###"
uv run python scripts/encode_mil_embeddings.py \
  ckpt_path=$CKPT model_experiment=$EXP run_tag=$TAG \
  feature_source=$FS node_agg=$NA source_tag=$SRC
echo "### LABEL-JOIN ###"
uv run python scripts/build_abmil_meta.py \
  --encode_dir $B/downstream/$TAG/mil_embeddings/$SRC \
  --melted $B/data/clinical/melted_table_images.csv \
  --out_dir $B/downstream/$TAG/meta_tables/$SRC --splits train test
echo "### 10-FOLD CV ###"
uv run python scripts/run_abmil.py run_tag=$TAG source_tag=$SRC
echo "### DONE ###"
