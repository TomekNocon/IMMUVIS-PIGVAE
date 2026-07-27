#!/bin/bash
#SBATCH --job-name=pigvae_downstream
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/downstream_%j.log
set -e
CKPT=$1; EXP=$2; TAG=$3
B=/raid_encrypted/immucan/embeddings/tnocon
cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE
echo "### ENCODE ($EXP -> $TAG) ###"
uv run python scripts/encode_mil_embeddings.py ckpt_path=$CKPT model_experiment=$EXP run_tag=$TAG
echo "### LABEL-JOIN ###"
uv run python scripts/build_abmil_meta.py \
  --encode_dir $B/downstream/$TAG/mil_embeddings \
  --melted $B/data/clinical/melted_table_images.csv \
  --out_dir $B/downstream/$TAG/meta_tables --splits train test
echo "### 10-FOLD CV ###"
uv run python scripts/run_abmil.py run_tag=$TAG
echo "### DONE ###"
