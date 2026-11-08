#!/bin/bash
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --job-name=findruns
#SBATCH --output=logs/slurm/%j_%x.out
R=/raid_encrypted/immucan/embeddings/tnocon/logs/train/runs
for id in cuogzab1 d391t2al; do
  d=$(ls -d $R/*/wandb/*-$id 2>/dev/null | head -1)
  run=${d%/wandb/*}
  echo "== $id -> $run"
  ls $run/checkpoints 2>/dev/null | head
  grep -n "grid_size\|center_crop_size\|zscore\|clip_range\|num_pca\|neighborhood_radius\|input_size\|pos_bias\|vae:\|node_z_dim\|experiment" $run/.hydra/config.yaml $run/.hydra/overrides.yaml 2>/dev/null | head -30
done
