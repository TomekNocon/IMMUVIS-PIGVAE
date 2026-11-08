#!/bin/bash
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=probe_ds
#SBATCH --output=logs/slurm/%j_%x.out
D=/raid_encrypted/immucan/embeddings/tnocon/downstream/vae16_s0p01_film_2026-07-31
ls $D; ls $D/meta_tables 2>/dev/null; ls $D/results 2>/dev/null | head
find $D -maxdepth 6 -name "*.ckpt" | grep -i zglobal | head -30
find $D -maxdepth 6 -name "*.ckpt" | wc -l
ls $D/meta_tables/zglobal 2>/dev/null | head
head -3 $D/meta_tables/zglobal/cords_train_metadata.csv 2>/dev/null
