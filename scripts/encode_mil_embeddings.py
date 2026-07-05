# scripts/encode_mil_embeddings.py
import os

import hydra
import rootutils
from omegaconf import DictConfig

# Repo convention (see src/train.py, src/eval.py, scripts/diagnose_model.py): put the
# project root on sys.path so `from src...` resolves when this file is invoked directly
# as `python scripts/encode_mil_embeddings.py` (its own dir would otherwise be sys.path[0]).
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.downstream.encode import load_frozen_pigvae, build_pca_layer, encode_h5

@hydra.main(version_base=None, config_path="../configs", config_name="downstream/encode")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    gae = load_frozen_pigvae(cfg.ckpt_path, cfg.experiment)
    pca = build_pca_layer(cfg.pca_pkl, cfg.pca_stats)
    for split in cfg.splits:
        emb = f"{cfg.out_dir}/cords_{split}_embeddings.npy"
        meta = f"{cfg.out_dir}/cords_{split}_metadata.csv"
        if os.path.exists(emb) and os.path.exists(meta):
            print(f"skip {split} (exists)"); continue
        encode_h5(f"{cfg.data_root}/{split}.h5", gae, pca, emb, meta, cfg.device, cfg.batch_size)
        print(f"wrote {emb}")

if __name__ == "__main__":
    main()
