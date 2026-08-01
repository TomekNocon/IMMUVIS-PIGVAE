# scripts/encode_mil_embeddings.py
import os

import hydra
import rootutils
from omegaconf import DictConfig

# Repo convention (see src/train.py, src/eval.py, scripts/diagnose_model.py): put the
# project root on sys.path so `from src...` resolves when this file is invoked directly
# as `python scripts/encode_mil_embeddings.py` (its own dir would otherwise be sys.path[0]).
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.downstream.encode import load_frozen_pigvae, build_pca_layer, encode_h5, source_subdir

@hydra.main(version_base=None, config_path="../configs", config_name="downstream/encode")
def main(cfg: DictConfig) -> None:
    feature_source = getattr(cfg, "feature_source", "zglobal")
    node_agg = getattr(cfg, "node_agg", "mean")
    # Guard against a source_tag / feature_source mismatch silently landing this
    # feature in the wrong bundle dir (build_abmil_meta + run_abmil key off source_tag).
    expected_tag = source_subdir(feature_source, node_agg)
    if cfg.source_tag != expected_tag:
        raise ValueError(
            f"source_tag={cfg.source_tag!r} does not match "
            f"source_subdir({feature_source!r}, {node_agg!r})={expected_tag!r}"
        )
    os.makedirs(cfg.out_dir, exist_ok=True)
    print(f"[encode] feature_source={feature_source} node_agg={node_agg} -> {cfg.out_dir}")

    # The raw baseline is a plain spatial mean-pool of the input patch: no encoder,
    # no PCA -- skip loading them entirely.
    if feature_source == "raw":
        gae, pca = None, None
    else:
        gae = load_frozen_pigvae(cfg.ckpt_path, cfg.model_experiment)
        pca = build_pca_layer(cfg.pca_pkl, cfg.pca_stats)

    for split in cfg.splits:
        emb = f"{cfg.out_dir}/cords_{split}_embeddings.npy"
        meta = f"{cfg.out_dir}/cords_{split}_metadata.csv"
        if os.path.exists(emb) and os.path.exists(meta):
            print(f"skip {split} (exists)"); continue
        encode_h5(f"{cfg.data_root}/{split}.h5", gae, pca, emb, meta, cfg.device,
                  cfg.batch_size, feature_source, node_agg)
        print(f"wrote {emb}")

if __name__ == "__main__":
    main()
