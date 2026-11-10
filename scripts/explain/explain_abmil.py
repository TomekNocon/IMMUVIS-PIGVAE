"""Decision-level attribution on a trained gated-ABMIL head.

Loads a saved fold checkpoint (from the downstream CV) + rebuilds the per-image z_global bags,
then reports, per image: the gated attention over crops, Integrated-Gradients crop importance,
and a deletion/insertion faithfulness check of the attention map. This answers "which crops
drive the clinical decision" — the crisp, decision-level counterpart to the diffuse encoder map.

Usage:
    uv run python scripts/explain/explain_abmil.py \
    abmil_ckpt=/raid_.../results/zglobal/cords/DX.name/fold_0/epoch=...ckpt \
    meta_dir=/raid_.../meta_tables/zglobal feature=DX.name
"""

import os

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.utils.resolvers import register_resolvers

register_resolvers()

from src.downstream.abmil.data import (
    balance_meta_df,
    build_image_bags_with_sources,
    drop_nan_labels,
    load_pooled_meta,
)
from src.downstream.abmil.lit import AbmilLitModule
from src.explainability import viz
from src.explainability.abmil_attribution import abmil_attention, abmil_ig, crop_importance
from src.explainability.faithfulness import faithfulness_score


def load_channel0_tiles(ef_list, idx_arr, data_root, channel=0):
    """Reload each crop's input patch from the source h5 and return its channel-`channel`
    tile (16x16). `embedding_idx` is the row index into <data_root>/<split>.h5['embeddings']."""
    import h5py

    tiles = [None] * len(idx_arr)
    # group indices by split (train/test), derived from the embeddings_file basename
    by_split: dict[str, list[int]] = {}
    for k, ef in enumerate(ef_list):
        base = os.path.basename(ef)
        split = "train" if "train" in base else ("test" if "test" in base else "train")
        by_split.setdefault(split, []).append(k)
    for split, ks in by_split.items():
        with h5py.File(f"{data_root}/{split}.h5", "r") as f:
            emb = f["embeddings"]
            for k in ks:
                tiles[k] = np.asarray(emb[int(idx_arr[k])][channel], dtype=np.float32)  # [16,16]
    return tiles


@hydra.main(version_base="1.3", config_path="../../configs", config_name="explain/abmil")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Pooled train+test meta of this feature, filtered exactly as in the downstream CV so the
    # class set matches the trained head.
    meta = load_pooled_meta(cfg.meta_dir, cfg.dataset)
    if "feature" in meta.columns:
        meta = meta[meta["feature"] == cfg.feature].copy()
    meta = drop_nan_labels(meta)
    meta = balance_meta_df(meta, min_class_freq=float(cfg.min_class_freq))
    classes = sorted(meta["feature_value"].astype(str).unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(
        f"[abmil] feature={cfg.feature} (min_class_freq={cfg.min_class_freq}) classes={class_to_idx}"
    )

    bags, labels, coords, srcs = build_image_bags_with_sources(meta, class_to_idx)
    print(f"[abmil] {len(bags)} image bags, emb_dim={bags[0].shape[1]}")

    # z-score with statistics of ALL bags: an approximation of the CV, which normalises each
    # fold with its training-fold statistics only. Kept as is (the thesis figure uses it).
    allc = np.concatenate(bags, axis=0)
    mu, sd = allc.mean(0), allc.std(0) + 1e-6
    bags = [(np.asarray(b, "float32") - mu) / sd for b in bags]

    lit = AbmilLitModule.load_from_checkpoint(cfg.abmil_ckpt, map_location="cpu")
    model = lit.model.eval()

    rows = []
    n = min(int(cfg.num_samples), len(bags))
    for i in range(n):
        x = torch.from_numpy(np.asarray(bags[i], "float32")).unsqueeze(0)  # [1, S, D]
        a = abmil_attention(model, x)[0, :, 0]  # [S]
        attr = abmil_ig(model, x, target=0, n_steps=int(cfg.n_steps))  # [1, S, D]
        imp = crop_importance(attr)[0]  # [S]

        def fwd(bag: torch.Tensor) -> torch.Tensor:
            return model(bag.unsqueeze(0))[0][0, 0]

        fs = faithfulness_score(fwd, x[0], a, steps=min(20, x.shape[1]))
        logit = float(model(x)[0].item())
        print(
            f"[abmil] img {i}: crops={x.shape[1]} label={int(labels[i])} logit={logit:+.3f} "
            f"top-attn={float(a.max()):.3f} attn_faithfulness={fs['faithfulness']:+.4f}"
        )
        torch.save(
            {
                "attention": a,
                "ig_importance": imp,
                "logit": logit,
                "label": int(labels[i]),
                "faithfulness": fs,
            },
            os.path.join(cfg.out_dir, f"abmil_img{i}.pt"),
        )
        viz.plot_abmil_crops(
            a,
            imp,
            os.path.join(cfg.out_dir, f"abmil_img{i}.png"),
            title=f"img {i} · {cfg.feature} · label={int(labels[i])} logit={logit:+.2f}",
        )
        # CLAM-style spatial heatmaps: crops placed at their tissue coords, coloured by importance.
        viz.plot_crop_importance(
            coords[i],
            a,
            os.path.join(cfg.out_dir, f"spatial_attn_img{i}.png"),
            title=f"img {i} · attention (logit {logit:+.2f})",
        )
        viz.plot_crop_importance(
            coords[i],
            imp,
            os.path.join(cfg.out_dir, f"spatial_ig_img{i}.png"),
            title=f"img {i} · IG importance (logit {logit:+.2f})",
        )
        # Stitched PoC: reload channel-0 input tiles from the h5, stitch by coords, overlay importance.
        if cfg.get("data_root"):
            ef_list, idx_arr = srcs[i]
            tiles = load_channel0_tiles(ef_list, idx_arr, cfg.data_root, channel=int(cfg.channel))
            viz.plot_stitched_importance(
                tiles,
                coords[i],
                a,
                os.path.join(cfg.out_dir, f"stitch_attn_img{i}.png"),
                title=f"img {i} · ch{cfg.channel} + attention (logit {logit:+.2f})",
            )
            viz.plot_stitched_importance(
                tiles,
                coords[i],
                imp,
                os.path.join(cfg.out_dir, f"stitch_ig_img{i}.png"),
                title=f"img {i} · ch{cfg.channel} + IG importance (logit {logit:+.2f})",
            )
        rows.append({
            "img": i,
            "label": int(labels[i]),
            "logit": logit,
            "faithfulness": fs["faithfulness"],
        })

    torch.save(rows, os.path.join(cfg.out_dir, "abmil_summary.pt"))
    print(f"[abmil] wrote outputs to {os.path.abspath(cfg.out_dir)}")


if __name__ == "__main__":
    main()
