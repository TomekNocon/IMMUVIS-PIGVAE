"""Hydra entrypoint for DECISION-level explainability on a trained gated-ABMIL head.

Loads a saved fold checkpoint (from the downstream CV) + rebuilds the per-image z_global bags,
then reports, per image: the gated attention over crops, Integrated-Gradients crop importance,
and a deletion/insertion faithfulness check of the attention map. This answers "which crops
drive the clinical decision" — the crisp, decision-level counterpart to the diffuse encoder map.

Example:
  uv run scripts/explain_abmil.py \
    abmil_ckpt=/raid_.../results/zglobal/cords/DX.name/fold_0/epoch=...ckpt \
    meta_dir=/raid_.../meta_tables/zglobal feature=DX.name paths=szary
"""

import operator
import os

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("multiply", lambda x, y: operator.mul(int(x), int(y)), replace=True)
OmegaConf.register_new_resolver("divide", lambda x, y: int(x) // int(y), replace=True)

from src.downstream.abmil.data import _load_memmap, build_image_bags
from src.downstream.abmil.lit import AbmilLitModule
from src.explainability import viz
from src.explainability.abmil_attribution import abmil_attention, abmil_ig, crop_importance
from src.explainability.faithfulness import faithfulness_score


def _drop_nan_labels(meta_df):
    valid = meta_df.dropna(subset=["feature_value"])["img_path"].unique()
    return meta_df[meta_df["img_path"].isin(valid)]


def build_bags_with_coords(meta_df, class_to_idx):
    """Like build_image_bags, but also returns each crop's (coords0, coords1) in bag order,
    so per-crop importance can be placed back on the tissue (CLAM-style heatmap)."""
    bags, labels, coords = [], [], []
    for _img, g in meta_df.groupby("img_path", sort=False):
        raw = g["feature_value"].iloc[0]
        lab = class_to_idx.get(str(raw))
        if lab is None or pd.isna(raw):
            continue
        rows, crd = [], []
        for ef, gg in g.groupby("embeddings_file"):
            rows.append(_load_memmap(ef)[gg["embedding_idx"].values])
            crd.append(gg[["coords0", "coords1"]].values)
        bags.append(np.concatenate(rows, axis=0))
        labels.append(lab)
        coords.append(np.concatenate(crd, axis=0))
    return bags, labels, coords


def _balance_meta_df(meta_df, min_class_freq=0.05):
    """Image-level class-frequency filter (replicates run_abmil.balance_meta_df) so the class
    set matches the trained head (e.g. DX.name -> binary Adeno vs Squamous)."""
    img_labels = meta_df.groupby("img_path", sort=False)["feature_value"].first()
    img_labels = img_labels[~pd.isna(img_labels)].astype(str)
    if len(img_labels) == 0:
        return meta_df.iloc[0:0]
    classes, counts = np.unique(img_labels.values, return_counts=True)
    keep = set(classes[(counts / counts.sum()) >= float(min_class_freq)])
    keep_imgs = img_labels[img_labels.isin(keep)].index
    return meta_df[meta_df["img_path"].isin(keep_imgs)]


@hydra.main(version_base="1.3", config_path="../configs", config_name="explain_abmil")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    # ── pooled meta for this feature ──
    frames = []
    for split in ("train", "test"):
        p = os.path.join(cfg.meta_dir, f"cords_{split}_metadata.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    meta = pd.concat(frames, ignore_index=True)
    if "feature" in meta.columns:
        meta = meta[meta["feature"] == cfg.feature].copy()
    # Match the downstream CV: drop-nan then class-frequency balance -> the head's class set.
    meta = _drop_nan_labels(meta)
    meta = _balance_meta_df(meta, min_class_freq=float(cfg.min_class_freq))
    classes = sorted(meta["feature_value"].astype(str).unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"[abmil] feature={cfg.feature} (min_class_freq={cfg.min_class_freq}) classes={class_to_idx}")

    bags, labels, coords = build_bags_with_coords(meta, class_to_idx)
    print(f"[abmil] {len(bags)} image bags, emb_dim={bags[0].shape[1]}")

    # ── zscore (global approximation of the per-fold cv_train normalisation) ──
    allc = np.concatenate(bags, axis=0)
    mu, sd = allc.mean(0), allc.std(0) + 1e-6
    bags = [(np.asarray(b, "float32") - mu) / sd for b in bags]

    # ── head ──
    lit = AbmilLitModule.load_from_checkpoint(cfg.abmil_ckpt, map_location="cpu")
    model = lit.model.eval()

    rows = []
    n = min(int(cfg.num_samples), len(bags))
    for i in range(n):
        x = torch.from_numpy(np.asarray(bags[i], "float32")).unsqueeze(0)  # [1, S, D]
        a = abmil_attention(model, x)[0, :, 0]                              # [S]
        attr = abmil_ig(model, x, target=0, n_steps=int(cfg.n_steps))       # [1, S, D]
        imp = crop_importance(attr)[0]                                      # [S]
        fwd = lambda bag: model(bag.unsqueeze(0))[0][0, 0]
        fs = faithfulness_score(fwd, x[0], a, steps=min(20, x.shape[1]))
        logit = float(model(x)[0].item())
        print(f"[abmil] img {i}: crops={x.shape[1]} label={int(labels[i])} logit={logit:+.3f} "
              f"top-attn={float(a.max()):.3f} attn_faithfulness={fs['faithfulness']:+.4f}")
        torch.save(
            {"attention": a, "ig_importance": imp, "logit": logit, "label": int(labels[i]), "faithfulness": fs},
            os.path.join(cfg.out_dir, f"abmil_img{i}.pt"),
        )
        viz.plot_abmil_crops(
            a, imp, os.path.join(cfg.out_dir, f"abmil_img{i}.png"),
            title=f"img {i} · {cfg.feature} · label={int(labels[i])} logit={logit:+.2f}",
        )
        # CLAM-style spatial heatmaps: crops placed at their tissue coords, coloured by importance.
        viz.plot_crop_importance(
            coords[i], a, os.path.join(cfg.out_dir, f"spatial_attn_img{i}.png"),
            title=f"img {i} · attention (logit {logit:+.2f})",
        )
        viz.plot_crop_importance(
            coords[i], imp, os.path.join(cfg.out_dir, f"spatial_ig_img{i}.png"),
            title=f"img {i} · IG importance (logit {logit:+.2f})",
        )
        rows.append({"img": i, "label": int(labels[i]), "logit": logit, "faithfulness": fs["faithfulness"]})

    torch.save(rows, os.path.join(cfg.out_dir, "abmil_summary.pt"))
    print(f"[abmil] wrote outputs to {os.path.abspath(cfg.out_dir)}")


if __name__ == "__main__":
    main()
