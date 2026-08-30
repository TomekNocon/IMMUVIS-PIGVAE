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

from src.downstream.abmil.data import build_image_bags
from src.downstream.abmil.lit import AbmilLitModule
from src.explainability.abmil_attribution import abmil_attention, abmil_ig, crop_importance
from src.explainability.faithfulness import faithfulness_score


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
    valid = meta.dropna(subset=["feature_value"])["img_path"].unique()
    meta = meta[meta["img_path"].isin(valid)]
    classes = sorted(meta["feature_value"].astype(str).unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"[abmil] feature={cfg.feature} classes={class_to_idx}")

    bags, labels = build_image_bags(meta, class_to_idx)
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
        rows.append({"img": i, "label": int(labels[i]), "logit": logit, "faithfulness": fs["faithfulness"]})

    torch.save(rows, os.path.join(cfg.out_dir, "abmil_summary.pt"))
    print(f"[abmil] wrote outputs to {os.path.abspath(cfg.out_dir)}")


if __name__ == "__main__":
    main()
