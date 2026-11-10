"""Quantify the ABMIL empty-crop shortcut without retraining (open item 15).

For the reference run's z_global downstream:
  1. tissue density of every crop from the raw TIFF (mean arcsinh(x/5) over channels and pixels); a
     crop is EMPTY if density <= max(0.1, 0.4 * median density of its image's crops), with a stricter
     absolute rule (<= 0.1) as a sensitivity check -> ``figures/_cache/empty_shortcut/crop_density.csv``;
  2. per clinical target, the bags and folds exactly as run_abmil / run_cv built them, each fold's saved
     head, and on the held-out bags: the attention mass on empty crops, and the out-of-fold AUC with all
     crops (reproduces the reported AUC) and with empty crops masked out of the attention softmax
     (same frozen head) -> ``attention_<target>.csv`` and ``summary.csv`` in ``--out-dir``.

Usage:
    uv run python scripts/thesis_figures/quantify_empty_shortcut.py
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import numpy as np
import pandas as pd
import tifffile
import torch

from figlib import abmil, imc, paths
from src.downstream.abmil.cv import _auc, apply_zscore_to_bags, zscore_stats_from_bags

TARGETS = ["Grade", "Relapse", "DX.name", "Stage"]
P = imc.CROP


def image_density(job) -> tuple[str, list[float]]:
    image, coords = job
    a = np.arcsinh(tifffile.imread(os.path.join(paths.TIFF_FULL, image)).astype(np.float32) / 5.0)
    return image, [
        float(a[:, r : r + P, c : c + P].mean())
        if r + P <= a.shape[1] and c + P <= a.shape[2]
        else np.nan
        for r, c in ((int(r), int(c)) for r, c in coords)
    ]


def crop_density(pooled: pd.DataFrame) -> pd.DataFrame:
    """Tissue density and empty flags of every crop (computed once, then read from the cache)."""
    if not abmil.CROP_DENSITY.exists():
        crops = pooled.drop_duplicates(["img_path", "coords0", "coords1"])[
            ["img_path", "coords0", "coords1"]
        ]
        jobs = [(image, g[["coords0", "coords1"]].values) for image, g in crops.groupby("img_path")]
        rows = []
        with ProcessPoolExecutor(8) as pool:
            for image, densities in pool.map(image_density, jobs, chunksize=4):
                g = crops[crops.img_path == image]
                rows += [
                    (image, r, c, d)
                    for (r, c), d in zip(g[["coords0", "coords1"]].values, densities, strict=True)
                ]
        dens = pd.DataFrame(rows, columns=["img_path", "coords0", "coords1", "density"])
        median = dens.groupby("img_path").density.transform("median")
        dens["empty"] = dens.density <= np.maximum(0.1, 0.4 * median)
        dens["empty_abs"] = dens.density <= 0.1
        dens.to_csv(abmil.CROP_DENSITY, index=False)
    dens = pd.read_csv(abmil.CROP_DENSITY)
    print(
        f"crops {len(dens)}  NaN density {dens.density.isna().sum()}  empty (rule) {dens['empty'].mean():.3%}  "
        f"empty (abs<=0.1) {dens.empty_abs.mean():.3%}",
        flush=True,
    )
    return dens


def pad(bags, extra_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad bags to one tensor; the mask is True on padding (and on crops flagged in extra_mask)."""
    size, dim = max(b.shape[0] for b in bags), bags[0].shape[1]
    x = torch.zeros(len(bags), size, dim)
    mask = torch.ones(len(bags), size, dtype=torch.bool)
    for i, b in enumerate(bags):
        x[i, : len(b)] = torch.from_numpy(np.asarray(b, "float32"))
        mask[i, : len(b)] = False
        if extra_mask is not None:
            mask[i, : len(b)] |= torch.from_numpy(extra_mask[i])
    return x, mask


def target_summary(target: str, pooled: pd.DataFrame, out_dir: Path) -> dict:
    cohort = abmil.load_cohort(target, pooled)
    flags = {
        "rule": [cohort.empty(i) for i in range(len(cohort.bags))],
        "abs": [cohort.density(i).empty_abs.values.astype(bool) for i in range(len(cohort.bags))],
    }
    n, n_classes = len(cohort.bags), len(cohort.classes)
    logit_dim = 1 if n_classes <= 2 else n_classes
    oof = {k: np.full((n, logit_dim), np.nan, np.float32) for k in ["all", "rule", "abs"]}
    attention_rows = []
    for f, (train, val) in enumerate(cohort.folds):
        head = abmil.fold_head(target, f)
        mu, sd = zscore_stats_from_bags(cohort.bags, indices=train)
        val_bags = [apply_zscore_to_bags([cohort.bags[i]], mu, sd)[0] for i in val]
        with torch.no_grad():
            x, mask = pad(val_bags)
            logits, att = abmil.gated_attention(head, x, mask)
            oof["all"][val] = logits.numpy().reshape(len(val), logit_dim)
            for rule in ["rule", "abs"]:
                empty = [flags[rule][i] for i in val]
                empty = [
                    e if (~e).any() else np.zeros_like(e) for e in empty
                ]  # an all-empty bag stays as is
                x2, mask2 = pad(val_bags, empty)
                oof[rule][val] = (
                    abmil.gated_attention(head, x2, mask2)[0].numpy().reshape(len(val), logit_dim)
                )
            mean_att = att.mean(1).numpy()  # [B, S], head-averaged
        for j, i in enumerate(val):
            s, e, e_abs = len(cohort.bags[i]), flags["rule"][i], flags["abs"][i]
            attention_rows.append({
                "bag": i,
                "fold": f,
                "n": s,
                "frac_empty": e.mean(),
                "att_empty": mean_att[j, :s][e].sum(),
                "frac_empty_abs": e_abs.mean(),
                "att_empty_abs": mean_att[j, :s][e_abs].sum(),
                "top_is_empty": bool(e[np.argmax(mean_att[j, :s])]),
            })
    att = pd.DataFrame(attention_rows)
    att.to_csv(out_dir / f"attention_{target}.csv", index=False)
    has_empty = att.frac_empty > 0
    reported = (
        pd.read_csv(paths.REF_RESULTS / "results.csv").set_index("feature").loc[target, "auc"]
    )
    y = cohort.y.astype(int)
    return {
        "feature": target,
        "bags": n,
        "classes": n_classes,
        "folds": len(cohort.folds),
        "reported_auc": reported,
        "auc_all": _auc(y, oof["all"], n_classes),
        "auc_no_empty": _auc(y, oof["rule"], n_classes),
        "auc_no_empty_abs": _auc(y, oof["abs"], n_classes),
        "frac_crops_empty": float(np.concatenate(flags["rule"]).mean()),
        "bags_with_empty": float(has_empty.mean()),
        "att_mass_empty": float(att.att_empty.mean()),
        "att_mass_empty_in_bags_with_empty": float(att[has_empty].att_empty.mean()),
        "frac_empty_in_bags_with_empty": float(att[has_empty].frac_empty.mean()),
        "enrichment": float(att[has_empty].att_empty.sum() / att[has_empty].frac_empty.sum()),
        "top_crop_empty": float(att.top_is_empty.mean()),
        "att_mass_empty_abs": float(att.att_empty_abs.mean()),
    }


def main(targets: list[str], out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pooled = abmil.pooled_meta()
    crop_density(pooled)
    summary = []
    for target in targets:
        summary.append(target_summary(target, pooled, out))
        print(summary[-1], flush=True)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    print(pd.DataFrame(summary).round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--out-dir", default=str(paths.CACHE / "empty_shortcut"))
    a = ap.parse_args()
    main(a.targets, a.out_dir)
