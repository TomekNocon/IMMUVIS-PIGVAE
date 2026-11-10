"""The reference ABMIL transfer setup, rebuilt exactly as ``run_abmil.py`` / ``run_cv`` ran it.

Same label filtering, bag order, ``StratifiedKFold(shuffle=True, random_state=42)`` and train-fold
z-score, so every head used here is the out-of-fold head of the reported CV.
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from figlib import paths
from src.downstream.abmil.cv import apply_zscore_to_bags, zscore_stats_from_bags
from src.downstream.abmil.data import (
    balance_meta_df,
    build_image_bags,
    drop_nan_labels,
    load_pooled_meta,
)
from src.downstream.abmil.lit import AbmilLitModule

DATASET = "cords"
CROP_DENSITY = paths.CACHE / "empty_shortcut" / "crop_density.csv"


def pooled_meta() -> pd.DataFrame:
    return load_pooled_meta(str(paths.REF_META), DATASET)


@dataclass
class Cohort:
    """Bags, labels and CV folds of one clinical target."""

    feature: str
    classes: list[str]
    bags: list[np.ndarray]
    y: np.ndarray
    rows: list[pd.DataFrame]  # per bag, crop metadata in bag-row order
    folds: list[tuple[np.ndarray, np.ndarray]]
    fold_of: np.ndarray = field(repr=False)

    @property
    def names(self) -> list[str]:
        return [r.img_path.iloc[0] for r in self.rows]

    def coords(self, i: int) -> np.ndarray:
        return self.rows[i][["coords0", "coords1"]].values.astype(int)

    @cached_property
    def _density(self) -> pd.DataFrame:
        return pd.read_csv(CROP_DENSITY).set_index(["img_path", "coords0", "coords1"])

    def density(self, i: int) -> pd.DataFrame:
        """Tissue density and empty flag of bag ``i``'s crops (quantify_empty_shortcut.py)."""
        r = self.rows[i]
        return self._density.loc[list(zip(r.img_path, r.coords0, r.coords1, strict=True))]

    def empty(self, i: int) -> np.ndarray:
        return self.density(i)["empty"].values.astype(bool)


def load_cohort(feature: str = "DX.name", pooled: pd.DataFrame | None = None) -> Cohort:
    pooled = pooled_meta() if pooled is None else pooled
    meta = balance_meta_df(drop_nan_labels(pooled[pooled.feature == feature]), min_class_freq=0.05)
    classes = sorted(meta.feature_value.astype(str).unique())
    bags, labels = build_image_bags(meta, {c: i for i, c in enumerate(classes)})
    y = labels.numpy()
    rows = [
        pd.concat([part for _, part in g.groupby("embeddings_file")])
        for _, g in meta.groupby("img_path", sort=False)
    ]
    n_folds = min(10, int(np.bincount(y).min()), len(bags))
    folds = list(
        StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(
            np.zeros(len(bags)), y
        )
    )
    fold_of = np.empty(len(bags), int)
    for f, (_, val) in enumerate(folds):
        fold_of[val] = f
    return Cohort(feature, classes, bags, y, rows, folds, fold_of)


def fold_head(feature: str, fold: int, device: str = "cpu") -> torch.nn.Module:
    """The saved gated-ABMIL head of one CV fold (latest checkpoint), frozen, in eval mode."""
    ckpt_dir = paths.REF_RESULTS / DATASET / feature / f"fold_{fold}"
    ckpt = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)[-1]
    head = (
        AbmilLitModule.load_from_checkpoint(str(ckpt), map_location="cpu").model.eval().to(device)
    )
    head.requires_grad_(False)
    return head


@dataclass
class FoldModels:
    """Per-fold heads and train-fold z-score statistics."""

    heads: dict[int, torch.nn.Module]
    stats: dict[int, tuple[np.ndarray, np.ndarray]]

    def zscored_bag(self, cohort: Cohort, i: int, device: str) -> torch.Tensor:
        """Bag ``i`` z-scored with its out-of-fold statistics: [1, n, D]."""
        mu, sd = self.stats[cohort.fold_of[i]]
        return (
            torch
            .from_numpy(apply_zscore_to_bags([cohort.bags[i]], mu, sd)[0])
            .float()[None]
            .to(device)
        )


def load_fold_models(cohort: Cohort, device: str) -> FoldModels:
    heads, stats = {}, {}
    for f, (train, _) in enumerate(cohort.folds):
        heads[f] = fold_head(cohort.feature, f, device)
        stats[f] = zscore_stats_from_bags(cohort.bags, indices=train)
    return FoldModels(heads, stats)


@torch.no_grad()
def oof_logits(cohort: Cohort, models: FoldModels, device: str) -> np.ndarray:
    """Out-of-fold decision logit of every bag (> 0 = second class)."""
    out = np.zeros(len(cohort.bags))
    for f, (_, val) in enumerate(cohort.folds):
        for i in val:
            out[i] = models.heads[f](models.zscored_bag(cohort, i, device))[0].item()
    return out


def gated_attention(model, x: torch.Tensor, mask: torch.Tensor | None = None):
    """Gated-ABMIL forward that also returns the attention: (logits [B, C], attention [B, H, S])."""
    v, u = torch.tanh(model.V(x)), torch.sigmoid(model.U(x))
    a = model.W(v * u)
    if mask is not None:
        a = a.masked_fill(mask.unsqueeze(2), -1e9)
    a = torch.softmax(a, dim=1).transpose(1, 2)
    return model.classifier(torch.bmm(a, x).reshape(x.size(0), -1)), a
