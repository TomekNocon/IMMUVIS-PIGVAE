#!/usr/bin/env python3
"""
EDA Outlier Detection
=====================
Detects outlier samples / nodes that may destabilise training.

Methods
-------
1. Z-score (per-feature, flag nodes with |z| > threshold)
2. IQR (per-feature, 1.5×IQR rule)
3. Isolation Forest (sample-level)
4. Local Outlier Factor (sample-level)

Outputs
-------
- ``eda_output/outlier_report.txt``
- ``eda_output/pca_scatter_outliers.png``
- ``eda_output/zscore_outlier_fraction.png``
- ``eda_output/iqr_outlier_fraction.png``
- ``eda_output/feature_tail_distributions.png``
- ``eda_output/outlier_sample_indices.npz``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm

from eda_config import EDAConfig, build_train_dataloader, set_seed


# ────────────────────────────────────────────────────────────────────
# Report helper
# ────────────────────────────────────────────────────────────────────

class Report:
    def __init__(self, path: Path):
        self.path = path
        self.lines: list[str] = []

    def __call__(self, msg: str = ""):
        print(msg)
        self.lines.append(msg)

    def section(self, title: str):
        sep = "=" * 72
        self.__call__("")
        self.__call__(sep)
        self.__call__(f"  {title}")
        self.__call__(sep)

    def save(self):
        self.path.write_text("\n".join(self.lines), encoding="utf-8")


# ────────────────────────────────────────────────────────────────────
# Data collection helpers
# ────────────────────────────────────────────────────────────────────

def collect_data(cfg: EDAConfig, max_batches: int | None = None):
    """Return (all_nodes, sample_means).

    all_nodes : (total_nodes, D)  – every node vector flattened
    sample_means : (n_samples, D) – mean feature per sample (graph)
    sample_node_features : list of (N, D) arrays per sample
    """
    loader = build_train_dataloader(cfg, shuffle=False)
    total = len(loader) if max_batches is None else min(max_batches, len(loader))
    node_chunks = []
    sample_means = []
    sample_node_features = []

    for i, batch in enumerate(tqdm(loader, desc="Collecting data for outlier detection", total=total)):
        nf = batch.node_features.numpy()  # (B*8, N, D)
        for j in range(nf.shape[0]):
            sample_node_features.append(nf[j])
            sample_means.append(nf[j].mean(axis=0))
        node_chunks.append(nf.reshape(-1, cfg.num_pca_components))
        if max_batches is not None and i >= max_batches - 1:
            break

    all_nodes = np.concatenate(node_chunks, axis=0)
    sample_means = np.stack(sample_means, axis=0)
    return all_nodes, sample_means, sample_node_features


# ────────────────────────────────────────────────────────────────────
# 1. Z-Score Outlier Detection
# ────────────────────────────────────────────────────────────────────

def zscore_outliers(cfg: EDAConfig, rpt: Report, all_nodes: np.ndarray, threshold: float = 4.0):
    rpt.section("1. Z-Score Outlier Detection")

    mean = all_nodes.mean(axis=0)
    std = all_nodes.std(axis=0) + 1e-12
    z = np.abs((all_nodes - mean) / std)

    outlier_mask = z > threshold
    n_outlier_nodes = outlier_mask.any(axis=1).sum()
    n_total = all_nodes.shape[0]
    frac = n_outlier_nodes / n_total

    rpt(f"  Threshold: |z| > {threshold}")
    rpt(f"  Outlier nodes: {n_outlier_nodes:,} / {n_total:,} ({frac * 100:.2f}%)")

    # Per-channel outlier fraction
    per_ch_frac = outlier_mask.mean(axis=0)
    rpt(f"\n  Top-10 channels by outlier fraction:")
    top_ch = np.argsort(per_ch_frac)[::-1][:10]
    for ch in top_ch:
        rpt(f"    Ch{ch:4d}: {per_ch_frac[ch] * 100:.3f}%")

    # Plot
    out = cfg.output_dir
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(per_ch_frac)), per_ch_frac * 100, alpha=0.7)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="1% threshold")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Outlier Fraction (%)")
    ax.set_title(f"Z-Score Outlier Fraction per Channel (|z| > {threshold})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "zscore_outlier_fraction.png", dpi=150)
    plt.close(fig)
    rpt(f"\n  Saved: {out / 'zscore_outlier_fraction.png'}")

    return outlier_mask


# ────────────────────────────────────────────────────────────────────
# 2. IQR Outlier Detection
# ────────────────────────────────────────────────────────────────────

def iqr_outliers(cfg: EDAConfig, rpt: Report, all_nodes: np.ndarray, k: float = 1.5):
    rpt.section("2. IQR Outlier Detection")

    q1 = np.percentile(all_nodes, 25, axis=0)
    q3 = np.percentile(all_nodes, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    outlier_mask = (all_nodes < lower) | (all_nodes > upper)
    n_outlier_nodes = outlier_mask.any(axis=1).sum()
    n_total = all_nodes.shape[0]

    rpt(f"  IQR multiplier k: {k}")
    rpt(f"  Outlier nodes: {n_outlier_nodes:,} / {n_total:,} ({n_outlier_nodes / n_total * 100:.2f}%)")

    per_ch_frac = outlier_mask.mean(axis=0)
    rpt(f"\n  Top-10 channels by IQR outlier fraction:")
    top_ch = np.argsort(per_ch_frac)[::-1][:10]
    for ch in top_ch:
        rpt(f"    Ch{ch:4d}: {per_ch_frac[ch] * 100:.3f}%")

    out = cfg.output_dir
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(per_ch_frac)), per_ch_frac * 100, alpha=0.7, color="orange")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Outlier Fraction (%)")
    ax.set_title(f"IQR Outlier Fraction per Channel (k={k})")
    fig.tight_layout()
    fig.savefig(out / "iqr_outlier_fraction.png", dpi=150)
    plt.close(fig)
    rpt(f"\n  Saved: {out / 'iqr_outlier_fraction.png'}")

    return outlier_mask


# ────────────────────────────────────────────────────────────────────
# 3. Isolation Forest (sample-level)
# ────────────────────────────────────────────────────────────────────

def isolation_forest_outliers(cfg: EDAConfig, rpt: Report, sample_means: np.ndarray):
    rpt.section("3. Isolation Forest (sample-level)")

    contamination = 0.05
    rpt(f"  contamination: {contamination}")
    rpt(f"  Input shape: {sample_means.shape}")

    # Sub-sample for speed if very large
    max_fit = 20000
    if sample_means.shape[0] > max_fit:
        idx = np.random.choice(sample_means.shape[0], max_fit, replace=False)
        fit_data = sample_means[idx]
        rpt(f"  Sub-sampled to {max_fit} samples for fitting")
    else:
        fit_data = sample_means
        idx = np.arange(sample_means.shape[0])

    clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = clf.fit_predict(fit_data)  # -1 = outlier, 1 = inlier
    scores = clf.decision_function(fit_data)

    n_outliers = (preds == -1).sum()
    rpt(f"  Outliers detected: {n_outliers} / {len(preds)} ({n_outliers / len(preds) * 100:.2f}%)")
    rpt(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    outlier_indices = idx[preds == -1] if sample_means.shape[0] > max_fit else np.where(preds == -1)[0]

    return outlier_indices, scores, preds


# ────────────────────────────────────────────────────────────────────
# 4. Local Outlier Factor (sample-level)
# ────────────────────────────────────────────────────────────────────

def lof_outliers(cfg: EDAConfig, rpt: Report, sample_means: np.ndarray):
    rpt.section("4. Local Outlier Factor (sample-level)")

    contamination = 0.05
    rpt(f"  contamination: {contamination}")

    max_fit = 20000
    if sample_means.shape[0] > max_fit:
        idx = np.random.choice(sample_means.shape[0], max_fit, replace=False)
        fit_data = sample_means[idx]
        rpt(f"  Sub-sampled to {max_fit} samples for fitting")
    else:
        fit_data = sample_means
        idx = np.arange(sample_means.shape[0])

    clf = LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=-1)
    preds = clf.fit_predict(fit_data)
    scores = clf.negative_outlier_factor_

    n_outliers = (preds == -1).sum()
    rpt(f"  Outliers detected: {n_outliers} / {len(preds)} ({n_outliers / len(preds) * 100:.2f}%)")
    rpt(f"  LOF score range: [{scores.min():.4f}, {scores.max():.4f}]")

    outlier_indices = idx[preds == -1] if sample_means.shape[0] > max_fit else np.where(preds == -1)[0]

    return outlier_indices, scores, preds


# ────────────────────────────────────────────────────────────────────
# 5. PCA Scatter with Outliers
# ────────────────────────────────────────────────────────────────────

def plot_pca_scatter(
    cfg: EDAConfig,
    rpt: Report,
    sample_means: np.ndarray,
    if_outliers: np.ndarray,
    lof_outliers_idx: np.ndarray,
):
    rpt.section("5. PCA Scatter Plot with Outlier Highlighting")

    out = cfg.output_dir

    # 2D PCA of sample means
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(sample_means)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Isolation Forest
    ax = axes[0]
    mask_if = np.zeros(len(sample_means), dtype=bool)
    mask_if[if_outliers] = True
    ax.scatter(coords[~mask_if, 0], coords[~mask_if, 1], s=3, alpha=0.3, label="Inlier")
    ax.scatter(coords[mask_if, 0], coords[mask_if, 1], s=8, alpha=0.7, c="red", label="Outlier")
    ax.set_title(f"Isolation Forest ({mask_if.sum()} outliers)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)

    # LOF
    ax = axes[1]
    mask_lof = np.zeros(len(sample_means), dtype=bool)
    mask_lof[lof_outliers_idx] = True
    ax.scatter(coords[~mask_lof, 0], coords[~mask_lof, 1], s=3, alpha=0.3, label="Inlier")
    ax.scatter(coords[mask_lof, 0], coords[mask_lof, 1], s=8, alpha=0.7, c="red", label="Outlier")
    ax.set_title(f"LOF ({mask_lof.sum()} outliers)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)

    fig.suptitle("Sample-Level PCA Scatter with Outliers", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "pca_scatter_outliers.png", dpi=150)
    plt.close(fig)
    rpt(f"  Saved: {out / 'pca_scatter_outliers.png'}")


# ────────────────────────────────────────────────────────────────────
# 6. Feature Tail Distributions
# ────────────────────────────────────────────────────────────────────

def plot_feature_tails(cfg: EDAConfig, rpt: Report, all_nodes: np.ndarray):
    rpt.section("6. Feature Tail Distribution Analysis")

    out = cfg.output_dir
    n_channels = all_nodes.shape[1]
    n_show = min(16, n_channels)

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for i, ax in enumerate(axes.flat):
        if i >= n_show:
            ax.set_visible(False)
            continue
        col = all_nodes[:, i]
        # Show the extreme tails (beyond 3 std)
        m, s = col.mean(), col.std()
        tail_mask = np.abs(col - m) > 3 * s
        ax.hist(col, bins=100, density=True, alpha=0.5, label="All", color="steelblue")
        if tail_mask.any():
            ax.hist(col[tail_mask], bins=50, density=True, alpha=0.7, label=f"Tail ({tail_mask.sum()})",
                    color="red")
        ax.axvline(m - 3 * s, color="red", linestyle="--", linewidth=0.8)
        ax.axvline(m + 3 * s, color="red", linestyle="--", linewidth=0.8)
        ax.set_title(f"Ch{i} (tail: {tail_mask.mean() * 100:.1f}%)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    fig.suptitle("Feature Tail Distributions (red = beyond 3σ)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "feature_tail_distributions.png", dpi=150)
    plt.close(fig)
    rpt(f"  Saved: {out / 'feature_tail_distributions.png'}")

    # Global tail statistics
    for thresh in [3, 4, 5]:
        m = all_nodes.mean(axis=0)
        s = all_nodes.std(axis=0)
        z = np.abs((all_nodes - m) / (s + 1e-12))
        frac = (z > thresh).mean()
        rpt(f"  Fraction of nodes beyond {thresh}σ: {frac * 100:.3f}%")


# ────────────────────────────────────────────────────────────────────
# 7. Summary
# ────────────────────────────────────────────────────────────────────

def outlier_summary(
    cfg: EDAConfig,
    rpt: Report,
    zscore_mask: np.ndarray,
    iqr_mask: np.ndarray,
    if_outliers: np.ndarray,
    lof_outliers_idx: np.ndarray,
    n_samples: int,
):
    rpt.section("7. Outlier Detection Summary")

    rpt(f"  {'Method':<25s}  {'Outlier Count':>15s}  {'Fraction':>10s}")
    rpt(f"  {'-' * 55}")

    z_count = zscore_mask.any(axis=1).sum()
    iqr_count = iqr_mask.any(axis=1).sum()
    n_nodes = zscore_mask.shape[0]

    rpt(f"  {'Z-Score (nodes)':<25s}  {z_count:>15,}  {z_count / n_nodes * 100:>9.2f}%")
    rpt(f"  {'IQR (nodes)':<25s}  {iqr_count:>15,}  {iqr_count / n_nodes * 100:>9.2f}%")
    rpt(f"  {'Isolation Forest (samples)':<25s}  {len(if_outliers):>15,}  {len(if_outliers) / n_samples * 100:>9.2f}%")
    rpt(f"  {'LOF (samples)':<25s}  {len(lof_outliers_idx):>15,}  {len(lof_outliers_idx) / n_samples * 100:>9.2f}%")

    # Save outlier indices
    out = cfg.output_dir
    np.savez(
        out / "outlier_sample_indices.npz",
        isolation_forest=if_outliers,
        lof=lof_outliers_idx,
    )
    rpt(f"\n  Saved outlier indices to: {out / 'outlier_sample_indices.npz'}")

    # Assess severity
    if_frac = len(if_outliers) / n_samples
    lof_frac = len(lof_outliers_idx) / n_samples
    if if_frac > 0.1 or lof_frac > 0.1:
        rpt(f"\n  ⚠️  HIGH outlier fraction (>10%) – outliers likely dominate some batches")
    elif if_frac > 0.05 or lof_frac > 0.05:
        rpt(f"\n  ⚠️  MODERATE outlier fraction (5-10%) – may affect convergence")
    else:
        rpt(f"\n  ✅ Outlier fraction is within normal range (<5%)")


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    cfg = EDAConfig()
    set_seed(cfg.seed)
    rpt = Report(cfg.output_dir / "outlier_report.txt")

    rpt.section("EDA OUTLIER DETECTION")

    max_batches = None  # Set to e.g. 100 for quick testing
    all_nodes, sample_means, sample_feats = collect_data(cfg, max_batches=max_batches)
    n_samples = sample_means.shape[0]

    rpt(f"  Node vectors: {all_nodes.shape}")
    rpt(f"  Sample means:  {sample_means.shape}")

    zscore_mask = zscore_outliers(cfg, rpt, all_nodes)
    iqr_mask = iqr_outliers(cfg, rpt, all_nodes)
    if_idx, if_scores, if_preds = isolation_forest_outliers(cfg, rpt, sample_means)
    lof_idx, lof_scores, lof_preds = lof_outliers(cfg, rpt, sample_means)
    plot_pca_scatter(cfg, rpt, sample_means, if_idx, lof_idx)
    plot_feature_tails(cfg, rpt, all_nodes)
    outlier_summary(cfg, rpt, zscore_mask, iqr_mask, if_idx, lof_idx, n_samples)

    rpt.section("OUTLIER DETECTION COMPLETE")
    rpt.save()
    print(f"\n✅ Full report saved to: {rpt.path}")


if __name__ == "__main__":
    main()
