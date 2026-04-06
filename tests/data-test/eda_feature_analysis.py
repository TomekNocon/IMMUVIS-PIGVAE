#!/usr/bin/env python3
"""
EDA Feature Analysis
====================
Comprehensive statistical analysis of the training dataset features.

Sections
--------
1. Dataset-level summary (num samples, nodes, channels, sparsity, NaNs, duplicates)
2. Per-feature (per-channel) distribution analysis
3. Per-node analysis (across dataset)
4. Per-channel analysis (variance, importance, correlation)
5. Convergence risk diagnostics

Outputs
-------
- ``eda_output/feature_report.txt``
- ``eda_output/feature_histograms.png``
- ``eda_output/feature_boxplots.png``
- ``eda_output/feature_kde.png``
- ``eda_output/node_mean_std.png``
- ``eda_output/channel_variance_importance.png``
- ``eda_output/channel_correlation.png``
- ``eda_output/channel_covariance.png``
- ``eda_output/feature_statistics.csv``
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as sp_stats
from tqdm import tqdm

from eda_config import EDAConfig, build_train_dataloader, set_seed


# ────────────────────────────────────────────────────────────────────
# Report helper (same pattern as pipeline check)
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
# Data collection
# ────────────────────────────────────────────────────────────────────

def collect_all_features(
    cfg: EDAConfig,
    max_batches: int | None = None,
) -> np.ndarray:
    """Collect all post-PCA node features from the training set.

    Returns
    -------
    all_features : np.ndarray, shape (total_nodes, num_pca_components)
    """
    loader = build_train_dataloader(cfg, shuffle=False)
    total = len(loader) if max_batches is None else min(max_batches, len(loader))
    chunks = []
    for i, batch in enumerate(tqdm(loader, desc="Collecting features", total=total)):
        nf = batch.node_features  # (B*8, N, D)
        chunks.append(nf.reshape(-1, cfg.num_pca_components).numpy())
        if max_batches is not None and i >= max_batches - 1:
            break
    return np.concatenate(chunks, axis=0)


def collect_per_sample_features(
    cfg: EDAConfig,
    max_batches: int | None = None,
) -> list[np.ndarray]:
    """Return list of per-sample arrays, each shape (N, D)."""
    loader = build_train_dataloader(cfg, shuffle=False)
    total = len(loader) if max_batches is None else min(max_batches, len(loader))
    samples = []
    for i, batch in enumerate(tqdm(loader, desc="Collecting per-sample", total=total)):
        nf = batch.node_features.numpy()  # (B*8, N, D)
        for j in range(nf.shape[0]):
            samples.append(nf[j])
        if max_batches is not None and i >= max_batches - 1:
            break
    return samples


# ────────────────────────────────────────────────────────────────────
# 1. Dataset-Level Summary
# ────────────────────────────────────────────────────────────────────

def dataset_level_summary(cfg: EDAConfig, rpt: Report, all_feats: np.ndarray, samples: list[np.ndarray]):
    rpt.section("1. Dataset-Level Summary")

    n_total_nodes = all_feats.shape[0]
    n_channels = all_feats.shape[1]
    n_samples = len(samples)
    nodes_per_sample = samples[0].shape[0] if samples else 0

    rpt(f"  Number of graph samples (B*8 per batch): {n_samples}")
    rpt(f"  Nodes per sample: {nodes_per_sample}")
    rpt(f"  Number of feature channels: {n_channels}")
    rpt(f"  Total node vectors: {n_total_nodes}")

    # Sparsity
    zero_count = (all_feats == 0.0).sum()
    sparsity = zero_count / all_feats.size
    rpt(f"\n  Exact zeros: {zero_count:,} / {all_feats.size:,} ({sparsity * 100:.2f}%)")
    near_zero = (np.abs(all_feats) < 1e-6).sum()
    rpt(f"  Near-zero (|x| < 1e-6): {near_zero:,} ({near_zero / all_feats.size * 100:.2f}%)")

    # Missing / invalid values
    nan_count = np.isnan(all_feats).sum()
    inf_count = np.isinf(all_feats).sum()
    rpt(f"\n  NaN count: {nan_count}  {'⚠️' if nan_count > 0 else '✅'}")
    rpt(f"  Inf count: {inf_count}  {'⚠️' if inf_count > 0 else '✅'}")

    # Duplicate detection (sample-level)
    rpt(f"\n  Checking for duplicate samples...")
    sample_hashes = set()
    dups = 0
    for s in samples:
        h = hash(s.tobytes())
        if h in sample_hashes:
            dups += 1
        sample_hashes.add(h)
    rpt(f"  Duplicate samples: {dups}  {'⚠️' if dups > 0 else '✅'}")

    # Global statistics
    rpt(f"\n  Global feature statistics:")
    rpt(f"    mean:  {all_feats.mean():.6f}")
    rpt(f"    std:   {all_feats.std():.6f}")
    rpt(f"    min:   {all_feats.min():.6f}")
    rpt(f"    max:   {all_feats.max():.6f}")
    rpt(f"    |max|: {np.abs(all_feats).max():.6f}")


# ────────────────────────────────────────────────────────────────────
# 2. Per-Feature (Channel) Distribution
# ────────────────────────────────────────────────────────────────────

def per_feature_analysis(cfg: EDAConfig, rpt: Report, all_feats: np.ndarray):
    rpt.section("2. Per-Feature (Channel) Distribution Analysis")

    n_channels = all_feats.shape[1]
    out = cfg.output_dir
    stats_rows = []

    rpt(f"\n  {'Ch':>4s}  {'mean':>10s}  {'std':>10s}  {'min':>10s}  {'max':>10s}  "
        f"{'skew':>8s}  {'kurt':>8s}  {'Flags':s}")

    warnings = []
    for c in range(n_channels):
        col = all_feats[:, c]
        m = col.mean()
        s = col.std()
        mn = col.min()
        mx = col.max()
        sk = float(sp_stats.skew(col))
        ku = float(sp_stats.kurtosis(col))
        flags = []
        if s < 1e-6:
            flags.append("NEAR-CONST")
        if abs(sk) > 3:
            flags.append("HEAVY-SKEW")
        if ku > 10:
            flags.append("HEAVY-TAIL")
        if abs(mx - mn) > 50:
            flags.append("WIDE-RANGE")
        flag_str = ", ".join(flags) if flags else ""
        rpt(f"  {c:4d}  {m:10.4f}  {s:10.4f}  {mn:10.4f}  {mx:10.4f}  "
            f"{sk:8.3f}  {ku:8.3f}  {flag_str}")
        stats_rows.append({
            "channel": c, "mean": m, "std": s, "min": mn, "max": mx,
            "skewness": sk, "kurtosis": ku, "flags": flag_str,
        })
        if flags:
            warnings.append((c, flags))

    # Save CSV
    csv_path = out / "feature_statistics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=stats_rows[0].keys())
        w.writeheader()
        w.writerows(stats_rows)
    rpt(f"\n  Saved: {csv_path}")

    # Warnings summary
    rpt(f"\n  Channels with warnings: {len(warnings)} / {n_channels}")
    for ch, fl in warnings[:20]:
        rpt(f"    Ch {ch}: {', '.join(fl)}")
    if len(warnings) > 20:
        rpt(f"    ... and {len(warnings) - 20} more")

    # ── Histograms (first 16 channels) ──
    n_show = min(16, n_channels)
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for i, ax in enumerate(axes.flat):
        if i >= n_show:
            ax.set_visible(False)
            continue
        col = all_feats[:, i]
        ax.hist(col, bins=80, density=True, alpha=0.7, edgecolor="none", color="steelblue")
        ax.set_title(f"Ch{i} (μ={col.mean():.2f}, σ={col.std():.2f})", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle("Feature Histograms (first 16 channels)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "feature_histograms.png", dpi=150)
    plt.close(fig)
    rpt(f"  Saved: {out / 'feature_histograms.png'}")

    # ── KDE plots (first 16 channels) ──
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for i, ax in enumerate(axes.flat):
        if i >= n_show:
            ax.set_visible(False)
            continue
        col = all_feats[:, i]
        # Sub-sample for KDE performance
        sub = col[::max(1, len(col) // 5000)]
        try:
            kde = sp_stats.gaussian_kde(sub)
            x_grid = np.linspace(sub.min(), sub.max(), 200)
            ax.plot(x_grid, kde(x_grid), color="darkorange", linewidth=1.5)
            ax.fill_between(x_grid, kde(x_grid), alpha=0.3, color="orange")
        except Exception:
            ax.hist(sub, bins=60, density=True, alpha=0.5)
        ax.set_title(f"Ch{i}", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle("Feature KDE Distributions (first 16 channels)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "feature_kde.png", dpi=150)
    plt.close(fig)
    rpt(f"  Saved: {out / 'feature_kde.png'}")

    # ── Boxplots ──
    # For many channels, show all using a compact horizontal boxplot
    fig, ax = plt.subplots(figsize=(14, max(6, n_channels * 0.12)))
    bp = ax.boxplot(
        [all_feats[:, c] for c in range(n_channels)],
        vert=False, patch_artist=True, showfliers=True,
        flierprops=dict(marker=".", markersize=1, alpha=0.3),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax.set_yticklabels([str(c) for c in range(n_channels)], fontsize=5)
    ax.set_xlabel("Feature Value")
    ax.set_title("Feature Boxplots (all channels)")
    fig.tight_layout()
    fig.savefig(out / "feature_boxplots.png", dpi=150)
    plt.close(fig)
    rpt(f"  Saved: {out / 'feature_boxplots.png'}")

    return stats_rows


# ────────────────────────────────────────────────────────────────────
# 3. Per-Node Analysis
# ────────────────────────────────────────────────────────────────────

def per_node_analysis(cfg: EDAConfig, rpt: Report, samples: list[np.ndarray]):
    rpt.section("3. Per-Node Analysis (across dataset)")

    if not samples:
        rpt("  No samples collected.")
        return

    n_nodes = samples[0].shape[0]
    n_channels = samples[0].shape[1]
    n_samples = len(samples)
    out = cfg.output_dir

    # Stack all samples: (n_samples, n_nodes, n_channels)
    all_samples = np.stack(samples, axis=0)

    # Per-node statistics (mean/std/var across samples and channels)
    node_means = all_samples.mean(axis=(0, 2))  # (n_nodes,)
    node_stds = all_samples.std(axis=(0, 2))
    node_vars = all_samples.var(axis=(0, 2))

    rpt(f"\n  Nodes: {n_nodes}, Samples: {n_samples}, Channels: {n_channels}")
    rpt(f"\n  {'Node':>6s}  {'mean':>10s}  {'std':>10s}  {'var':>10s}  {'Flags':s}")

    constant_nodes = []
    high_var_nodes = []
    low_var_nodes = []

    for n in range(n_nodes):
        node_data = all_samples[:, n, :]  # (n_samples, n_channels)
        m = node_data.mean()
        s = node_data.std()
        v = node_data.var()
        flags = []
        if s < 1e-5:
            flags.append("CONSTANT")
            constant_nodes.append(n)
        if v > 100:
            flags.append("HIGH-VAR")
            high_var_nodes.append(n)
        if 0 < v < 0.01:
            flags.append("LOW-VAR")
            low_var_nodes.append(n)
        flag_str = ", ".join(flags)
        rpt(f"  {n:6d}  {m:10.4f}  {s:10.4f}  {v:10.4f}  {flag_str}")

    rpt(f"\n  Summary:")
    rpt(f"    Constant nodes (std < 1e-5): {len(constant_nodes)}  "
        f"{'⚠️' if constant_nodes else '✅'}")
    rpt(f"    High-variance nodes (var > 100): {len(high_var_nodes)}  "
        f"{'⚠️' if high_var_nodes else '✅'}")
    rpt(f"    Low-variance nodes (0 < var < 0.01): {len(low_var_nodes)}  "
        f"{'⚠️' if low_var_nodes else '✅'}")

    # Plot node mean and std
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Mean across dataset per node
    per_node_mean = all_samples.mean(axis=0)  # (n_nodes, n_channels)
    ax1.imshow(per_node_mean.T, aspect="auto", cmap="RdBu_r")
    ax1.set_xlabel("Node Index")
    ax1.set_ylabel("Channel")
    ax1.set_title("Mean Activation per Node × Channel")

    per_node_std = all_samples.std(axis=0)  # (n_nodes, n_channels)
    im2 = ax2.imshow(per_node_std.T, aspect="auto", cmap="hot")
    ax2.set_xlabel("Node Index")
    ax2.set_ylabel("Channel")
    ax2.set_title("Std Activation per Node × Channel")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    fig.tight_layout()
    fig.savefig(out / "node_mean_std.png", dpi=150)
    plt.close(fig)
    rpt(f"\n  Saved: {out / 'node_mean_std.png'}")

    # Node variance bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(n_nodes), node_vars, alpha=0.7)
    ax.set_xlabel("Node Index")
    ax.set_ylabel("Variance (across samples & channels)")
    ax.set_title("Per-Node Variance")
    fig.tight_layout()
    fig.savefig(out / "node_variance.png", dpi=150)
    plt.close(fig)
    rpt(f"  Saved: {out / 'node_variance.png'}")


# ────────────────────────────────────────────────────────────────────
# 4. Per-Channel Analysis (correlation, importance)
# ────────────────────────────────────────────────────────────────────

def per_channel_analysis(cfg: EDAConfig, rpt: Report, all_feats: np.ndarray):
    rpt.section("4. Per-Channel Analysis")

    n_channels = all_feats.shape[1]
    out = cfg.output_dir

    # Variance per channel
    chan_var = all_feats.var(axis=0)
    chan_var_sorted_idx = np.argsort(chan_var)[::-1]

    rpt(f"\n  Channel variance (sorted by importance):")
    rpt(f"  {'Rank':>5s}  {'Ch':>4s}  {'Variance':>12s}  {'% Total':>8s}")
    total_var = chan_var.sum()
    for rank, idx in enumerate(chan_var_sorted_idx[:20]):
        pct = chan_var[idx] / total_var * 100
        rpt(f"  {rank:5d}  {idx:4d}  {chan_var[idx]:12.6f}  {pct:8.2f}%")

    # Zero-variance channels
    zero_var = (chan_var < 1e-8).sum()
    rpt(f"\n  Zero-variance channels (< 1e-8): {zero_var}  "
        f"{'⚠️  DEAD FEATURES' if zero_var > 0 else '✅'}")

    # Channel variance bar chart
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(n_channels), chan_var, alpha=0.7)
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("Variance")
    ax.set_title("Feature Variance per Channel (importance proxy)")
    fig.tight_layout()
    fig.savefig(out / "channel_variance_importance.png", dpi=150)
    plt.close(fig)
    rpt(f"\n  Saved: {out / 'channel_variance_importance.png'}")

    # Correlation matrix
    rpt(f"\n  Computing correlation matrix...")
    corr = np.corrcoef(all_feats.T)  # (D, D)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_title("Channel Correlation Matrix")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "channel_correlation.png", dpi=150)
    plt.close(fig)
    rpt(f"  Saved: {out / 'channel_correlation.png'}")

    # Highly correlated feature pairs
    np.fill_diagonal(corr, 0)
    high_corr = np.argwhere(np.abs(corr) > 0.5)
    rpt(f"\n  Highly correlated channel pairs (|r| > 0.5): {len(high_corr) // 2}")
    if len(high_corr) > 0:
        rpt(f"    Max |correlation|: {np.abs(corr).max():.4f}")
        # Show top 10
        flat_idx = np.argsort(np.abs(corr).ravel())[::-1]
        shown = set()
        count = 0
        for fi in flat_idx:
            r, c = divmod(fi, n_channels)
            if r >= c:
                continue
            pair = (min(r, c), max(r, c))
            if pair in shown:
                continue
            shown.add(pair)
            rpt(f"    Ch{r} ↔ Ch{c}: r={corr[r, c]:.4f}")
            count += 1
            if count >= 10:
                break

    # Covariance matrix
    cov = np.cov(all_feats.T)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cov, cmap="viridis", aspect="auto")
    ax.set_title("Channel Covariance Matrix")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "channel_covariance.png", dpi=150)
    plt.close(fig)
    rpt(f"  Saved: {out / 'channel_covariance.png'}")

    # Feature scale disparity
    scales = all_feats.std(axis=0)
    scale_ratio = scales.max() / (scales.min() + 1e-12)
    rpt(f"\n  Feature scale disparity (max_std / min_std): {scale_ratio:.2f}  "
        f"{'⚠️  DRASTIC SCALE DIFFERENCE' if scale_ratio > 100 else '✅'}")


# ────────────────────────────────────────────────────────────────────
# 5. Convergence Risk Diagnostics
# ────────────────────────────────────────────────────────────────────

def convergence_diagnostics(cfg: EDAConfig, rpt: Report, all_feats: np.ndarray, stats_rows: list[dict]):
    rpt.section("5. Convergence Risk Diagnostics")

    risks = []

    # 5a. Feature scales differ drastically
    scales = all_feats.std(axis=0)
    scale_ratio = scales.max() / (scales.min() + 1e-12)
    if scale_ratio > 50:
        risks.append(f"Feature scale ratio {scale_ratio:.1f}x (max_std/min_std) – may cause gradient imbalance")

    # 5b. PCA outputs contain exploding values
    abs_max = np.abs(all_feats).max()
    if abs_max > 20:
        risks.append(f"Max |feature value| = {abs_max:.2f} – potential gradient explosion")

    # 5c. Extremely skewed distributions
    skewed = [r for r in stats_rows if abs(r["skewness"]) > 5]
    if skewed:
        risks.append(f"{len(skewed)} channels with |skewness| > 5 – heavy asymmetry")

    # 5d. Heavy tails
    heavy_tail = [r for r in stats_rows if r["kurtosis"] > 20]
    if heavy_tail:
        risks.append(f"{len(heavy_tail)} channels with kurtosis > 20 – extreme outliers likely")

    # 5e. Zero-variance features
    zero_var = (scales < 1e-6).sum()
    if zero_var > 0:
        risks.append(f"{zero_var} channels with near-zero variance – dead features")

    # 5f. Highly correlated features
    corr = np.corrcoef(all_feats.T)
    np.fill_diagonal(corr, 0)
    n_high = (np.abs(corr) > 0.7).sum() // 2
    if n_high > 0:
        risks.append(f"{n_high} feature pairs with |correlation| > 0.7 – redundancy")

    # 5g. NaN / Inf
    if np.isnan(all_feats).any():
        risks.append("NaN values detected in features")
    if np.isinf(all_feats).any():
        risks.append("Inf values detected in features")

    # Summary
    if risks:
        rpt(f"\n  ⚠️  {len(risks)} convergence risk(s) detected:\n")
        for i, r in enumerate(risks, 1):
            rpt(f"    {i}. {r}")
    else:
        rpt(f"\n  ✅ No major convergence risks detected in feature distributions.")

    rpt(f"\n  Recommendation:")
    if scale_ratio > 50:
        rpt(f"    - Consider additional feature scaling or clipping")
    if abs_max > 20:
        rpt(f"    - Consider clamping post-PCA features or adjusting Welford statistics")
    if skewed or heavy_tail:
        rpt(f"    - Consider robust scaling (quantile) or arcsinh transform")

    return risks


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    cfg = EDAConfig()
    set_seed(cfg.seed)
    rpt = Report(cfg.output_dir / "feature_report.txt")

    rpt.section("EDA FEATURE ANALYSIS")
    rpt(f"  Collecting training data features (post-PCA, z-scored)...")

    # Use max_batches=None for full analysis, or a smaller number for quick checks
    max_batches = None  # Set to e.g. 100 for quick debugging
    all_feats = collect_all_features(cfg, max_batches=max_batches)
    samples = collect_per_sample_features(cfg, max_batches=max_batches)

    rpt(f"  Total feature matrix shape: {all_feats.shape}")

    dataset_level_summary(cfg, rpt, all_feats, samples)
    stats_rows = per_feature_analysis(cfg, rpt, all_feats)
    per_node_analysis(cfg, rpt, samples)
    per_channel_analysis(cfg, rpt, all_feats)
    convergence_diagnostics(cfg, rpt, all_feats, stats_rows)

    rpt.section("FEATURE ANALYSIS COMPLETE")
    rpt.save()
    print(f"\n✅ Full report saved to: {rpt.path}")


if __name__ == "__main__":
    main()
