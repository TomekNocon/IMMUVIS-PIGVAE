#!/usr/bin/env python3
"""
EDA Batch-Level Analysis + Graph-Specific Checks
=================================================
Since training occurs in mini-batches, this script analyses batch-to-batch
variation and checks for batches that are dominated by outliers.

Also includes graph-specific diagnostics (adjacency, degree, connectivity).

Sections
--------
1. Batch-level feature statistics
2. Batch variance comparison
3. Extreme-value batches
4. Outlier-dominated batch detection
5. Graph-specific checks (adjacency, degree, connectivity, isolated nodes)
6. Node feature variance per graph

Outputs
-------
- ``eda_output/batch_report.txt``
- ``eda_output/batch_mean_distribution.png``
- ``eda_output/batch_std_distribution.png``
- ``eda_output/batch_extreme_values.png``
- ``eda_output/graph_degree_distribution.png``
- ``eda_output/graph_node_feature_variance.png``
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from eda_config import EDAConfig, build_train_dataloader, build_train_val_test, set_seed
from src.data.components.graphs_datamodules import GridGraphDataset


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
# 1–4. Batch-Level Analysis
# ────────────────────────────────────────────────────────────────────

def batch_level_analysis(cfg: EDAConfig, rpt: Report, max_batches: int | None = None):
    rpt.section("1–4. Batch-Level Feature Analysis")

    loader = build_train_dataloader(cfg, shuffle=False)
    total = len(loader) if max_batches is None else min(max_batches, len(loader))

    batch_means = []
    batch_stds = []
    batch_mins = []
    batch_maxs = []
    batch_abs_maxs = []
    batch_nan_counts = []
    batch_inf_counts = []

    for i, batch in enumerate(tqdm(loader, desc="Analysing batches", total=total)):
        nf = batch.node_features.numpy()  # (B*8, N, D)
        flat = nf.reshape(-1, cfg.num_pca_components)

        batch_means.append(flat.mean(axis=0))
        batch_stds.append(flat.std(axis=0))
        batch_mins.append(flat.min(axis=0))
        batch_maxs.append(flat.max(axis=0))
        batch_abs_maxs.append(np.abs(flat).max())
        batch_nan_counts.append(np.isnan(flat).sum())
        batch_inf_counts.append(np.isinf(flat).sum())

        if max_batches is not None and i >= max_batches - 1:
            break

    n_batches = len(batch_means)
    batch_means = np.stack(batch_means)  # (n_batches, D)
    batch_stds = np.stack(batch_stds)
    batch_mins = np.stack(batch_mins)
    batch_maxs = np.stack(batch_maxs)
    batch_abs_maxs = np.array(batch_abs_maxs)

    rpt(f"\n  Number of batches analysed: {n_batches}")
    rpt(f"  Batch size (effective): {cfg.batch_size * cfg.num_aug_per_sample}")

    # ── Per-batch summary ──
    rpt(f"\n  Batch mean (across channels):")
    global_mean_of_means = batch_means.mean()
    rpt(f"    Grand mean: {global_mean_of_means:.6f}")
    rpt(f"    Std of batch means: {batch_means.mean(axis=1).std():.6f}")

    rpt(f"\n  Batch std (across channels):")
    rpt(f"    Mean batch std: {batch_stds.mean():.6f}")
    rpt(f"    Std of batch stds: {batch_stds.mean(axis=1).std():.6f}")

    # ── NaN / Inf per batch ──
    nan_total = sum(batch_nan_counts)
    inf_total = sum(batch_inf_counts)
    rpt(f"\n  Total NaN across all batches: {nan_total}  {'⚠️' if nan_total > 0 else '✅'}")
    rpt(f"  Total Inf across all batches: {inf_total}  {'⚠️' if inf_total > 0 else '✅'}")

    # ── Batch extreme values ──
    rpt(f"\n  Max |value| per batch:")
    rpt(f"    Mean: {batch_abs_maxs.mean():.4f}")
    rpt(f"    Max:  {batch_abs_maxs.max():.4f}")
    rpt(f"    Std:  {batch_abs_maxs.std():.4f}")

    extreme_batches = np.where(batch_abs_maxs > batch_abs_maxs.mean() + 3 * batch_abs_maxs.std())[0]
    rpt(f"  Batches with extreme max |value| (>3σ): {len(extreme_batches)}")
    if len(extreme_batches) > 0:
        rpt(f"    Indices: {extreme_batches[:20].tolist()}")
        rpt(f"    ⚠️  These batches may cause gradient spikes")

    # ── Variance across batches (per channel) ──
    rpt(f"\n  Cross-batch variance per channel (how much batch means fluctuate):")
    cross_batch_var = batch_means.var(axis=0)
    high_var_channels = np.where(cross_batch_var > 0.1)[0]
    rpt(f"    Channels with cross-batch mean variance > 0.1: {len(high_var_channels)}")
    if len(high_var_channels) > 0:
        rpt(f"    ⚠️  High batch-to-batch variation suggests inconsistent sampling")

    # ── Outlier-dominated batches ──
    # A batch is "outlier-dominated" if its mean deviates far from the global mean
    batch_mean_per_batch = batch_means.mean(axis=1)  # scalar per batch
    global_mean = batch_mean_per_batch.mean()
    global_std = batch_mean_per_batch.std()
    outlier_batches = np.where(np.abs(batch_mean_per_batch - global_mean) > 3 * global_std)[0]
    rpt(f"\n  Outlier-dominated batches (batch mean >3σ from global): {len(outlier_batches)}")
    if len(outlier_batches) > 0:
        rpt(f"    Indices: {outlier_batches[:20].tolist()}")
        for idx in outlier_batches[:5]:
            rpt(f"    Batch {idx}: mean={batch_mean_per_batch[idx]:.4f}")

    # ── Plots ──
    out = cfg.output_dir

    # Batch mean distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(batch_mean_per_batch, bins=50, alpha=0.7, edgecolor="none")
    ax.axvline(global_mean, color="red", linestyle="--", label=f"μ={global_mean:.3f}")
    ax.axvline(global_mean + 3 * global_std, color="orange", linestyle="--", label="3σ")
    ax.axvline(global_mean - 3 * global_std, color="orange", linestyle="--")
    ax.set_title("Distribution of Batch Means")
    ax.set_xlabel("Batch Mean")
    ax.legend()

    ax = axes[1]
    ax.plot(batch_mean_per_batch, alpha=0.7, linewidth=0.5)
    ax.axhline(global_mean, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Batch Mean over Time (batch index)")
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Batch Mean")

    fig.tight_layout()
    fig.savefig(out / "batch_mean_distribution.png", dpi=150)
    plt.close(fig)
    rpt(f"\n  Saved: {out / 'batch_mean_distribution.png'}")

    # Batch std distribution
    batch_std_per_batch = batch_stds.mean(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(batch_std_per_batch, bins=50, alpha=0.7, edgecolor="none", color="orange")
    ax.set_title("Distribution of Batch Stds")
    ax.set_xlabel("Batch Std (mean across channels)")

    ax = axes[1]
    ax.plot(batch_std_per_batch, alpha=0.7, linewidth=0.5, color="orange")
    ax.set_title("Batch Std over Time (batch index)")
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Batch Std")

    fig.tight_layout()
    fig.savefig(out / "batch_std_distribution.png", dpi=150)
    plt.close(fig)
    rpt(f"  Saved: {out / 'batch_std_distribution.png'}")

    # Batch extreme values
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(batch_abs_maxs, alpha=0.7, linewidth=0.5, color="crimson")
    ax.axhline(
        batch_abs_maxs.mean() + 3 * batch_abs_maxs.std(),
        color="red", linestyle="--", alpha=0.5, label="3σ threshold",
    )
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Max |value| in batch")
    ax.set_title("Extreme Values per Batch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "batch_extreme_values.png", dpi=150)
    plt.close(fig)
    rpt(f"  Saved: {out / 'batch_extreme_values.png'}")

    return {
        "batch_means": batch_means,
        "batch_stds": batch_stds,
        "batch_abs_maxs": batch_abs_maxs,
        "extreme_batches": extreme_batches,
        "outlier_batches": outlier_batches,
    }


# ────────────────────────────────────────────────────────────────────
# 5. Graph-Specific Checks
# ────────────────────────────────────────────────────────────────────

def graph_specific_checks(cfg: EDAConfig, rpt: Report, max_samples: int = 200):
    rpt.section("5. Graph-Specific Checks")

    grid_size = cfg.grid_size
    n_nodes = grid_size ** 2
    rpt(f"  Grid size: {grid_size}×{grid_size} = {n_nodes} nodes")

    # Build a reference grid graph
    g = nx.grid_graph((grid_size, grid_size))
    rpt(f"\n  Reference grid graph:")
    rpt(f"    Nodes: {g.number_of_nodes()}")
    rpt(f"    Edges: {g.number_of_edges()}")
    rpt(f"    Connected: {nx.is_connected(g)}")
    rpt(f"    Density: {nx.density(g):.6f}")

    # Adjacency matrix
    adj = nx.adjacency_matrix(g).toarray()
    rpt(f"    Adjacency matrix shape: {adj.shape}")
    rpt(f"    Adjacency density: {adj.sum() / (adj.shape[0] * adj.shape[1]):.6f}")

    # Degree distribution
    degrees = [d for _, d in g.degree()]
    rpt(f"\n  Degree distribution:")
    rpt(f"    Min degree: {min(degrees)}")
    rpt(f"    Max degree: {max(degrees)}")
    rpt(f"    Mean degree: {np.mean(degrees):.2f}")

    # Isolated nodes
    isolated = list(nx.isolates(g))
    rpt(f"    Isolated nodes: {len(isolated)}  {'⚠️' if isolated else '✅'}")

    # Degree histogram
    from collections import Counter
    deg_count = Counter(degrees)
    rpt(f"    Degree counts: {dict(sorted(deg_count.items()))}")

    out = cfg.output_dir
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Degree distribution plot
    ax = axes[0]
    degs = sorted(deg_count.keys())
    counts = [deg_count[d] for d in degs]
    ax.bar(degs, counts, alpha=0.7)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.set_title(f"Degree Distribution ({grid_size}×{grid_size} grid)")

    # Adjacency matrix plot
    ax = axes[1]
    ax.imshow(adj, cmap="binary", aspect="equal")
    ax.set_title("Adjacency Matrix")

    fig.tight_layout()
    fig.savefig(out / "graph_degree_distribution.png", dpi=150)
    plt.close(fig)
    rpt(f"\n  Saved: {out / 'graph_degree_distribution.png'}")

    # Check corner/edge/center node statistics
    rpt(f"\n  Grid topology analysis:")
    corner_nodes = []
    edge_nodes = []
    center_nodes = []
    for idx, (row, col) in enumerate([(r, c) for r in range(grid_size) for c in range(grid_size)]):
        is_corner = (row in [0, grid_size - 1]) and (col in [0, grid_size - 1])
        is_edge = (row in [0, grid_size - 1]) or (col in [0, grid_size - 1])
        if is_corner:
            corner_nodes.append(idx)
        elif is_edge:
            edge_nodes.append(idx)
        else:
            center_nodes.append(idx)
    rpt(f"    Corner nodes: {len(corner_nodes)} (degree 2)")
    rpt(f"    Edge nodes: {len(edge_nodes)} (degree 3)")
    rpt(f"    Center nodes: {len(center_nodes)} (degree 4)")


# ────────────────────────────────────────────────────────────────────
# 6. Node Feature Variance per Graph
# ────────────────────────────────────────────────────────────────────

def node_feature_variance_per_graph(cfg: EDAConfig, rpt: Report, max_batches: int | None = None):
    rpt.section("6. Node Feature Variance per Graph")

    loader = build_train_dataloader(cfg, shuffle=False)
    total = len(loader) if max_batches is None else min(max_batches, len(loader))

    per_graph_var = []
    per_graph_range = []

    for i, batch in enumerate(tqdm(loader, desc="Computing per-graph variance", total=total)):
        nf = batch.node_features.numpy()  # (B*8, N, D)
        for j in range(nf.shape[0]):
            graph_features = nf[j]  # (N, D)
            per_graph_var.append(graph_features.var())
            per_graph_range.append(graph_features.max() - graph_features.min())
        if max_batches is not None and i >= max_batches - 1:
            break

    per_graph_var = np.array(per_graph_var)
    per_graph_range = np.array(per_graph_range)

    rpt(f"\n  Per-graph feature variance:")
    rpt(f"    Mean: {per_graph_var.mean():.6f}")
    rpt(f"    Std:  {per_graph_var.std():.6f}")
    rpt(f"    Min:  {per_graph_var.min():.6f}")
    rpt(f"    Max:  {per_graph_var.max():.6f}")

    rpt(f"\n  Per-graph feature range:")
    rpt(f"    Mean: {per_graph_range.mean():.6f}")
    rpt(f"    Std:  {per_graph_range.std():.6f}")
    rpt(f"    Max:  {per_graph_range.max():.6f}")

    # Detect graphs with very low or very high variance
    low_var = (per_graph_var < per_graph_var.mean() * 0.1).sum()
    high_var = (per_graph_var > per_graph_var.mean() * 5).sum()
    rpt(f"\n  Graphs with very low variance (<10% of mean): {low_var}")
    rpt(f"  Graphs with very high variance (>5x mean): {high_var}")
    if low_var > 0 or high_var > 0:
        rpt(f"  ⚠️  High variance disparity between graphs may cause training instability")

    out = cfg.output_dir
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(per_graph_var, bins=80, alpha=0.7, edgecolor="none")
    ax.set_xlabel("Feature Variance")
    ax.set_title("Distribution of Per-Graph Feature Variance")

    ax = axes[1]
    ax.hist(per_graph_range, bins=80, alpha=0.7, edgecolor="none", color="orange")
    ax.set_xlabel("Feature Range (max - min)")
    ax.set_title("Distribution of Per-Graph Feature Range")

    fig.tight_layout()
    fig.savefig(out / "graph_node_feature_variance.png", dpi=150)
    plt.close(fig)
    rpt(f"\n  Saved: {out / 'graph_node_feature_variance.png'}")


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    cfg = EDAConfig()
    set_seed(cfg.seed)
    rpt = Report(cfg.output_dir / "batch_report.txt")

    rpt.section("EDA BATCH-LEVEL & GRAPH ANALYSIS")

    max_batches = None  # Set to e.g. 100 for quick testing
    batch_stats = batch_level_analysis(cfg, rpt, max_batches=max_batches)
    graph_specific_checks(cfg, rpt)
    node_feature_variance_per_graph(cfg, rpt, max_batches=max_batches)

    rpt.section("BATCH & GRAPH ANALYSIS COMPLETE")
    rpt.save()
    print(f"\n✅ Full report saved to: {rpt.path}")


if __name__ == "__main__":
    main()
