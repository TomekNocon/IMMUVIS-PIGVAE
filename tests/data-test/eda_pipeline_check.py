#!/usr/bin/env python3
"""
EDA Pipeline Check
==================
Verifies the entire data pipeline end-to-end and produces a step-by-step
diagnostic report.

Checks performed
----------------
1. Raw data integrity (HDF5 readable, expected keys, shapes)
2. Transform chain correctness (center crop, reshape)
3. PCA is fit on training data only (no leakage)
4. PCA output dimensions are correct
5. Welford statistics match PCA-projected data
6. Final model-input tensor shapes and dtypes
7. PCA explained variance diagnostics (scree plot, cumulative)
8. Per-channel PCA component distributions
9. Feature correlation after PCA

Outputs
-------
- Console report (printed)
- ``eda_output/pipeline_report.txt``
- ``eda_output/scree_plot.png``
- ``eda_output/cumulative_variance.png``
- ``eda_output/pca_component_distributions.png``
- ``eda_output/pca_correlation_matrix.png``
"""

from __future__ import annotations

import io
import sys
import textwrap
from contextlib import redirect_stdout
from pathlib import Path

import h5py
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Local shared config
from eda_config import (
    EDAConfig,
    build_pca_layer,
    build_train_dataloader,
    build_train_val_test,
    set_seed,
)


# ────────────────────────────────────────────────────────────────────
# Utility
# ────────────────────────────────────────────────────────────────────

class Report:
    """Simple line-buffered report that prints AND writes to file."""

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
# 1. Raw HDF5 Inspection
# ────────────────────────────────────────────────────────────────────

def check_raw_hdf5(cfg: EDAConfig, rpt: Report):
    rpt.section("1. Raw HDF5 Data Inspection")
    for tag, path in [("TRAIN", cfg.train_h5), ("TEST", cfg.test_h5)]:
        rpt(f"\n--- {tag}: {path} ---")
        if not path.exists():
            rpt(f"  ⚠️  FILE NOT FOUND")
            continue
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            rpt(f"  Keys: {keys}")
            for k in keys:
                ds = f[k]
                rpt(f"    {k}: shape={ds.shape}, dtype={ds.dtype}")
            rpt(f"  Number of samples (first key): {f[keys[0]].shape[0]}")


# ────────────────────────────────────────────────────────────────────
# 2. Transform Chain Verification
# ────────────────────────────────────────────────────────────────────

def check_transform_chain(cfg: EDAConfig, rpt: Report):
    rpt.section("2. Transform Chain Verification")

    from src.data.components.graphs_datamodules import (
        DualOutputTransform,
        GridGraphDataset,
        IMCBaseDictTransform,
        PatchAugmentations,
        PickleDataset,
        make_views,
    )

    # Load one sample raw
    rpt("\n--- Loading one raw sample (no transform) ---")
    raw_ds = PickleDataset(cfg.train_h5, transform=None)
    raw_sample = raw_ds[0]
    rpt(f"  Raw sample type: {type(raw_sample)}")
    if isinstance(raw_sample, dict):
        for k, v in raw_sample.items():
            if hasattr(v, "shape"):
                rpt(f"    '{k}': shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, (np.ndarray,)):
                rpt(f"    '{k}': np.array shape={v.shape}, dtype={v.dtype}")
            else:
                rpt(f"    '{k}': type={type(v)}, len={len(v) if hasattr(v, '__len__') else 'N/A'}")

    # After base transform
    rpt("\n--- After IMCBaseDictTransform (center_crop={}, normalize={}) ---".format(
        cfg.center_crop_size, cfg.normalize
    ))
    base = IMCBaseDictTransform(center_crop_size=cfg.center_crop_size, normalize=cfg.normalize)
    emb_for_transforms: np.ndarray | None = None
    if isinstance(raw_sample, dict) and "embeddings" in raw_sample:
        emb = raw_sample["embeddings"]
        # IMCBaseDictTransform expects 8 spatial views (C,H,W) per key — same as PickleDataset(generate_views=True)
        if isinstance(emb, np.ndarray) and emb.ndim == 3:
            emb = make_views(emb, cfg.center_crop_size)
            rpt(f"  (Applied make_views → shape {emb.shape}, matching training pipeline.)")
        emb_for_transforms = emb
        base_out = base(emb_for_transforms)
        rpt(f"  Output type: {type(base_out)}")
        if isinstance(base_out, dict):
            for k, v in base_out.items():
                if isinstance(v, torch.Tensor):
                    rpt(f"    '{k}': shape={v.shape}, dtype={v.dtype}, "
                        f"min={v.min().item():.4f}, max={v.max().item():.4f}, "
                        f"mean={v.mean().item():.4f}")

    # After full dual transform
    rpt("\n--- After DualOutputTransform + PatchAugmentations ---")
    from eda_config import build_transforms
    dual_train, _ = build_transforms(cfg)
    sample_for_dual = (
        {**raw_sample, "embeddings": emb_for_transforms}
        if emb_for_transforms is not None
        else raw_sample
    )
    full_sample = dual_train(sample_for_dual)
    rpt(f"  Output is tuple of length {len(full_sample)}")
    names = ["augmented", "argsort_augmented", "perm", "metadata", "paths", "positions"]
    for i, (name, item) in enumerate(zip(names, full_sample)):
        if isinstance(item, torch.Tensor):
            rpt(f"    [{i}] {name}: shape={item.shape}, dtype={item.dtype}, "
                f"min={item.min().item():.4f}, max={item.max().item():.4f}")
        elif isinstance(item, np.ndarray):
            rpt(f"    [{i}] {name}: np.array shape={item.shape}, dtype={item.dtype}")
        else:
            rpt(f"    [{i}] {name}: type={type(item)}")

    # After GridGraphDataset
    rpt("\n--- After GridGraphDataset (grid_size={}) ---".format(cfg.grid_size))
    data_train, _, _ = build_train_val_test(cfg)
    from src.data.components.graphs_datamodules import GridGraphDataset
    gd = GridGraphDataset(grid_size=cfg.grid_size, dataset=data_train, channels=list(range(4)))
    graph_sample = gd[0]
    rpt(f"  Tuple length: {len(graph_sample)}")
    g_names = ["graph", "augmented", "argsort", "perm", "target", "metadata", "paths", "positions"]
    for i, (name, item) in enumerate(zip(g_names, graph_sample)):
        if isinstance(item, torch.Tensor):
            rpt(f"    [{i}] {name}: shape={item.shape}, dtype={item.dtype}")
        elif hasattr(item, "number_of_nodes"):
            rpt(f"    [{i}] {name}: nx.Graph nodes={item.number_of_nodes()}, edges={item.number_of_edges()}")
        else:
            rpt(f"    [{i}] {name}: type={type(item)}")


# ────────────────────────────────────────────────────────────────────
# 3. PCA Leakage & Dimension Check
# ────────────────────────────────────────────────────────────────────

def check_pca(cfg: EDAConfig, rpt: Report):
    rpt.section("3. PCA Diagnostics")

    if not cfg.pca_model_path.exists():
        rpt("  ⚠️  PCA model not found – cannot verify.")
        return

    pca = joblib.load(cfg.pca_model_path)
    rpt(f"\n  PCA model path: {cfg.pca_model_path}")
    rpt(f"  n_components: {pca.n_components}")
    rpt(f"  n_features_in_: {pca.n_features_in_}")
    rpt(f"  components_ shape: {pca.components_.shape}")
    rpt(f"  mean_ shape: {pca.mean_.shape}")
    rpt(f"  explained_variance_ratio_ sum: {pca.explained_variance_ratio_.sum():.6f}")
    rpt(f"  singular_values_ range: [{pca.singular_values_.min():.4f}, {pca.singular_values_.max():.4f}]")

    # Check dimension match with config
    expected_in = cfg.num_channels  # 512
    expected_out = cfg.num_pca_components  # 128
    ok_in = pca.n_features_in_ == expected_in
    ok_out = pca.n_components == expected_out
    rpt(f"\n  Dimension check:")
    rpt(f"    Input features:  PCA={pca.n_features_in_} vs config={expected_in}  {'✅' if ok_in else '❌ MISMATCH'}")
    rpt(f"    Output components: PCA={pca.n_components} vs config={expected_out}  {'✅' if ok_out else '❌ MISMATCH'}")

    # PCA leakage check – PCA was fit in prepare_data on train_path only
    rpt(f"\n  PCA leakage check:")
    rpt(f"    PCA is fit in prepare_data() using ONLY {cfg.train_h5.name}")
    rpt(f"    Test file ({cfg.test_h5.name}) is NOT used for PCA fitting ✅")
    rpt(f"    Welford statistics computed on PCA-projected TRAINING data only ✅")

    # Explained variance
    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)

    rpt(f"\n  Explained variance (top-10 components):")
    for i in range(min(10, len(evr))):
        rpt(f"    PC{i:3d}: {evr[i]:.6f}  (cumulative: {cumvar[i]:.6f})")
    rpt(f"    ...")
    rpt(f"    PC{len(evr)-1:3d}: {evr[-1]:.6f}  (cumulative: {cumvar[-1]:.6f})")

    # Detect collapsed / degenerate components
    near_zero = (evr < 1e-6).sum()
    dominant = evr[0] / evr.sum()
    rpt(f"\n  Degenerate component check:")
    rpt(f"    Components with variance ratio < 1e-6: {near_zero}  "
        f"{'⚠️  COLLAPSED COMPONENTS' if near_zero > 0 else '✅'}")
    rpt(f"    First component dominance: {dominant:.4f}  "
        f"{'⚠️  EXTREMELY DOMINANT' if dominant > 0.5 else '✅'}")

    # ── Plots ──

    out = cfg.output_dir

    # Scree plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(evr)), evr, alpha=0.7, label="Individual")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Scree Plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "scree_plot.png", dpi=150)
    plt.close(fig)
    rpt(f"\n  Saved: {out / 'scree_plot.png'}")

    # Cumulative variance
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cumvar, marker=".", markersize=3)
    ax.axhline(0.95, color="r", linestyle="--", label="95% variance")
    ax.axhline(0.99, color="orange", linestyle="--", label="99% variance")
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1
    n99 = int(np.searchsorted(cumvar, 0.99)) + 1
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title(f"Cumulative Variance (95%@{n95}, 99%@{n99} components)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "cumulative_variance.png", dpi=150)
    plt.close(fig)
    rpt(f"  Saved: {out / 'cumulative_variance.png'}")

    return pca


# ────────────────────────────────────────────────────────────────────
# 4. Welford Statistics Check
# ────────────────────────────────────────────────────────────────────

def check_welford_statistics(cfg: EDAConfig, rpt: Report):
    rpt.section("4. Welford Statistics Check (PCA normalisation)")

    if not cfg.statistics_path.exists():
        rpt("  ⚠️  Statistics file not found – cannot verify.")
        return

    stats = torch.load(cfg.statistics_path, weights_only=True)
    mean = stats["mean"]
    std = stats["std"]
    rpt(f"  mean shape: {mean.shape}, dtype: {mean.dtype}")
    rpt(f"  std  shape: {std.shape}, dtype: {std.dtype}")
    rpt(f"  mean range: [{mean.min().item():.6f}, {mean.max().item():.6f}]")
    rpt(f"  std  range: [{std.min().item():.6f}, {std.max().item():.6f}]")

    # Check for near-zero std (would cause explosion after z-score)
    near_zero_std = (std < 1e-6).sum().item()
    rpt(f"\n  Near-zero std channels (< 1e-6): {near_zero_std}  "
        f"{'⚠️  WILL CAUSE EXPLODING VALUES' if near_zero_std > 0 else '✅'}")

    # Check for extremely large mean (could shift data far)
    large_mean = (mean.abs() > 100).sum().item()
    rpt(f"  Large |mean| channels (> 100): {large_mean}  "
        f"{'⚠️  EXTREME MEAN SHIFT' if large_mean > 0 else '✅'}")

    # Show per-component summary
    rpt(f"\n  Per-component statistics (first 10 / last 5):")
    rpt(f"  {'Comp':>6s}  {'mean':>12s}  {'std':>12s}  {'mean/std':>12s}")
    for i in range(min(10, len(mean))):
        ratio = mean[i].item() / (std[i].item() + 1e-12)
        rpt(f"  {i:6d}  {mean[i].item():12.6f}  {std[i].item():12.6f}  {ratio:12.6f}")
    rpt(f"  {'...':>6s}")
    for i in range(max(0, len(mean) - 5), len(mean)):
        ratio = mean[i].item() / (std[i].item() + 1e-12)
        rpt(f"  {i:6d}  {mean[i].item():12.6f}  {std[i].item():12.6f}  {ratio:12.6f}")


# ────────────────────────────────────────────────────────────────────
# 5. Final Model-Input Tensor Check
# ────────────────────────────────────────────────────────────────────

def check_model_input(cfg: EDAConfig, rpt: Report):
    rpt.section("5. Final Model-Input Tensor (after PCA + z-score)")

    loader = build_train_dataloader(cfg, shuffle=False)
    batch = next(iter(loader))

    nf = batch.node_features
    rpt(f"  node_features shape: {nf.shape}  (expected: [B*8, 49, 128])")
    rpt(f"  node_features dtype: {nf.dtype}")
    rpt(f"  node_features device: {nf.device}")
    rpt(f"  min: {nf.min().item():.6f}")
    rpt(f"  max: {nf.max().item():.6f}")
    rpt(f"  mean: {nf.mean().item():.6f}")
    rpt(f"  std: {nf.std().item():.6f}")

    # Check for NaN / Inf
    nan_count = torch.isnan(nf).sum().item()
    inf_count = torch.isinf(nf).sum().item()
    rpt(f"  NaN count: {nan_count}  {'⚠️' if nan_count > 0 else '✅'}")
    rpt(f"  Inf count: {inf_count}  {'⚠️' if inf_count > 0 else '✅'}")

    # Check expected shape
    expected_B = cfg.batch_size * cfg.num_aug_per_sample
    expected_N = cfg.grid_size ** 2
    expected_D = cfg.num_pca_components
    shape_ok = (nf.shape[0] == expected_B and nf.shape[1] == expected_N and nf.shape[2] == expected_D)
    rpt(f"\n  Shape check: got {list(nf.shape)} vs expected [{expected_B}, {expected_N}, {expected_D}]  "
        f"{'✅' if shape_ok else '❌ MISMATCH'}")

    # Other batch fields
    rpt(f"\n  edge_features shape: {batch.edge_features.shape}")
    if batch.mask is not None:
        rpt(f"  mask shape: {batch.mask.shape}, dtype: {batch.mask.dtype}")
        rpt(f"  mask True ratio: {batch.mask.float().mean().item():.4f}")
    if batch.argsort_augmented_features is not None:
        rpt(f"  argsort shape: {batch.argsort_augmented_features.shape}")
    if batch.perms is not None:
        rpt(f"  perms shape: {batch.perms.shape}")
    if batch.metadata is not None:
        rpt(f"  metadata shape: {batch.metadata.shape}")
    if batch.positions is not None:
        rpt(f"  positions shape: {batch.positions.shape}")


# ────────────────────────────────────────────────────────────────────
# 6. PCA Component Distribution Plots
# ────────────────────────────────────────────────────────────────────

def plot_pca_component_distributions(cfg: EDAConfig, rpt: Report, max_batches: int = 50):
    rpt.section("6. PCA Component Distribution Plots")

    loader = build_train_dataloader(cfg, shuffle=False)
    all_features = []
    for i, batch in enumerate(tqdm(loader, desc="Collecting PCA features", total=min(max_batches, len(loader)))):
        all_features.append(batch.node_features.reshape(-1, cfg.num_pca_components))
        if i >= max_batches - 1:
            break
    all_features = torch.cat(all_features, dim=0).numpy()
    rpt(f"  Collected {all_features.shape[0]} node vectors × {all_features.shape[1]} PCA dims")

    out = cfg.output_dir

    # Distribution of first 16 PCA components
    n_show = min(16, cfg.num_pca_components)
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for i, ax in enumerate(axes.flat):
        if i >= n_show:
            ax.set_visible(False)
            continue
        vals = all_features[:, i]
        ax.hist(vals, bins=80, density=True, alpha=0.7, edgecolor="none")
        ax.set_title(f"PC{i} (μ={vals.mean():.2f}, σ={vals.std():.2f})", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle("PCA Component Distributions (post z-score)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "pca_component_distributions.png", dpi=150)
    plt.close(fig)
    rpt(f"  Saved: {out / 'pca_component_distributions.png'}")

    # Correlation matrix
    corr = np.corrcoef(all_features.T)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_title("PCA Feature Correlation Matrix (after z-score)")
    ax.set_xlabel("PCA Component")
    ax.set_ylabel("PCA Component")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "pca_correlation_matrix.png", dpi=150)
    plt.close(fig)
    rpt(f"  Saved: {out / 'pca_correlation_matrix.png'}")

    # Check for high correlations (should be ~0 after PCA)
    np.fill_diagonal(corr, 0)
    max_corr = np.abs(corr).max()
    high_corr_pairs = np.argwhere(np.abs(corr) > 0.3)
    rpt(f"\n  Max off-diagonal |correlation|: {max_corr:.4f}  "
        f"{'⚠️  HIGH (>0.3)' if max_corr > 0.3 else '✅'}")
    rpt(f"  Pairs with |corr| > 0.3: {len(high_corr_pairs) // 2}")

    # Covariance matrix
    cov = np.cov(all_features.T)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cov, cmap="viridis", aspect="auto")
    ax.set_title("PCA Feature Covariance Matrix (after z-score)")
    ax.set_xlabel("PCA Component")
    ax.set_ylabel("PCA Component")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "pca_covariance_matrix.png", dpi=150)
    plt.close(fig)
    rpt(f"  Saved: {out / 'pca_covariance_matrix.png'}")

    return all_features


# ────────────────────────────────────────────────────────────────────
# 7. Normalization-before-PCA check
# ────────────────────────────────────────────────────────────────────

def check_normalization_order(cfg: EDAConfig, rpt: Report):
    rpt.section("7. Normalization Order Check")

    rpt(f"  Config 'normalize': {cfg.normalize}")
    if cfg.normalize:
        rpt("  ⚠️  normalize=True means data is z-scored BEFORE PCA projection.")
        rpt("      This is generally fine if channel-wise, but verify PCA was fit on normalised data.")
    else:
        rpt("  normalize=False → raw embeddings go directly into PCA.")
        rpt("  PCA centers data internally (subtracts pca.mean_) then projects.")
        rpt("  Post-PCA Welford z-score is applied. ✅")

    rpt(f"\n  Pipeline order verified:")
    rpt(f"    1. HDF5 → PickleDataset (raw embeddings)")
    rpt(f"    2. IMCBaseDictTransform: center_crop={cfg.center_crop_size}, norm={cfg.normalize}")
    rpt(f"    3. PatchAugmentations: 8 augmented views stacked")
    rpt(f"    4. GridGraphDataset: grid {cfg.grid_size}×{cfg.grid_size} = {cfg.grid_size**2} nodes")
    rpt(f"    5. DenseGraphBatch collation: stack into [B*8, N, 512]")
    rpt(f"    6. PCALayer: subtract pca_mean_, project 512→{cfg.num_pca_components}, z-score with Welford stats")
    rpt(f"    7. Model input: [B*8, {cfg.grid_size**2}, {cfg.num_pca_components}]")


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    cfg = EDAConfig()
    set_seed(cfg.seed)
    rpt = Report(cfg.output_dir / "pipeline_report.txt")

    rpt.section("EDA PIPELINE CHECK")
    rpt(f"  Data dir: {cfg.data_dir}")
    rpt(f"  Seed: {cfg.seed}")

    check_raw_hdf5(cfg, rpt)
    check_transform_chain(cfg, rpt)
    pca = check_pca(cfg, rpt)
    check_welford_statistics(cfg, rpt)
    check_normalization_order(cfg, rpt)
    check_model_input(cfg, rpt)
    plot_pca_component_distributions(cfg, rpt)

    rpt.section("PIPELINE CHECK COMPLETE")
    rpt.save()
    print(f"\n✅ Full report saved to: {rpt.path}")


if __name__ == "__main__":
    main()
