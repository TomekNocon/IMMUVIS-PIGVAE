# %% [markdown]
# 📊 Comprehensive EDA Report – IMMUVIS-PIGVAE Data Pipeline

# **Goal**: Prove or disprove that training instability comes from the dataset rather than the model architecture.

# This notebook orchestrates all EDA scripts and produces a unified report with:
# - Statistical summaries
# - Plots
# - Warnings about suspicious data patterns
# - Potential causes of training instability

# ---

# ## Data Pipeline Summary

# ```
# Raw HDF5 (nsclc2_panel1_train.h5)
#   → PickleDataset (per-sample dict with 8 augmented views, each [1, C=512, H, W])
#   → IMCBaseDictTransform (center crop 7×7, optional norm, reshape [N, C])
#   → PatchAugmentations (stack 8 views → [8, N=49, C=512])
#   → DualOutputTransform
#   → random_split (train / leftover, seed=42)
#   → GridGraphDataset (networkx 7×7 grid graph per sample)
#   → DenseGraphDataLoader + PCADenseGraphCollator
#       → DenseGraphBatch.from_sparse_graph_list (collation)
#       → PCALayer.forward (project 512→128, Welford z-score)
#   → Model receives DenseGraphBatch with node_features [B*8, 49, 128]
# ```

# %% 

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Ensure we can import from tests/data-test/
SCRIPT_DIR = os.path.dirname(os.path.abspath('__file__'))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from eda_config import EDAConfig, set_seed, build_train_dataloader, build_pca_layer, build_train_val_test

cfg = EDAConfig()
set_seed(cfg.seed)
print(f"Data dir: {cfg.data_dir}")
print(f"Output dir: {cfg.output_dir}")
print(f"Train H5: {cfg.train_h5}")
print(f"PCA model: {cfg.pca_model_path}")
print(f"Statistics: {cfg.statistics_path}")

# %%[markdown]
# ---
## Z-Score Amplification Check

# Compare raw PCA output (no z-score) vs z-scored output for the heavy-tailed channels.
# This shows whether the extreme ±35 values are real or artifacts of dividing by small std.
# %%
import torch
import numpy as np
import joblib
from tqdm import tqdm
from eda_config import EDAConfig, build_train_dataloader, build_pca_layer, build_raw_train_dataloader

cfg = EDAConfig()

# --- Load Welford statistics and PCA model ---
statistics = torch.load(cfg.statistics_path)
welford_std = statistics["std"]  # shape: (128,)
welford_mean = statistics["mean"]

pca = joblib.load(cfg.pca_model_path)
pca_mean = torch.tensor(pca.mean_, dtype=torch.float32)
components = torch.tensor(pca.components_, dtype=torch.float32)

print("Welford std per PCA component (first 10 / last 10):")
print(f"  PC  0-9:  {welford_std[:10].numpy().round(4)}")
print(f"  PC118-127: {welford_std[118:].numpy().round(4)}")
print(f"  Std ratio (PC0 / PC127): {welford_std[0].item():.2f} / {welford_std[127].item():.4f} = {welford_std[0].item() / welford_std[127].item():.1f}x")

# --- Collect raw PCA features (before z-score) and z-scored features ---
raw_loader = build_raw_train_dataloader(cfg, shuffle=False)
MAX_BATCHES = 100

raw_pca_all = []
zscored_all = []

for i, batch in enumerate(tqdm(raw_loader, total=MAX_BATCHES, desc="Collecting")):
    if i >= MAX_BATCHES:
        break
    x = batch.node_features  # (B*8, 36, 768)
    
    # Raw PCA projection (no z-score)
    x_centered = x - pca_mean
    x_pca_raw = torch.matmul(x_centered, components.t())  # (B*8, 36, 128)
    
    # Z-scored
    x_zscored = (x_pca_raw - welford_mean) / (welford_std + 1e-8)
    
    raw_pca_all.append(x_pca_raw.reshape(-1, 128))
    zscored_all.append(x_zscored.reshape(-1, 128))

raw_pca_all = torch.cat(raw_pca_all, dim=0).numpy()
zscored_all = torch.cat(zscored_all, dim=0).numpy()

print(f"\nTotal vectors: {raw_pca_all.shape[0]}")

# --- Compare heavy-tail channels ---
heavy_tail_channels = [48, 66, 85, 103, 104, 105, 111, 120, 121]
normal_channels = [0, 1, 2, 3, 4]

print("\n" + "=" * 90)
print(f"  {'Channel':>8}  {'Welford std':>12}  {'Raw PCA min':>12}  {'Raw PCA max':>12}  {'Z-score min':>12}  {'Z-score max':>12}  {'Amplification':>14}")
print("=" * 90)

for ch in normal_channels + heavy_tail_channels:
    raw_min = raw_pca_all[:, ch].min()
    raw_max = raw_pca_all[:, ch].max()
    z_min = zscored_all[:, ch].min()
    z_max = zscored_all[:, ch].max()
    std = welford_std[ch].item()
    label = "HEAVY-TAIL" if ch in heavy_tail_channels else ""
    print(f"  {ch:>8}  {std:>12.4f}  {raw_min:>12.4f}  {raw_max:>12.4f}  {z_min:>12.4f}  {z_max:>12.4f}  {1/std:>13.1f}x  {label}")

print("=" * 90)
print("\n↑ 'Amplification' = 1/std = how much z-scoring multiplies raw PCA values.")
print("  High amplification on low-variance PCs turns small raw outliers into extreme z-scores.")
# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Heavy-Tail Channels: Raw PCA (blue) vs Z-Scored (orange)", fontsize=14, y=1.01)

for ax, ch in zip(axes.flat, heavy_tail_channels):
    raw_vals = raw_pca_all[:, ch]
    z_vals = zscored_all[:, ch]
    
    ax_raw = ax
    ax_raw.hist(raw_vals, bins=200, alpha=0.7, color="steelblue", density=True, label=f"Raw PCA (std={welford_std[ch]:.3f})")
    ax_raw.set_title(f"Channel {ch}", fontsize=11)
    ax_raw.set_ylabel("Density (raw)", color="steelblue")
    ax_raw.legend(loc="upper left", fontsize=8)
    
    ax_z = ax_raw.twinx()
    ax_z.hist(z_vals, bins=200, alpha=0.5, color="darkorange", density=True, label=f"Z-scored (÷{welford_std[ch]:.3f})")
    ax_z.set_ylabel("Density (z-scored)", color="darkorange")
    ax_z.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig(str(cfg.output_dir / "zscore_amplification.png"), dpi=150, bbox_inches="tight")
plt.show()

print(f"\nSaved: {cfg.output_dir / 'zscore_amplification.png'}")