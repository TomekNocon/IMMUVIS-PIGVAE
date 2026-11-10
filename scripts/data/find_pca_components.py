"""Explained variance of an IncrementalPCA on the training crops.

Fits IncrementalPCA on the full training split with the same preprocessing as ``IMCDataModule``
(center crop -> arcsinh -> flatten), reports how many components reach each cumulative
explained-variance threshold, and saves the variance plot to the run directory. Crop size,
channels and data root come from the data / experiment configs.

Usage:
    uv run python scripts/data/find_pca_components.py
    uv run python scripts/data/find_pca_components.py experiment=pigvae thresholds=[0.9,0.99]
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.graphs_datamodules import (
    DualOutputTransform,
    IMCBaseDictTransform,
    PatchAugmentations,
    PickleDataset,
)


def build_loader(
    h5_path: Path,
    center_crop_size: int,
    normalize: bool,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Build a DataLoader with the same transforms used by ``IMCDataModule.prepare_data``."""
    base_tf = IMCBaseDictTransform(
        center_crop_size=center_crop_size,
        normalize=normalize,
    )
    aug_tf = PatchAugmentations(
        prob=1.0,
        size=center_crop_size,
        patch_size=1,
        is_validation=True,
    )
    dual_tf = DualOutputTransform(base_tf, aug_tf)

    dataset = PickleDataset(
        h5_path,
        transform=dual_tf,
        generate_views=True,
        center_crop_size=center_crop_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )


def fit_incremental_pca(
    loader: DataLoader,
    num_channels: int,
    batch_size: int,
    max_components: int | None,
) -> IncrementalPCA:
    """Fit IncrementalPCA over the full dataset, returning the fitted model."""
    n_components = max_components or num_channels
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    for batch in tqdm(loader, desc="Fitting IncrementalPCA"):
        node_features = batch[0]
        x = node_features[:batch_size].reshape(-1, num_channels).numpy()
        if x.shape[0] < n_components:
            continue
        ipca.partial_fit(x)

    return ipca


def report_thresholds(
    cumvar: np.ndarray,
    thresholds: list[float],
) -> dict[float, int]:
    """For each threshold, find the number of components needed."""
    results: dict[float, int] = {}
    for t in sorted(thresholds):
        idx = int(np.searchsorted(cumvar, t))
        n_comp = min(idx + 1, len(cumvar))
        results[t] = n_comp
    return results


def plot_explained_variance(
    explained_ratio: np.ndarray,
    cumvar: np.ndarray,
    thresholds: dict[float, int],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax1.bar(range(1, len(explained_ratio) + 1), explained_ratio, alpha=0.7, color="steelblue")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("Individual Explained Variance")
    ax1.set_xlim(0, len(explained_ratio) + 1)

    ax2 = axes[1]
    ax2.plot(range(1, len(cumvar) + 1), cumvar, color="steelblue", linewidth=2)
    colors = plt.cm.Set1(np.linspace(0, 1, len(thresholds)))
    for (thresh, n_comp), color in zip(thresholds.items(), colors, strict=True):
        ax2.axhline(y=thresh, linestyle="--", alpha=0.5, color=color)
        ax2.axvline(x=n_comp, linestyle=":", alpha=0.5, color=color)
        ax2.annotate(
            f"{thresh * 100:.1f}% → {n_comp}",
            xy=(n_comp, thresh),
            xytext=(n_comp + len(cumvar) * 0.02, thresh - 0.02),
            fontsize=9,
            color=color,
            fontweight="bold",
        )
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("Cumulative Explained Variance")
    ax2.set_xlim(0, len(cumvar) + 1)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="data_prep/find_pca")
def main(cfg: DictConfig) -> None:
    h5_path = Path(cfg.h5_path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 not found at {h5_path}")
    output_dir = Path(cfg.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset:          {cfg.dataset}")
    print(f"HDF5 path:        {h5_path}")
    print(f"Center crop size: {cfg.center_crop_size}")
    print(f"Normalize:        {cfg.normalize}")
    print(f"Num channels:     {cfg.num_channels}")
    print(f"Batch size:       {cfg.batch_size}")
    print(f"Thresholds:       {list(cfg.thresholds)}")
    print()

    loader = build_loader(
        h5_path,
        center_crop_size=cfg.center_crop_size,
        normalize=cfg.normalize,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    t0 = time.perf_counter()
    ipca = fit_incremental_pca(
        loader,
        num_channels=cfg.num_channels,
        batch_size=cfg.batch_size,
        max_components=cfg.max_components,
    )
    print(f"\nIncrementalPCA fitted in {time.perf_counter() - t0:.1f}s")

    explained_ratio = ipca.explained_variance_ratio_
    cumvar = np.cumsum(explained_ratio)

    print(f"\nTotal components fitted: {len(explained_ratio)}")
    print(f"Total variance explained: {cumvar[-1] * 100:.2f}%")

    results = report_thresholds(cumvar, list(OmegaConf.to_container(cfg.thresholds)))

    print(f"\n{'Threshold':>12s}  {'Components':>12s}  {'Cumul. Var.':>12s}")
    print("-" * 40)
    for thresh, n_comp in results.items():
        actual_var = cumvar[n_comp - 1] if n_comp <= len(cumvar) else cumvar[-1]
        print(f"{thresh * 100:>11.1f}%  {n_comp:>12d}  {actual_var * 100:>11.2f}%")

    plot_path = output_dir / f"pca_explained_variance_crop{cfg.center_crop_size}.png"
    plot_explained_variance(explained_ratio, cumvar, results, plot_path)

    print("\nTop-20 individual component variance contributions:")
    for i, ratio in enumerate(explained_ratio[:20]):
        bar = "█" * int(ratio / explained_ratio[0] * 40)
        print(f"  PC{i + 1:>3d}: {ratio * 100:>7.3f}%  cumul={cumvar[i] * 100:>7.3f}%  {bar}")


if __name__ == "__main__":
    main()
