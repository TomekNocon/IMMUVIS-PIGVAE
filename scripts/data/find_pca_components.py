#!/usr/bin/env python3
"""Determine the optimal number of PCA components for the cords dataset.

Fits IncrementalPCA on the **full** training split using the same preprocessing
pipeline as ``IMCDataModule`` (center-crop → arcsinh → flatten) and reports how
many components are needed to explain a given fraction of variance.

Usage
-----
    python scripts/data/find_pca_components.py [OPTIONS]

Examples
--------
    # Use defaults (center_crop=6, normalize=False, batch_size=64)
    python scripts/data/find_pca_components.py

    # Custom data root and thresholds
    python scripts/data/find_pca_components.py \
        --data-root /my/data/IMC \
        --thresholds 0.90 0.95 0.99 0.999
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.components.graphs_datamodules import (
    DualOutputTransform,
    IMCBaseDictTransform,
    PatchAugmentations,
    PickleDataset,
)

DEFAULT_DATA_ROOT = Path("/raid_encrypted/immucan/embeddings/tnocon/data/IMC")
DATASET = "cords"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find optimal number of PCA components for the cords IMC dataset.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing <dataset>/{train,test}.h5",
    )
    parser.add_argument("--dataset", type=str, default=DATASET)
    parser.add_argument("--center-crop-size", type=int, default=6)
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Apply channel-wise z-score before arcsinh (should match your training config).",
    )
    parser.add_argument("--num-channels", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=7)
    parser.add_argument(
        "--max-components",
        type=int,
        default=None,
        help="Max PCA components to fit. Defaults to min(num_channels, num_samples_seen).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.85, 0.90, 0.95, 0.99],
        help="Cumulative explained-variance thresholds to report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save the plot. Defaults to <data-root>/<dataset>/.",
    )
    return parser.parse_args()


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
    for (thresh, n_comp), color in zip(thresholds.items(), colors):
        ax2.axhline(y=thresh, linestyle="--", alpha=0.5, color=color)
        ax2.axvline(x=n_comp, linestyle=":", alpha=0.5, color=color)
        ax2.annotate(
            f"{thresh*100:.1f}% → {n_comp}",
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


def main() -> None:
    args = parse_args()

    h5_path = args.data_root / args.dataset / "train.h5"
    if not h5_path.is_file():
        print(f"ERROR: HDF5 not found at {h5_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or (args.data_root / args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset:          {args.dataset}")
    print(f"HDF5 path:        {h5_path}")
    print(f"Center crop size: {args.center_crop_size}")
    print(f"Normalize:        {args.normalize}")
    print(f"Num channels:     {args.num_channels}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Thresholds:       {args.thresholds}")
    print()

    loader = build_loader(
        h5_path,
        center_crop_size=args.center_crop_size,
        normalize=args.normalize,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    t0 = time.perf_counter()
    ipca = fit_incremental_pca(
        loader,
        num_channels=args.num_channels,
        batch_size=args.batch_size,
        max_components=args.max_components,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nIncrementalPCA fitted in {elapsed:.1f}s")

    explained_ratio = ipca.explained_variance_ratio_
    cumvar = np.cumsum(explained_ratio)

    print(f"\nTotal components fitted: {len(explained_ratio)}")
    print(f"Total variance explained: {cumvar[-1]*100:.2f}%")

    results = report_thresholds(cumvar, args.thresholds)

    print(f"\n{'Threshold':>12s}  {'Components':>12s}  {'Cumul. Var.':>12s}")
    print("-" * 40)
    for thresh, n_comp in results.items():
        actual_var = cumvar[n_comp - 1] if n_comp <= len(cumvar) else cumvar[-1]
        print(f"{thresh*100:>11.1f}%  {n_comp:>12d}  {actual_var*100:>11.2f}%")

    plot_path = output_dir / f"pca_explained_variance_crop{args.center_crop_size}.png"
    plot_explained_variance(explained_ratio, cumvar, results, plot_path)

    print("\nTop-20 individual component variance contributions:")
    for i, ratio in enumerate(explained_ratio[:20]):
        bar = "█" * int(ratio / explained_ratio[0] * 40)
        print(f"  PC{i+1:>3d}: {ratio*100:>7.3f}%  cumul={cumvar[i]*100:>7.3f}%  {bar}")


if __name__ == "__main__":
    main()
