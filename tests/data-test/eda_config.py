"""
Shared configuration and data loading utilities for EDA scripts.

This module provides a single, reproducible way to instantiate the exact same
data pipeline used during training so every EDA script analyses identical data.

Data Pipeline (documented):
    Raw HDF5 (nsclc2_panel1_train.h5)
      → PickleDataset (loads per-sample dicts with 8 augmented views, each [1, C, H, W])
      → IMCBaseDictTransform (center crop 7×7, optional norm, reshape to [N, C])
      → PatchAugmentations (stack 8 views → [8, N, C], argsort, perm)
      → DualOutputTransform (combines base + aug)
      → random_split (train / leftover)
      → GridGraphDataset (creates networkx grid graph per sample)
      → DenseGraphDataLoader + PCADenseGraphCollator
          → DenseGraphBatch.from_sparse_graph_list (collates into dense batch)
          → PCALayer.forward (project 512→128 dims, then Welford z-score)
      → Model receives DenseGraphBatch with node_features [B*8, 49, 128]
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import rootutils

# ---------------------------------------------------------------------------
# Project root setup (same as training scripts)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
rootutils.setup_root(str(PROJECT_ROOT), indicator=".project-root", pythonpath=True)

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.data.components.graphs_datamodules import (
    DenseGraphDataLoader,
    DualOutputTransform,
    GridGraphDataset,
    IMCBaseDictTransform,
    PatchAugmentations,
    PCADenseGraphCollator,
    PCALayer,
    PickleDataset,
)


# ---------------------------------------------------------------------------
# Configuration dataclass – mirrors configs/data/mnist.yaml values
# ---------------------------------------------------------------------------
@dataclass
class EDAConfig:
    """All paths and hyper-parameters needed to reproduce the training data pipeline."""

    data_dir: str = "/raid_encrypted/immucan/embeddings/tnocon/data"
    batch_size: int = 8
    train_val_test_split: list[int] = field(default_factory=lambda: [40_843, 10_197, 0, 0])
    num_workers: int = 0  # keep 0 for deterministic single-process loading
    pin_memory: bool = False
    grid_size: int = 6
    size: int = 6
    patch_size: int = 1
    augmentation_prob: float = 1.0
    is_contrastive: bool = True
    num_aug_per_sample: int = 8
    center_crop_size: int = 6
    num_channels: int = 768
    num_pca_components: int = 128
    num_node_features: int = 128
    normalize: bool = False  # PCA handles normalisation

    # Derived paths
    @property
    def train_h5(self) -> Path:
        return Path(self.data_dir) / "IMC" / "cords" / "train.h5"

    @property
    def test_h5(self) -> Path:
        return Path(self.data_dir) /"IMC"/ "cords" / "test.h5"

    @property
    def pca_model_path(self) -> Path:
        return (
            Path(self.data_dir)
            / "IMC" / "cords"
            / f"pca_model_{self.num_pca_components}_center_crop_{self.center_crop_size}.pkl"
        )

    @property
    def statistics_path(self) -> Path:
        return Path(self.data_dir) / "IMC" / "cords" / f"imc_statistics_{self.num_pca_components}_center_crop_{self.center_crop_size}.pt"

    # Output directory for EDA artefacts
    @property
    def output_dir(self) -> Path:
        d = PROJECT_ROOT / "tests" / "data-test" / "eda_output"
        d.mkdir(parents=True, exist_ok=True)
        return d

    seed: int = 42


# ---------------------------------------------------------------------------
# Helper: build transforms identical to IMCDataModule.__init__
# ---------------------------------------------------------------------------

def build_transforms(cfg: EDAConfig):
    """Return (dual_train, dual_val) transform pipelines."""
    base = IMCBaseDictTransform(
        center_crop_size=cfg.center_crop_size, normalize=cfg.normalize
    )
    aug_train = PatchAugmentations(
        prob=cfg.augmentation_prob,
        size=cfg.size,
        patch_size=cfg.patch_size,
    )
    aug_val = PatchAugmentations(
        prob=cfg.augmentation_prob,
        size=cfg.size,
        patch_size=cfg.patch_size,
        is_validation=True,
    )
    dual_train = DualOutputTransform(base, aug_train)
    dual_val = DualOutputTransform(base, aug_val)
    return dual_train, dual_val


# ---------------------------------------------------------------------------
# Helper: build datasets + splits identical to IMCDataModule.setup()
# ---------------------------------------------------------------------------

def build_train_val_test(cfg: EDAConfig):
    """Return (data_train, data_val, data_test) Subset objects."""
    dual_train, dual_val = build_transforms(cfg)

    trainset = PickleDataset(cfg.train_h5, transform=dual_train, generate_views=True, center_crop_size=cfg.center_crop_size)
    testset = PickleDataset(cfg.test_h5, transform=dual_val, generate_views=True, center_crop_size=cfg.center_crop_size)

    train_ratio, val_ratio, test_ratio, _ = cfg.train_val_test_split
    size_trainset = len(trainset)
    size_testset = len(testset)

    data_train, _ = random_split(
        dataset=trainset,
        lengths=[train_ratio, size_trainset - train_ratio],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    data_val, data_test, _ = random_split(
        dataset=testset,
        lengths=[val_ratio, test_ratio, size_testset - val_ratio - test_ratio],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    return data_train, data_val, data_test


# ---------------------------------------------------------------------------
# Helper: build the PCA layer (same as IMCDataModule.setup)
# ---------------------------------------------------------------------------

def build_pca_layer(cfg: EDAConfig) -> PCALayer:
    """Load the fitted PCA layer used during training."""
    return PCALayer(cfg.pca_model_path, cfg.statistics_path)


# ---------------------------------------------------------------------------
# Helper: build a full training dataloader identical to training
# ---------------------------------------------------------------------------

def build_train_dataloader(cfg: EDAConfig, shuffle: bool = False):
    """Return a DataLoader that produces the exact same batches as training."""
    data_train, _, _ = build_train_val_test(cfg)
    pca_layer = build_pca_layer(cfg)

    train_dataset = GridGraphDataset(
        grid_size=cfg.grid_size,
        dataset=data_train,
        channels=list(range(4)),
    )
    collate_fn = PCADenseGraphCollator(pca_layer)
    loader = DenseGraphDataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return loader


# ---------------------------------------------------------------------------
# Helper: build a raw (pre-PCA) training dataloader for inspecting raw features
# ---------------------------------------------------------------------------

def build_raw_train_dataloader(cfg: EDAConfig, shuffle: bool = False):
    """Return a DataLoader without PCA (raw 768-dim features)."""
    data_train, _, _ = build_train_val_test(cfg)

    train_dataset = GridGraphDataset(
        grid_size=cfg.grid_size,
        dataset=data_train,
        channels=list(range(4)),
    )
    # Use default collate (no PCA)
    loader = DenseGraphDataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        shuffle=shuffle,
    )
    return loader


# ---------------------------------------------------------------------------
# Helper: build embeddings-only dataloader for PCA diagnostics
# ---------------------------------------------------------------------------

def build_embeddings_only_loader(cfg: EDAConfig):
    """Dataloader that returns only augmented tensors (no graph), for PCA fitting checks.

    Each batch element is ``batch[0]`` after default collate (the stacked augmented tensor).
    """
    dual_train, _ = build_transforms(cfg)
    dataset = PickleDataset(
        cfg.train_h5,
        transform=dual_train,
        generate_views=True,
        center_crop_size=cfg.center_crop_size,
        only_embeddings=True,
    )

    train_ratio = cfg.train_val_test_split[0]
    size = len(dataset)
    data_train, _ = random_split(
        dataset=dataset,
        lengths=[train_ratio, size - train_ratio],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    return DataLoader(
        data_train,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
