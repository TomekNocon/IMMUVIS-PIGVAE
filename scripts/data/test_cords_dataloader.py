#!/usr/bin/env python3
"""Test the full dataloader pipeline for the cords HDF5 dataset."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.components.graphs_datamodules import (
    DenseGraphDataLoader,
    DualOutputTransform,
    GridGraphDataset,
    IMCBaseDictTransform,
    PatchAugmentations,
    PickleDataset,
    make_views,
)

DATA_ROOT = Path("/raid_encrypted/immucan/embeddings/tnocon/data/IMC")
DATASET = "cords"


def inspect_h5(split: str) -> None:
    path = DATA_ROOT / DATASET / f"{split}.h5"
    print(f"\n{'='*60}")
    print(f"  {split.upper()} — {path}")
    print(f"{'='*60}")
    with h5py.File(path, "r") as f:
        for key in f.keys():
            ds = f[key]
            print(f"  {key:15s}  shape={str(ds.shape):20s}  dtype={ds.dtype}")
        n = len(f["embeddings"])
        print(f"  Total samples: {n}")

        sample = f["embeddings"][0]
        print(f"  First embedding: shape={sample.shape}  "
              f"min={sample.min():.4f}  max={sample.max():.4f}  "
              f"mean={sample.mean():.4f}")

        path_sample = f["paths"][0]
        pos_sample = f["positions"][0]
        print(f"  First path:     {path_sample}")
        print(f"  First position: {pos_sample}")


def test_pickle_dataset(split: str) -> None:
    path = DATA_ROOT / DATASET / f"{split}.h5"
    print(f"\n--- PickleDataset (generate_views=True) [{split}] ---")
    ds = PickleDataset(path, generate_views=True)
    print(f"  Length: {len(ds)}")

    item = ds[0]
    print(f"  Keys: {list(item.keys())}")
    for k, v in item.items():
        if isinstance(v, np.ndarray):
            print(f"    {k:15s}  shape={str(v.shape):20s}  dtype={v.dtype}")
        else:
            print(f"    {k:15s}  {type(v).__name__}: {v}")

    emb = item["embeddings"]
    assert emb.ndim == 4 and emb.shape[0] == 8, (
        f"Expected (8,C,H,W), got {emb.shape}"
    )
    print(f"  [OK] embeddings shape: {emb.shape}")


def test_transform_chain(split: str) -> None:
    path = DATA_ROOT / DATASET / f"{split}.h5"
    print(f"\n--- DualOutputTransform chain [{split}] ---")

    base_tf = IMCBaseDictTransform(center_crop_size=7, normalize=False)
    aug_tf = PatchAugmentations(prob=1.0, size=6, patch_size=1, is_validation=(split == "test"))
    dual_tf = DualOutputTransform(base_tf, aug_tf)

    ds = PickleDataset(path, transform=dual_tf, generate_views=True)
    print(f"  Length: {len(ds)}")

    out = ds[0]
    print(f"  Output is a {type(out).__name__} of length {len(out)}")
    augmented, argsort, perm, metadata, paths, positions = out
    print(f"    augmented:  {augmented.shape}  dtype={augmented.dtype}")
    print(f"    argsort:    {argsort.shape}")
    print(f"    perm:       {perm.shape}")
    print(f"    metadata:   {type(metadata).__name__}  shape={metadata.shape if hasattr(metadata, 'shape') else 'N/A'}")
    print(f"    paths:      {type(paths).__name__}")
    print(f"    positions:  {type(positions).__name__}  shape={positions.shape if hasattr(positions, 'shape') else 'N/A'}")
    print(f"  [OK] Transform chain works")


def test_full_dataloader(split: str, num_batches: int = 3) -> None:
    path = DATA_ROOT / DATASET / f"{split}.h5"
    print(f"\n--- Full DataLoader [{split}], {num_batches} batches ---")

    base_tf = IMCBaseDictTransform(center_crop_size=6, normalize=False)
    is_val = split == "test"
    aug_tf = PatchAugmentations(prob=1.0, size=6, patch_size=1, is_validation=is_val)
    dual_tf = DualOutputTransform(base_tf, aug_tf)

    ds = PickleDataset(path, transform=dual_tf, generate_views=True)
    grid_ds = GridGraphDataset(grid_size=6, dataset=ds, channels=list(range(4)))

    loader = DenseGraphDataLoader(
        dataset=grid_ds,
        batch_size=16,
        num_workers=0,
        shuffle=False,
    )

    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        print(f"  batch {i}: type={type(batch).__name__}  ", end="")
        if hasattr(batch, "x"):
            print(f"x={batch.x.shape}  ", end="")
        if hasattr(batch, "edge_index"):
            print(f"edge_index={batch.edge_index.shape}  ", end="")
        if hasattr(batch, "batch"):
            print(f"batch_vec max={batch.batch.max().item()}  ", end="")
        print()
    dt = time.perf_counter() - t0

    node_features = batch.node_features
    x1 = node_features[0]
    x2 = node_features[16]
    print(x1.shape)
    print(torch.allclose(x1.flatten().sort()[0], x2.flatten().sort()[0], atol=1e-6))
    print((x1.flatten().sort()[0] - x2.flatten().sort()[0]).abs().max())
    
    print(f"  {num_batches} batches loaded in {dt:.2f}s ({dt / num_batches:.3f}s/batch)")
    print(f"  [OK] Full DataLoader works")


def count_samples() -> None:
    print(f"\n{'='*60}")
    print("  SAMPLE COUNTS")
    print(f"{'='*60}")
    total = 0
    for dataset_dir in sorted(DATA_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        ds_name = dataset_dir.name
        for split in ("train", "test"):
            h5_path = dataset_dir / f"{split}.h5"
            if not h5_path.is_file():
                continue
            with h5py.File(h5_path, "r") as f:
                n = len(f["embeddings"])
            total += n
            print(f"  {ds_name:20s}  {split:5s}  {n:>8,} samples")
    print(f"  {'─'*45}")
    print(f"  {'TOTAL':20s}         {total:>8,} samples")


if __name__ == "__main__":
    count_samples()

    for split in ("train", "test"):
        # inspect_h5(split)
        # test_pickle_dataset(split)
        # test_transform_chain(split)
        test_full_dataloader(split, num_batches=1)

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
