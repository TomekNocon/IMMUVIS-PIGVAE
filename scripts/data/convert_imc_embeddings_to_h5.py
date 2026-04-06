#!/usr/bin/env python3
"""Convert .npy embeddings + metadata .csv into per-dataset HDF5 (single-view).

The 8 spatial views (rotations + flips) are generated on the fly at training time,
so this script stores only the original (N, C, H, W) embeddings, keeping output
size close to the raw input.

Input layout::

    {input_dir}/{split}/
        {prefix}_{split}_image_patches_embeddings_batch_0.npy   # (N, C, H, W)
        {prefix}_{split}_image_patches_metadata_batch_0.csv     # img_path, panel, coords0, coords1
        ...

Output (one HDF5 per dataset x split)::

    {out_root}/{panel}/train.h5   (and test.h5)

Each HDF5 contains:

* ``embeddings``  (N, C, H, W)  - original single view
* ``metadata``    (N, 8, 3)     - zeros placeholder
* ``positions``   (N, 4)        - parsed from coords0/coords1
* ``paths``       (N,)          - variable-length UTF-8 strings

Example::

    python scripts/data/convert_imc_embeddings_to_h5.py \\
        --input-dir /raid_encrypted/immucan/embeddings/tnocon/ImmuVis-768-UN-475 \\
        --out-root  /raid/tnocon/data/IMC
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

WRITE_CHUNK = 512


def _batch_id(name: str) -> int:
    m = re.search(r"batch_(\d+)", name)
    if m is None:
        raise ValueError(f"Cannot parse batch id from {name!r}")
    return int(m.group(1))


def pair_npy_csv(split_dir: Path) -> list[tuple[Path, Path]]:
    npys = {_batch_id(p.name): p for p in split_dir.glob("*embeddings_batch_*.npy")}
    csvs = {_batch_id(p.name): p for p in split_dir.glob("*metadata_batch_*.csv")}
    batch_ids = sorted(set(npys) & set(csvs))
    if not batch_ids:
        raise FileNotFoundError(f"No matching npy/csv batch pairs in {split_dir}")
    return [(npys[b], csvs[b]) for b in batch_ids]


def parse_coords(s: str) -> tuple[float, float]:
    val = ast.literal_eval(s)
    return float(val[0]), float(val[1])


def _open_h5(path: Path, c: int, h: int, w: int, chunk_rows: int) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    cr = max(1, min(chunk_rows, 1024))
    f.create_dataset(
        "embeddings",
        shape=(0, c, h, w),
        maxshape=(None, c, h, w),
        chunks=(cr, c, h, w),
        dtype="float32",
    )
    f.create_dataset(
        "metadata",
        shape=(0, 8, 3),
        maxshape=(None, 8, 3),
        chunks=(cr, 8, 3),
        dtype="float32",
    )
    f.create_dataset(
        "positions",
        shape=(0, 4),
        maxshape=(None, 4),
        chunks=(cr, 4),
        dtype="float32",
    )
    f.create_dataset(
        "paths",
        shape=(0,),
        maxshape=(None,),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    return f


def _append(
    f: h5py.File, emb: np.ndarray, pos: np.ndarray, paths: list[str]
) -> None:
    n = emb.shape[0]
    for name in ("embeddings", "metadata", "positions", "paths"):
        d = f[name]
        d.resize(d.shape[0] + n, axis=0)
    f["embeddings"][-n:] = emb
    f["metadata"][-n:] = np.zeros((n, 8, 3), dtype=np.float32)
    f["positions"][-n:] = pos
    f["paths"][-n:] = np.array(paths, dtype=object)


def _parse_positions(df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    n = len(indices)
    pos = np.zeros((n, 4), dtype=np.float32)
    if "coords0" in df.columns and "coords1" in df.columns:
        for j, idx in enumerate(indices):
            c0a, c0b = parse_coords(df.at[idx, "coords0"])
            c1a, c1b = parse_coords(df.at[idx, "coords1"])
            pos[j] = [c0a, c0b, c1a, c1b]
    return pos


def process_split(
    split_dir: Path,
    out_root: Path,
    split: str,
    chunk_rows: int,
    write_chunk: int,
) -> dict[str, int]:
    pairs = pair_npy_csv(split_dir)
    handles: dict[str, h5py.File] = {}
    counts: dict[str, int] = defaultdict(int)

    try:
        for batch_idx, (npy_path, csv_path) in enumerate(pairs):
            emb = np.load(npy_path, mmap_mode="r")
            if emb.ndim != 4:
                raise ValueError(f"Expected (N,C,H,W) in {npy_path}, got shape {emb.shape}")
            df = pd.read_csv(csv_path)
            if len(df) != emb.shape[0]:
                raise ValueError(
                    f"Row mismatch: {npy_path} has {emb.shape[0]} rows, "
                    f"{csv_path} has {len(df)} rows"
                )

            _, c, h, w = emb.shape

            for panel_name, group in df.groupby("panel"):
                panel_name = str(panel_name)
                indices = group.index.to_numpy()
                total_panel = len(indices)

                if panel_name not in handles:
                    out_path = out_root / panel_name / f"{split}.h5"
                    handles[panel_name] = _open_h5(out_path, c, h, w, chunk_rows)

                for start in range(0, total_panel, write_chunk):
                    end = min(start + write_chunk, total_panel)
                    chunk_idx = indices[start:end]

                    sub_emb = np.asarray(emb[chunk_idx], dtype=np.float32)
                    pos = _parse_positions(df, chunk_idx)
                    img_paths = df.loc[chunk_idx, "img_path"].astype(str).tolist()

                    _append(handles[panel_name], sub_emb, pos, img_paths)
                    del sub_emb, pos
                    counts[panel_name] += end - start

                    print(
                        f"    batch {batch_idx + 1}/{len(pairs)} | "
                        f"{panel_name} chunk {start // write_chunk + 1}/"
                        f"{(total_panel + write_chunk - 1) // write_chunk} "
                        f"({end}/{total_panel} rows)",
                        flush=True,
                    )

            print(
                f"  [{batch_idx + 1}/{len(pairs)}] {npy_path.name} done "
                f"({emb.shape[0]} rows)",
                flush=True,
            )
    finally:
        for f in handles.values():
            f.close()

    return dict(counts)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root with train/ and test/ subdirs containing npy+csv batches.",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root. Creates {out_root}/{panel}/train.h5 and test.h5.",
    )
    p.add_argument("--chunk-rows", type=int, default=64, help="HDF5 chunk row size.")
    p.add_argument(
        "--write-chunk",
        type=int,
        default=WRITE_CHUNK,
        help="Rows processed in memory at once.",
    )
    args = p.parse_args()

    for split in ("train", "test"):
        split_dir = args.input_dir / split
        if not split_dir.is_dir():
            print(f"Skipping {split}: {split_dir} not found")
            continue
        print(f"Processing {split} from {split_dir} ...", flush=True)
        counts = process_split(
            split_dir, args.out_root, split, args.chunk_rows, args.write_chunk
        )
        for name, n in sorted(counts.items()):
            out_h5 = args.out_root / name / f"{split}.h5"
            print(f"  -> {out_h5}  ({n} samples)")

    print("Done.")


if __name__ == "__main__":
    main()
