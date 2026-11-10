"""Bags of crop embeddings per image, and image-level filtering of the metadata table."""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_MEMMAP_CACHE: dict[str, np.ndarray] = {}


class MILDataset(Dataset):
    """``(bag [S, D], label)`` pairs."""

    def __init__(self, bags: list[np.ndarray], labels: torch.Tensor):
        self.bags, self.labels = bags, labels

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        return self.bags[i], int(self.labels[i])


def mil_collate(
    batch: list[tuple[np.ndarray, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-pad bags to the longest one; ``mask`` is True at the padding positions."""
    bags, labels = zip(*batch, strict=True)
    max_size, dim = max(b.shape[0] for b in bags), bags[0].shape[1]
    out = torch.zeros(len(bags), max_size, dim)
    mask = torch.zeros(len(bags), max_size, dtype=torch.bool)
    for i, bag in enumerate(bags):
        size = bag.shape[0]
        out[i, :size] = torch.from_numpy(np.asarray(bag, "float32"))
        mask[i, size:] = True
    return out, mask, torch.tensor(labels, dtype=torch.long)


def _load_memmap(path: str) -> np.ndarray:
    if path not in _MEMMAP_CACHE:
        _MEMMAP_CACHE[path] = np.load(path, mmap_mode="r")
    return _MEMMAP_CACHE[path]


def _labelled_image_groups(meta_df: pd.DataFrame, class_to_idx: dict[str, int]):
    """Yield ``(label, [(embeddings_file, rows), ...])`` per image, in table order.

    Images without a label, or with a class outside ``class_to_idx``, are skipped. Within an
    image the crops are grouped by embeddings file (sorted), which fixes the crop order of a bag.
    """
    for _img, group in meta_df.groupby("img_path", sort=False):
        raw = group["feature_value"].iloc[0]
        if pd.isna(raw):
            continue
        label = class_to_idx.get(str(raw))
        if label is None:
            continue
        yield label, list(group.groupby("embeddings_file"))


def build_image_bags(
    meta_df: pd.DataFrame, class_to_idx: dict[str, int]
) -> tuple[list[np.ndarray], torch.Tensor]:
    """One bag per image (``img_path``) with its label; unlabelled/unknown classes are skipped."""
    bags, labels = [], []
    for label, files in _labelled_image_groups(meta_df, class_to_idx):
        rows = [
            _load_memmap(file)[rows_of_file["embedding_idx"].values] for file, rows_of_file in files
        ]
        bags.append(np.concatenate(rows, axis=0))
        labels.append(label)
    return bags, torch.tensor(labels, dtype=torch.long)


def build_image_bags_with_sources(
    meta_df: pd.DataFrame, class_to_idx: dict[str, int]
) -> tuple[list[np.ndarray], torch.Tensor, list[np.ndarray], list[tuple[list[str], np.ndarray]]]:
    """:func:`build_image_bags` plus, per bag and in bag order, each crop's tissue coordinates
    ``[S, 2]`` (coords0, coords1) and its source ``(embeddings_file per crop, embedding_idx)``,
    so per-crop scores can be mapped back onto the slide and the input patches reloaded."""
    bags, labels, coords, sources = [], [], [], []
    for label, files in _labelled_image_groups(meta_df, class_to_idx):
        rows = [
            _load_memmap(file)[rows_of_file["embedding_idx"].values] for file, rows_of_file in files
        ]
        bags.append(np.concatenate(rows, axis=0))
        labels.append(label)
        coords.append(np.concatenate([r[["coords0", "coords1"]].values for _, r in files], axis=0))
        sources.append((
            [file for file, r in files for _ in range(len(r))],
            np.concatenate([r["embedding_idx"].values for _, r in files]),
        ))
    return bags, torch.tensor(labels, dtype=torch.long), coords, sources


def drop_nan_labels(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Drop every crop of an image whose label is missing."""
    valid_imgs = meta_df.dropna(subset=["feature_value"])["img_path"].unique()
    return meta_df[meta_df["img_path"].isin(valid_imgs)]


def balance_meta_df(meta_df: pd.DataFrame, min_class_freq: float = 0.05) -> pd.DataFrame:
    """Image-level class-frequency filter: keep classes with freq >= min_class_freq."""
    img_labels = meta_df.groupby("img_path", sort=False)["feature_value"].first()
    img_labels = img_labels[~pd.isna(img_labels)].astype(str)
    if len(img_labels) == 0:
        return meta_df.iloc[0:0]
    unique_classes, counts = np.unique(img_labels.values, return_counts=True)
    freq = counts / counts.sum()
    keep_classes = set(unique_classes[freq >= float(min_class_freq)])
    keep_imgs = img_labels[img_labels.isin(keep_classes)].index
    return meta_df[meta_df["img_path"].isin(keep_imgs)]


def load_pooled_meta(meta_dir: str, dataset: str) -> pd.DataFrame:
    """Pool the train+test metadata CSVs for one dataset (CV runs over both)."""
    frames = []
    for split in ("train", "test"):
        path = os.path.join(meta_dir, f"{dataset}_{split}_metadata.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"no {dataset}_{{train,test}}_metadata.csv under {meta_dir}")
    pooled = pd.concat(frames, ignore_index=True)
    # The encode stage writes [img_path, coords*, embeddings_file, embedding_idx]; the clinical
    # label (`feature_value`) is joined in by build_meta, so fail early if it is missing.
    if "feature_value" not in pooled.columns:
        raise KeyError(
            f"{dataset} metadata under {meta_dir} has no 'feature_value' column -- join "
            f"clinical labels onto the encode output before running run_abmil (columns: "
            f"{list(pooled.columns)})."
        )
    return pooled
