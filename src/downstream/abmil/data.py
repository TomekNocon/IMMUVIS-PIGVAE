import os

import numpy as np, torch
import pandas as pd
from torch.utils.data import Dataset

class MILDataset(Dataset):
    def __init__(self, bags, labels): self.bags, self.labels = bags, labels
    def __len__(self): return len(self.bags)
    def __getitem__(self, i): return self.bags[i], int(self.labels[i])

def mil_collate(batch):
    bags, labels = zip(*batch)
    max_s = max(b.shape[0] for b in bags); d = bags[0].shape[1]
    out = torch.zeros(len(bags), max_s, d); mask = torch.zeros(len(bags), max_s, dtype=torch.bool)
    for i, b in enumerate(bags):
        s = b.shape[0]; out[i, :s] = torch.from_numpy(np.asarray(b, "float32")); mask[i, s:] = True
    return out, mask, torch.tensor(labels, dtype=torch.long)


_MEMMAP_CACHE = {}
def _load_memmap(path):
    a = _MEMMAP_CACHE.get(path)
    if a is None:
        a = np.load(path, mmap_mode="r"); _MEMMAP_CACHE[path] = a
    return a

def build_image_bags(meta_df, class_to_idx):
    bags, labels = [], []
    for _img, g in meta_df.groupby("img_path", sort=False):
        raw = g["feature_value"].iloc[0]
        if pd.isna(raw): continue
        lab = class_to_idx.get(str(raw))
        if lab is None: continue
        rows = []
        for ef, gg in g.groupby("embeddings_file"):
            rows.append(_load_memmap(ef)[gg["embedding_idx"].values])
        bags.append(np.concatenate(rows, axis=0)); labels.append(lab)
    return bags, torch.tensor(labels, dtype=torch.long)


# Image-level filtering on the pooled (train+test) meta table, applied per dataset x feature
# before building bags (mirrors gated_abmil's run_abmil.drop_nan_labels / balance_meta_df).


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
    # The encode stage writes only [img_path, coords*, embeddings_file, embedding_idx];
    # `feature_value` (the clinical label) must be joined in as a separate manual step.
    # Fail loud here rather than KeyError-ing deep in drop_nan_labels.
    if "feature_value" not in pooled.columns:
        raise KeyError(
            f"{dataset} metadata under {meta_dir} has no 'feature_value' column -- join "
            f"clinical labels onto the encode output before running run_abmil (columns: "
            f"{list(pooled.columns)})."
        )
    return pooled
