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
