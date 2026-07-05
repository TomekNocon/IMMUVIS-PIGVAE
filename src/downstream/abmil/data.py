import numpy as np, torch
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
