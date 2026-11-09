"""Run the trained PIGVAE on the pipeline-figure crop and save its reconstruction (szary).

Input = the crop's PCA-128 node features from the pipeline cache (same projection the dataloader
applies: zscore=False, no clip), in training node order (row-major). Deterministic (sample=False).
"""
import argparse, sys
import numpy as np, torch
sys.path.insert(0, ".")
from src.downstream.encode import load_frozen_pigvae
from src.data.components.graphs_datamodules import DenseGraphBatch

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="thesis_figures/_cache/pipeline_7775_384_256.npz")
ap.add_argument("--ckpt", default="/raid_encrypted/immucan/embeddings/tnocon/logs/train/runs/2026-07-31_10-29-22/checkpoints/last.ckpt")
ap.add_argument("--experiment", default="pigvae")
a = ap.parse_args()

d = dict(np.load(a.npz))
z = d["z"]                                                   # [128, 16, 16]
nodes = torch.from_numpy(z.reshape(128, -1).T.copy()).float()[None]   # [1, 256, 128]
gae = load_frozen_pigvae(a.ckpt, a.experiment)
batch = DenseGraphBatch(node_features=nodes, edge_features=torch.empty(0),
                        mask=torch.ones(1, 256, dtype=torch.bool))
with torch.no_grad():
    z_nodes, z_global, *_ = gae.encode(batch, sample=False)
    xhat = gae.decode(z_nodes, z_global, batch.mask).node_features[0]   # [256, 128]
xhat = xhat.numpy().T.reshape(128, 16, 16)
err = ((xhat - z) ** 2).mean() / (z ** 2).mean()
print("relative MSE", err, " corr", np.corrcoef(xhat.ravel(), z.ravel())[0, 1])
d["xhat"] = xhat
np.savez_compressed(a.npz, **d)
print("saved xhat into", a.npz)
