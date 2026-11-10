"""Add the trained PIGVAE's reconstruction of the pipeline-figure crop to its cache (szary).

Input = the crop's PCA-128 node features from the pipeline cache (the projection the dataloader
applies: zscore=False, no clip), in training node order (row-major). Deterministic (sample=False).
Writes the reconstruction as ``xhat`` [128, 16, 16] into the npz (in place unless ``--out``).

Usage:
    uv run python scripts/thesis_figures/extract_reconstruction.py
"""

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import numpy as np
import torch

from figlib import paths
from src.data.components.graphs_datamodules import DenseGraphBatch
from src.downstream.encode import load_frozen_pigvae


@torch.no_grad()
def reconstruct(z: np.ndarray, ckpt: str, experiment: str) -> np.ndarray:
    """Node features [128, 16, 16] -> deterministic reconstruction [128, 16, 16]."""
    nodes = torch.from_numpy(z.reshape(128, -1).T.copy()).float()[None]  # [1, 256, 128]
    gae = load_frozen_pigvae(ckpt, experiment)
    batch = DenseGraphBatch(
        node_features=nodes, edge_features=torch.empty(0), mask=torch.ones(1, 256, dtype=torch.bool)
    )
    z_nodes, z_global, *_ = gae.encode(batch, sample=False)
    xhat = gae.decode(z_nodes, z_global, batch.mask).node_features[0]  # [256, 128]
    return xhat.numpy().T.reshape(128, 16, 16)


def main(npz: str, ckpt: str, experiment: str, out: str | None) -> None:
    d = dict(np.load(npz))
    d["xhat"] = reconstruct(d["z"], ckpt, experiment)
    rel_mse = ((d["xhat"] - d["z"]) ** 2).mean() / (d["z"] ** 2).mean()
    print(
        f"relative MSE {rel_mse:.4f} | corr {np.corrcoef(d['xhat'].ravel(), d['z'].ravel())[0, 1]:.4f}"
    )
    np.savez_compressed(out or npz, **d)
    print("saved xhat into", out or npz)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", default=str(paths.CACHE / "pipeline_7775_384_256.npz"))
    ap.add_argument("--ckpt", default=str(paths.REF_CKPT))
    ap.add_argument("--experiment", default=paths.REF_EXPERIMENT)
    ap.add_argument(
        "--out", default=None, help="write to this npz instead of updating --npz in place"
    )
    a = ap.parse_args()
    main(a.npz, a.ckpt, a.experiment, a.out)
