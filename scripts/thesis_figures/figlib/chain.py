"""The reference representation chain: stored ImmuVis map -> PCA-128 -> PIGVAE -> z_global.

Optionally prefixed by the ImmuVis encoder itself (raw crop -> ImmuVis map), for analyses that
perturb the raw markers.
"""

import h5py
import joblib
import numpy as np
import torch

from figlib import immuvis, paths
from src.data.components.graphs_datamodules import DenseGraphBatch
from src.downstream.encode import load_frozen_pigvae

GRID = 16
NODES = GRID * GRID
EMB_DIM = 768


class ReferenceChain:
    """Frozen PIGVAE (experiment=pigvae, run 81zasevr) with the training PCA-128 projection."""

    def __init__(self, device: str, with_immuvis: bool = False):
        self.device = device
        self.gae = load_frozen_pigvae(str(paths.REF_CKPT), paths.REF_EXPERIMENT).to(device).eval()
        self.gae.requires_grad_(False)
        pca = joblib.load(paths.PCA_128)
        self.components = torch.tensor(
            pca.components_, dtype=torch.float32, device=device
        )  # [128, 768]
        self.pca_mean = torch.tensor(pca.mean_, dtype=torch.float32, device=device)
        self._h5 = None
        self.immuvis = None
        if with_immuvis:
            self.immuvis, (missing, unexpected) = immuvis.load_immuvis(device)
            self.immuvis.requires_grad_(False)
            print(
                f"ImmuVis load: {len(missing)} missing / {len(unexpected)} unexpected keys",
                flush=True,
            )
            ids, self.markers = immuvis.channel_ids("cords")
            self.channel_ids = ids.to(device)

    # ---- PCA node features -> z_global ----------------------------------------------------------
    def batch(self, node_features: torch.Tensor) -> DenseGraphBatch:
        b, n = node_features.shape[:2]
        return DenseGraphBatch(
            node_features=node_features,
            edge_features=torch.empty(0, device=self.device),
            mask=torch.ones(b, n, dtype=torch.bool, device=self.device),
        )

    def z_global(self, node_features: torch.Tensor) -> torch.Tensor:
        """``[B, 256, 128]`` node features -> deterministic z_global ``[B, D]``."""
        return self.gae.encode(self.batch(node_features), sample=False)[1]

    def project(self, latent: torch.Tensor) -> torch.Tensor:
        """ImmuVis map ``[B, 768, 16, 16]`` -> PCA node features ``[B, 256, 128]`` (row-major nodes)."""
        nodes = latent.reshape(latent.shape[0], EMB_DIM, NODES).transpose(1, 2)
        return (nodes - self.pca_mean) @ self.components.T

    # ---- raw crop -> z_global (needs with_immuvis) ------------------------------------------------
    def encode_crops(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocessed crops ``[B, 43, 128, 128]`` -> (z_global ``[B, D]``, ImmuVis map)."""
        latent = immuvis.encode(self.immuvis, x, self.channel_ids[None].expand(x.shape[0], -1))
        return self.z_global(self.project(latent)), latent

    # ---- stored embeddings ----------------------------------------------------------------------
    @property
    def h5(self) -> dict:
        if self._h5 is None:
            self._h5 = {s: h5py.File(paths.CORDS_DATA / f"{s}.h5", "r") for s in ("train", "test")}
            self._h5_images = {
                s: np.array([p.decode().split("/")[-1] for p in f["paths"][:]])
                for s, f in self._h5.items()
            }
        return self._h5

    def stored_embedding(self, image: str, row: int, col: int) -> torch.Tensor:
        """The stored ImmuVis map ``[768, 16, 16]`` of the crop at (row, col) of ``image``."""
        h5 = self.h5
        split = "train" if (self._h5_images["train"] == image).any() else "test"
        rows = np.where(self._h5_images[split] == image)[0]
        positions = h5[split]["positions"][rows[0] : rows[-1] + 1]
        hit = rows[[j for j, p in enumerate(positions) if int(p[0]) == row and int(p[1]) == col]][0]
        return torch.from_numpy(h5[split]["embeddings"][hit]).float().to(self.device)

    def stored_nodes(self, image: str, row: int, col: int) -> torch.Tensor:
        """PCA node features ``[256, 128]`` of a stored crop embedding."""
        return self.project(self.stored_embedding(image, row, col)[None])[0]
