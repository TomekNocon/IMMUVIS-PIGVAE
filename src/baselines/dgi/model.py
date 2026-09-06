"""Deep Graph Infomax (DGI) baseline for the IMC crop-graphs.

A self-supervised GRAPH baseline to compare against PIGVAE: DGI trains node embeddings by
maximising mutual information between per-node patch representations and a global graph summary
(Velickovic et al., 2019), a contrastive/MI objective rather than PIGVAE's reconstruction. On our
fixed 16x16 grid graphs the mean-readout summary is D4-invariant (a D4 rotation is a grid
automorphism, so it commutes with the grid adjacency -- see tests), so the summary is a fair
invariant crop embedding for the downstream ABMIL transfer, exactly like z_global.

The DGI network itself is reused verbatim from the reference implementation vendored as the
``external/dgi`` submodule (petarv-/DGI); this module only adds the grid adjacency, the per-graph
feature corruption for DGI's negatives, the MI loss, and the LightningModule wiring.
"""

from __future__ import annotations

import os
import sys

import lightning as L
import networkx as nx
import torch
import torch.nn as nn

# Reuse the reference DGI network from the submodule (no copy): put its flat package dir on
# sys.path so `from models.dgi import DGI` (which does `from layers import ...`) resolves.
_DGI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "external/dgi")
if _DGI_DIR not in sys.path:
    sys.path.insert(0, _DGI_DIR)

from models.dgi import DGI  # noqa: E402 (submodule import after sys.path tweak)


def grid_adjacency(grid_size: int, normalize: bool = True) -> torch.Tensor:
    """Dense adjacency ``[N, N]`` of a ``grid_size`` x ``grid_size`` grid, row-major node order.

    Adds self-loops (GCN needs ``A + I``); with ``normalize`` applies the symmetric GCN
    normalisation ``D^-1/2 (A + I) D^-1/2``. A D4 rotation of the grid is an automorphism, so this
    matrix commutes with the corresponding node permutation -> the mean-readout summary is
    D4-invariant.
    """
    n = grid_size
    order = [(i, j) for i in range(n) for j in range(n)]
    idx = {node: k for k, node in enumerate(order)}
    N = n * n
    A = torch.zeros(N, N)
    for u, v in nx.grid_2d_graph(n, n).edges():
        A[idx[u], idx[v]] = A[idx[v], idx[u]] = 1.0
    A = A + torch.eye(N)  # self-loops
    if normalize:
        d = A.sum(1)
        A = d.pow(-0.5).unsqueeze(1) * A * d.pow(-0.5).unsqueeze(0)
    return A


def corrupt_features(x: torch.Tensor) -> torch.Tensor:
    """DGI corruption: shuffle each graph's node rows independently (the negative samples).

    ``x`` is ``[B, N, F]``; the returned tensor shuffles the ``N`` node rows within each graph
    (row multiset preserved per graph), destroying the node<->position structure the discriminator
    must tell apart from the real graph.
    """
    B, N, _ = x.shape
    out = torch.empty_like(x)
    for b in range(B):
        out[b] = x[b][torch.randperm(N, device=x.device)]
    return out


class DGILitModule(L.LightningModule):
    """LightningModule wrapping the reference DGI for our crop-graphs.

    ``embed(x)`` returns the mean-readout summary ``[B, n_h]`` -- the per-crop invariant embedding
    for ABMIL. ``mi_loss(x)`` is the DGI mutual-information (BCE) objective.
    """

    def __init__(self, n_in: int, n_h: int, grid_size: int, lr: float = 1e-3,
                 activation: str = "prelu"):
        super().__init__()
        self.save_hyperparameters()
        self.dgi = DGI(n_in, n_h, activation)
        self.register_buffer("adj", grid_adjacency(grid_size))  # [N, N], normalised
        self.lr = lr
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _adj_batch(self, batch_size: int) -> torch.Tensor:
        return self.adj.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, N] for the dense bmm path

    def mi_loss(self, x: torch.Tensor, msk: torch.Tensor | None = None) -> torch.Tensor:
        B, N, _ = x.shape
        x_corrupt = corrupt_features(x)
        logits = self.dgi(x, x_corrupt, self._adj_batch(B), False, msk, None, None)  # [B, 2N]
        labels = torch.cat([torch.ones(B, N), torch.zeros(B, N)], dim=1).to(logits)
        return self.loss_fn(logits, labels)

    def embed(self, x: torch.Tensor, msk: torch.Tensor | None = None) -> torch.Tensor:
        _h, c = self.dgi.embed(x, self._adj_batch(x.shape[0]), False, msk)
        return c  # [B, n_h] mean-readout summary

    def training_step(self, batch, batch_idx):
        loss = self.mi_loss(batch.node_features, getattr(batch, "mask", None))
        self.log("train/mi_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.mi_loss(batch.node_features, getattr(batch, "mask", None))
        self.log("val/mi_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
