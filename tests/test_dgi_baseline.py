"""Tests for the DGI graph baseline (src/baselines/dgi/model.py).

Synthetic + CPU-only. Covers the pieces the baseline's correctness rests on: the normalized grid
adjacency (and that it commutes with a D4 rotation -> the invariance foundation), per-graph
feature corruption for DGI's negatives, the D4-invariant crop summary used for ABMIL, and that the
mutual-information loss is a real scalar that decreases on a toy.
"""

import networkx as nx
import numpy as np
import pytest
import torch

from src.baselines.dgi.model import DGILitModule, corrupt_features, grid_adjacency


def _rotation_perm(grid_size: int) -> torch.Tensor:
    """Row-major node index permutation for a 90-degree grid rotation (a D4 automorphism)."""
    n = grid_size
    idx = {(i, j): i * n + j for i in range(n) for j in range(n)}
    p = np.empty(n * n, dtype=np.int64)
    for (i, j), k in idx.items():
        p[k] = idx[(j, n - 1 - i)]
    return torch.from_numpy(p)


def test_grid_adjacency_shape_and_symmetry():
    A = grid_adjacency(4)                      # 4x4 grid -> 16 nodes
    assert A.shape == (16, 16)
    assert torch.allclose(A, A.t(), atol=1e-6)  # undirected -> symmetric


def test_grid_adjacency_has_self_loops():
    A = grid_adjacency(4, normalize=False)
    assert (A.diag() > 0).all()                # self-loops added (GCN needs A + I)


def test_grid_adjacency_commutes_with_d4_rotation():
    # The invariance foundation: a 90-degree rotation is a grid automorphism, so P A P^T == A.
    A = grid_adjacency(4)
    p = _rotation_perm(4)
    P = torch.zeros(16, 16)
    P[torch.arange(16), p] = 1.0
    assert (P @ A @ P.t() - A).abs().max().item() < 1e-6


def test_corrupt_features_shape_and_per_graph_shuffle():
    torch.manual_seed(0)
    x = torch.randn(3, 16, 8)                  # [B, N, F]
    xc = corrupt_features(x)
    assert xc.shape == x.shape
    # each graph's corrupted rows are a permutation of its own rows (row multiset preserved)
    for b in range(x.shape[0]):
        orig = {tuple(r.tolist()) for r in x[b]}
        corr = {tuple(r.tolist()) for r in xc[b]}
        assert orig == corr
    assert not torch.allclose(xc, x)           # actually shuffled


def test_embed_returns_summary_of_hidden_width():
    torch.manual_seed(0)
    m = DGILitModule(n_in=8, n_h=6, grid_size=4)
    c = m.embed(torch.randn(3, 16, 8))
    assert c.shape == (3, 6)                    # per-crop summary, hidden width


def test_embed_is_d4_invariant():
    # The crop summary of a rotated crop must equal the original's (mean readout + automorphism).
    torch.manual_seed(0)
    m = DGILitModule(n_in=8, n_h=6, grid_size=4).eval()
    x = torch.randn(1, 16, 8)
    p = _rotation_perm(4)
    c = m.embed(x)
    c_rot = m.embed(x[:, p, :])
    assert (c - c_rot).abs().max().item() < 1e-5


def test_mi_loss_is_scalar_and_grad():
    torch.manual_seed(0)
    m = DGILitModule(n_in=8, n_h=6, grid_size=4)
    loss = m.mi_loss(torch.randn(4, 16, 8))
    assert loss.ndim == 0 and loss.requires_grad


def test_mi_loss_decreases_on_toy():
    torch.manual_seed(0)
    m = DGILitModule(n_in=8, n_h=6, grid_size=4)
    x = torch.randn(8, 16, 8)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    first = m.mi_loss(x).item()
    for _ in range(50):
        opt.zero_grad(); loss = m.mi_loss(x); loss.backward(); opt.step()
    assert loss.item() < first                  # the contrastive objective actually trains
