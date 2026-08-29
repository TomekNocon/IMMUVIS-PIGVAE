"""Visualisation helpers for explainability outputs (matplotlib, headless-safe).

- ``node_importance_to_grid`` reshapes per-node importance to the 2D grid for a spatial heatmap.
- ``plot_node_heatmap`` overlays per-node importance on the grid (which sub-regions of a crop).
- ``plot_marker_attribution`` ranks the marker channels (the biological "which proteins" read).
- ``plot_crop_importance`` scatters crops at their tissue coordinates coloured by decision weight.

Node ordering assumption: the encoder flattens the grid **row-major**, so node ``i`` sits at
``(row=i // grid_size, col=i % grid_size)``. If the pipeline's flatten order ever changes, only
``node_importance_to_grid`` needs updating.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def node_importance_to_grid(importance, grid_size: int) -> np.ndarray:
    """Reshape per-node importance ``[N]`` (``N == grid_size**2``) to ``[grid_size, grid_size]``."""
    arr = _to_np(importance).reshape(-1)
    assert arr.size == grid_size * grid_size, f"expected {grid_size**2} nodes, got {arr.size}"
    return arr.reshape(grid_size, grid_size)


def plot_node_heatmap(importance, grid_size: int, out_path: str, title: str | None = None) -> None:
    grid = node_importance_to_grid(importance, grid_size)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(grid, cmap="magma")
    ax.set_title(title or "per-node attribution")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_marker_attribution(marker_attr, out_path: str, marker_names: list[str] | None = None, top_k: int = 20) -> None:
    """Bar chart of the top-``k`` marker channels by absolute attribution (summed over nodes)."""
    m = _to_np(marker_attr)
    if m.ndim > 1:  # [.., C] -> aggregate over leading dims
        m = m.reshape(-1, m.shape[-1]).sum(axis=0)
    order = np.argsort(np.abs(m))[::-1][:top_k]
    names = [str(i) if marker_names is None else marker_names[i] for i in order]
    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(order))))
    ax.barh(range(len(order)), m[order][::-1], color="#3b7dd8")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(names[::-1], fontsize=7)
    ax.set_xlabel("attribution"); ax.set_title(f"top-{top_k} markers")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_crop_importance(coords, importance, out_path: str, title: str | None = None) -> None:
    """Scatter crops at ``coords`` ``[S, 2]`` coloured by per-crop ``importance`` ``[S]``."""
    c = _to_np(coords); imp = _to_np(importance).reshape(-1)
    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(c[:, 0], c[:, 1], c=imp, cmap="magma", s=18)
    ax.set_title(title or "per-crop decision weight"); ax.invert_yaxis(); ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
