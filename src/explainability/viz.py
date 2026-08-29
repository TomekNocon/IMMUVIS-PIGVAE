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


def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-9)


def plot_crop_panel(attribution, feature_map, grid_size: int, top_channels, out_path: str, title: str | None = None) -> None:
    """Grid comparing the attribution with 3 feature-map projections (for choosing a viz).

    Panels: [IG attribution | aggregate intensity | false-colour(top-3)] on the top row, and the
    top-3 individual channel maps on the bottom row. NB: ``feature_map`` is the *embedding*
    (768-d), so these channels are upstream-encoder dims, not the 40 real IMC markers.

    Args:
        attribution: per-node importance ``[N]``.
        feature_map: reconstructed per-node features ``[N, C]`` (inverse-PCA of the encoder input).
        grid_size: side of the grid (16).
        top_channels: iterable of channel indices to render (>= 3 used).
    """
    attr = node_importance_to_grid(attribution, grid_size)
    feat = _to_np(feature_map).reshape(grid_size, grid_size, -1)
    ch = list(top_channels)[:3]
    while len(ch) < 3:
        ch.append(ch[-1])

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    # top-left: attribution
    im = axes[0, 0].imshow(attr, cmap="magma"); axes[0, 0].set_title("IG attribution")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)
    # top-mid: aggregate intensity (mean over channels)
    im = axes[0, 1].imshow(feat.mean(axis=-1), cmap="viridis"); axes[0, 1].set_title("aggregate intensity")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)
    # top-right: false-colour composite of the top-3 channels
    rgb = np.stack([_norm01(feat[..., c]) for c in ch], axis=-1)
    axes[0, 2].imshow(rgb); axes[0, 2].set_title(f"false-colour ch {ch}")
    # bottom: individual channel maps
    for j, c in enumerate(ch):
        im = axes[1, j].imshow(feat[..., c], cmap="magma"); axes[1, j].set_title(f"channel {c}")
        fig.colorbar(im, ax=axes[1, j], fraction=0.046)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    if title:
        fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_crop_importance(coords, importance, out_path: str, title: str | None = None) -> None:
    """Scatter crops at ``coords`` ``[S, 2]`` coloured by per-crop ``importance`` ``[S]``."""
    c = _to_np(coords); imp = _to_np(importance).reshape(-1)
    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(c[:, 0], c[:, 1], c=imp, cmap="magma", s=18)
    ax.set_title(title or "per-crop decision weight"); ax.invert_yaxis(); ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
