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

from collections.abc import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ArrayLike = torch.Tensor | np.ndarray | Sequence[float]


def _to_np(x: ArrayLike) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-9)


def _no_ticks(ax: Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def _save(fig: Figure, out_path: str) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def node_importance_to_grid(importance: ArrayLike, grid_size: int) -> np.ndarray:
    """Reshape per-node importance ``[N]`` (``N == grid_size**2``) to ``[grid_size, grid_size]``."""
    arr = _to_np(importance).reshape(-1)
    if arr.size != grid_size * grid_size:
        raise ValueError(f"expected {grid_size**2} nodes, got {arr.size}")
    return arr.reshape(grid_size, grid_size)


def plot_node_heatmap(
    importance: ArrayLike, grid_size: int, out_path: str, title: str | None = None
) -> None:
    """Per-node importance as a ``grid_size x grid_size`` heatmap."""
    grid = node_importance_to_grid(importance, grid_size)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(grid, cmap="magma")
    ax.set_title(title or "per-node attribution")
    _no_ticks(ax)
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out_path)


def plot_marker_attribution(
    marker_attr: ArrayLike,
    out_path: str,
    marker_names: list[str] | None = None,
    top_k: int = 20,
) -> None:
    """Bar chart of the top-``k`` marker channels by absolute attribution (summed over nodes)."""
    m = _to_np(marker_attr)
    if m.ndim > 1:  # [.., C] -> aggregate over leading dims
        m = m.reshape(-1, m.shape[-1]).sum(axis=0)
    order = np.argsort(np.abs(m))[::-1][:top_k]
    names = [str(i) if marker_names is None else marker_names[i] for i in order]
    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(order))))
    ax.barh(range(len(order)), m[order][::-1], color="#3b7dd8")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names[::-1], fontsize=7)
    ax.set_xlabel("attribution")
    ax.set_title(f"top-{top_k} markers")
    _save(fig, out_path)


def plot_crop_panel(
    attribution: ArrayLike,
    feature_map: ArrayLike,
    grid_size: int,
    top_channels: Iterable[int],
    out_path: str,
    title: str | None = None,
) -> None:
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
    im = axes[0, 0].imshow(attr, cmap="magma")
    axes[0, 0].set_title("IG attribution")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)

    im = axes[0, 1].imshow(feat.mean(axis=-1), cmap="viridis")
    axes[0, 1].set_title("aggregate intensity")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)

    axes[0, 2].imshow(np.stack([_norm01(feat[..., c]) for c in ch], axis=-1))
    axes[0, 2].set_title(f"false-colour ch {ch}")

    for j, c in enumerate(ch):
        im = axes[1, j].imshow(feat[..., c], cmap="magma")
        axes[1, j].set_title(f"channel {c}")
        fig.colorbar(im, ax=axes[1, j], fraction=0.046)
    for ax in axes.ravel():
        _no_ticks(ax)
    if title:
        fig.suptitle(title)
    _save(fig, out_path)


def plot_abmil_crops(
    attention: ArrayLike,
    importance: ArrayLike,
    out_path: str,
    title: str | None = None,
    top_k: int = 5,
) -> None:
    """Per-crop decision view: attention bars (top) + IG-importance bars (bottom) over crops.

    The top-``k`` crops by attention are highlighted: these are the tissue regions the
    gated-ABMIL pools most when making the clinical call.
    """
    a = _to_np(attention).reshape(-1)
    imp = _to_np(importance).reshape(-1)
    s = len(a)
    top = set(np.argsort(a)[::-1][:top_k].tolist())
    fig, axes = plt.subplots(2, 1, figsize=(max(5, s * 0.18), 5), sharex=True)
    axes[0].bar(range(s), a, color=["#e4572e" if i in top else "#3b7dd8" for i in range(s)])
    axes[0].set_ylabel("attention")
    axes[0].set_title(title or "gated-ABMIL per-crop")
    axes[1].bar(range(s), imp, color="#d8763b")
    axes[1].set_ylabel("IG importance")
    axes[1].set_xlabel("crop index")
    _save(fig, out_path)


def plot_stitched_importance(
    tiles: Sequence[ArrayLike],
    coords: ArrayLike,
    importance: ArrayLike,
    out_path: str,
    title: str | None = None,
    tile: int = 16,
    alpha: float = 0.55,
) -> None:
    """Stitch per-crop channel-0 tiles at their coords, overlay per-crop importance (CLAM style).

    Background = the crops' first-channel ``tile x tile`` maps placed on the tissue grid (shows
    each crop's shape); overlay = the importance heatmap (magma, alpha-blended), so the regions
    that drive the decision glow. ``tiles``: list of ``[tile, tile]``; ``coords``: ``[N, 2]``
    (coords0=row, coords1=col); ``importance``: ``[N]``.
    """
    c = np.asarray(coords)
    imp = _to_np(importance).reshape(-1)
    ys = {v: i for i, v in enumerate(sorted(set(c[:, 0].tolist())))}
    xs = {v: i for i, v in enumerate(sorted(set(c[:, 1].tolist())))}
    H, W = len(ys) * tile, len(xs) * tile
    bg = np.full((H, W), np.nan, dtype=np.float32)
    hm = np.full((H, W), np.nan, dtype=np.float32)
    for k in range(len(tiles)):
        r, cc = ys[c[k, 0]] * tile, xs[c[k, 1]] * tile
        t = _to_np(tiles[k]).astype(np.float32)
        bg[r : r + tile, cc : cc + tile] = _norm01(t)  # per-tile normalise -> show shape
        hm[r : r + tile, cc : cc + tile] = imp[k]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.nan_to_num(bg), cmap="gray", interpolation="nearest")
    ax.imshow(hm, cmap="magma", alpha=alpha, interpolation="nearest")  # NaN cells transparent
    ax.set_title(title or "stitched crops · importance overlay")
    _no_ticks(ax)
    _save(fig, out_path)


def plot_crop_importance(
    coords: ArrayLike, importance: ArrayLike, out_path: str, title: str | None = None
) -> None:
    """Scatter crops at ``coords`` ``[S, 2]`` coloured by per-crop ``importance`` ``[S]``."""
    c = _to_np(coords)
    imp = _to_np(importance).reshape(-1)
    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(c[:, 0], c[:, 1], c=imp, cmap="magma", s=18)
    ax.set_title(title or "per-crop decision weight")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, fraction=0.046)
    _save(fig, out_path)
