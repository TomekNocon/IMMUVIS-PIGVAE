"""Professional IMC (Imaging Mass Cytometry) input visualisation.

The upstream pipeline feeds raw multiplex images (``C`` markers x H x W, ``C`` ~40) through the
ImmuVis encoder; before any of that, the raw multiplex input is what a pathologist actually reads.
This module renders that raw input two publication-standard ways:

- ``plot_marker_grid`` -- every marker as its own labelled, percentile-contrasted panel (the
  multiplex "contact sheet"; which proteins are where).
- ``plot_false_colour`` -- a handful of markers mapped to distinct colours and blended into a
  single RGB overlay (the classic multiplexed-immunofluorescence look), with a legend.

No model or checkpoint is needed: this reads the raw pixels only. Inputs may be a whole-image
``.tiff`` (via ``tifffile``) or a pre-cut ``.npy`` patch; both are normalised to channel-first
``[C, H, W]``. Preprocessing matches the repo's known recipe ``arcsinh(x / 5)`` (see
``src/data/components/graphs_datamodules.py`` line ~288) so the rendered intensities match what
the encoder is fed. Tiling (``get_all_patches``) mirrors the upstream ``generate_patches`` recipe
(non-overlapping 128, drop the right/bottom remainder) so a rendered patch lines up 1:1 with the
PIGVAE embedding rows.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Default false-colour palette: perceptually distinct, ordered so the first (often a DNA/nuclear
# channel) reads as cool blue. RGB in [0, 1].
DEFAULT_COLOURS: list[tuple[float, float, float]] = [
    (0.10, 0.45, 0.95),  # blue
    (0.95, 0.25, 0.20),  # red
    (0.20, 0.85, 0.35),  # green
    (0.95, 0.75, 0.10),  # amber
    (0.75, 0.30, 0.95),  # violet
    (0.20, 0.90, 0.90),  # cyan
]

# Marker-name hints for auto-picking a composite when names are available (case-insensitive
# substring match, in priority order). DNA first so it lands on the blue slot.
_COMPOSITE_HINTS: list[str] = ["dna", "dapi", "ir19", "histone", "panck", "keratin",
                               "cd8", "cd3", "cd68", "cd20", "sma", "vimentin", "cd31"]


def _to_np(x) -> np.ndarray:
    """Detach a torch tensor to numpy, or pass a numpy array through."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _norm01(a) -> np.ndarray:
    """Normalise to [0, 1] by min/max (flat input -> zeros)."""
    a = _to_np(a).astype(np.float32)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _to_chw(img: np.ndarray) -> np.ndarray:
    """Normalise a 3D multiplex array to channel-first ``[C, H, W]``.

    IMC images have far fewer markers than pixels, so the channel axis is the smallest one;
    whole-image TIFFs are already ``[C, H, W]`` while ``.npy`` patches are often ``[H, W, C]``.
    """
    a = np.asarray(img)
    if a.ndim != 3:
        raise ValueError(f"expected a 3D multiplex array [C,H,W] or [H,W,C], got shape {a.shape}")
    c_axis = int(np.argmin(a.shape))
    return np.moveaxis(a, c_axis, 0) if c_axis != 0 else a


def load_imc_image(path: str, arcsinh_scale: float | None = 5.0) -> np.ndarray:
    """Load a raw multiplex input (``.tiff`` or ``.npy``) as float32 ``[C, H, W]``.

    ``arcsinh_scale`` applies the repo's ``arcsinh(x / scale)`` intensity compression (set it to
    ``None`` for inputs that are already arcsinh-transformed, e.g. the ``_onlyarcsinh`` patches).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import tifffile

        raw = tifffile.imread(path)
    elif ext == ".npy":
        raw = np.load(path)
    else:
        raise ValueError(f"unsupported extension {ext!r} (expected .tiff or .npy)")
    img = _to_chw(raw).astype(np.float32)
    if arcsinh_scale is not None:
        img = np.arcsinh(img / float(arcsinh_scale))
    return img


def butterworth_denoise(img: np.ndarray, cutoff_frequency_ratio: float = 0.2) -> np.ndarray:
    """Per-channel Butterworth low-pass filter -- the pipeline's denoiser.

    Matches the recipe the PIGVAE embeddings were made with (``use_butterworth_filter=True``,
    ``cutoff_frequency_ratio=0.2``, low-pass): it removes the high-frequency single-ion-count
    speckle inherent to raw IMC, so the rendered image reflects what the encoder actually sees.
    Input/output are channel-first ``[C, H, W]``.
    """
    from skimage.filters import butterworth

    img = _to_chw(img)
    out = [butterworth(img[c], cutoff_frequency_ratio=cutoff_frequency_ratio, high_pass=False)
           for c in range(img.shape[0])]
    return np.stack(out).astype(np.float32)


def get_all_patches(img: np.ndarray, patch_size: int = 128):
    """Tile ``[C, H, W]`` into non-overlapping ``patch_size`` patches, dropping the remainder.

    Mirrors the upstream ``generate_patches.get_all_patches`` (the recipe the PIGVAE embeddings
    were cut with), so patch ``k`` here corresponds to embedding row ``k``. Returns
    ``(patches, coords)`` with ``patches[k]`` a ``[C, patch_size, patch_size]`` array and
    ``coords[k] == [(i, j), (i + patch_size, j + patch_size)]``.
    """
    img = _to_chw(img)
    _, H, W = img.shape
    patches, coords = [], []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            if i + patch_size > H or j + patch_size > W:
                continue
            patches.append(img[:, i:i + patch_size, j:j + patch_size])
            coords.append([(i, j), (i + patch_size, j + patch_size)])
    return patches, coords


def _percentile_clip(ch: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Contrast-stretch one channel to ``[0, 1]`` between its ``lo``/``hi`` percentiles."""
    a = np.asarray(ch, dtype=np.float32)
    plo, phi = np.percentile(a, lo), np.percentile(a, hi)
    if phi <= plo:
        return np.zeros_like(a)
    return np.clip((a - plo) / (phi - plo), 0.0, 1.0)


def distinct_colours(n: int, cmap: str = "nipy_spectral") -> list[tuple[float, float, float]]:
    """Return ``n`` visually distinct RGB colours sampled across a spread colormap.

    Used for an all-marker composite (too many channels for the hand-picked palette). The
    endpoints are trimmed off ``nipy_spectral`` so no marker lands on near-black/near-white.
    """
    if n <= len(DEFAULT_COLOURS):
        return DEFAULT_COLOURS[:n]
    cm = plt.get_cmap(cmap)
    return [tuple(cm(t)[:3]) for t in np.linspace(0.05, 0.95, n)]


def _resolve_channels(names, channels) -> list[int]:
    """Resolve a list of channel selectors (int indices or marker-name strings) to int indices."""
    out = []
    for c in channels:
        if isinstance(c, (int, np.integer)):
            out.append(int(c))
        elif names is not None and c in names:
            out.append(list(names).index(c))
        else:
            raise ValueError(f"channel {c!r} is not an int index and not in marker_names")
    return out


def default_composite_channels(marker_names=None, num_channels: int | None = None, k: int = 5) -> list[int]:
    """Pick ~``k`` channels for a false-colour composite.

    Prefers well-known lineage markers by name (``_COMPOSITE_HINTS``); falls back to evenly
    spaced channel indices when names are unavailable.
    """
    if marker_names is not None:
        lower = [str(m).lower() for m in marker_names]
        picked: list[int] = []
        for hint in _COMPOSITE_HINTS:
            for idx, nm in enumerate(lower):
                if hint in nm and idx not in picked:
                    picked.append(idx)
                    break
            if len(picked) >= k:
                break
        if picked:
            return picked[:k]
        num_channels = len(marker_names)
    if not num_channels:
        raise ValueError("need marker_names or num_channels to pick composite channels")
    return list(np.linspace(0, num_channels - 1, min(k, num_channels)).round().astype(int))


def _add_scale_bar(ax, length_um: float, pixel_size_um: float, color="white") -> None:
    """Draw a µm scale bar in the lower-right of ``ax`` (image axes, pixel coordinates)."""
    W = ax.get_xlim()[1]
    H = ax.get_ylim()[0]  # inverted y (image), so ylim[0] is the bottom
    bar_px = length_um / pixel_size_um
    x1 = W * 0.96
    x0 = x1 - bar_px
    y = H * 0.94
    ax.plot([x0, x1], [y, y], color=color, lw=3, solid_capstyle="butt")
    ax.text((x0 + x1) / 2, y - H * 0.02, f"{length_um:g} µm", color=color,
            ha="center", va="bottom", fontsize=8)


def plot_marker_grid(img, marker_names=None, out_path: str = "imc_grid.png", cmap: str = "magma",
                     pctl: tuple[float, float] = (1.0, 99.0), ncols: int = 8,
                     pixel_size_um: float | None = None, scale_bar_um: float = 100.0,
                     title: str | None = None) -> None:
    """Publication "contact sheet": every marker as its own labelled, percentile-contrasted panel."""
    img = _to_chw(img)
    C = img.shape[0]
    names = list(marker_names) if marker_names is not None else [str(i) for i in range(C)]
    ncols = min(ncols, C)
    nrows = (C + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.7 * ncols, 1.8 * nrows), facecolor="black")
    axes = np.atleast_1d(axes).ravel()
    for c in range(C):
        ax = axes[c]
        ax.imshow(_percentile_clip(img[c], *pctl), cmap=cmap, interpolation="nearest")
        ax.set_title(names[c], color="white", fontsize=8, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    if pixel_size_um and C:
        _add_scale_bar(axes[0], scale_bar_um, pixel_size_um)
    for c in range(C, len(axes)):
        axes[c].axis("off")
    if title:
        fig.suptitle(title, color="white", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def compose_false_colour(img, channels, colours=None, pctl: tuple[float, float] = (1.0, 99.0),
                         blend: str = "max") -> np.ndarray:
    """Blend selected channels into an ``[H, W, 3]`` RGB image in ``[0, 1]``.

    ``blend="max"`` keeps markers visually separable (per-pixel channel that dominates wins);
    ``blend="add"`` sums then clips (brighter where markers co-localise).
    """
    img = _to_chw(img)
    ch = list(channels)
    cols = list(colours) if colours is not None else DEFAULT_COLOURS
    if len(cols) < len(ch):
        raise ValueError(f"need >= {len(ch)} colours for {len(ch)} channels, got {len(cols)}")
    H, W = img.shape[1:]
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for c, col in zip(ch, cols):
        contrib = _percentile_clip(img[c], *pctl)[..., None] * np.asarray(col, dtype=np.float32)
        rgb = np.maximum(rgb, contrib) if blend == "max" else rgb + contrib
    return np.clip(rgb, 0.0, 1.0)


def plot_false_colour(img, channels, colours=None, marker_names=None,
                      out_path: str = "imc_composite.png", pctl: tuple[float, float] = (1.0, 99.0),
                      blend: str = "max", pixel_size_um: float | None = None,
                      scale_bar_um: float = 100.0, title: str | None = None) -> np.ndarray:
    """Render the classic multiplexed-IF composite (markers -> colours, blended) with a legend.

    ``channels`` may be int indices or marker-name strings (resolved via ``marker_names``).
    Returns the RGB array so callers can reuse it.
    """
    img = _to_chw(img)
    idx = _resolve_channels(marker_names, channels)
    cols = list(colours) if colours is not None else distinct_colours(len(idx))
    rgb = compose_false_colour(img, idx, cols, pctl=pctl, blend=blend)
    names = ([str(marker_names[i]) for i in idx] if marker_names is not None
             else [f"ch {i}" for i in idx])
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="black")
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if pixel_size_um:
        _add_scale_bar(ax, scale_bar_um, pixel_size_um)
    handles = [mpatches.Patch(color=cols[k], label=names[k]) for k in range(len(idx))]
    # Many markers -> compact multi-column legend placed OUTSIDE the image (right).
    many = len(idx) > 12
    leg = ax.legend(handles=handles,
                    loc="upper left" if many else "upper right",
                    bbox_to_anchor=(1.01, 1.0) if many else None,
                    ncol=2 if len(idx) > 24 else 1,
                    framealpha=0.0 if many else 0.35, facecolor="black", edgecolor="none",
                    fontsize=6 if many else 9, labelspacing=0.25, handlelength=1.0)
    for txt in leg.get_texts():
        txt.set_color("white")
    if title:
        ax.set_title(title, color="white", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    return rgb


def _upsample_nearest(a: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour upsample a small 2D map (e.g. 16x16) to ``out_hw`` (e.g. 128x128)."""
    a = _to_np(a)
    H, W = out_hw
    ri = (np.arange(H) * a.shape[0] // H).clip(0, a.shape[0] - 1)
    ci = (np.arange(W) * a.shape[1] // W).clip(0, a.shape[1] - 1)
    return a[ri][:, ci]


def overlay_heatmap_on_rgb(rgb, heat, out_path: str, alpha: float = 0.5, title: str | None = None,
                           pixel_size_um: float | None = None, scale_bar_um: float = 20.0) -> None:
    """Overlay per-region importance ``heat`` (any 2D, e.g. 16x16) on a false-colour crop ``rgb``.

    ``heat`` is nearest-upsampled to the crop size, normalised, magma-blended over the tissue --
    'where on this crop' the attribution concentrates. For image-2 (representation on raw crop).
    """
    rgb = _to_np(rgb)
    H, W = rgb.shape[:2]
    hm = _norm01(_upsample_nearest(heat, (H, W)))
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
    ax.imshow(rgb, interpolation="nearest")
    ax.imshow(hm, cmap="magma", alpha=alpha, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if pixel_size_um:
        _add_scale_bar(ax, scale_bar_um, pixel_size_um)
    if title:
        ax.set_title(title, color="white", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path, dpi=200, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def stitch_composite_importance(crops_rgb, coords, importance, out_path: str, alpha: float = 0.45,
                                title: str | None = None, highlight: int | None = None,
                                cell: int = 32) -> None:
    """Stitch false-colour crops at their tissue coords and lay a CLAM importance heatmap over them.

    ``crops_rgb``: list of ``[h,w,3]`` false-colour crops; ``coords``: ``[N,2]`` (row,col px);
    ``importance``: ``[N]`` per-crop decision weight. The crops form the (fully-visible) tissue
    mosaic; a translucent, smoothly-interpolated jet heatmap of the importance is overlaid at
    constant ``alpha`` so the tissue always shows through -- 'which region drove the decision'.
    ``highlight`` outlines the decisive crop.
    """
    c = np.asarray(coords)
    raw_imp = _to_np(importance).reshape(-1)
    # Rank-normalise so the map is discriminative (uniform 0..1 by rank) even when raw IG values
    # are bunched -- the decisive crop lands at 1.0 (red), the least relevant at 0 (blue).
    order = np.argsort(np.argsort(raw_imp)).astype(np.float32)
    imp = order / max(len(raw_imp) - 1, 1)
    ys = {v: i for i, v in enumerate(sorted(set(c[:, 0].tolist())))}
    xs = {v: i for i, v in enumerate(sorted(set(c[:, 1].tolist())))}
    nr, nc = len(ys), len(xs)
    H, W = nr * cell, nc * cell
    bg = np.zeros((H, W, 3), dtype=np.float32)
    coarse = np.full((nr, nc), np.nan, dtype=np.float32)  # one importance value per crop
    present = np.zeros((nr, nc), dtype=np.float32)         # 1 where the crop has real tissue
    for k in range(len(crops_rgb)):
        gy, gx = ys[c[k, 0]], xs[c[k, 1]]
        thumb = np.stack([_upsample_nearest(_to_np(crops_rgb[k])[..., ch], (cell, cell))
                          for ch in range(3)], axis=-1)
        bg[gy * cell:gy * cell + cell, gx * cell:gx * cell + cell, :] = thumb
        coarse[gy, gx] = imp[k]
        present[gy, gx] = 1.0 if float(np.mean(thumb)) > 0.02 else 0.0

    fig, ax = plt.subplots(figsize=(7.6, 7), facecolor="black")
    ax.imshow(bg, interpolation="nearest")
    # CLAM heatmap: tissue stays FULLY visible; a translucent, smoothly-interpolated importance
    # field (jet: blue=low -> red=high) is laid over it. Alpha is 0 on empty cells AND cells with
    # no tissue, so the overlay never washes the background (both arrays interpolate together).
    filled = np.where(np.isnan(coarse), 0.0, coarse)
    alpha_arr = (np.where(np.isnan(coarse), 0.0, float(alpha)) * present).astype(np.float32)
    im = ax.imshow(filled, cmap="jet", vmin=0.0, vmax=1.0, alpha=alpha_arr,
                   interpolation="bilinear", extent=(0, W, H, 0), aspect="auto", zorder=2)
    if highlight is not None:
        r, cc = ys[c[highlight, 0]] * cell, xs[c[highlight, 1]] * cell
        ax.add_patch(plt.Rectangle((cc, r), cell, cell, fill=False, edgecolor="white",
                                   lw=2.5, zorder=3))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("decision importance", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    if title:
        ax.set_title(title, color="white", fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=200, facecolor="black", bbox_inches="tight")
    plt.close(fig)
