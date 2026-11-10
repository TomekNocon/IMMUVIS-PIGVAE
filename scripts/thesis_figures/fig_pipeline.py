"""The data pipeline end to end, drawn from real data for one cords crop (Sec. 3.5.1).

slide -> 128x128x43 crop -> ImmuVis encoder -> 16x16x768 feature map -> PCA-128 -> 16x16 grid graph.
Data: extract_pipeline_data.py (szary); rendering runs anywhere.

Usage:
    uv run python scripts/thesis_figures/fig_pipeline.py
"""

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

from figlib import imc, io, paths, style
from figlib.nodes import HIGHLIGHT, TRACKED_CELL, node_rgb, tinted

EDGE = "#333333"
COMPOSITE_GAIN = {"DNA1": 0.35, "panCK": 0.8, "CD8a": 1.0, "SMA": 0.8}
PANEL = 15  # edge of a square panel (data units)
BOTTOM = 10  # bottom of the image row
LAYER_OFFSET = 0.6  # offset between the layers of a deck
GAP = 4.2  # gap between a panel and the next arrow
CAPTION_Y = 1.0
MID_Y = BOTTOM + PANEL / 2


def composite(img: np.ndarray, names: list[str], lo: float = 1, hi: float = 99.8) -> np.ndarray:
    """Additive false-colour RGB of the composite markers, percentile-stretched per channel."""
    out = np.zeros((*img.shape[1:], 3))
    for m, gain in COMPOSITE_GAIN.items():
        ch = img[names.index(m)].astype(np.float32)
        out += gain * imc.stretch(ch, lo, hi)[..., None] * imc.rgb(style.MARKER_COLOURS[m])
    return np.clip(out, 0, 1)


def deck(ax, images, x: float, y: float, w: float, n_ghost: int = 0) -> float:
    """Draw a deck of square images back to front (grey ghosts behind); return its total width."""
    layers = [None] * n_ghost + list(images)
    for k, im in enumerate(layers):
        ox = x + (len(layers) - 1 - k) * LAYER_OFFSET
        oy = y + (len(layers) - 1 - k) * LAYER_OFFSET
        if im is None:
            ax.add_patch(Rectangle((ox, oy), w, w, fc="#d9d9d9", ec=EDGE, lw=0.4, zorder=k))
        else:
            ax.imshow(im, extent=(ox, ox + w, oy, oy + w), zorder=k, interpolation="nearest")
            ax.add_patch(Rectangle((ox, oy), w, w, fill=False, ec=EDGE, lw=0.5, zorder=k + 0.5))
    return w + (len(layers) - 1) * LAYER_OFFSET


def arrow(ax, x0: float, x1: float, y: float, text: str | None = None) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=EDGE, zorder=50
        )
    )
    if text:
        ax.text((x0 + x1) / 2, y + 0.9, text, ha="center", va="bottom", fontsize=6.5)


def caption(ax, xc: float, title: str, sub: str) -> None:
    ax.text(xc, CAPTION_Y + 2.9, title, ha="center", fontsize=7.5, weight="bold")
    ax.text(xc, CAPTION_Y, sub, ha="center", fontsize=6.5)


def mark_cell(ax, x: float, y: float, w: float, n: int = 16, cell=TRACKED_CELL) -> None:
    """Outline grid cell ``cell`` of an n x n square image drawn at (x, y) with edge w."""
    s, (r, c) = w / n, cell
    for ec, lw in [("white", 1.6), (HIGHLIGHT, 0.8)]:
        ax.add_patch(
            Rectangle((x + c * s, y + w - (r + 1) * s), s, s, fill=False, ec=ec, lw=lw, zorder=60)
        )


def grid_graph(ax, z: np.ndarray, x: float, y: float, w: float, cell=TRACKED_CELL) -> None:
    """The 16x16 grid graph with node colours, one node and its four neighbours highlighted."""
    n = z.shape[1]
    s = w / (n - 1)
    colours = node_rgb(z)
    xs = x + np.arange(n) * s
    ys = y + w - np.arange(n) * s  # row 0 at the top, like the image
    for r in range(n):
        ax.plot([xs[0], xs[-1]], [ys[r]] * 2, color="#a0a0a0", lw=0.3, zorder=1)
        ax.plot([xs[r]] * 2, [ys[0], ys[-1]], color="#a0a0a0", lw=0.3, zorder=1)
    gx, gy = np.meshgrid(xs, ys)
    ax.scatter(gx.ravel(), gy.ravel(), s=4.2, c=colours.reshape(-1, 3), ec=EDGE, lw=0.12, zorder=2)
    r, c = cell
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ax.plot([xs[c], xs[c + dc]], [ys[r], ys[r + dr]], color="white", lw=2.0, zorder=3)
        ax.plot([xs[c], xs[c + dc]], [ys[r], ys[r + dr]], color=HIGHLIGHT, lw=1.0, zorder=3.1)
    ax.scatter([xs[c]], [ys[r]], s=16, c=[colours[r, c]], ec=HIGHLIGHT, lw=1.0, zorder=4)


def draw_slide(ax, slide, names, i: int, j: int) -> float:
    """(a) the slide with the 128-px tiling and the chosen crop; returns the right edge."""
    height, width = slide.shape[1:]
    sw = PANEL * 1.25
    sh = sw * height / width
    px = sw / width
    x0, y0 = 0.5, MID_Y - sh / 2
    ax.imshow(composite(slide, names), extent=(x0, x0 + sw, y0, y0 + sh), interpolation="bilinear")
    for t in range(0, height - 127, 128):
        for u in range(0, width - 127, 128):
            ax.add_patch(
                Rectangle(
                    (x0 + u * px, y0 + sh - (t + 128) * px),
                    128 * px,
                    128 * px,
                    fill=False,
                    ec="white",
                    lw=0.25,
                    alpha=0.5,
                )
            )
    ax.add_patch(
        Rectangle(
            (x0 + j * px, y0 + sh - (i + 128) * px),
            128 * px,
            128 * px,
            fill=False,
            ec=HIGHLIGHT,
            lw=1.1,
            zorder=5,
        )
    )
    caption(ax, x0 + sw / 2, "cords slide", rf"${height}\times{width}$ px")
    return x0 + sw


def draw_crop(ax, crop, names, x: float) -> float:
    """(b) the crop as a deck of marker channels, with the 8x8-px footprint of each feature cell."""
    xc = x + GAP + 2.5
    arrow(ax, x + 0.7, xc - 0.7, MID_Y)
    layers = [
        tinted(crop[names.index(m)], style.MARKER_COLOURS[m])
        for m in ["DNA1", "SMA", "CD8a", "panCK"]
    ]
    width = deck(ax, layers, xc, BOTTOM, PANEL, n_ghost=3)
    s16 = PANEL / 16
    for k in range(1, 16):
        ax.plot(
            [xc + k * s16] * 2,
            [BOTTOM, BOTTOM + PANEL],
            color="white",
            lw=0.15,
            alpha=0.45,
            zorder=20,
        )
        ax.plot(
            [xc, xc + PANEL], [BOTTOM + k * s16] * 2, color="white", lw=0.15, alpha=0.45, zorder=20
        )
    mark_cell(ax, xc, BOTTOM, PANEL)
    caption(ax, xc + width / 2, "crop", r"$128\times128\times43$")
    return xc + width


def draw_encoder(ax, x: float) -> float:
    """(c) the frozen ImmuVis encoder, one forward pass over the crop."""
    xe, ew = x + GAP, 9.5
    arrow(ax, x + 0.5, xe - 0.5, MID_Y)
    ax.add_patch(
        Polygon(
            [
                (xe, BOTTOM - 0.5),
                (xe, BOTTOM + PANEL + 0.5 + LAYER_OFFSET * 4),
                (xe + ew, BOTTOM + PANEL * 0.8),
                (xe + ew, BOTTOM + PANEL * 0.25),
            ],
            closed=True,
            fc="#eef2f8",
            ec=EDGE,
            lw=0.7,
        )
    )
    ax.text(
        xe + ew / 2,
        MID_Y + 0.6,
        "ImmuVis\nencoder",
        ha="center",
        va="center",
        fontsize=7,
        linespacing=1.15,
    )
    caption(ax, xe + ew / 2, "one pass", "frozen")
    return xe + ew


def draw_feature_map(ax, emb, x: float) -> float:
    """(d) the 16x16x768 feature map; front slices are the three most variable channels."""
    xf = x + GAP
    arrow(ax, x + 0.5, xf - 0.5, MID_Y)
    top = np.argsort(emb.reshape(768, -1).var(1))[::-1][:3]
    slices = [plt.cm.viridis(imc.stretch(emb[k], 2, 98))[..., :3] for k in top]
    width = deck(ax, slices, xf, BOTTOM, PANEL, n_ghost=5)
    mark_cell(ax, xf, BOTTOM, PANEL)
    caption(ax, xf + width / 2, "feature map", r"$16\times16\times768$")
    return xf + width


def draw_node_features(ax, z, x: float) -> float:
    """(e) PCA-128 node features; the front layer shows three components as RGB."""
    xp = x + GAP + 3.0
    arrow(ax, x + 0.5, xp - 0.5, MID_Y, text=r"PCA")
    width = deck(ax, [node_rgb(z)], xp, BOTTOM, PANEL, n_ghost=3)
    mark_cell(ax, xp, BOTTOM, PANEL)
    caption(ax, xp + width / 2, "node features", r"$16\times16\times128$")
    return xp + width


def draw_graph(ax, z, x: float) -> float:
    """(f) the grid graph with four-neighbour adjacency."""
    xg = x + GAP + 0.8
    arrow(ax, x + 0.5, xg - 0.9, MID_Y)
    grid_graph(ax, z, xg, BOTTOM, PANEL)
    caption(ax, xg + PANEL / 2, "grid graph", r"$N=256$, 4-neighbour")
    return xg + PANEL


def main(npz: str, out: str) -> None:
    d = np.load(npz)
    names, slide = list(d["markers"]), d["slide"]
    i, j = d["crop_ij"]
    crop = slide[:, i : i + 128, j : j + 128]

    style.use_serif(7.5)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_aspect("equal")
    x = draw_slide(ax, slide, names, i, j)
    x = draw_crop(ax, crop, names, x)
    x = draw_encoder(ax, x)
    x = draw_feature_map(ax, d["emb"], x)
    x = draw_node_features(ax, d["z"], x)
    x = draw_graph(ax, d["z"], x)

    right, top = x + 3.2, BOTTOM + PANEL + 8 * LAYER_OFFSET + 0.6
    ax.set_xlim(0, right)
    ax.set_ylim(0, top)
    fig.set_size_inches(6.3, 6.3 * top / right)
    io.save(fig, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", default=str(paths.CACHE / "pipeline_7775_384_256.npz"))
    ap.add_argument("--out", default=str(paths.figure("fig_pipeline")))
    a = ap.parse_args()
    main(a.npz, a.out)
