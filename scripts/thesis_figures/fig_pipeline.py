"""Thesis figure (Sec. 3.5.1): the pipeline end to end, drawn from real data for one cords crop.

slide -> 128x128x43 crop -> ImmuVis encoder -> 16x16x768 feature map -> PCA-128 -> 16x16 grid graph.
Reads the npz written by extract_pipeline_data.py (szary); rendering itself runs anywhere.
"""
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
from fig_cords_crop import COLOURS, stretch

COMPOSITE = ["DNA1", "panCK", "CD8a", "SMA"]
EDGE = "#333333"
HL = "#d62728"      # the one cell followed through the pipeline
CELL = (5, 10)      # its (row, col) in the 16x16 grid


GAIN = {"DNA1": 0.35, "panCK": 0.8, "CD8a": 1.0, "SMA": 0.8}


def composite(img, names, lo=1, hi=99.8):
    """Additive false-colour RGB of the COMPOSITE markers, percentile-stretched per channel."""
    rgb = np.zeros(img.shape[1:] + (3,))
    for m in COMPOSITE:
        ch = img[names.index(m)].astype(np.float32)
        rgb += GAIN[m] * stretch(ch, lo, hi)[..., None] * np.array(matplotlib.colors.to_rgb(COLOURS[m]))
    return np.clip(rgb, 0, 1)


def tinted(ch, colour):
    return stretch(ch.astype(np.float32))[..., None] * np.array(matplotlib.colors.to_rgb(colour))


def pc_rgb(z):
    """First three principal components -> RGB, robustly scaled (z: [128, 16, 16])."""
    out = np.stack([stretch(z[k], 2, 98) for k in range(3)], -1)
    return out


FEAT_CMAP = "seismic"   # same map as the wandb feature-map panels (src/models/components/plot.py)


PC = 1                  # which principal component to show (0-based); PC2 tracks panCK here (r=-0.98)
NODE_RGB = [(-1, 1), (1, 0), (1, 3)]  # (sign, component) -> R, G, B: -PC2, PC1, PC4


def node_rgb(z, lo=2, hi=98, ref=None):
    """Node features -> RGB from three principal components (robust 2nd-98th pct per channel).

    -PC2 drives red (tracks panCK, so tumour reads warm), PC1 green, PC4 blue.
    `ref` (e.g. the input X when colouring a reconstruction) fixes the scale, so the two are comparable.
    """
    ref = z if ref is None else ref
    ch = []
    for sign, k in NODE_RGB:
        v = sign * np.asarray(z[k], dtype=np.float32)
        r = sign * np.asarray(ref[k], dtype=np.float32)
        a, b = np.percentile(r, lo), np.percentile(r, hi)
        ch.append(np.clip((v - a) / (b - a), 0, 1))
    return np.stack(ch, -1)


def feat_map(v, lo=2, hi=98):
    """One feature channel -> RGB on the seismic map.

    Like the wandb panels (vmin/vmax = data range, white = midpoint), but with a robust
    2nd-98th percentile range so a few extreme cells do not wash out the rest.
    """
    v = np.asarray(v, dtype=np.float32)
    a, b = np.percentile(v, lo), np.percentile(v, hi)
    t = np.clip((v - a) / (b - a), 0, 1) if b > a else np.full_like(v, 0.5)
    return plt.get_cmap(FEAT_CMAP)(t)[..., :3]


def stack(ax, images, x, y, w, dx, dy, n_ghost=0, edge=EDGE):
    """Draw a deck of square images, back to front, offset by (dx, dy) per layer."""
    layers = [None] * n_ghost + list(images)
    for k, im in enumerate(layers):
        ox, oy = x + (len(layers) - 1 - k) * dx, y + (len(layers) - 1 - k) * dy
        if im is None:
            ax.add_patch(Rectangle((ox, oy), w, w, fc="#d9d9d9", ec=edge, lw=0.4, zorder=k))
        else:
            ax.imshow(im, extent=(ox, ox + w, oy, oy + w), zorder=k, interpolation="nearest")
            ax.add_patch(Rectangle((ox, oy), w, w, fill=False, ec=edge, lw=0.5, zorder=k + 0.5))
    return len(layers)


def arrow(ax, x0, x1, y, label=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=8,
                                 lw=0.9, color=EDGE, zorder=50))
    if label:
        ax.text((x0 + x1) / 2, y + 0.9, label, ha="center", va="bottom", fontsize=6.5)


def grid_graph(ax, z, x, y, w, cell=CELL):
    n = z.shape[1]
    s = w / (n - 1)
    cols = node_rgb(z)
    xs = x + np.arange(n) * s
    ys = y + w - np.arange(n) * s                       # row 0 at the top, like the image
    for r in range(n):
        ax.plot([xs[0], xs[-1]], [ys[r]] * 2, color="#a0a0a0", lw=0.3, zorder=1)
        ax.plot([xs[r]] * 2, [ys[0], ys[-1]], color="#a0a0a0", lw=0.3, zorder=1)
    X, Y = np.meshgrid(xs, ys)
    ax.scatter(X.ravel(), Y.ravel(), s=4.2, c=cols.reshape(-1, 3), ec=EDGE, lw=0.12, zorder=2)
    r, c = cell                                          # one node and its four neighbours
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ax.plot([xs[c], xs[c + dc]], [ys[r], ys[r + dr]], color="white", lw=2.0, zorder=3)
        ax.plot([xs[c], xs[c + dc]], [ys[r], ys[r + dr]], color=HL, lw=1.0, zorder=3.1)
    ax.scatter([xs[c]], [ys[r]], s=16, c=[cols[r, c]], ec=HL, lw=1.0, zorder=4)


def mark_cell(ax, x, y, w, n=16, cell=CELL, z=60):
    """Outline grid cell `cell` of an n x n square image drawn at (x, y) with edge w."""
    s = w / n; r, c = cell
    for ec, lw in [("white", 1.6), (HL, 0.8)]:
        ax.add_patch(Rectangle((x + c * s, y + w - (r + 1) * s), s, s, fill=False, ec=ec, lw=lw, zorder=z))


def label(ax, xc, title, sub, y):
    ax.text(xc, y + 2.9, title, ha="center", fontsize=7.5, weight="bold")
    ax.text(xc, y, sub, ha="center", fontsize=6.5)


def main(npz, out):
    d = np.load(npz)
    names = list(d["markers"]); slide = d["slide"]; i, j = d["crop_ij"]
    emb, z = d["emb"], d["z"]
    crop = slide[:, i:i + 128, j:j + 128]

    plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "mathtext.fontset": "dejavuserif"})
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_aspect("equal")
    yb, w = 10, 15          # bottom of the image row, edge of a square panel
    off, gap = 0.6, 4.2     # deck layer offset, gap between a panel and an arrow
    ycap = 1.0
    ymid = yb + w / 2

    def deck_width(n_layers):
        return w + (n_layers - 1) * off

    # (a) slide with the 128-px tiling and the chosen crop
    H, W = slide.shape[1:]
    sw = w * 1.25; sh = sw * H / W; px = sw / W
    x0 = 0.5; y0 = ymid - sh / 2
    ax.imshow(composite(slide, names), extent=(x0, x0 + sw, y0, y0 + sh), interpolation="bilinear")
    for t in range(0, H - 127, 128):
        for u in range(0, W - 127, 128):
            ax.add_patch(Rectangle((x0 + u * px, y0 + sh - (t + 128) * px), 128 * px, 128 * px,
                                   fill=False, ec="white", lw=0.25, alpha=0.5))
    ax.add_patch(Rectangle((x0 + j * px, y0 + sh - (i + 128) * px), 128 * px, 128 * px,
                           fill=False, ec=HL, lw=1.1, zorder=5))
    label(ax, x0 + sw / 2, "cords slide", rf"${H}\times{W}$ px", ycap)
    x = x0 + sw

    # (b) the crop as a deck of single marker channels (front to back: panCK, CD8a, SMA, DNA1,
    # then grey layers standing in for the remaining channels), as in the crop figure
    xc = x + gap + 2.5
    arrow(ax, x + 0.7, xc - 0.7, ymid)
    layers = [tinted(crop[names.index(m)], COLOURS[m]) for m in ["DNA1", "SMA", "CD8a", "panCK"]]
    nl = stack(ax, layers, xc, yb, w, off, off, n_ghost=3)
    s16 = w / 16
    for k in range(1, 16):   # the 8x8-px footprint of each feature-map cell
        ax.plot([xc + k * s16] * 2, [yb, yb + w], color="white", lw=0.15, alpha=0.45, zorder=20)
        ax.plot([xc, xc + w], [yb + k * s16] * 2, color="white", lw=0.15, alpha=0.45, zorder=20)
    mark_cell(ax, xc, yb, w)
    cw = deck_width(nl)
    label(ax, xc + cw / 2, "crop", r"$128\times128\times43$", ycap)
    x = xc + cw

    # (c) the ImmuVis encoder, a single forward pass over the whole crop
    xe = x + gap; ew = 9.5
    arrow(ax, x + 0.5, xe - 0.5, ymid)
    ax.add_patch(Polygon([(xe, yb - 0.5), (xe, yb + w + 0.5 + off * 4), (xe + ew, yb + w * 0.8),
                          (xe + ew, yb + w * 0.25)], closed=True, fc="#eef2f8", ec=EDGE, lw=0.7))
    ax.text(xe + ew / 2, ymid + 0.6, "ImmuVis\nencoder", ha="center", va="center", fontsize=7, linespacing=1.15)
    label(ax, xe + ew / 2, "one pass", "frozen", ycap)
    x = xe + ew

    # (d) the 16x16x768 feature map (front slices: three individual feature channels)
    xf = x + gap
    arrow(ax, x + 0.5, xf - 0.5, ymid)
    top = np.argsort(emb.reshape(768, -1).var(1))[::-1][:3]
    sl = [plt.cm.viridis(stretch(emb[k], 2, 98))[..., :3] for k in top]
    nl = stack(ax, sl, xf, yb, w, off, off, n_ghost=5)
    mark_cell(ax, xf, yb, w)
    fw = deck_width(nl)
    label(ax, xf + fw / 2, "feature map", r"$16\times16\times768$", ycap)
    x = xf + fw

    # (e) PCA-128 node features (front: first three components as RGB)
    xp = x + gap + 3.0
    arrow(ax, x + 0.5, xp - 0.5, ymid, label=r"PCA")
    nl = stack(ax, [node_rgb(z)], xp, yb, w, off, off, n_ghost=3)
    mark_cell(ax, xp, yb, w)
    pw = deck_width(nl)
    label(ax, xp + pw / 2, "node features", r"$16\times16\times128$", ycap)
    x = xp + pw

    # (f) the grid graph, four-neighbour adjacency
    xg = x + gap + 0.8; gw = w
    arrow(ax, x + 0.5, xg - 0.9, ymid)
    grid_graph(ax, z, xg, yb, gw)
    label(ax, xg + gw / 2, "grid graph", r"$N=256$, 4-neighbour", ycap)
    x = xg + gw

    right, top_y = x + 3.2, yb + w + 8 * off + 0.6
    ax.set_xlim(0, right); ax.set_ylim(0, top_y)
    width_in = 6.3
    fig.set_size_inches(width_in, width_in * top_y / right)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
    print("size (in):", fig.get_size_inches())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="figures/_cache/pipeline_7775_384_256.npz")
    ap.add_argument("--out", default="figures/fig_pipeline")
    a = ap.parse_args(); main(a.npz, a.out)
