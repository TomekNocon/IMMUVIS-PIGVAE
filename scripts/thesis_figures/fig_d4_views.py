"""Thesis figure (Sec. 3.5.3): the eight D4 views of one feature map, as a strip.

Views are produced by the dataloader's own `_spatial_view` (np.rot90 counter-clockwise, then an
optional horizontal flip), applied to the real PCA-128 node features of the pipeline-figure crop.
The highlighted cell is the one followed in the pipeline figure, tracked through every view.
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.data.components.graphs_datamodules import _spatial_view
from fig_cords_crop import COLOURS, stretch
from fig_pipeline import node_rgb, tinted, HL, CELL

ORDER = ["r0_nf", "r90_nf", "r180_nf", "r270_nf", "r0_f", "r90_f", "r180_f", "r270_f"]


def view_rgb(z, key):
    return node_rgb(_spatial_view(z, key))


def cell_after(key, n=16, cell=CELL):
    m = np.zeros((1, n, n)); m[0][cell] = 1
    return tuple(int(v[0]) for v in np.nonzero(_spatial_view(m, key)[0]))


def main(npz, out, with_crop):
    d = np.load(npz)
    names = list(d["markers"]); i, j = d["crop_ij"]; z = d["z"]
    crop = d["slide"][names.index("panCK"), i:i + 128, j:j + 128].astype(np.float32)[None]

    plt.rcParams.update({"font.family": "serif", "font.size": 7.5})
    rows = 2 if with_crop else 1
    W = 6.3                                   # all sizes below in inches
    left = 0.42 if with_crop else 0.03
    sp, gap, head, vsp, bottom = 0.05, 0.22, 0.33, 0.06, 0.03
    pw = (W - left - 0.03 - 7 * sp - gap) / 8
    H = head + rows * pw + (rows - 1) * vsp + bottom
    fig = plt.figure(figsize=(W, H))
    xs = [left + k * (pw + sp) + (gap if k >= 4 else 0) for k in range(8)]
    for k, key in enumerate(ORDER):
        for r in range(rows):
            y = H - head - (r + 1) * pw - r * vsp
            ax = fig.add_axes([xs[k] / W, y / H, pw / W, pw / H])
            if with_crop and r == 0:
                ax.imshow(tinted(_spatial_view(crop, key)[0], COLOURS["panCK"]), interpolation="nearest")
                s = 8
            else:
                ax.imshow(view_rgb(z, key), interpolation="nearest")
                s = 1
            rr, cc = cell_after(key)
            for ec, lw in [("white", 1.3), (HL, 0.7)]:   # white halo keeps red visible on magenta
                ax.add_patch(Rectangle((cc * s - 0.5, rr * s - 0.5), s, s, fill=False, ec=ec, lw=lw))
            ax.set_xticks([]); ax.set_yticks([])
            for sp_ in ax.spines.values(): sp_.set_linewidth(0.6)
            if r == 0:
                ax.set_title(f"{key.split('_')[0][1:]}°", fontsize=7.5, pad=2)
            if k == 0 and with_crop:
                ax.set_ylabel("crop\n(panCK)" if r == 0 else "feature\nmap", fontsize=7, labelpad=3)
    y_grp = (H - 0.03) / H
    fig.text((xs[1] + xs[2] + pw) / 2 / W, y_grp, "rotation", ha="center", va="top", fontsize=7.5, weight="bold")
    fig.text((xs[5] + xs[6] + pw) / 2 / W, y_grp, "rotation + horizontal flip", ha="center", va="top",
             fontsize=7.5, weight="bold")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="figures/_cache/pipeline_7775_384_256.npz")
    ap.add_argument("--out", default="figures/fig_d4_views")
    ap.add_argument("--with-crop", action="store_true")
    a = ap.parse_args(); main(a.npz, a.out, a.with_crop)
