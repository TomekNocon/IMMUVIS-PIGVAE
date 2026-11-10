"""The eight D4 views of one feature map, as a strip (Sec. 3.5.3).

Views come from the dataloader's own ``_spatial_view`` (np.rot90 counter-clockwise, then an optional
horizontal flip), applied to the real PCA-128 node features of the pipeline-figure crop. The
highlighted cell is the one followed in the pipeline figure, tracked through every view.

Usage:
    uv run python scripts/thesis_figures/fig_d4_views.py
    uv run python scripts/thesis_figures/fig_d4_views.py --with-crop --out figures/fig_d4_views_with_crop
"""

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from figlib import io, paths, style
from figlib.nodes import HIGHLIGHT, TRACKED_CELL, node_rgb, tinted
from src.data.components.graphs_datamodules import _spatial_view

VIEWS = ["r0_nf", "r90_nf", "r180_nf", "r270_nf", "r0_f", "r90_f", "r180_f", "r270_f"]
WIDTH = 6.3  # inches; all layout sizes below are in inches


def cell_after(view: str, n: int = 16, cell=TRACKED_CELL) -> tuple[int, int]:
    """Where ``cell`` of an n x n grid lands under ``view``."""
    m = np.zeros((1, n, n))
    m[0][cell] = 1
    return tuple(int(v[0]) for v in np.nonzero(_spatial_view(m, view)[0]))


def main(npz: str, out: str, with_crop: bool) -> None:
    d = np.load(npz)
    names, z = list(d["markers"]), d["z"]
    i, j = d["crop_ij"]
    crop = d["slide"][names.index("panCK"), i : i + 128, j : j + 128].astype(np.float32)[None]

    style.use_serif(7.5, mathtext=False)
    rows = 2 if with_crop else 1
    left = 0.42 if with_crop else 0.03
    sp, gap, head, vsp, bottom = 0.05, 0.22, 0.33, 0.06, 0.03
    pw = (WIDTH - left - 0.03 - 7 * sp - gap) / 8
    height = head + rows * pw + (rows - 1) * vsp + bottom
    fig = plt.figure(figsize=(WIDTH, height))
    xs = [left + k * (pw + sp) + (gap if k >= 4 else 0) for k in range(8)]

    for k, view in enumerate(VIEWS):
        for r in range(rows):
            y = height - head - (r + 1) * pw - r * vsp
            ax = fig.add_axes([xs[k] / WIDTH, y / height, pw / WIDTH, pw / height])
            if with_crop and r == 0:
                ax.imshow(
                    tinted(_spatial_view(crop, view)[0], style.MARKER_COLOURS["panCK"]),
                    interpolation="nearest",
                )
                scale = 8  # crop pixels per feature-map cell
            else:
                ax.imshow(node_rgb(_spatial_view(z, view)), interpolation="nearest")
                scale = 1
            rr, cc = cell_after(view)
            for ec, lw in [
                ("white", 1.3),
                (HIGHLIGHT, 0.7),
            ]:  # white halo keeps red visible on magenta
                ax.add_patch(
                    Rectangle(
                        (cc * scale - 0.5, rr * scale - 0.5), scale, scale, fill=False, ec=ec, lw=lw
                    )
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
            if r == 0:
                ax.set_title(f"{view.split('_')[0][1:]}°", fontsize=7.5, pad=2)
            if k == 0 and with_crop:
                ax.set_ylabel("crop\n(panCK)" if r == 0 else "feature\nmap", fontsize=7, labelpad=3)

    y_group = (height - 0.03) / height
    fig.text(
        (xs[1] + xs[2] + pw) / 2 / WIDTH,
        y_group,
        "rotation",
        ha="center",
        va="top",
        fontsize=7.5,
        weight="bold",
    )
    fig.text(
        (xs[5] + xs[6] + pw) / 2 / WIDTH,
        y_group,
        "rotation + horizontal flip",
        ha="center",
        va="top",
        fontsize=7.5,
        weight="bold",
    )
    io.save(fig, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", default=str(paths.CACHE / "pipeline_7775_384_256.npz"))
    ap.add_argument("--out", default=str(paths.figure("fig_d4_views")))
    ap.add_argument(
        "--with-crop", action="store_true", help="add the panCK crop row above the feature maps"
    )
    a = ap.parse_args()
    main(a.npz, a.out, a.with_crop)
