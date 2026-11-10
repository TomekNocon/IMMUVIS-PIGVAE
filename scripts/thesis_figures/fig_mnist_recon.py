"""PIGVAE on MNIST / Fashion-MNIST, per D4 view (Sec. 5.2.1).

For each selected example: the ground truth (left), then for each of the 8 D4 views the
permutation matrix the permuter chose (top) and the reconstruction restored to the canonical
orientation (bottom), as in the wandb validation panels. A red frame marks a view whose permutation
is inconsistent with the other views of the same image (not explained by the majority view->class
mapping composed with one image-specific D4 offset). Data: extract_mnist_recon.py.

Usage:
    uv run python scripts/thesis_figures/fig_mnist_recon.py --examples 1,3,8
    uv run python scripts/thesis_figures/fig_mnist_recon.py \
        --npz figures/_cache/fmnist_recon_rttifh86.npz --examples 0,7,9 --out figures/fig_fmnist_recon_ex0_7_9
"""

import argparse
import json

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from figlib import io, paths, style

VIEW_LABELS = [
    "id",
    "flip",
    "rot 90°",
    "rot 90°\n+flip",
    "rot 180°",
    "rot 180°\n+flip",
    "rot 270°",
    "rot 270°\n+flip",
]
INCONSISTENT = "#c0392b"
WIDTH = 6.3  # inches; all layout sizes below are in inches


def frame(ax, colour: str, lw: float) -> None:
    ax.add_patch(
        Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=colour, lw=lw, clip_on=False)
    )


def image_axes(fig, x: float, y: float, size: float, height: float):
    ax = fig.add_axes([x / WIDTH, y / height, size / WIDTH, size / height])
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def main(npz: str, out: str, examples: list[int] | None, gt_per_view: bool, title: str) -> None:
    d = np.load(npz)
    meta = json.loads(str(d["meta"]))
    examples = examples or list(range(min(3, d["pred"].shape[0])))
    rows_per = 3 if gt_per_view else 2
    row_names = ["Perm.", "Recon.", "Truth"][:rows_per]

    style.use_serif(7.5, mathtext=False)
    left, right, sp, gap, head, bottom = 0.03, 0.03, 0.04, 0.26, 0.36, 0.05
    vsp, block_gap = 0.04, 0.14
    first_col = 0 if gt_per_view else 1
    ncols = 8 + first_col
    pw = (WIDTH - left - right - (ncols - 1) * sp - (gap if first_col else 0)) / ncols
    block = rows_per * pw + (rows_per - 1) * vsp
    height = head + len(examples) * block + (len(examples) - 1) * block_gap + bottom
    fig = plt.figure(figsize=(WIDTH, height))
    xs = [left + k * (pw + sp) + (gap if (first_col and k >= 1) else 0) for k in range(ncols)]
    vmin = min(d["gt"][examples].min(), d["pred"][examples].min())
    vmax = max(d["gt"][examples].max(), d["pred"][examples].max())

    for b, i in enumerate(examples):
        top = height - head - b * (block + block_gap)
        for r in range(rows_per):
            y = top - (r + 1) * pw - r * vsp
            fig.text(
                (xs[first_col] - 0.04) / WIDTH,
                (y + pw / 2) / height,
                row_names[r],
                rotation=90,
                ha="right",
                va="center",
            )
            if not gt_per_view and r == 0:  # one ground-truth cell spanning both rows
                span = 2 * pw + vsp
                ax = image_axes(fig, xs[0], top - span + (span - pw) / 2, pw, height)
                ax.imshow(d["gt"][i, 0], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
                if b == 0:
                    ax.set_title("Ground\ntruth", pad=3)
            for k in range(8):
                ax = image_axes(fig, xs[k + first_col], y, pw, height)
                if r == 0:
                    ax.imshow(
                        1 - d["perm"][i, k], cmap="gray", vmin=0, vmax=1, interpolation="nearest"
                    )
                else:
                    ax.imshow(
                        d["pred" if r == 1 else "gt"][i, k],
                        cmap="gray",
                        vmin=vmin,
                        vmax=vmax,
                        interpolation="nearest",
                    )
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)
                if r < 2 and not d["correct"][i, k]:
                    frame(ax, INCONSISTENT, 1.4)
                if b == 0 and r == 0:
                    ax.set_title(VIEW_LABELS[k], pad=3)

    if title:
        fig.suptitle(title, y=0.995, fontsize=8)
    io.save(fig, out)
    acc = meta["d4_accuracy"]["soft"]
    print(
        f"epoch {meta['epoch']} | D4 view accuracy (val, n={acc['n_images']}): frame-independent "
        f"{acc.get('equiv_accuracy', float('nan')):.4f} (all-8 {acc.get('equiv_all8_correct_frac', float('nan')):.4f}),"
        f" majority mapping {acc['accuracy']:.4f} | examples {examples}, "
        f"inconsistent views shown: {int((~d['correct'][examples]).sum())}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", default=str(paths.CACHE / "mnist_recon_f1twm4pw.npz"))
    ap.add_argument("--out", default=str(paths.figure("fig_mnist_recon")))
    ap.add_argument("--examples", type=lambda s: [int(x) for x in s.split(",")], default=None)
    ap.add_argument(
        "--gt-per-view", action="store_true", help="repeat the ground truth under every view"
    )
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    main(a.npz, a.out, a.examples, a.gt_per_view, a.title)
