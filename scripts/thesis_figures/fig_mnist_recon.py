"""Thesis figure (Sec. 5.2.1): PIGVAE on MNIST / Fashion-MNIST, per D4 view.

For each selected example: the ground truth (left), then for each of the 8 D4 views the
permutation matrix chosen by the permuter (top) and the reconstruction restored to the
canonical orientation (bottom), exactly as in the wandb validation panels. A red frame marks
a view whose chosen permutation is inconsistent with the other views of the same image (i.e. not
explained by the majority view->class mapping composed with one image-specific D4 offset). Data: extract_mnist_recon.py.
"""
import argparse, json, os

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

VIEW_LABELS = ["id", "flip", "rot 90°", "rot 90°\n+flip", "rot 180°", "rot 180°\n+flip", "rot 270°", "rot 270°\n+flip"]
WRONG = "#c0392b"


def frame(ax, colour, lw):
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=colour, lw=lw, clip_on=False))


def main(npz, out, examples, gt_per_view, title):
    d = np.load(npz)
    meta = json.loads(str(d["meta"]))
    ex = examples if examples else list(range(min(3, d["pred"].shape[0])))
    rows_per = 3 if gt_per_view else 2

    plt.rcParams.update({"font.family": "serif", "font.size": 7.5})
    W = 6.3
    left, right, sp, gap, head, bottom = 0.03, 0.03, 0.04, 0.26, 0.36, 0.05
    vsp, block_gap = 0.04, 0.14
    first_col = 0 if gt_per_view else 1
    ncols = 8 + first_col
    pw = (W - left - right - (ncols - 1) * sp - (gap if first_col else 0)) / ncols
    H = head + len(ex) * (rows_per * pw + (rows_per - 1) * vsp) + (len(ex) - 1) * block_gap + bottom
    fig = plt.figure(figsize=(W, H))
    xs = [left + k * (pw + sp) + (gap if (first_col and k >= 1) else 0) for k in range(ncols)]

    vmin = min(d["gt"][ex].min(), d["pred"][ex].min()); vmax = max(d["gt"][ex].max(), d["pred"][ex].max())
    row_names = (["Perm.", "Recon.", "Truth"] if gt_per_view else ["Perm.", "Recon."])

    for b, i in enumerate(ex):
        top = H - head - b * (rows_per * pw + (rows_per - 1) * vsp + block_gap)
        for r in range(rows_per):
            y = top - (r + 1) * pw - r * vsp
            fig.text((xs[first_col] - 0.04) / W, (y + pw / 2) / H, row_names[r], rotation=90, ha="right", va="center")
            if not gt_per_view and r == 0:   # single ground-truth cell spanning both rows
                h2 = 2 * pw + vsp
                ax = fig.add_axes([xs[0] / W, (top - h2 + (h2 - pw) / 2) / H, pw / W, pw / H])
                ax.imshow(d["gt"][i, 0], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if b == 0: ax.set_title("Ground\ntruth", pad=3)
            for k in range(8):
                ax = fig.add_axes([xs[k + first_col] / W, y / H, pw / W, pw / H])
                if r == 0:
                    ax.imshow(1 - d["perm"][i, k], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                elif r == 1:
                    ax.imshow(d["pred"][i, k], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
                else:
                    ax.imshow(d["gt"][i, k], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values(): s.set_linewidth(0.5)
                if r < 2 and not d["correct"][i, k]:
                    frame(ax, WRONG, 1.4)
                if b == 0 and r == 0:
                    ax.set_title(VIEW_LABELS[k], pad=3)

    acc = meta["d4_accuracy"]["soft"]
    if title:
        fig.suptitle(title, y=0.995, fontsize=8)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
    print(f"saved {out}.pdf/.png | epoch {meta['epoch']} | D4 view accuracy (val, n={acc['n_images']}): "
          f"frame-independent {acc.get('equiv_accuracy', float('nan')):.4f} "
          f"(all-8 {acc.get('equiv_all8_correct_frac', float('nan')):.4f}), "
          f"vs majority mapping {acc['accuracy']:.4f} | examples {ex}, "
          f"wrong views shown: {int((~d['correct'][ex]).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="figures/_cache/mnist_recon_f1twm4pw.npz")
    ap.add_argument("--out", default="figures/fig_mnist_recon")
    ap.add_argument("--examples", type=lambda s: [int(x) for x in s.split(",")], default=None)
    ap.add_argument("--gt-per-view", action="store_true", help="repeat ground truth under every view (wandb layout)")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    main(a.npz, a.out, a.examples, a.gt_per_view, a.title)
