"""What limits reconstruction (Sec. 5.3): PCA truncation and per-node error on the grid.

fig_pca_variance   cumulative explained variance of the fitted PCAs (6x6 development crop and
                   16x16 final grid), marking 128 and 256 components.
fig_recon_error    per-node reconstruction error on the 10,197 validation crops for the radius-1
                   (cuogzab1) and radius-2 (d391t2al) runs, as a ratio to each run's interior mean,
                   plus error against per-node target magnitude.
Data: ``figures/_cache/recon_error_*.npz`` (extract_recon_error.py).

Usage:
    uv run python scripts/thesis_figures/fig_recon_limits.py
"""

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from figlib import io, paths, style

RADIUS_RUNS = [
    ("r1_cuogzab1", r"radius $r=1$", style.BLUE),
    ("r2_d391t2al", r"radius $r=2$", style.GOLD),
]
ERROR_CMAP = LinearSegmentedColormap.from_list("err", ["#ffffff", "#f6dfa0", "#d4962a", "#8a4b0f"])
RATIO_RANGE = (0.98, 1.10)


def recon_error(tag: str):
    return np.load(paths.CACHE / f"recon_error_{tag}.npz")


def ring_index(g: int) -> np.ndarray:
    """Distance of every grid cell to the border (0 = outer ring)."""
    d = np.minimum(np.arange(g), g - 1 - np.arange(g))
    return np.minimum.outer(d, d)


def plot_pca_variance(out: str) -> None:
    d = recon_error(paths.REF_WANDB_ID)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    curves = [
        ("evr_256_center_crop_16", r"$16\times16$ grid (final model)", style.BLUE, "-"),
        ("evr_192_center_crop_6", r"$6\times6$ centre crop (development)", style.GOLD, "--"),
    ]
    for key, label, colour, ls in curves:
        cumulative = 100 * np.cumsum(d[key])
        ax.plot(
            np.arange(1, len(cumulative) + 1), cumulative, color=colour, lw=1.3, ls=ls, label=label
        )
    marks = [
        (128, 100 * d["evr_128_center_crop_16"].sum(), style.BLUE, (6, -11)),
        (256, 100 * d["evr_256_center_crop_16"].sum(), style.BLUE, (-34, -12)),
        (128, 100 * d["evr_128_center_crop_6"].sum(), style.GOLD, (6, 4)),
    ]
    for k, value, colour, offset in marks:
        ax.plot([k, k], [0, value], color=colour, lw=0.6, ls=":")
        ax.plot([0, k], [value, value], color=colour, lw=0.6, ls=":")
        ax.scatter([k], [value], s=14, color=colour, zorder=5)
        ax.annotate(
            f"{value:.1f}%",
            (k, value),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            color=colour,
        )
    ax.set_xlim(0, 260)
    ax.set_ylim(40, 100)
    ax.set_xticks([0, 64, 128, 192, 256])
    ax.set_xlabel("number of principal components")
    ax.set_ylabel("cumulative explained variance (%)")
    ax.legend(frameon=False, loc="lower right", fontsize=6.8)
    fig.tight_layout()
    io.save(fig, out)


def plot_error_grids(fig) -> None:
    """(a) per-node error relative to the interior mean, one annotated grid per radius."""
    for i, (tag, title, colour) in enumerate(RADIUS_RUNS):
        err = recon_error(tag)["err"]
        g = err.shape[0]
        rings = ring_index(g)
        ratio = err / err[rings == rings.max()].mean()
        ax = fig.add_axes([0.01 + i * 0.255, 0.12, 0.22, 0.70])
        im = ax.imshow(ratio, cmap=ERROR_CMAP, vmin=RATIO_RANGE[0], vmax=RATIO_RANGE[1])
        for r in range(g):
            for c in range(g):
                ax.text(
                    c,
                    r,
                    f"{ratio[r, c]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color="white" if ratio[r, c] > 1.07 else style.INK,
                )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
        ax.set_title(f"{title}\nmean error {err.mean():.3f}", fontsize=7.2, pad=3, color=colour)
    cbar = fig.colorbar(im, cax=fig.add_axes([0.515, 0.12, 0.010, 0.70]))
    cbar.ax.tick_params(labelsize=6, width=0.5, length=2)
    cbar.outline.set_linewidth(0.4)
    cbar.set_label("error / interior mean", fontsize=6.5)
    fig.text(0.005, 0.96, "(a) per-node error on the grid", fontsize=7.5, va="top")


def plot_error_vs_magnitude(fig) -> None:
    """(b) per-node error against the node's target magnitude."""
    ax = fig.add_axes([0.70, 0.18, 0.29, 0.64])
    for tag, title, colour in RADIUS_RUNS:
        d = recon_error(tag)
        bins, populated = d["mag_bins"], d["mag_n"] > 200
        centres = ((bins[:-1] + bins[1:]) / 2)[populated]
        ax.plot(
            centres, d["mag_err"][populated], color=colour, lw=1.3, marker="o", ms=2.2, label=title
        )
    ax.set_xlabel(r"node target magnitude $\|x_i\|/\sqrt{128}$", fontsize=6.8)
    ax.set_ylabel("per-node error", fontsize=6.8)
    ax.tick_params(labelsize=6.3)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.text(0.635, 0.96, "(b) error grows with target magnitude", fontsize=7.5, va="top")


def plot_recon_error(out: str) -> None:
    fig = plt.figure(figsize=(6.3, 2.45))
    plot_error_grids(fig)
    plot_error_vs_magnitude(fig)
    io.save(fig, out)


def main(out_dir: str) -> None:
    style.use_serif(7.5, thin_axes=True, open_axes=True)
    plot_pca_variance(f"{out_dir}/fig_pca_variance")
    plot_recon_error(f"{out_dir}/fig_recon_error")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(paths.FIGURES))
    main(ap.parse_args().out_dir)
