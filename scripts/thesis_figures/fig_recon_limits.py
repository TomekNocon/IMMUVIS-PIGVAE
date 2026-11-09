"""Thesis figures for Sec. 5.3 (what limits reconstruction).

fig_pca_variance : cumulative explained variance of the fitted PCAs (6x6 development crop and
                   16x16 final grid), marking 128 and 256 components.
fig_recon_error  : per-node reconstruction error on the 10,197 validation crops for the radius-1
                   (cuogzab1) and radius-2 (d391t2al) runs, as a ratio to each run's interior mean,
                   plus error vs per-node target magnitude.
Inputs: figures/_cache/recon_error_*.npz (extract_recon_error*.py, szary).
"""
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, BLUE, GOLD, RED, GREY = "#2b2b2b", "#2f6db5", "#a87f10", "#c8323e", "#8a8a8a"
C = "figures/_cache/"
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "mathtext.fontset": "dejavuserif",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def fig_pca(out):
    d = np.load(C + "recon_error_81zasevr.npz")
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    curves = [("evr_256_center_crop_16", r"$16\times16$ grid (final model)", BLUE, "-"),
              ("evr_192_center_crop_6", r"$6\times6$ centre crop (development)", GOLD, "--")]
    for key, lab, col, ls in curves:
        c = 100 * np.cumsum(d[key]); k = np.arange(1, len(c) + 1)
        ax.plot(k, c, color=col, lw=1.3, ls=ls, label=lab)
    marks = [(128, 100 * d["evr_128_center_crop_16"].sum(), BLUE, (6, -11)),
             (256, 100 * d["evr_256_center_crop_16"].sum(), BLUE, (-34, -12)),
             (128, 100 * d["evr_128_center_crop_6"].sum(), GOLD, (6, 4))]
    for k, v, col, off in marks:
        ax.plot([k, k], [0, v], color=col, lw=0.6, ls=":"); ax.plot([0, k], [v, v], color=col, lw=0.6, ls=":")
        ax.scatter([k], [v], s=14, color=col, zorder=5)
        ax.annotate(f"{v:.1f}%", (k, v), xytext=off, textcoords="offset points", fontsize=7, color=col)
    ax.set_xlim(0, 260); ax.set_ylim(40, 100)
    ax.set_xticks([0, 64, 128, 192, 256])
    ax.set_xlabel("number of principal components"); ax.set_ylabel("cumulative explained variance (%)")
    ax.legend(frameon=False, loc="lower right", fontsize=6.8)
    fig.tight_layout(); fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300); plt.close(fig)


def fig_err(out):
    runs = [("r1_cuogzab1", r"radius $r=1$", BLUE), ("r2_d391t2al", r"radius $r=2$", GOLD)]
    cmap = LinearSegmentedColormap.from_list("err", ["#ffffff", "#f6dfa0", "#d4962a", "#8a4b0f"])
    fig = plt.figure(figsize=(6.3, 2.45))
    vmin, vmax = 0.98, 1.10
    for i, (tag, title, col) in enumerate(runs):
        d = np.load(C + f"recon_error_{tag}.npz"); e = d["err"]; g = e.shape[0]
        ring = np.minimum.outer(np.minimum(np.arange(g), g - 1 - np.arange(g)), np.minimum(np.arange(g), g - 1 - np.arange(g)))
        ratio = e / e[ring == ring.max()].mean()
        ax = fig.add_axes([0.01 + i * 0.255, 0.12, 0.22, 0.70])
        im = ax.imshow(ratio, cmap=cmap, vmin=vmin, vmax=vmax)
        for r in range(g):
            for c in range(g):
                ax.text(c, r, f"{ratio[r, c]:.2f}", ha="center", va="center", fontsize=5.2,
                        color="white" if ratio[r, c] > 1.07 else INK)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.5)
        ax.set_title(f"{title}\nmean error {e.mean():.3f}", fontsize=7.2, pad=3, color=col)
    cax = fig.add_axes([0.515, 0.12, 0.010, 0.70])
    cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=6, width=0.5, length=2); cb.outline.set_linewidth(0.4)
    cb.set_label("error / interior mean", fontsize=6.5)
    fig.text(0.005, 0.96, "(a) per-node error on the grid", fontsize=7.5, va="top")
    # (b) error vs target magnitude
    ax = fig.add_axes([0.70, 0.18, 0.29, 0.64])
    for tag, title, col in runs:
        d = np.load(C + f"recon_error_{tag}.npz"); b = d["mag_bins"]; ok = d["mag_n"] > 200
        x = ((b[:-1] + b[1:]) / 2)[ok]
        ax.plot(x, d["mag_err"][ok], color=col, lw=1.3, marker="o", ms=2.2, label=title)
    ax.set_xlabel(r"node target magnitude $\|x_i\|/\sqrt{128}$", fontsize=6.8)
    ax.set_ylabel("per-node error", fontsize=6.8); ax.tick_params(labelsize=6.3)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.text(0.635, 0.96, "(b) error grows with target magnitude", fontsize=7.5, va="top")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    fig_pca("figures/fig_pca_variance")
    fig_err("figures/fig_recon_error")
