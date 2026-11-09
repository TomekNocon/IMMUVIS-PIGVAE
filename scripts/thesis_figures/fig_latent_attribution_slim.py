"""Slim latent-attribution figure (Sec. 6.4, item 17): panels (a) and (b) only.

(a) ABMIL IG per tissue crop and z_global dimension (top 24 dims by |IG| on the slide), decisive crop boxed.
(b) decisive crop: top 10 dims by |IG|, with the sum over all dims.
The node-level maps of fig_latent_faithfulness.py are dropped (not validated, Sec. 6.5).
Reads figures/_cache/latent_faith/example_<img>.npz. Writes figures/fig_latent_attribution_slim_<img>.{pdf,png}.
"""
import rootutils
import glob

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
LF = ROOT / "figures" / "_cache" / "latent_faith"
OUT = ROOT / "figures"
GOLD, GREY = "#a87f10", "#8a8a8a"
plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})

for exf in sorted(glob.glob(str(LF / "example_*.npz"))):
    ex = np.load(exf); img = str(ex["img"])[:-5]
    M, e, k = ex["M"], ex["empty"], int(ex["decisive"])
    pred = "squamous" if ex["logit"] > 0 else "adeno"
    fig = plt.figure(figsize=(6.3, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.55)

    # (a) crop x latent-dim matrix
    ax = fig.add_subplot(gs[0])
    real = np.where(~e)[0]; cs = real[np.argsort(-M[real].sum(1))]
    ds = np.argsort(-np.abs(M[real]).sum(0))[:24]
    sub = M[np.ix_(cs, ds)]; v = np.abs(sub).max()
    im = ax.imshow(sub, aspect="auto", cmap="PuOr_r", vmin=-v, vmax=v, interpolation="nearest")
    ax.set_xticks(range(len(ds))); ax.set_xticklabels(ds, rotation=90, fontsize=5.5)
    ax.set_ylabel(f"tissue crops (n={len(real)}), sorted by total IG")
    ax.set_xlabel("z_global dimension (top 24 by |IG| on this slide)")
    ka = int(np.where(cs == k)[0][0])
    ax.add_patch(Rectangle((-0.5, ka - 0.5), len(ds), 1, fill=False, ec="k", lw=0.9))
    ax.set_yticks([ka]); ax.set_yticklabels(["decisive\ncrop"], fontsize=6)
    ax.set_title(f"(a) Decision IG per crop and latent dimension, slide {img}", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label(f"IG toward prediction ({pred})", fontsize=6.5); cb.ax.tick_params(labelsize=6)

    # (b) decisive crop: top dims
    ax = fig.add_subplot(gs[1])
    md = ex["M_dec"]; o = np.argsort(-np.abs(md))[:10]
    ax.barh(range(len(o)), 100 * md[o], color=[GOLD if md[j] >= 0 else GREY for j in o], height=0.75)
    ax.set_yticks(range(len(o))); ax.set_yticklabels([f"dim {j}" for j in o]); ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    share = np.abs(md[o]).sum() / np.abs(md).sum()
    ax.set_title("(b) Decisive crop: top 10 dims", loc="left")
    ax.set_xlabel(f"IG toward prediction (×10⁻²)\nall {M.shape[1]} dims sum to {md.sum():+.2f};\ntop 10 = {share:.0%} of |IG|")

    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.25)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_latent_attribution_slim_{img}.{ext}", dpi=300)
    plt.close(fig)
    print(f"wrote fig_latent_attribution_slim_{img} | decisive crop sum {md.sum():+.3f}, top-10 share {share:.1%}, "
          f"dims needed for 50% of |IG|: {int(np.searchsorted(np.cumsum(np.sort(np.abs(md))[::-1]) / np.abs(md).sum(), 0.5)) + 1}")
