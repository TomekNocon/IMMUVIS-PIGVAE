"""Attention vs IG per crop: a faithful example (img2) next to a cautionary one (img0).

Reads figures/_cache/abmil_dx/abmil_img{i}.pt (attention, ig_importance, logit, label,
faithfulness). Writes figures/fig_abmil_attention_vs_ig.{pdf,png}.
"""
import rootutils

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
SRC = ROOT / "figures" / "_cache" / "abmil_dx"
OUT = ROOT / "figures"

BLUE = "#2f6db5"    # DNA1 — attention
RED = "#c8323e"     # CD45 — top-k attention highlight
GOLD = "#a87f10"    # panCK — IG supports the call
GREY = "#8a8a8a"    # IG opposes the call
TOPK = 5

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


def load(i):
    d = torch.load(SRC / f"abmil_img{i}.pt", weights_only=False)
    att = np.asarray(d["attention"], float)
    ig = np.asarray(d["ig_importance"], float)
    # IG is signed toward the squamous logit; orient toward the predicted class.
    sign = 1.0 if d["logit"] > 0 else -1.0
    return att, ig * sign, d


cases = [
    (2, "(a) Faithful explanation"),
    (0, "(b) Cautionary example"),
]

fig, axes = plt.subplots(2, 2, figsize=(6.3, 3.6), sharex="col",
                         gridspec_kw=dict(height_ratios=[1, 1], hspace=0.18, wspace=0.28))

for col, (i, head) in enumerate(cases):
    att, ig, d = load(i)
    n = len(att)
    x = np.arange(n)
    top = set(np.argsort(att)[::-1][:TOPK])
    f = d["faithfulness"]
    pred = "squamous" if d["logit"] > 0 else "adeno"
    true = "squamous" if d["label"] == 1 else "adeno"

    ax = axes[0, col]
    ax.bar(x, att, color=[RED if k in top else BLUE for k in x], width=0.8)
    ax.set_title(f"{head}\ntrue {true} · pred {pred} · logit {d['logit']:+.2f}\n"
                 f"faithfulness (ins − del AUC) {f['faithfulness']:+.2f}", loc="left")
    if col == 0:
        ax.set_ylabel("ABMIL attention")
    ax.set_ylim(0, att.max() * 1.15)

    ax = axes[1, col]
    ax.bar(x, ig, color=[GOLD if v >= 0 else GREY for v in ig], width=0.8)
    ax.axhline(0, color="k", lw=0.6)
    if col == 0:
        ax.set_ylabel("IG toward prediction")
    ax.set_xlabel("crop index")
    ax.set_xticks(x[::5])

    # mark the IG argmax |.| crop on both rows
    k = int(np.argmax(np.abs(ig)))
    axes[1, col].annotate(f"crop {k}", (k, ig[k]), xytext=(6, -2 if ig[k] < 0 else 2),
                          textcoords="offset points", fontsize=7,
                          va="top" if ig[k] < 0 else "bottom")

handles = [
    plt.Rectangle((0, 0), 1, 1, color=RED, label=f"top-{TOPK} attention"),
    plt.Rectangle((0, 0), 1, 1, color=BLUE, label="other crops"),
    plt.Rectangle((0, 0), 1, 1, color=GOLD, label="IG supports prediction"),
    plt.Rectangle((0, 0), 1, 1, color=GREY, label="IG opposes prediction"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(left=0.09, right=0.99, top=0.81, bottom=0.17)

OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_abmil_attention_vs_ig.{ext}", dpi=300)
print("wrote", OUT / "fig_abmil_attention_vs_ig.pdf")
