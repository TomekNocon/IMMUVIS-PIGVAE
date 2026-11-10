"""ABMIL attention vs Integrated Gradients per crop: a faithful example next to a cautionary one.

Reads ``figures/_cache/abmil_dx/abmil_img{i}.pt`` (attention, ig_importance, logit, label,
faithfulness) written by ``scripts/explain/explain_abmil.py``.

Usage:
    uv run python scripts/thesis_figures/fig_abmil_attention_vs_ig.py
"""

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import matplotlib.pyplot as plt
import numpy as np
import torch

from figlib import io, paths, style

CACHE = paths.CACHE / "abmil_dx"
TOP_K = 5
CASES = [(2, "(a) Faithful explanation"), (0, "(b) Cautionary example")]


def load_case(i: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Attention and IG of image ``i``, with IG oriented toward the predicted class."""
    d = torch.load(CACHE / f"abmil_img{i}.pt", weights_only=False)
    sign = 1.0 if d["logit"] > 0 else -1.0  # IG is signed toward the squamous logit
    return np.asarray(d["attention"], float), np.asarray(d["ig_importance"], float) * sign, d


def plot_case(ax_att, ax_ig, i: int, heading: str, first: bool) -> None:
    att, ig, d = load_case(i)
    x = np.arange(len(att))
    top = set(np.argsort(att)[::-1][:TOP_K])
    pred = "squamous" if d["logit"] > 0 else "adeno"
    true = "squamous" if d["label"] == 1 else "adeno"

    ax_att.bar(x, att, color=[style.RED if k in top else style.BLUE for k in x], width=0.8)
    ax_att.set_title(
        f"{heading}\ntrue {true} · pred {pred} · logit {d['logit']:+.2f}\n"
        f"faithfulness (ins − del AUC) {d['faithfulness']['faithfulness']:+.2f}",
        loc="left",
    )
    if first:
        ax_att.set_ylabel("ABMIL attention")
    ax_att.set_ylim(0, att.max() * 1.15)

    ax_ig.bar(x, ig, color=[style.GOLD if v >= 0 else style.GREY for v in ig], width=0.8)
    ax_ig.axhline(0, color="k", lw=0.6)
    if first:
        ax_ig.set_ylabel("IG toward prediction")
    ax_ig.set_xlabel("crop index")
    ax_ig.set_xticks(x[::5])

    k = int(np.argmax(np.abs(ig)))  # the crop with the largest |IG|
    ax_ig.annotate(
        f"crop {k}",
        (k, ig[k]),
        xytext=(6, -2 if ig[k] < 0 else 2),
        textcoords="offset points",
        fontsize=7,
        va="top" if ig[k] < 0 else "bottom",
    )


def main(out: str) -> None:
    style.use_sans(8, title=8.5, label=8, tick=7, legend=7)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(6.3, 3.6),
        sharex="col",
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.18, "wspace": 0.28},
    )
    for col, (i, heading) in enumerate(CASES):
        plot_case(axes[0, col], axes[1, col], i, heading, first=col == 0)

    patches = [
        (style.RED, f"top-{TOP_K} attention"),
        (style.BLUE, "other crops"),
        (style.GOLD, "IG supports prediction"),
        (style.GREY, "IG opposes prediction"),
    ]
    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=t) for c, t in patches],
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.81, bottom=0.17)
    io.save(fig, out, pdf_dpi=300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(paths.figure("fig_abmil_attention_vs_ig")))
    main(ap.parse_args().out)
