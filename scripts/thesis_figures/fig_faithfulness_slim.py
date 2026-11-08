"""Slim faithfulness figure (Sec. 6.5, item 18): crop-level panels + one win-rate summary.

The node-level median curves of fig_latent_faithfulness.py (old panels d/e) are dropped; the node result
(all node methods ~ chance) is kept only as bars in panel (d).
  (a) crop deletion AUC   (b) crop insertion AUC   (slides with |logit| > 1)
  (c) slide 7064 crop curves
  (d) % of slides / crops where a method beats a random order (deletion AUC), crops vs nodes
Reads thesis_figures/_cache/latent_faith/{cohort,faith_nodes,example_7064}.npz.
Writes thesis_figures/fig_faithfulness_slim.{pdf,png}.
"""
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LF = ROOT / "thesis_figures" / "_cache" / "latent_faith"
OUT = ROOT / "thesis_figures"
RED, GOLD, GREY = "#c8323e", "#a87f10", "#8a8a8a"
METH_COL = {"ig": GOLD, "attention": RED, "random": GREY}
plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})
rng = np.random.default_rng(0)


def boot_ci(v, n=2000):
    v = np.asarray(v)[np.isfinite(v)]
    m = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def win(a, b):
    w = (a < b).astype(float); bs = [rng.choice(w, len(w)).mean() for _ in range(2000)]
    return w.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


coh = np.load(LF / "cohort.npz"); fn = np.load(LF / "faith_nodes.npz"); ex = np.load(LF / "example_7064.npz")
b1 = coh["b1"]; cols = list(coh["b1_cols"]); sel = np.abs(coh["logit"]) > 1

fig = plt.figure(figsize=(6.3, 4.9))
gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.38, width_ratios=[1, 1.15])

# (a)/(b) crop-level AUC distributions
for j, mode in enumerate(["deletion", "insertion"]):
    ax = fig.add_subplot(gs[0, j])
    vals = []
    for q, meth in enumerate(["ig", "attention", "random"]):
        v = b1[sel, cols.index(f"{meth}_{mode}")]; v = v[np.isfinite(v)]; vals.append(v)
        m, lo, hi = boot_ci(v)
        ax.boxplot(v, positions=[q], widths=0.55, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=METH_COL[meth], alpha=0.35, lw=0.6), medianprops=dict(color="k", lw=1),
                   whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        ax.errorbar(q, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=3, color="k", lw=1)
        ax.text(q + 0.33, m, f"{m:.2f}", va="center", fontsize=6)
        print(f"{mode:9s} {meth:9s} mean AUC {m:.3f} [{lo:.3f}, {hi:.3f}]")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["IG", "attention", "random"])
    ax.set_xlim(-0.5, 2.75)
    ax.set_ylim(np.percentile(np.concatenate(vals), [1, 99]) + np.array([-0.1, 0.1]))
    ax.set_ylabel(f"{mode} AUC ({'lower' if mode == 'deletion' else 'higher'} = better)")
    ax.set_title(f"({'ab'[j]}) Crop {mode}, {sel.sum()} slides", loc="left")

# (c) example curves, slide 7064
ax = fig.add_subplot(gs[1, 0])
for meth, key in [("ig", "ig"), ("attention", "attention"), ("random", "random0")]:
    for mode, ls in [("deletion", "-"), ("insertion", "--")]:
        cv = ex[f"crop_curve_{key}_{mode}"]
        ax.plot(np.linspace(0, 1, len(cv)), cv, ls=ls, color=METH_COL[meth], lw=1.1)
ax.axhline(0, color="k", lw=0.4); ax.axhline(1, color="k", lw=0.4, ls=":")
ax.set_xlabel("fraction of crops removed / inserted"); ax.set_ylabel("normalised score\n(1 = full slide, 0 = baseline)")
ax.set_title("(c) Example curves, slide 7064", loc="left")

# (d) win rate vs random: crop level vs node level
ax = fig.add_subplot(gs[1, 1])
bars = [("IG", win(b1[sel, cols.index("ig_deletion")], b1[sel, cols.index("random_deletion")]), GOLD),
        ("attention", win(b1[sel, cols.index("attention_deletion")], b1[sel, cols.index("random_deletion")]), RED)]
for t, lab in [("t1", "dim"), ("t2", "logit")]:
    ok = np.abs(fn[f"{t}_dfx"]) > 1e-3
    bars.append((f"IG\n→{lab}", win(fn[f"{t}_ig_deletion"][ok], fn[f"{t}_random_deletion"][ok]), GOLD))
    bars.append((f"rollout\n→{lab}", win(fn[f"{t}_rollout_deletion"][ok], fn[f"{t}_random_deletion"][ok]), RED))
xs = np.r_[0, 1, 2.6, 3.6, 4.6, 5.6]
for x, (lab, (m, lo, hi), col) in zip(xs, bars):
    ax.bar(x, 100 * m, color=col, alpha=0.85, width=0.75)
    ax.errorbar(x, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], fmt="none", color="k", lw=0.8)
    ax.text(x, 100 * hi + 2, (f"{100 * m:.1f}" if m > 0.995 else f"{100 * m:.0f}"), ha="center", va="bottom", fontsize=6)
    print(f"win rate {lab!r}: {m:.1%} [{lo:.1%}, {hi:.1%}]")
ax.axhline(50, color="k", lw=0.6, ls="--")
ax.text(7.25, 51.5, "chance", ha="right", va="bottom", fontsize=6)
ax.axvline(1.8, color="#cccccc", lw=0.6)
ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars], fontsize=6)
for x, txt in [(0.5, "crop level"), (4.1, "node level (within a crop)")]:
    ax.text(x, -0.24, txt, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5,
            fontweight="bold")
ax.set_xlim(-0.6, 7.3); ax.set_ylim(0, 112)
ax.set_ylabel("% beating a random order\n(deletion AUC)")
ax.set_title("(d) Better than random?", loc="left")

handles = [plt.Line2D([], [], color=GOLD, lw=2, label="Integrated Gradients"),
           plt.Line2D([], [], color=RED, lw=2, label="attention (crops) / rollout (nodes)"),
           plt.Line2D([], [], color=GREY, lw=2, label="random order"),
           plt.Line2D([], [], color="k", ls="-", label="deletion"),
           plt.Line2D([], [], color="k", ls="--", label="insertion")]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0),
           handlelength=1.8, columnspacing=1.0)
fig.subplots_adjust(left=0.1, right=0.97, top=0.94, bottom=0.17)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_faithfulness_slim.{ext}", dpi=300)
print("wrote", OUT / "fig_faithfulness_slim.pdf")
