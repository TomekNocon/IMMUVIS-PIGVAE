"""Figures for items 17 (Sec. 6.4, decision -> latent) and 18 (Sec. 6.5, faithfulness).

Reads figures/_cache/latent_faith/{cohort,faith_nodes,example_<img>}.npz (extract_latent_faithfulness.py)
and the Butterworth-filtered slide from _cache/spatial_v3 for the crop image.
Writes
  figures/fig_latent_attribution_<img>.{pdf,png}
  figures/fig_faithfulness.{pdf,png}
and prints the numbers quoted in the text (means, 95% bootstrap CIs, paired win rates).
"""
import glob, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
C = ROOT / "figures" / "_cache"
LF = C / "latent_faith"
OUT = ROOT / "figures"
P = 128
BLUE, RED, GOLD, GREY = "#2f6db5", "#c8323e", "#a87f10", "#8a8a8a"
METH_COL = {"ig": GOLD, "attention": RED, "rollout": RED, "random": GREY}
METH_LAB = {"ig": "Integrated Gradients", "attention": "ABMIL attention", "rollout": "attention rollout",
            "random": "random order"}
COMPOSITE = [("DNA1", "#4C9BE8"), ("panCK", "#E8C34C"), ("CD45", "#E8584C"), ("SMA", "#5CCB7A")]
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


def stretch(ch, ref):
    a, b = np.percentile(ref, 1), np.percentile(ref, 99.5)
    return np.clip((ch - a) / (b - a), 0, 1) if b > a else np.zeros_like(ch)


def composite(slide, markers, r0, c0):
    out = 0
    for m, col in COMPOSITE:
        j = markers.index(m)
        out = out + stretch(slide[j, r0:r0 + P, c0:c0 + P], slide[j])[..., None] * np.array(matplotlib.colors.to_rgb(col))
    return np.clip(0.6 * out, 0, 1)


# ============================================================ item 17: decision -> latent
coh = np.load(LF / "cohort.npz")
classes = list(coh["classes"])
for exf in sorted(glob.glob(str(LF / "example_*.npz"))):
    ex = np.load(exf); img = str(ex["img"])[:-5]
    sp = np.load(glob.glob(str(C / "spatial_v3" / f"*_{img}.npz"))[0])
    markers = list(sp["markers"]); slide = sp["slide"].astype(np.float32)
    M, e, k = ex["M"], ex["empty"], int(ex["decisive"]); r0, c0 = ex["coords"][k]
    top_d = list(ex["top_dims"])
    fig = plt.figure(figsize=(6.3, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.45)
    g0 = gs[0].subgridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.6)
    g1 = gs[1].subgridspec(1, 5, wspace=0.08)

    # (a) crop x latent-dim matrix: real crops sorted by decision IG, top-24 dims by slide |IG|
    ax = fig.add_subplot(g0[0])
    real = np.where(~e)[0]; cs = real[np.argsort(-M[real].sum(1))]
    ds = np.argsort(-np.abs(M[real]).sum(0))[:24]
    sub = M[np.ix_(cs, ds)]; v = np.abs(sub).max()
    im = ax.imshow(sub, aspect="auto", cmap="PuOr_r", vmin=-v, vmax=v, interpolation="nearest")
    ax.set_xticks(range(len(ds))); ax.set_xticklabels(ds, rotation=90, fontsize=5.5)
    ax.set_yticks([]); ax.set_ylabel(f"tissue crops (n={len(real)}), sorted by decision IG")
    ax.set_xlabel("z_global dimension (top 24 by |IG| on this slide)")
    ka = int(np.where(cs == k)[0][0])
    ax.add_patch(Rectangle((-0.5, ka - 0.5), len(ds), 1, fill=False, ec="k", lw=0.9))
    ax.set_title(f"(a) Decision IG per crop and latent dimension, slide {img}", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("IG toward prediction", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    # (b) decisive crop: top dims
    ax = fig.add_subplot(g0[1])
    md = ex["M_dec"]; o = np.argsort(-np.abs(md))[:10]
    ax.barh(range(len(o)), md[o], color=[GOLD if md[j] >= 0 else GREY for j in o], height=0.75)
    ax.set_yticks(range(len(o))); ax.set_yticklabels([f"dim {j}" for j in o]); ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("IG toward prediction"); ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.set_title(f"(b) Decisive crop: top dims\n(sum of all {M.shape[1]}: {md.sum():+.2f})", loc="left")

    # (c) crop + encoder-IG node maps for the top-3 decision dims + rollout
    ax = fig.add_subplot(g1[0]); ax.imshow(composite(slide, markers, r0, c0)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("decisive crop", fontsize=7)
    ax.text(0, 1.32, "(c) Which grid nodes write the top-3 supporting dimensions (encoder IG, 16×16 nodes)",
            transform=ax.transAxes, fontsize=8)
    maps = [(f"dim {d}", ex["ig_t1"][j]) for j, d in enumerate(top_d)] + [("rollout (CLS)", ex["rollout"])]
    for t, (title, mp) in enumerate(maps):
        ax = fig.add_subplot(g1[t + 1])
        g = mp.reshape(16, 16)
        if title.startswith("rollout"):
            ax.imshow(g, cmap="Reds", vmin=0, vmax=g.max())
        else:
            vv = np.abs(g).max(); ax.imshow(g, cmap="PuOr_r", vmin=-vv, vmax=vv)
            ax.set_title(f"{title}", fontsize=7)
        if title.startswith("rollout"):
            ax.set_title(title, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.04)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_latent_attribution_{img}.{ext}", dpi=300)
    plt.close(fig)
    corrs = [np.corrcoef(ex["ig_t1"][j], ex["rollout"])[0, 1] for j in range(len(top_d))]
    print(f"[17] {img}: decisive crop {k}, top dims {top_d} IG {np.round(md[top_d], 3)}; "
          f"corr(encoder IG, rollout) {np.round(corrs, 2)}")

# cohort: are the decision dims shared across slides?
dt, yy = coh["dimtot"], coh["y"]
pred = (coh["logit"] > 0).astype(int)
for c in [0, 1]:
    s = (yy == c) & (pred == c)
    mean = dt[s].mean(0); o = np.argsort(-mean)[:8]
    frac = (dt[s][:, o] > 0).mean(0)
    print(f"[17] cohort {classes[c]} (n={s.sum()} correct): top decision dims {list(o)} "
          f"mean IG {np.round(mean[o], 2)} positive in {np.round(frac, 2)} of slides")
print(f"[17] ABMIL IG completeness: median rel err {np.median(coh['comp_err']):.2e}, max {coh['comp_err'].max():.2e}")

# ============================================================ item 18: faithfulness
fig = plt.figure(figsize=(6.3, 4.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.62, wspace=0.42)

# (a) crop level (ABMIL): AUC distributions, slides with a clear decision (|logit| > 1)
b1 = coh["b1"]; cols = list(coh["b1_cols"]); sel = np.abs(coh["logit"]) > 1
print(f"[18] crop level, {sel.sum()} / {len(sel)} slides with |logit| > 1")
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
        print(f"     {mode:9s} {meth:9s} mean AUC {m:.3f} [{lo:.3f}, {hi:.3f}]  median {np.median(v):.3f}")
    for a_, b_ in [(0, 1), (0, 2), (1, 2)]:
        better = (vals[a_] < vals[b_]) if mode == "deletion" else (vals[a_] > vals[b_])
        print(f"     {mode:9s} {['ig','attention','random'][a_]} better than {['ig','attention','random'][b_]} "
              f"in {better.mean():.2%} of slides")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["IG", "attention", "random"])
    ax.set_ylim(np.percentile(np.concatenate(vals), [1, 99]) + np.array([-0.1, 0.1]))
    ax.set_ylabel(f"{mode} AUC ({'lower' if mode == 'deletion' else 'higher'} = better)")
    ax.set_title(f"({'ab'[j]}) Crops, {mode}", loc="left")

# example crop-level curve (7064)
ex = np.load(LF / "example_7064.npz")
ax = fig.add_subplot(gs[0, 2])
for meth, key in [("ig", "ig"), ("attention", "attention"), ("random", "random0")]:
    for mode, ls in [("deletion", "-"), ("insertion", "--")]:
        cv = ex[f"crop_curve_{key}_{mode}"]
        ax.plot(np.linspace(0, 1, len(cv)), cv, ls=ls, color=METH_COL[meth], lw=1)
ax.axhline(0, color="k", lw=0.4); ax.axhline(1, color="k", lw=0.4, ls=":")
ax.set_xlabel("fraction of crops"); ax.set_ylabel("normalised score")
ax.set_title("(c) Slide 7064 curves", loc="left")

# node level: mean normalised curves with bootstrap bands, two targets
fn = np.load(LF / "faith_nodes.npz")
for j, (t, title) in enumerate([("t1", "(d) Nodes → top decision dim"), ("t2", "(e) Nodes → decision logit")]):
    ok = np.abs(fn[f"{t}_dfx"]) > 1e-3
    print(f"[18] node level {t}: n={ok.sum()} crops, median IG completeness rel err {np.median(fn[f'{t}_comp_err'][ok]):.3f}, "
          f"median corr(IG, rollout) {np.nanmedian(fn[f'corr_ig_roll_{t}'][ok]):+.2f}")
    ax = fig.add_subplot(gs[1, j])
    for meth in ["ig", "rollout", "random"]:
        for mode, ls in [("deletion", "-"), ("insertion", "--")]:
            cv = fn[f"{t}_{meth}_{mode}_curve"][ok]
            x = np.linspace(0, 1, cv.shape[1]); m = np.nanmedian(cv, 0)
            lo, hi = np.nanpercentile(cv, 25, 0), np.nanpercentile(cv, 75, 0)
            ax.plot(x, m, ls=ls, color=METH_COL[meth], lw=1)
            if mode == "deletion": ax.fill_between(x, lo, hi, color=METH_COL[meth], alpha=0.12, lw=0)
            a = fn[f"{t}_{meth}_{mode}"][ok]; mm, l_, h_ = boot_ci(a)
            print(f"     {mode:9s} {meth:8s} mean AUC {mm:.3f} [{l_:.3f}, {h_:.3f}]  median {np.nanmedian(a):.3f}")
        if meth != "random":
            for mode in ["deletion", "insertion"]:
                a, b = fn[f"{t}_{meth}_{mode}"][ok], fn[f"{t}_random_{mode}"][ok]
                w = (a < b) if mode == "deletion" else (a > b)
                print(f"     {mode:9s} {meth} better than random in {np.nanmean(w):.2%} of crops")
    a, b = fn[f"{t}_ig_deletion"][ok], fn[f"{t}_rollout_deletion"][ok]
    print(f"     deletion IG better than rollout in {np.nanmean(a < b):.2%} of crops")
    ax.axhline(0, color="k", lw=0.4); ax.axhline(1, color="k", lw=0.4, ls=":")
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("fraction of nodes"); ax.set_ylabel("normalised score")
    ax.set_title(title, loc="left")

# (f) paired win rate vs random order (deletion AUC lower than the slide's/crop's random baseline)
ax = fig.add_subplot(gs[1, 2])
def win(a, b):
    w = (a < b).astype(float); bs = [rng.choice(w, len(w)).mean() for _ in range(2000)]
    return w.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
bars = [("IG\ncrops", win(b1[sel, cols.index("ig_deletion")], b1[sel, cols.index("random_deletion")]), GOLD),
        ("att.\ncrops", win(b1[sel, cols.index("attention_deletion")], b1[sel, cols.index("random_deletion")]), RED)]
for t, lab in [("t1", "dim"), ("t2", "logit")]:
    ok = np.abs(fn[f"{t}_dfx"]) > 1e-3
    bars.append((f"IG\nnodes\n→{lab}", win(fn[f"{t}_ig_deletion"][ok], fn[f"{t}_random_deletion"][ok]), GOLD))
    bars.append((f"roll.\nnodes\n→{lab}", win(fn[f"{t}_rollout_deletion"][ok], fn[f"{t}_random_deletion"][ok]), RED))
for q, (lab, (m, lo, hi), col) in enumerate(bars):
    ax.bar(q, 100 * m, color=col, alpha=0.85, width=0.7)
    ax.errorbar(q, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], fmt="none", color="k", lw=0.8)
ax.axhline(50, color="k", lw=0.6, ls="--"); ax.text(1.5, 46, "chance", ha="center", va="top", fontsize=6)
ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=5.5)
ax.set_ylim(0, 105); ax.set_ylabel("% beating random order\n(deletion AUC)")
ax.set_title("(f) Win rate vs random", loc="left")

handles = [plt.Line2D([], [], color=GOLD, label="Integrated Gradients"),
           plt.Line2D([], [], color=RED, label="attention (crops) / rollout (nodes)"),
           plt.Line2D([], [], color=GREY, label="random order"),
           plt.Line2D([], [], color="k", ls="-", label="deletion"),
           plt.Line2D([], [], color="k", ls="--", label="insertion")]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.14)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_faithfulness.{ext}", dpi=300)
print("wrote figures")
