"""Decision -> latent attribution (Sec. 6.4, item 17) and attribution faithfulness (Sec. 6.5, item 18).

Writes per example slide ``fig_latent_attribution_<img>`` (decision IG per crop and z_global
dimension, the decisive crop's top dimensions, and encoder-IG / rollout node maps) and
``fig_faithfulness`` (crop- and node-level deletion / insertion AUCs, example curves, win rates vs a
random order), and prints the numbers quoted in the text (means, 95% bootstrap CIs, win rates).

``--slim`` writes the thesis versions instead: ``fig_latent_attribution_slim_<img>`` without the
node maps (not validated, Sec. 6.5) and ``fig_faithfulness_slim`` with the node-level result kept
only as win-rate bars.

Data: ``figures/_cache/latent_faith/{cohort,faith_nodes,example_<img>}.npz``
(extract_latent_faithfulness.py) and the slide from ``figures/_cache/spatial_v3``.

Usage:
    uv run python scripts/thesis_figures/fig_latent_faithfulness.py
    uv run python scripts/thesis_figures/fig_latent_faithfulness.py --slim
"""

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from figlib import imc, io, paths, style
from figlib.style import GOLD, GREY, METHOD_COLOURS, RED

CACHE = paths.CACHE / "latent_faith"
P = imc.CROP
N_DIMS_SHOWN = 24
CROP_METHODS = ["ig", "attention", "random"]
BOOTSTRAP = 2000


# ---- statistics ---------------------------------------------------------------------------------
def boot_ci(rng, v) -> tuple[float, float, float]:
    """Mean and 95% bootstrap CI of the finite values of ``v``."""
    v = np.asarray(v)[np.isfinite(v)]
    means = [rng.choice(v, len(v)).mean() for _ in range(BOOTSTRAP)]
    return v.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def win_rate(rng, a, b) -> tuple[float, float, float]:
    """Fraction (with 95% bootstrap CI) where deletion AUC ``a`` beats the random baseline ``b``."""
    wins = (a < b).astype(float)
    means = [rng.choice(wins, len(wins)).mean() for _ in range(BOOTSTRAP)]
    return wins.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def node_ok(nodes, target: str) -> np.ndarray:
    """Crops whose target actually moves between baseline and input."""
    return np.abs(nodes[f"{target}_dfx"]) > 1e-3


# ---- item 17: decision -> latent ------------------------------------------------------------------
def crop_composite(slide, markers, r0, c0) -> np.ndarray:
    out = 0
    for m in style.SLIDE_COMPOSITE:
        j = markers.index(m)
        out = out + imc.stretch(slide[j, r0 : r0 + P, c0 : c0 + P], ref=slide[j])[
            ..., None
        ] * imc.rgb(style.MARKER_COLOURS[m])
    return np.clip(0.6 * out, 0, 1)


def plot_dim_matrix(fig, ax, ex, image: str, slim: bool) -> None:
    """(a) decision IG per tissue crop (sorted) and z_global dimension (top 24 by |IG| on the slide)."""
    attribution, empty, decisive = ex["M"], ex["empty"], int(ex["decisive"])
    real = np.where(~empty)[0]
    crop_order = real[np.argsort(-attribution[real].sum(1))]
    dims = np.argsort(-np.abs(attribution[real]).sum(0))[:N_DIMS_SHOWN]
    sub = attribution[np.ix_(crop_order, dims)]
    v = np.abs(sub).max()
    im = ax.imshow(sub, aspect="auto", cmap="PuOr_r", vmin=-v, vmax=v, interpolation="nearest")
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, rotation=90, fontsize=5.5)
    row = int(np.where(crop_order == decisive)[0][0])
    if slim:
        ax.set_ylabel(f"tissue crops (n={len(real)}), sorted by total IG")
    else:
        ax.set_yticks([])
        ax.set_ylabel(f"tissue crops (n={len(real)}), sorted by decision IG")
    ax.set_xlabel("z_global dimension (top 24 by |IG| on this slide)")
    ax.add_patch(Rectangle((-0.5, row - 0.5), len(dims), 1, fill=False, ec="k", lw=0.9))
    if slim:
        ax.set_yticks([row])
        ax.set_yticklabels(["decisive\ncrop"], fontsize=6)
    ax.set_title(f"(a) Decision IG per crop and latent dimension, slide {image}", loc="left")
    if slim:
        pred = "squamous" if ex["logit"] > 0 else "adeno"
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_label(f"IG toward prediction ({pred})", fontsize=6.5)
    else:
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("IG toward prediction", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)


def plot_top_dims(ax, ex, slim: bool) -> float:
    """(b) the decisive crop's 10 dimensions with the largest |IG|; returns their share of |IG|."""
    per_dim = ex["M_dec"]
    order = np.argsort(-np.abs(per_dim))[:10]
    scale = 100 if slim else 1
    ax.barh(
        range(len(order)),
        scale * per_dim[order],
        color=[GOLD if per_dim[j] >= 0 else GREY for j in order],
        height=0.75,
    )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"dim {j}" for j in order])
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    share = np.abs(per_dim[order]).sum() / np.abs(per_dim).sum()
    n_dims = ex["M"].shape[1]
    if slim:
        ax.set_title("(b) Decisive crop: top 10 dims", loc="left")
        ax.set_xlabel(
            f"IG toward prediction (×10⁻²)\nall {n_dims} dims sum to {per_dim.sum():+.2f};\n"
            f"top 10 = {share:.0%} of |IG|"
        )
    else:
        ax.set_xlabel("IG toward prediction")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.set_title(
            f"(b) Decisive crop: top dims\n(sum of all {n_dims}: {per_dim.sum():+.2f})", loc="left"
        )
    return share


def plot_node_maps(fig, cells, ex, slide, markers) -> None:
    """(c) the decisive crop and encoder-IG node maps of its top-3 decision dims, plus rollout."""
    r0, c0 = ex["coords"][int(ex["decisive"])]
    ax = fig.add_subplot(cells[0])
    ax.imshow(crop_composite(slide, markers, r0, c0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("decisive crop", fontsize=7)
    ax.text(
        0,
        1.32,
        "(c) Which grid nodes write the top-3 supporting dimensions (encoder IG, 16×16 nodes)",
        transform=ax.transAxes,
        fontsize=8,
    )
    maps = [(f"dim {d}", ex["ig_t1"][j], "PuOr_r") for j, d in enumerate(ex["top_dims"])]
    maps.append(("rollout (CLS)", ex["rollout"], "Reds"))
    for t, (title, node_map, cmap) in enumerate(maps):
        ax = fig.add_subplot(cells[t + 1])
        grid = node_map.reshape(16, 16)
        if cmap == "Reds":
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=grid.max())
        else:
            v = np.abs(grid).max()
            ax.imshow(grid, cmap=cmap, vmin=-v, vmax=v)
        ax.set_title(title, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])


def latent_attribution(example_file, slim: bool) -> None:
    ex = np.load(example_file)
    image = str(ex["img"])[:-5]
    per_dim, decisive = ex["M_dec"], int(ex["decisive"])
    if slim:
        fig = plt.figure(figsize=(6.3, 3.3))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.55)
        plot_dim_matrix(fig, fig.add_subplot(gs[0]), ex, image, slim)
        share = plot_top_dims(fig.add_subplot(gs[1]), ex, slim)
        fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.25)
        io.save(fig, paths.figure(f"fig_latent_attribution_slim_{image}"), pdf_dpi=300)
        cumulative = np.cumsum(np.sort(np.abs(per_dim))[::-1]) / np.abs(per_dim).sum()
        print(
            f"[17] {image}: decisive crop sum {per_dim.sum():+.3f}, top-10 share {share:.1%}, "
            f"dims needed for 50% of |IG|: {int(np.searchsorted(cumulative, 0.5)) + 1}"
        )
        return

    spatial = np.load(next((paths.CACHE / "spatial_v3").glob(f"*_{image}.npz")))
    fig = plt.figure(figsize=(6.3, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.45)
    top = gs[0].subgridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.6)
    plot_dim_matrix(fig, fig.add_subplot(top[0]), ex, image, slim)
    plot_top_dims(fig.add_subplot(top[1]), ex, slim)
    plot_node_maps(
        fig,
        gs[1].subgridspec(1, 5, wspace=0.08),
        ex,
        spatial["slide"].astype(np.float32),
        list(spatial["markers"]),
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.04)
    io.save(fig, paths.figure(f"fig_latent_attribution_{image}"), pdf_dpi=300)
    top_dims = ex["top_dims"]
    corrs = [np.corrcoef(ex["ig_t1"][j], ex["rollout"])[0, 1] for j in range(len(top_dims))]
    print(
        f"[17] {image}: decisive crop {decisive}, top dims {list(top_dims)} IG {np.round(per_dim[top_dims], 3)}; "
        f"corr(encoder IG, rollout) {np.round(corrs, 2)}"
    )


def print_cohort_dims(cohort) -> None:
    """Are the decision dimensions shared across slides?"""
    totals, y, classes = cohort["dimtot"], cohort["y"], list(cohort["classes"])
    pred = (cohort["logit"] > 0).astype(int)
    for c in [0, 1]:
        correct = (y == c) & (pred == c)
        mean = totals[correct].mean(0)
        order = np.argsort(-mean)[:8]
        positive = (totals[correct][:, order] > 0).mean(0)
        print(
            f"[17] cohort {classes[c]} (n={correct.sum()} correct): top decision dims {list(order)} "
            f"mean IG {np.round(mean[order], 2)} positive in {np.round(positive, 2)} of slides"
        )
    print(
        f"[17] ABMIL IG completeness: median rel err {np.median(cohort['comp_err']):.2e}, "
        f"max {cohort['comp_err'].max():.2e}"
    )


# ---- item 18: faithfulness ------------------------------------------------------------------------
def crop_aucs(cohort):
    """Crop-level AUC table, its column names, and the slides with a clear decision (|logit| > 1)."""
    return cohort["b1"], list(cohort["b1_cols"]), np.abs(cohort["logit"]) > 1


def plot_crop_auc(ax, rng, cohort, mode: str, letter: str, slim: bool) -> None:
    """Crop-level deletion or insertion AUC per method (boxes) with mean and bootstrap CI."""
    table, cols, clear = crop_aucs(cohort)
    values = []
    for q, method in enumerate(CROP_METHODS):
        v = table[clear, cols.index(f"{method}_{mode}")]
        v = v[np.isfinite(v)]
        values.append(v)
        m, lo, hi = boot_ci(rng, v)
        ax.boxplot(
            v,
            positions=[q],
            widths=0.55,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": METHOD_COLOURS[method], "alpha": 0.35, "lw": 0.6},
            medianprops={"color": "k", "lw": 1},
            whiskerprops={"lw": 0.6},
            capprops={"lw": 0.6},
        )
        ax.errorbar(q, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=3, color="k", lw=1)
        if slim:
            ax.text(q + 0.33, m, f"{m:.2f}", va="center", fontsize=6)
            print(f"{mode:9s} {method:9s} mean AUC {m:.3f} [{lo:.3f}, {hi:.3f}]")
        else:
            print(
                f"     {mode:9s} {method:9s} mean AUC {m:.3f} [{lo:.3f}, {hi:.3f}]  median {np.median(v):.3f}"
            )
    if not slim:
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            better = (values[a] < values[b]) if mode == "deletion" else (values[a] > values[b])
            print(
                f"     {mode:9s} {CROP_METHODS[a]} better than {CROP_METHODS[b]} in {better.mean():.2%} of slides"
            )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["IG", "attention", "random"])
    if slim:
        ax.set_xlim(-0.5, 2.75)
    ax.set_ylim(np.percentile(np.concatenate(values), [1, 99]) + np.array([-0.1, 0.1]))
    ax.set_ylabel(f"{mode} AUC ({'lower' if mode == 'deletion' else 'higher'} = better)")
    title = f"({letter}) Crop {mode}, {clear.sum()} slides" if slim else f"({letter}) Crops, {mode}"
    ax.set_title(title, loc="left")


def plot_example_curves(ax, example, slim: bool) -> None:
    """Deletion / insertion curves over crops for slide 7064."""
    for method, key in [("ig", "ig"), ("attention", "attention"), ("random", "random0")]:
        for mode, ls in [("deletion", "-"), ("insertion", "--")]:
            curve = example[f"crop_curve_{key}_{mode}"]
            ax.plot(
                np.linspace(0, 1, len(curve)),
                curve,
                ls=ls,
                color=METHOD_COLOURS[method],
                lw=1.1 if slim else 1,
            )
    ax.axhline(0, color="k", lw=0.4)
    ax.axhline(1, color="k", lw=0.4, ls=":")
    if slim:
        ax.set_xlabel("fraction of crops removed / inserted")
        ax.set_ylabel("normalised score\n(1 = full slide, 0 = baseline)")
        ax.set_title("(c) Example curves, slide 7064", loc="left")
    else:
        ax.set_xlabel("fraction of crops")
        ax.set_ylabel("normalised score")
        ax.set_title("(c) Slide 7064 curves", loc="left")


def plot_node_curves(ax, rng, nodes, target: str, title: str) -> None:
    """Median normalised node-level curves (IQR band for deletion) for one target."""
    ok = node_ok(nodes, target)
    print(
        f"[18] node level {target}: n={ok.sum()} crops, median IG completeness rel err "
        f"{np.median(nodes[f'{target}_comp_err'][ok]):.3f}, median corr(IG, rollout) "
        f"{np.nanmedian(nodes[f'corr_ig_roll_{target}'][ok]):+.2f}"
    )
    for method in ["ig", "rollout", "random"]:
        for mode, ls in [("deletion", "-"), ("insertion", "--")]:
            curves = nodes[f"{target}_{method}_{mode}_curve"][ok]
            x = np.linspace(0, 1, curves.shape[1])
            ax.plot(x, np.nanmedian(curves, 0), ls=ls, color=METHOD_COLOURS[method], lw=1)
            if mode == "deletion":
                ax.fill_between(
                    x,
                    np.nanpercentile(curves, 25, 0),
                    np.nanpercentile(curves, 75, 0),
                    color=METHOD_COLOURS[method],
                    alpha=0.12,
                    lw=0,
                )
            aucs = nodes[f"{target}_{method}_{mode}"][ok]
            m, lo, hi = boot_ci(rng, aucs)
            print(
                f"     {mode:9s} {method:8s} mean AUC {m:.3f} [{lo:.3f}, {hi:.3f}]  median {np.nanmedian(aucs):.3f}"
            )
        if method != "random":
            for mode in ["deletion", "insertion"]:
                a, b = nodes[f"{target}_{method}_{mode}"][ok], nodes[f"{target}_random_{mode}"][ok]
                wins = (a < b) if mode == "deletion" else (a > b)
                print(
                    f"     {mode:9s} {method} better than random in {np.nanmean(wins):.2%} of crops"
                )
    a, b = nodes[f"{target}_ig_deletion"][ok], nodes[f"{target}_rollout_deletion"][ok]
    print(f"     deletion IG better than rollout in {np.nanmean(a < b):.2%} of crops")
    ax.axhline(0, color="k", lw=0.4)
    ax.axhline(1, color="k", lw=0.4, ls=":")
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("fraction of nodes")
    ax.set_ylabel("normalised score")
    ax.set_title(title, loc="left")


def win_rate_bars(rng, cohort, nodes, slim: bool) -> list:
    table, cols, clear = crop_aucs(cohort)
    random = table[clear, cols.index("random_deletion")]
    bars = [
        (
            "IG" if slim else "IG\ncrops",
            win_rate(rng, table[clear, cols.index("ig_deletion")], random),
            GOLD,
        ),
        (
            "attention" if slim else "att.\ncrops",
            win_rate(rng, table[clear, cols.index("attention_deletion")], random),
            RED,
        ),
    ]
    for target, short in [("t1", "dim"), ("t2", "logit")]:
        ok = node_ok(nodes, target)
        rnd = nodes[f"{target}_random_deletion"][ok]
        bars.append((
            f"IG\n→{short}" if slim else f"IG\nnodes\n→{short}",
            win_rate(rng, nodes[f"{target}_ig_deletion"][ok], rnd),
            GOLD,
        ))
        bars.append((
            f"rollout\n→{short}" if slim else f"roll.\nnodes\n→{short}",
            win_rate(rng, nodes[f"{target}_rollout_deletion"][ok], rnd),
            RED,
        ))
    return bars


def plot_win_rates(ax, rng, cohort, nodes, slim: bool) -> None:
    """Share of slides / crops where a method's deletion AUC beats a random order."""
    bars = win_rate_bars(rng, cohort, nodes, slim)
    xs = np.r_[0, 1, 2.6, 3.6, 4.6, 5.6] if slim else np.arange(len(bars))
    for x, (label, (m, lo, hi), colour) in zip(xs, bars, strict=True):
        ax.bar(x, 100 * m, color=colour, alpha=0.85, width=0.75 if slim else 0.7)
        ax.errorbar(
            x, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], fmt="none", color="k", lw=0.8
        )
        if slim:
            ax.text(
                x,
                100 * hi + 2,
                f"{100 * m:.1f}" if m > 0.995 else f"{100 * m:.0f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
            print(f"win rate {label!r}: {m:.1%} [{lo:.1%}, {hi:.1%}]")
    ax.axhline(50, color="k", lw=0.6, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=6 if slim else 5.5)
    if slim:
        ax.text(7.25, 51.5, "chance", ha="right", va="bottom", fontsize=6)
        ax.axvline(1.8, color="#cccccc", lw=0.6)
        for x, text in [(0.5, "crop level"), (4.1, "node level (within a crop)")]:
            ax.text(
                x,
                -0.24,
                text,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6.5,
                fontweight="bold",
            )
        ax.set_xlim(-0.6, 7.3)
        ax.set_ylim(0, 112)
        ax.set_ylabel("% beating a random order\n(deletion AUC)")
        ax.set_title("(d) Better than random?", loc="left")
    else:
        ax.text(1.5, 46, "chance", ha="center", va="top", fontsize=6)
        ax.set_ylim(0, 105)
        ax.set_ylabel("% beating random order\n(deletion AUC)")
        ax.set_title("(f) Win rate vs random", loc="left")


def method_legend(fig, slim: bool) -> None:
    lw = {"lw": 2} if slim else {}
    handles = [
        plt.Line2D([], [], color=GOLD, label="Integrated Gradients", **lw),
        plt.Line2D([], [], color=RED, label="attention (crops) / rollout (nodes)", **lw),
        plt.Line2D([], [], color=GREY, label="random order", **lw),
        plt.Line2D([], [], color="k", ls="-", label="deletion"),
        plt.Line2D([], [], color="k", ls="--", label="insertion"),
    ]
    if slim:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
            handlelength=1.8,
            columnspacing=1.0,
        )
    else:
        fig.legend(
            handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01)
        )


def faithfulness(cohort, nodes, example, slim: bool) -> None:
    rng = np.random.default_rng(0)
    if slim:
        fig = plt.figure(figsize=(6.3, 4.9))
        gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.38, width_ratios=[1, 1.15])
        for j, mode in enumerate(["deletion", "insertion"]):
            plot_crop_auc(fig.add_subplot(gs[0, j]), rng, cohort, mode, "ab"[j], slim)
        plot_example_curves(fig.add_subplot(gs[1, 0]), example, slim)
        plot_win_rates(fig.add_subplot(gs[1, 1]), rng, cohort, nodes, slim)
        method_legend(fig, slim)
        fig.subplots_adjust(left=0.1, right=0.97, top=0.94, bottom=0.17)
        io.save(fig, paths.figure("fig_faithfulness_slim"), pdf_dpi=300)
        return

    fig = plt.figure(figsize=(6.3, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.62, wspace=0.42)
    _, _, clear = crop_aucs(cohort)
    print(f"[18] crop level, {clear.sum()} / {len(clear)} slides with |logit| > 1")
    for j, mode in enumerate(["deletion", "insertion"]):
        plot_crop_auc(fig.add_subplot(gs[0, j]), rng, cohort, mode, "ab"[j], slim)
    plot_example_curves(fig.add_subplot(gs[0, 2]), example, slim)
    for j, (target, title) in enumerate([
        ("t1", "(d) Nodes → top decision dim"),
        ("t2", "(e) Nodes → decision logit"),
    ]):
        plot_node_curves(fig.add_subplot(gs[1, j]), rng, nodes, target, title)
    plot_win_rates(fig.add_subplot(gs[1, 2]), rng, cohort, nodes, slim)
    method_legend(fig, slim)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.14)
    io.save(fig, paths.figure("fig_faithfulness"), pdf_dpi=300)


def main(slim: bool) -> None:
    style.use_sans(7.5, title=8, label=7.5, tick=6.5, legend=6.5)
    cohort = np.load(CACHE / "cohort.npz")
    for example_file in sorted(CACHE.glob("example_*.npz")):
        latent_attribution(example_file, slim)
    if not slim:
        print_cohort_dims(cohort)
    faithfulness(
        cohort, np.load(CACHE / "faith_nodes.npz"), np.load(CACHE / "example_7064.npz"), slim
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slim", action="store_true", help="write the slim thesis versions")
    main(ap.parse_args().slim)
