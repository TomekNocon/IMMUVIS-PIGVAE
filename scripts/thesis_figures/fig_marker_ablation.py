"""Which of the 43 IMC markers drive the ABMIL decision (Sec. 6.3 deep-dive).

Panels for one patient: slide composite with the decisive real-tissue crop boxed, the per-crop ABMIL
IG decision map (empty crops hatched), the top markers inside the decisive crop, slide-level marker
effects coloured by marker group, and cohort group effects (adenocarcinoma vs squamous, one dot per
slide). ``--with-attention`` adds the head-averaged ABMIL attention map next to the IG map, direction
labels and a group legend on the marker-effect panel.

Effects are oriented to the predicted class: > 0 means removing the marker weakens the call.
Data: ``figures/_cache/spatial_v3/<slide>.npz`` (extract_spatial_attribution.py) and
``figures/_cache/marker_ablation/*.npz`` (extract_marker_ablation.py).

Usage:
    uv run python scripts/thesis_figures/fig_marker_ablation.py --slide adeno_7064
    uv run python scripts/thesis_figures/fig_marker_ablation.py --slide squam_6577 --with-attention
"""

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

from figlib import imc, io, paths, style

P = imc.CROP
N_TILES, N_TOP, N_BOTTOM = 5, 10, 4
NOTE = "#555555"
GROUP_OF = {m: g for g, ms in style.MARKER_GROUPS.items() for m in ms}


def short(marker: str) -> str:
    return marker.replace("Carbonic Anhydrase", "Carb. anhydrase")


def crop_map(shape, coords, values, keep=None) -> np.ndarray:
    """Paint one value per 128-px crop onto a slide-sized grid (NaN outside / where not kept)."""
    grid = np.full(shape, np.nan)
    for j, (a, b) in enumerate(coords):
        grid[a : a + P, b : b + P] = values[j] if keep is None or keep[j] else np.nan
    return grid


def hatch_empty(ax, coords, empty, colour: str) -> None:
    for j, (a, b) in enumerate(coords):
        if empty[j]:
            ax.add_patch(
                Rectangle((b - 0.5, a - 0.5), P, P, fill=False, hatch="////", ec=colour, lw=0)
            )


def slide_axes(ax, slide) -> None:
    ax.set_xlim(-0.5, slide.shape[2] - 0.5)
    ax.set_ylim(slide.shape[1] - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


class Data:
    """Spatial attribution + marker ablation of one slide."""

    def __init__(self, slide_tag: str):
        self.sp = np.load(paths.CACHE / "spatial_v3" / f"{slide_tag}.npz")
        self.ab = np.load(paths.CACHE / "marker_ablation" / f"{slide_tag.split('_')[1]}.npz")
        if not (self.sp["coords"] == self.ab["coords"]).all():
            raise ValueError(f"{slide_tag}: spatial and ablation caches list different crops")
        self.markers = list(self.ab["markers"])
        self.slide = self.sp["slide"].astype(np.float32)
        self.coords, self.empty, self.ig = self.sp["coords"], self.sp["empty"], self.sp["ig"]
        self.decisive = int(self.sp["decisive"])
        self.r0, self.c0 = self.coords[self.decisive]
        self.logit = self.sp["logit"]
        self.pred = "squamous" if self.logit > 0 else "adenocarcinoma"
        self.label = str(self.ab["label"])
        self.image = str(self.ab["img"])[:-5]


def panel_slide(ax, d: Data, attention: bool) -> None:
    """(a) false-colour composite with the decisive crop boxed and a 100 um scale bar."""
    composite = sum(
        imc.tint(d.slide[d.markers.index(m)], style.MARKER_COLOURS[m])
        for m in style.SLIDE_COMPOSITE
    )
    ax.imshow(np.clip(0.55 * composite, 0, 1))
    ax.add_patch(Rectangle((d.c0 - 0.5, d.r0 - 0.5), P, P, fill=False, ec="white", lw=1.6))
    ax.set_xticks([])
    ax.set_yticks([])
    call = f"{d.label.split()[0].lower()}"
    if attention:
        ax.set_title(f"(a) Slide {d.image}", loc="left")
        ax.text(
            0.5,
            -0.03,
            f"{call}, logit {d.logit:+.1f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.5,
        )
    else:
        ax.set_title(f"(a) Slide {d.image}, {call} (logit {d.logit:+.1f})", loc="left")
    for i, m in enumerate(style.SLIDE_COMPOSITE):
        ax.text(
            0.01,
            0.99 - 0.055 * i,
            m,
            color=style.MARKER_COLOURS[m],
            transform=ax.transAxes,
            fontsize=6.5,
            va="top",
            fontweight="bold",
        )
    height, width = d.slide.shape[1:]
    ax.plot([width - 120, width - 20], [height - 25] * 2, color="white", lw=2)
    ax.text(width - 70, height - 35, "100 µm", color="white", ha="center", fontsize=6)


def panel_ig_map(fig, ax, d: Data, attention: bool) -> None:
    """(b) per-crop ABMIL IG toward the prediction; empty crops hatched."""
    real = ~d.empty
    im = ax.imshow(
        crop_map(d.slide.shape[1:], d.coords, d.ig, keep=real),
        cmap="cividis",
        vmin=0,
        vmax=np.abs(d.ig[real]).max(),
    )
    hatch_empty(ax, d.coords, d.empty, "#bbbbbb")
    ax.add_patch(Rectangle((d.c0 - 0.5, d.r0 - 0.5), P, P, fill=False, ec=style.RED, lw=1.6))
    slide_axes(ax, d.slide)
    if attention:
        ax.set_title("(b) ABMIL IG per crop", loc="left")
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03)
        cbar.set_label("IG toward prediction", fontsize=6.5, labelpad=1)
        cbar.ax.tick_params(labelsize=5.5)
    else:
        ax.set_title("(b) Decision map (ABMIL IG per crop)", loc="left")
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label("IG toward prediction", fontsize=6.5, labelpad=2)
        cbar.ax.tick_params(labelsize=6)


def panel_attention_map(fig, ax, d: Data) -> None:
    """(c) head-averaged ABMIL attention per crop (empty crops hatched but coloured)."""
    att = d.sp["att"]
    im = ax.imshow(crop_map(d.slide.shape[1:], d.coords, att), cmap="Reds", vmin=0, vmax=att.max())
    hatch_empty(ax, d.coords, d.empty, NOTE)
    ax.add_patch(Rectangle((d.c0 - 0.5, d.r0 - 0.5), P, P, fill=False, ec="k", lw=1.4, ls="--"))
    top = int(np.argmax(att))
    ta, tb = d.coords[top]
    ax.add_patch(Rectangle((tb - 0.5, ta - 0.5), P, P, fill=False, ec=style.BLUE, lw=1.6))
    slide_axes(ax, d.slide)
    ax.set_title("(c) ABMIL attention per crop", loc="left")
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03)
    cbar.set_label("attention (sums to 1)", fontsize=6.5, labelpad=1)
    cbar.ax.tick_params(labelsize=5.5)
    rank = int((att > att[d.decisive]).sum()) + 1
    print(f"attention: decisive crop rank {rank}, max crop {top} empty={bool(d.empty[top])}")


def panel_crop_tiles(fig, cells, d: Data, letter: str) -> np.ndarray:
    """Markers whose removal from the decisive crop alone weakens the call most."""
    effect = d.ab["crop_abl"][d.decisive]
    order = np.argsort(-effect)[:N_TILES]
    for t, mi in enumerate(order):
        ax = fig.add_subplot(cells[t])
        marker = d.markers[mi]
        colour = style.GROUP_COLOURS[GROUP_OF[marker]]
        ax.imshow(imc.tint(d.slide[mi, d.r0 : d.r0 + P, d.c0 : d.c0 + P], colour, ref=d.slide[mi]))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(colour)
            spine.set_linewidth(1.2)
        ax.set_title(f"{short(marker)}\n{effect[mi]:+.2f}", fontsize=7, color=colour)
        if t == 0:
            ax.text(
                0,
                1.42,
                f"({letter}) Markers whose removal (in this crop only) weakens the call most",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
            )
    return order


def direction_labels(ax, d: Data) -> None:
    """Arrows above the bars: left of 0 supports the other class, right supports the prediction."""
    other = "adeno" if d.pred == "squamous" else "squamous"
    predicted = "squamous" if d.pred == "squamous" else "adeno"
    xmin, xmax = ax.get_xlim()
    span = xmax - xmin
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    arrowprops = {"arrowstyle": "->", "color": NOTE, "lw": 0.8}
    ax.annotate(
        "",
        xy=(xmin + 0.02 * span, 1.035),
        xytext=(-0.05 * span, 1.035),
        xycoords=trans,
        arrowprops=arrowprops,
    )
    ax.annotate(
        "",
        xy=(xmax - 0.02 * span, 1.035),
        xytext=(0.05 * span, 1.035),
        xycoords=trans,
        arrowprops=arrowprops,
    )
    ax.text(
        -0.06 * span,
        1.06,
        f"marker supports {other}",
        transform=trans,
        ha="right",
        va="bottom",
        fontsize=6,
        color=NOTE,
    )
    ax.text(
        0.06 * span,
        1.06,
        f"marker supports {predicted}",
        transform=trans,
        ha="left",
        va="bottom",
        fontsize=6,
        color=NOTE,
    )
    ax.set_xlim(xmin, xmax)


def panel_slide_effects(ax, d: Data, letter: str, attention: bool) -> None:
    """Slide-level marker effects (marker removed from every crop), top and bottom markers."""
    effect = d.ab["slide_abl"]
    order = np.argsort(-effect)
    selected = np.r_[order[:N_TOP], order[-N_BOTTOM:]]
    y = np.r_[np.arange(N_TOP), np.arange(N_TOP, N_TOP + N_BOTTOM) + 0.8]
    ax.barh(
        y,
        effect[selected],
        color=[style.GROUP_COLOURS[GROUP_OF[d.markers[j]]] for j in selected],
        height=0.75,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([short(d.markers[j]) for j in selected])
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    if attention:
        ax.set_xlabel("Δ logit, marker removed from all crops")
        ax.set_title(f"({letter}) Marker effects, slide {d.image}", loc="left", pad=22)
        direction_labels(ax, d)
        ax.legend(
            handles=[
                Rectangle((0, 0), 1, 1, color=c, label=style.GROUP_SHORT[g])
                for g, c in style.GROUP_COLOURS.items()
            ],
            title="marker group",
            title_fontsize=6.5,
            loc="lower right",
            frameon=False,
            handlelength=1.0,
            handleheight=0.8,
            labelspacing=0.3,
            borderaxespad=0.2,
        )
    else:
        ax.set_xlabel("Δ logit toward prediction\n(marker removed from all crops)")
        ax.set_title(f"({letter}) Marker effects, slide {d.image}", loc="left", pad=16)


def panel_cohort(ax, d: Data, letter: str) -> None:
    """Group effects across the cohort of correctly classified slides, one dot per slide."""
    cohort = [np.load(f) for f in sorted((paths.CACHE / "marker_ablation").glob("*.npz"))]
    groups = list(d.ab["group_names"])
    rng = np.random.default_rng(0)
    classes = list(style.CLASS_COLOURS)
    for gi, g in enumerate(groups):
        for ci, cls in enumerate(classes):
            v = np.array([s["group_abl"][gi] for s in cohort if str(s["label"]) == cls])
            x = gi + (ci - 0.5) * 0.36
            ax.scatter(
                x + rng.uniform(-0.07, 0.07, len(v)),
                v,
                s=7,
                color=style.CLASS_COLOURS[cls],
                alpha=0.75,
                lw=0,
            )
            ax.hlines(v.mean(), x - 0.13, x + 0.13, color="k", lw=1.2)
        ax.axvspan(gi - 0.5, gi + 0.5, color=style.GROUP_COLOURS[g], alpha=0.07, lw=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([style.GROUP_SHORT[g] for g in groups], rotation=30, ha="right")
    for tick, g in zip(ax.get_xticklabels(), groups, strict=True):
        tick.set_color(style.GROUP_COLOURS[g])
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylabel("Δ logit toward prediction\n(marker group removed)")
    n_per_class = {cls: sum(str(s["label"]) == cls for s in cohort) for cls in classes}
    ax.set_title(f"({letter}) Cohort, correctly classified slides", loc="left", pad=16)
    ax.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                ls="",
                marker="o",
                ms=4,
                color=style.CLASS_COLOURS[cls],
                label=f"{cls.split()[0].lower()} (n={n_per_class[cls]})",
            )
            for cls in classes
        ],
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=2,
        frameon=False,
        handletextpad=0.2,
        columnspacing=0.8,
    )


def main(slide_tag: str, attention: bool, out: str | None) -> None:
    style.use_sans(7.5, title=8, label=7.5, tick=6.5, legend=6.5)
    d = Data(slide_tag)
    fig = plt.figure(figsize=(6.3, 7.7 if attention else 7.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.35, 1.25, 2.6], hspace=0.55 if attention else 0.38)
    top = gs[0].subgridspec(1, 3 if attention else 2, wspace=0.12 if attention else 0.08)
    middle = gs[1].subgridspec(1, N_TILES, wspace=0.06)
    bottom = gs[2].subgridspec(1, 2, wspace=0.45, width_ratios=[1.0, 1.15])
    letters = iter("cdef" if attention else "cde")

    panel_slide(fig.add_subplot(top[0]), d, attention)
    panel_ig_map(fig, fig.add_subplot(top[1]), d, attention)
    if attention:
        panel_attention_map(fig, fig.add_subplot(top[2]), d)
        next(letters)
    order = panel_crop_tiles(fig, middle, d, next(letters))
    panel_slide_effects(fig.add_subplot(bottom[0]), d, next(letters), attention)
    panel_cohort(fig.add_subplot(bottom[1]), d, next(letters))

    fig.subplots_adjust(left=0.12, right=0.93, top=0.95, bottom=0.08)
    name = (
        f"fig_marker_ablation_att_{slide_tag}" if attention else f"fig_marker_ablation_{slide_tag}"
    )
    io.save(fig, out or paths.figure(name), pdf_dpi=300)
    crop_effect = d.ab["crop_abl"][d.decisive]
    print(
        f"decisive crop {d.decisive} {(d.r0, d.c0)} | top in crop",
        [(d.markers[j], round(crop_effect[j], 2)) for j in order],
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--slide", default="adeno_7064", help="<class>_<image> of a spatial_v3 cache file"
    )
    ap.add_argument(
        "--with-attention", action="store_true", help="add the ABMIL attention map panel"
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output stem (default figures/fig_marker_ablation[_att]_<slide>)",
    )
    a = ap.parse_args()
    main(a.slide, a.with_attention, a.out)
