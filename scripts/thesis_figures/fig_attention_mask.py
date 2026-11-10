"""The encoder attention mask (Sec. 3.6).

(a) the r=1 Manhattan ball on the grid (interior and corner node) and the CLS asymmetry;
(b) the boolean mask itself, built by the encoder's own ``_create_neighborhood_mask`` (4x4 grid for
    legibility; rows = queries, columns = keys, token 0 = CLS);
(c) the receptive field of a content node after l = 1..4 layers on the real 16x16 grid.

Usage:
    uv run python scripts/thesis_figures/fig_attention_mask.py
"""

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle

from figlib import io, paths, style
from src.models.components.llama_graph_transformer import _create_neighborhood_mask

BLOCKED = "#f3c3c7"  # node -> CLS read (masked)
CELL_EDGE = "#b7c3d1"
NOTE = "#555555"


def manhattan(r: int, c: int, centre: tuple[int, int]) -> int:
    return abs(r - centre[0]) + abs(c - centre[1])


def grid(
    ax, n: int, colour_of, x0: float = 0, y0: float = 0, s: float = 1.0, lw: float = 0.4
) -> None:
    for r in range(n):
        for c in range(n):
            ax.add_patch(
                Rectangle(
                    (x0 + c * s, y0 + (n - 1 - r) * s),
                    s,
                    s,
                    fc=colour_of(r, c),
                    ec=CELL_EDGE,
                    lw=lw,
                    zorder=2,
                )
            )


def arrow(
    ax, p0, p1, colour: str, lw: float = 0.7, ms: float = 6, rad: float = 0.0, ls="-"
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=ms,
            lw=lw,
            color=colour,
            ls=ls,
            connectionstyle=f"arc3,rad={rad}",
            zorder=5,
            shrinkA=0,
            shrinkB=0,
        )
    )


def panel_ball(ax) -> None:
    """(a) neighbourhoods of an interior and a corner query, and the one-way CLS readout."""
    n = 7
    interior, corner = (3, 2), (6, 6)

    def colour(r, c):
        for query in (interior, corner):
            d = manhattan(r, c, query)
            if d == 0:
                return style.BLUE
            if d == 1:
                return style.BLUE_LIGHT
        return "white"

    grid(ax, n, colour)
    cx, cy = n / 2, n + 3.0  # the CLS token above the grid
    ax.add_patch(Rectangle((cx - 0.6, cy - 0.6), 1.2, 1.2, fc=style.RED, ec="none", zorder=4))
    ax.text(cx + 1.0, cy, "CLS", ha="left", va="center", fontsize=7, color=style.RED)
    for c in [0.5, 2.5, 4.5, 6.5]:
        arrow(ax, (c, n + 0.15), (cx + (c - cx) * 0.12, cy - 0.7), style.RED, lw=0.6, ms=5)
    ax.text(
        n + 0.35,
        n + 1.2,
        "CLS reads\nevery node",
        ha="left",
        va="center",
        fontsize=6,
        color=style.RED,
    )
    arrow(
        ax,
        (cx - 0.75, cy - 0.2),
        (-0.6, n - 0.5),
        "#9a9a9a",
        lw=0.6,
        ms=5,
        ls=(0, (2, 1.5)),
        rad=0.25,
    )
    ax.text(
        0.15,
        n + 1.55,
        "×",
        ha="center",
        va="center",
        fontsize=10,
        color=style.RED,
        weight="bold",
        zorder=6,
    )
    ax.text(
        -0.9, n + 2.4, "nodes cannot\nread CLS", ha="right", va="center", fontsize=6, color=NOTE
    )
    ax.text(
        -0.3,
        n - interior[0] - 0.5,
        "interior:\n4 neighbours\n+ self",
        ha="right",
        va="center",
        fontsize=5.8,
        color=style.BLUE,
    )
    ax.text(
        n + 0.3,
        0.5,
        "corner:\n2 neighbours\n+ self",
        ha="left",
        va="center",
        fontsize=5.8,
        color=style.BLUE,
    )
    ax.set_xlim(-4.2, n + 4.2)
    ax.set_ylim(-0.5, n + 4.2)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_mask(ax, g: int = 4) -> None:
    """(b) the encoder's boolean mask for a g x g grid plus the CLS token."""
    allowed = _create_neighborhood_mask(
        g * g + 1, True, "cpu", 1
    ).numpy()  # [(N+1), (N+1)], True = may attend
    code = np.where(allowed, 1, 0)
    code[0, :] = 2  # CLS row (query): reads all
    code[1:, 0] = 3  # CLS column (key) for content queries: blocked
    np.fill_diagonal(code[1:, 1:], 4)  # self
    cmap = ListedColormap(["white", style.BLUE_LIGHT, style.RED, BLOCKED, style.BLUE])
    ax.imshow(code, cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
    n = code.shape[0]
    for k in range(n + 1):
        ax.axhline(k - 0.5, color=CELL_EDGE, lw=0.25)
        ax.axvline(k - 0.5, color=CELL_EDGE, lw=0.25)
    ax.axhline(0.5, color=style.INK, lw=0.7)
    ax.axvline(0.5, color=style.INK, lw=0.7)
    ticks = list(range(n))
    labels = ["CLS"] + [str(i) for i in range(1, n)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, fontsize=4.2, rotation=90)
    ax.set_yticklabels(labels, fontsize=4.2)
    ax.tick_params(length=0, pad=1.5)
    ax.set_xlabel("key (attended to)", fontsize=6, labelpad=2)
    ax.set_ylabel("query (attending)", fontsize=6, labelpad=2)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_receptive_field(axes, n: int = 16, query: tuple[int, int] = (7, 7)) -> None:
    """(c) nodes within reach after 1..4 layers."""
    for layers, ax in enumerate(axes, start=1):

        def colour(r, c, layers=layers):
            d = manhattan(r, c, query)
            return style.BLUE if d == 0 else (style.BLUE_LIGHT if d <= layers else "white")

        grid(ax, n, colour, lw=0.2)
        reach = sum(1 for r in range(n) for c in range(n) if manhattan(r, c, query) <= layers)
        ax.text(
            n / 2,
            -0.8,
            f"after {layers} layer{'s' if layers > 1 else ''}: {reach} / {n * n} nodes",
            ha="center",
            va="top",
            fontsize=5.8,
            color=style.BLUE,
        )
        ax.set_xlim(-0.3, n + 0.3)
        ax.set_ylim(-3.2, n + 0.3)
        ax.set_aspect("equal")
        ax.axis("off")


def legend(fig) -> None:
    items = [
        (style.BLUE, "query node (self)"),
        (style.BLUE_LIGHT, r"attends: within $r$"),
        (style.RED, "CLS reads all"),
        (BLOCKED, "blocked: node reads CLS"),
        ("white", "masked"),
    ]
    for i, (colour, text) in enumerate(items):
        x = 0.03 + i * 0.195
        fig.patches.append(
            Rectangle(
                (x, 0.318), 0.014, 0.022, transform=fig.transFigure, fc=colour, ec=CELL_EDGE, lw=0.4
            )
        )
        fig.text(x + 0.019, 0.329, text, fontsize=5.8, va="center")


def main(out: str) -> None:
    style.use_serif(7)
    fig = plt.figure(figsize=(6.3, 4.45))
    fig.text(
        0.005, 0.975, r"(a) the $r=1$ Manhattan ball and the CLS asymmetry", fontsize=7.5, va="top"
    )
    fig.text(0.545, 0.975, r"(b) the mask itself, $4\times4$ grid ($N=16$)", fontsize=7.5, va="top")
    panel_ball(fig.add_axes([0.00, 0.40, 0.50, 0.55]))
    panel_mask(fig.add_axes([0.60, 0.395, 0.37, 0.47]))
    legend(fig)
    fig.text(
        0.005,
        0.275,
        r"(c) receptive field of a node after $\ell$ encoder layers ($r=1$, real $16\times16$ grid)",
        fontsize=7.5,
        va="center",
    )
    panel_receptive_field([fig.add_axes([0.03 + k * 0.245, 0.0, 0.205, 0.255]) for k in range(4)])
    io.save(fig, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(paths.figure("fig_attention_mask")))
    main(ap.parse_args().out)
