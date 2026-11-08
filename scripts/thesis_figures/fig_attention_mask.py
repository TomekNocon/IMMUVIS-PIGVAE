"""Thesis figure (Sec. 3.6, open item 5): the encoder attention mask.

(a) the r=1 Manhattan ball on the grid (interior and corner node) and the CLS asymmetry;
(b) the boolean mask itself, built by the encoder's own _create_neighborhood_mask (4x4 grid
    for legibility; rows = queries, columns = keys, token 0 = CLS);
(c) the receptive field of a content node after l = 1..4 layers on the real 16x16 grid.
Palette as in the architecture figure: encoder blue, CLS red.
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.models.components.llama_graph_transformer import _create_neighborhood_mask

INK = "#2b2b2b"
BLUE, BLUE_L, BLUE_F = "#2f6db5", "#9fc0e8", "#e3edf9"
RED, RED_L = "#c8323e", "#f3c3c7"
CELL_E = "#b7c3d1"


def grid(ax, n, colour_of, x0=0, y0=0, s=1.0, lw=0.4):
    for r in range(n):
        for c in range(n):
            ax.add_patch(Rectangle((x0 + c * s, y0 + (n - 1 - r) * s), s, s, fc=colour_of(r, c),
                                   ec=CELL_E, lw=lw, zorder=2))


def arrow(ax, p0, p1, color, lw=0.7, ms=6, rad=0.0, z=5, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def panel_a(ax):
    n = 7
    qi, qc = (3, 2), (6, 6)                       # interior query, corner query
    def col(r, c):
        for (qr, qq) in (qi, qc):
            d = abs(r - qr) + abs(c - qq)
            if d == 0: return BLUE
            if d == 1: return BLUE_L
        return "white"
    grid(ax, n, col)
    # CLS token above the grid: reads every content node (one-way)
    cx, cy = n / 2, n + 3.0
    ax.add_patch(Rectangle((cx - 0.6, cy - 0.6), 1.2, 1.2, fc=RED, ec="none", zorder=4))
    ax.text(cx + 1.0, cy, "CLS", ha="left", va="center", fontsize=7, color=RED)
    for c in [0.5, 2.5, 4.5, 6.5]:
        arrow(ax, (c, n + 0.15), (cx + (c - cx) * 0.12, cy - 0.7), RED, lw=0.6, ms=5)
    ax.text(n + 0.35, n + 1.2, "CLS reads\nevery node", ha="left", va="center", fontsize=6, color=RED)
    # blocked direction: node -> CLS read is masked
    p0, p1 = (cx - 0.75, cy - 0.2), (-0.6, n - 0.5)
    arrow(ax, p0, p1, "#9a9a9a", lw=0.6, ms=5, ls=(0, (2, 1.5)), rad=0.25)
    ax.text(0.15, n + 1.55, "×", ha="center", va="center", fontsize=10, color=RED, weight="bold", zorder=6)
    ax.text(-0.9, n + 2.4, "nodes cannot\nread CLS", ha="right", va="center", fontsize=6, color="#555555")
    ax.text(-0.3, n - qi[0] - 0.5, "interior:\n4 neighbours\n+ self", ha="right", va="center", fontsize=5.8, color=BLUE)
    ax.text(n + 0.3, 0.5, "corner:\n2 neighbours\n+ self", ha="left", va="center", fontsize=5.8, color=BLUE)
    ax.set_xlim(-4.2, n + 4.2); ax.set_ylim(-0.5, n + 4.2); ax.set_aspect("equal"); ax.axis("off")


def panel_b(ax, g=4):
    m = _create_neighborhood_mask(g * g + 1, True, "cpu", 1).numpy()   # [(N+1), (N+1)], True = may attend
    code = np.where(m, 1, 0)
    code[0, :] = 2                      # CLS row (query) - reads all
    code[1:, 0] = 3                     # CLS column (key) for content queries - blocked
    np.fill_diagonal(code[1:, 1:], 4)   # self
    cmap = ListedColormap(["white", BLUE_L, RED, RED_L, BLUE])
    ax.imshow(code, cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
    n = code.shape[0]
    for k in range(n + 1):
        ax.axhline(k - 0.5, color=CELL_E, lw=0.25); ax.axvline(k - 0.5, color=CELL_E, lw=0.25)
    ax.axhline(0.5, color=INK, lw=0.7); ax.axvline(0.5, color=INK, lw=0.7)
    ax.set_xticks([0] + list(range(1, n))); ax.set_yticks([0] + list(range(1, n)))
    labels = ["CLS"] + [str(i) for i in range(1, n)]
    ax.set_xticklabels(labels, fontsize=4.2, rotation=90); ax.set_yticklabels(labels, fontsize=4.2)
    ax.tick_params(length=0, pad=1.5)
    ax.set_xlabel("key (attended to)", fontsize=6, labelpad=2); ax.set_ylabel("query (attending)", fontsize=6, labelpad=2)
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    for sp in ax.spines.values(): sp.set_visible(False)


def panel_c(axes, n=16, q=(7, 7)):
    for L, ax in enumerate(axes, start=1):
        def col(r, c):
            d = abs(r - q[0]) + abs(c - q[1])
            return BLUE if d == 0 else (BLUE_L if d <= L else "white")
        grid(ax, n, col, lw=0.2)
        k = sum(1 for r in range(n) for c in range(n) if abs(r - q[0]) + abs(c - q[1]) <= L)
        ax.text(n / 2, -0.8, f"after {L} layer{'s' if L > 1 else ''}: {k} / {n*n} nodes", ha="center",
                va="top", fontsize=5.8, color=BLUE)
        ax.set_xlim(-0.3, n + 0.3); ax.set_ylim(-3.2, n + 0.3); ax.set_aspect("equal"); ax.axis("off")


def main(out):
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "mathtext.fontset": "dejavuserif"})
    fig = plt.figure(figsize=(6.3, 4.45))
    fig.text(0.005, 0.975, r"(a) the $r=1$ Manhattan ball and the CLS asymmetry", fontsize=7.5, va="top")
    fig.text(0.545, 0.975, r"(b) the mask itself, $4\times4$ grid ($N=16$)", fontsize=7.5, va="top")
    ax_a = fig.add_axes([0.00, 0.40, 0.50, 0.55])
    ax_b = fig.add_axes([0.60, 0.395, 0.37, 0.47])
    panel_a(ax_a); panel_b(ax_b)
    items = [(BLUE, "query node (self)"), (BLUE_L, r"attends: within $r$"), (RED, "CLS reads all"),
             (RED_L, "blocked: node reads CLS"), ("white", "masked")]
    for i, (c, t) in enumerate(items):
        x = 0.03 + i * 0.195
        fig.patches.append(Rectangle((x, 0.318), 0.014, 0.022, transform=fig.transFigure, fc=c, ec=CELL_E, lw=0.4))
        fig.text(x + 0.019, 0.329, t, fontsize=5.8, va="center")
    fig.text(0.005, 0.275, r"(c) receptive field of a node after $\ell$ encoder layers ($r=1$, real $16\times16$ grid)",
             fontsize=7.5, va="center")
    axes_c = [fig.add_axes([0.03 + k * 0.245, 0.0, 0.205, 0.255]) for k in range(4)]
    panel_c(axes_c)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="thesis_figures/fig_attention_mask")
    main(ap.parse_args().out)
