"""The PIGVAE encoder-decoder over the grid graph (Figure 3.1).

Drawn from ``src/models/components/modules.py`` (GraphAE, GraphEncoder, NodeBottleneckEncoder,
SimplePermuter, GraphDecoder, FiLMConditioner) and ``configs/model/model.yaml``. Encoder and decoder
are containers with explicit layer boxes, tokens are stacked-cell vectors, FiLM is
(1 + gamma) * norm(x) + beta. Input / output grids show -PC2 of the real node features of the example
crop (tracks panCK) on a black -> gold ramp; the output is the trained model's reconstruction
(extract_reconstruction.py).

Usage:
    uv run python scripts/thesis_figures/fig_architecture.py
"""

import argparse
from dataclasses import dataclass

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from figlib import imc, io, paths, style
from figlib.nodes import PC
from figlib.style import BLUE, BLUE_FILL, BLUE_LIGHT, GOLD, GOLD_FILL, GREY, INK, RED

DASH = (0, (3, 2))
NOTE = "#555555"
GRID_EDGE = "#9aa9bb"
X_RANGE, Y_RANGE = (-4.5, 161), (-6, 61)
Y_NODES, Y_GLOBAL = 20, 50  # centre of the node path and of the global path


# ---- drawing primitives ------------------------------------------------------------------------
def box(ax, x, y, w, h, fc, ec, lw=0.7, ls="-", r=0.6, z=3) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            fc=fc,
            ec=ec,
            lw=lw,
            ls=ls,
            zorder=z,
        )
    )


def arrow(ax, p0, p1, color=INK, lw=0.8, ls="-", rad=0.0, z=5, ms=7) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=ms,
            lw=lw,
            color=color,
            ls=ls,
            connectionstyle=f"arc3,rad={rad}",
            zorder=z,
            shrinkA=0,
            shrinkB=0,
        )
    )


def line(ax, xs, ys, color=INK, lw=0.8, ls="-", z=5) -> None:
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")


def vec(ax, x, y, color, n=4, cw=1.6, ch=1.25) -> None:
    """A token drawn as a column vector of n stacked cells; (x, y) = bottom-left."""
    for k in range(n):
        ax.add_patch(Rectangle((x, y + k * ch), cw, ch, fc=color, ec="white", lw=0.5, zorder=6))


def grid_glyph(ax, x, y, w, colours, dot, edges=None) -> None:
    n = colours.shape[0]
    s = w / (n - 1)
    xs, ys = x + np.arange(n) * s, y + w - np.arange(n) * s
    if edges:
        for k in range(n):
            line(ax, [xs[0], xs[-1]], [ys[k]] * 2, color=edges, lw=0.4, z=2)
            line(ax, [xs[k]] * 2, [ys[0], ys[-1]], color=edges, lw=0.4, z=2)
    gx, gy = np.meshgrid(xs, ys)
    ax.scatter(gx.ravel(), gy.ravel(), s=dot, c=colours.reshape(-1, 3), ec="none", zorder=3)


def node_grid(ax, x, y, w, color, n=6, dot=5.0, edge=GRID_EDGE) -> None:
    """A uniform n x n grid graph (one token per node)."""
    grid_glyph(ax, x, y, w, np.tile(imc.rgb(color), (n, n, 1)), dot, edges=edge)


def layer(ax, x, y, w, h, text, ec, fs=5.8, ls="-", fc="white") -> None:
    box(ax, x, y, w, h, fc, ec, lw=0.6, ls=ls, r=0.5, z=3)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        zorder=4,
        linespacing=1.12,
    )


def mask_stencil(ax, x, y, w) -> None:
    """Encoder attention mask: query cell (dark), 4 neighbours (light); CLS reads all nodes."""
    n = 5
    c = w / n
    for r in range(n):
        for k in range(n):
            d = abs(r - 2) + abs(k - 2)
            fc = BLUE if d == 0 else (BLUE_LIGHT if d == 1 else "white")
            ax.add_patch(
                Rectangle((x + k * c, y + r * c), c, c, fc=fc, ec=GRID_EDGE, lw=0.3, zorder=3)
            )
    cx = x + w + 2.4
    line(ax, [x + w + 0.3] * 2, [y + 0.2, y + w - 0.2], color=RED, lw=0.5, z=3)
    arrow(ax, (x + w + 0.3, y + w / 2), (cx, y + w / 2), color=RED, lw=0.5, z=3, ms=5)
    ax.add_patch(Rectangle((cx, y + w / 2 - c / 2), c, c, fc=RED, ec="none", zorder=3))
    ax.text(
        cx + c / 2, y + w / 2 - c / 2 - 0.4, "CLS", ha="center", va="top", fontsize=4.6, color=RED
    )


# ---- sections, left to right ----------------------------------------------------------------------
@dataclass
class Anchors:
    """Coordinates one section hands to the next."""

    layer_x: float = 0  # encoder layer column
    layer_w: float = 0
    cls_x: float = 0  # CLS token
    nodes_x: float = 0  # node-token grid H
    nodes_w: float = 0
    stats_x: float = 0  # s(H) branch
    z_global_x: float = 0
    vae_x: float = 0
    z_nodes_x: float = 0
    z_nodes_w: float = 0
    permuter_x: float = 0
    decoder_x: float = 0


def draw_input(ax, colours) -> None:
    grid_glyph(ax, 0, Y_NODES - 6, 12, colours, dot=4.6)
    ax.text(6, Y_NODES + 7.4, "grid graph", ha="center", va="bottom", fontsize=6.8, weight="bold")
    ax.text(6, Y_NODES - 7.6, r"$X\in\mathbb{R}^{N\times128}$", ha="center", va="top", fontsize=6.4)
    ax.text(6, Y_NODES - 10.6, r"$N=256\ (16\times16)$", ha="center", va="top", fontsize=5.6)


def draw_encoder(ax, a: Anchors) -> None:
    """Encoder container with its layers (bottom to top), the CLS token c and the node tokens H."""
    ex, ey, ew, eh = 16, 2, 27, 54
    box(ax, ex, ey, ew, eh, BLUE_FILL, BLUE, lw=0.9, r=1.2, z=1)
    ax.text(
        ex + ew / 2,
        ey + eh - 2.0,
        "Encoder",
        ha="center",
        va="top",
        fontsize=7.5,
        weight="bold",
        color=BLUE,
    )
    a.layer_x, a.layer_w = ex + 2, ew - 4
    lx, lw = a.layer_x, a.layer_w
    layer(ax, lx, 5.0, lw, 4.6, "Linear 128→512\n+ prepend CLS token", BLUE, fs=5.2)
    layer(ax, lx, 11.8, lw, 3.8, "Linear + SiLU", BLUE)
    layer(ax, lx, 17.8, lw, 6.2, "Transformer block ×4\nlocal attention, no PE", BLUE, fs=5.3)
    layer(ax, lx, 26.2, lw, 3.8, "LayerNorm", BLUE)
    for y0, y1 in [(9.6, 11.8), (15.6, 17.8), (24.0, 26.2)]:
        arrow(ax, (lx + lw / 2, y0), (lx + lw / 2, y1), lw=0.6, ms=5)
    mask_stencil(ax, ex + 8.0, 40.5, 6.0)
    ax.text(
        ex + ew / 2,
        38.6,
        "attention mask\nnode: 4 neighbours + self\nCLS: reads all nodes",
        ha="center",
        va="top",
        fontsize=4.8,
        color=BLUE,
        linespacing=1.12,
    )
    arrow(ax, (12.8, Y_NODES), (lx, 7.3))

    split = (ex + ew + 1.8, 28.1)  # the encoder output splits into c (red) and H (blue)
    line(ax, [lx + lw, split[0]], [split[1], split[1]])
    ax.scatter([split[0]], [split[1]], s=4, c=INK, zorder=6)
    a.cls_x = split[0] + 3.0
    vec(ax, a.cls_x, 33.0, RED)
    arrow(ax, split, (a.cls_x - 0.2, 35.5), color=RED, lw=0.7, rad=-0.2)
    ax.text(
        a.cls_x + 2.3,
        35.5,
        r"$c\in\mathbb{R}^{512}$",
        ha="left",
        va="center",
        fontsize=5.8,
        color=RED,
    )
    a.nodes_x, a.nodes_w = a.cls_x, 8.0
    node_grid(ax, a.nodes_x, Y_NODES - a.nodes_w / 2, a.nodes_w, BLUE, dot=5.5)
    arrow(ax, split, (a.nodes_x - 0.2, Y_NODES + 2.0), color=BLUE, lw=0.7, rad=0.25)
    ax.text(
        a.nodes_x + a.nodes_w / 2,
        Y_NODES - a.nodes_w / 2 - 1.2,
        r"$H\in\mathbb{R}^{N\times512}$",
        ha="center",
        va="top",
        fontsize=5.8,
        color=BLUE,
    )


def draw_global_path(ax, a: Anchors) -> None:
    """Global readout LN(c + W s(H)) -> z_global -> downstream tasks."""
    gx, gw, gh = 16 + 27 + 3.0, 32, 10
    box(ax, gx, Y_GLOBAL - gh / 2, gw, gh, "white", RED, lw=0.8)
    ax.text(
        gx + gw / 2,
        Y_GLOBAL + 2.9,
        "global readout",
        ha="center",
        va="center",
        fontsize=6.0,
        color=RED,
        zorder=4,
    )
    ax.text(
        gx + gw / 2,
        Y_GLOBAL - 0.1,
        r"$\mathrm{LN}(c + W\,s(H))$",
        ha="center",
        va="center",
        fontsize=6.8,
        zorder=4,
    )
    ax.text(
        gx + gw / 2,
        Y_GLOBAL - 3.1,
        r"$s(H)=[\mathrm{mean};\mathrm{var};\max]$ over nodes" + "\nLN without learned parameters",
        ha="center",
        va="center",
        fontsize=4.8,
        color=NOTE,
        zorder=4,
        linespacing=1.1,
    )
    arrow(ax, (a.cls_x + 0.8, 38.0), (a.cls_x + 0.8, Y_GLOBAL - gh / 2), color=RED)
    a.stats_x = a.nodes_x + a.nodes_w + 4.0  # s(H): pooled statistics over the node grid
    y_stats = Y_NODES + a.nodes_w / 2 - 0.5
    line(
        ax,
        [a.nodes_x + a.nodes_w + 0.4, a.stats_x],
        [y_stats, y_stats],
        color=BLUE,
        lw=0.55,
        ls=(0, (2, 1.5)),
    )
    arrow(
        ax,
        (a.stats_x, y_stats),
        (a.stats_x, Y_GLOBAL - gh / 2),
        color=BLUE,
        lw=0.55,
        ls=(0, (2, 1.5)),
    )
    ax.text(a.stats_x + 0.8, 38.5, r"$s(H)$", fontsize=5.6, color=BLUE, ha="left", va="center")

    zx = a.z_global_x = gx + gw + 4.5
    vec(ax, zx, Y_GLOBAL - 3.4, RED, n=5, cw=2.0, ch=1.36)
    ax.text(
        zx + 3.0,
        Y_GLOBAL + 1.4,
        r"$z_{\mathrm{global}}$",
        ha="left",
        va="center",
        fontsize=7.6,
        color=RED,
    )
    ax.text(
        zx + 3.0,
        Y_GLOBAL - 2.2,
        r"$\mathbb{R}^{512}$, invariant",
        ha="left",
        va="center",
        fontsize=5.4,
        color=RED,
    )
    ax.text(
        zx + 1.0,
        Y_GLOBAL + 6.4,
        r"$z_{\mathrm{global}}(gX)=z_{\mathrm{global}}(X)$",
        ha="center",
        va="center",
        fontsize=5.4,
        color=RED,
    )
    arrow(ax, (gx + gw, Y_GLOBAL), (zx - 0.3, Y_GLOBAL), color=RED)
    dx = zx + 26
    box(ax, dx, Y_GLOBAL - 2.6, 14, 5.2, "white", GREY)
    ax.text(dx + 7, Y_GLOBAL, "downstream\ntasks", ha="center", va="center", fontsize=6.0, zorder=4)
    arrow(ax, (zx + 17.0, Y_GLOBAL), (dx, Y_GLOBAL), color=RED, lw=1.0)


def draw_node_path(ax, a: Anchors) -> None:
    """VAE head (mu, sigma) -> sample -> z_nodes on the grid -> dormant permuter."""
    vx = a.vae_x = a.stats_x + 4.5
    y_mu, y_sigma = Y_NODES + 6.4, Y_NODES - 6.4
    for symbol, yy in [(r"$\mu$", y_mu), (r"$\sigma$", y_sigma)]:
        box(ax, vx, yy - 2.0, 5.2, 4.0, "white", BLUE, ls=DASH, lw=0.7)
        ax.text(vx + 2.6, yy, symbol, ha="center", va="center", fontsize=7.2, zorder=4)
    ax.text(vx + 2.6, y_sigma - 2.8, "VAE head", ha="center", va="top", fontsize=5.6, color=BLUE)
    arrow(
        ax,
        (a.nodes_x + a.nodes_w + 0.4, Y_NODES + 0.8),
        (vx, y_mu - 0.6),
        color=BLUE,
        lw=0.6,
        rad=-0.2,
    )
    arrow(
        ax,
        (a.nodes_x + a.nodes_w + 0.4, Y_NODES - 0.8),
        (vx, y_sigma + 0.6),
        color=BLUE,
        lw=0.6,
        rad=0.2,
    )

    sample_x = vx + 10.0
    line(ax, [vx + 5.2, sample_x, sample_x], [y_mu, y_mu, Y_NODES + 2.4], lw=0.7)
    arrow(ax, (sample_x, Y_NODES + 3.0), (sample_x, Y_NODES + 2.4), lw=0.7, ms=5)
    line(ax, [vx + 5.2, sample_x, sample_x], [y_sigma, y_sigma, Y_NODES - 2.4], lw=0.7)
    arrow(ax, (sample_x, Y_NODES - 3.0), (sample_x, Y_NODES - 2.4), lw=0.7, ms=5)
    ax.text(vx + 6.0, y_mu + 0.6, "sample", fontsize=4.8, ha="left", va="bottom", color=NOTE)
    ax.add_patch(plt.Circle((sample_x, Y_NODES), 2.4, fc="white", ec=INK, lw=0.6, zorder=6))
    ax.text(sample_x, Y_NODES, "~", ha="center", va="center", fontsize=7, zorder=7)

    a.z_nodes_x, a.z_nodes_w = sample_x + 4.5, 7.0
    znx, znw = a.z_nodes_x, a.z_nodes_w
    node_grid(ax, znx, Y_NODES - znw / 2, znw, BLUE_LIGHT, dot=4.2, edge="#b9c9dc")
    arrow(ax, (sample_x + 2.4, Y_NODES), (znx - 0.4, Y_NODES), lw=0.6, ms=5)
    ax.text(
        znx + znw / 2,
        Y_NODES + znw / 2 + 1.2,
        r"$z_{\mathrm{nodes}}\in\mathbb{R}^{N\times32}$",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=BLUE,
    )
    ax.text(
        vx - 1.0,
        5.2,
        r"$z=\mu+\sigma\odot\epsilon,\ \epsilon\sim\mathcal{N}(0,I)$;  $z=\mu$ at evaluation",
        ha="left",
        va="center",
        fontsize=5.0,
        color=NOTE,
    )
    ax.text(
        vx - 1.0,
        2.2,
        r"equivariant: $\mu(gX)=P_g\,\mu(X)$",
        ha="left",
        va="center",
        fontsize=5.2,
        color=BLUE,
    )

    a.permuter_x = znx + znw + 3.0
    box(ax, a.permuter_x, Y_NODES - 2.6, 11.5, 5.2, "#f4f4f4", GREY, ls=DASH)
    ax.text(
        a.permuter_x + 5.75,
        Y_NODES,
        "Permuter",
        ha="center",
        va="center",
        fontsize=5.8,
        color=NOTE,
        zorder=4,
    )
    arrow(ax, (znx + znw + 0.4, Y_NODES), (a.permuter_x, Y_NODES), color=BLUE, lw=0.7)


def draw_decoder(ax, a: Anchors, colours_out) -> None:
    """Decoder container (left to right), FiLM conditioning from z_global, and the reconstruction."""
    dcx, dcy, dcw, dch = a.permuter_x + 14.5, Y_NODES - 10, 39.5, 20
    box(ax, dcx, dcy, dcw, dch, GOLD_FILL, GOLD, lw=0.9, r=1.2, z=1)
    ax.text(
        dcx + 1.5,
        dcy + dch - 1.3,
        "Decoder",
        ha="left",
        va="top",
        fontsize=7.5,
        weight="bold",
        color=GOLD,
    )
    ly, lh = dcy + 2.0, dch - 7.0
    mlp_x, mlp_w = dcx + 1.8, 11.2
    tf_x, tf_w = mlp_x + mlp_w + 2.0, 14.0
    out_x, out_w = tf_x + tf_w + 2.0, 6.7
    layer(ax, mlp_x, ly, mlp_w, lh, "MLP\n32→512\n+ absolute\n2D PE\n(row, col)", GOLD, fs=5.4)
    layer(ax, tf_x, ly, tf_w, lh, "Transformer\nblock ×4\nfull attention", GOLD, fs=5.7)
    layer(ax, out_x, ly, out_w, lh, "Linear\n512\n→128", GOLD, fs=5.4)
    arrow(ax, (mlp_x + mlp_w, ly + lh / 2), (tf_x, ly + lh / 2), lw=0.6, ms=5)
    arrow(ax, (tf_x + tf_w, ly + lh / 2), (out_x, ly + lh / 2), lw=0.6, ms=5)
    arrow(ax, (a.permuter_x + 11.5, Y_NODES), (mlp_x, Y_NODES), color=BLUE, lw=0.7)

    # FiLM (optional): generator -> (gamma_l, beta_l) into every block
    fx, fw, fy = tf_x + tf_w / 2, 11.0, 41.0
    line(
        ax,
        [a.z_global_x + 1.0, a.z_global_x + 1.0, fx - fw / 2],
        [Y_GLOBAL - 3.4, fy, fy],
        color=RED,
        ls=DASH,
    )
    box(ax, fx - fw / 2, fy - 2.3, fw, 4.6, "white", RED, ls=DASH)
    ax.text(fx, fy, "FiLM gen.", ha="center", va="center", fontsize=5.5, color=RED, zorder=4)
    arrow(ax, (fx, fy - 2.3), (fx, ly + lh), color=RED, ls=DASH, ms=6)
    ax.text(
        fx - 1.0,
        34.6,
        r"$(1+\gamma_\ell)\,\mathrm{norm}(x)+\beta_\ell$" + "\n" + r"both pre-norms, $\ell=1..4$",
        ha="right",
        va="center",
        fontsize=5.0,
        color=RED,
        linespacing=1.15,
    )

    rx = dcx + dcw + 3.0
    grid_glyph(ax, rx, Y_NODES - 5, 10, colours_out, dot=3.2)
    ax.text(
        rx + 5,
        Y_NODES - 6.6,
        r"$\hat X\in\mathbb{R}^{N\times128}$",
        ha="center",
        va="top",
        fontsize=6.4,
    )
    arrow(ax, (out_x + out_w, Y_NODES), (rx - 0.6, Y_NODES))
    ax.text(
        dcx + dcw / 2 + 3,
        -2.6,
        r"$\mathcal{L}=\mathcal{L}_{\mathrm{rec}}(\hat X, X)\ +\ \lambda\,"
        r"\mathrm{KL}(q(z_{\mathrm{nodes}}\mid H)\,\|\,\mathcal{N}(0,I))$"
        "\n"
        r"$\mathcal{L}_{\mathrm{rec}}$ over all $N$ nodes and all 128 features",
        ha="center",
        va="center",
        fontsize=5.2,
        linespacing=1.3,
    )
    print("right edge:", rx + 10, "of", X_RANGE[1])


def draw_legend(ax) -> None:
    x, y = 16, -4.0
    line(ax, [x, x + 4], [y, y])
    ax.text(x + 5, y, "active", fontsize=5.2, va="center")
    line(ax, [x + 12, x + 16], [y, y], color=RED, ls=DASH)
    ax.text(x + 17, y, "optional (on in FiLM runs)", fontsize=5.2, va="center")
    ax.add_patch(Rectangle((x + 43, y - 1.0), 4, 2.0, fc="#f4f4f4", ec=GREY, lw=0.6, ls=DASH))
    ax.text(x + 48, y, "dormant in all IMC runs", fontsize=5.2, va="center")


def grid_colours(d) -> tuple[np.ndarray, np.ndarray]:
    """-PC2 of input and reconstruction on a black -> panCK ramp, both scaled by the input."""
    ramp = LinearSegmentedColormap.from_list("panck", ["#141414", style.MARKER_COLOURS["panCK"]])
    v_in = -d["z"][PC]
    lo, hi = np.percentile(v_in, 2), np.percentile(v_in, 98)

    def to_colours(v):
        return ramp(np.clip((v - lo) / (hi - lo), 0, 1))[..., :3]

    colours_in = to_colours(v_in)
    return colours_in, (to_colours(-d["xhat"][PC]) if "xhat" in d else colours_in)


def main(npz: str, out: str) -> None:
    colours_in, colours_out = grid_colours(np.load(npz))
    style.use_serif(7)
    fig = plt.figure(figsize=(6.3, 6.3 * (Y_RANGE[1] - Y_RANGE[0]) / (X_RANGE[1] - X_RANGE[0])))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.axis("off")
    ax.set_aspect("equal")

    anchors = Anchors()
    draw_input(ax, colours_in)
    draw_encoder(ax, anchors)
    draw_global_path(ax, anchors)
    draw_node_path(ax, anchors)
    draw_decoder(ax, anchors, colours_out)
    draw_legend(ax)
    io.save(fig, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", default=str(paths.CACHE / "pipeline_7775_384_256.npz"))
    ap.add_argument("--out", default=str(paths.figure("fig_architecture")))
    a = ap.parse_args()
    main(a.npz, a.out)
