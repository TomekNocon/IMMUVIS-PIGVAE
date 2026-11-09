"""Thesis Figure 3.1: the PIGVAE encoder-decoder over the grid graph.

Drawn from src/models/components/modules.py (GraphAE / GraphEncoder / NodeBottleneckEncoder /
SimplePermuter / GraphDecoder / FiLMConditioner) and configs/model/model.yaml.
Style: VAE-style trapezoids for encoder/decoder, tokens as stacked-cell vectors, FiLM as
(1+gamma)*norm(x)+beta. Palette from the IMC markers: DNA1 blue (encoder, node stream),
CD45 red (global / invariant stream), panCK gold (decoder, tissue glyphs).
Input/output grids: -PC2 of the real node features of the example crop (tracks panCK) on a
black->gold ramp; the output is the trained model's reconstruction (extract_reconstruction.py).
"""
import rootutils
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from fig_pipeline import PC

INK = "#2b2b2b"
BLUE, BLUE_F, BLUE_L = "#2f6db5", "#e3edf9", "#9fc0e8"    # encoder + node stream (DNA1)
RED, RED_F, RED_L = "#c8323e", "#fbe6e7", "#eea3a9"       # global / invariant stream (CD45)
GOLD, GOLD_F = "#a87f10", "#fbf2d6"                        # decoder (panCK)
GREY = "#8a8a8a"
DASH = (0, (3, 2))


def box(ax, x, y, w, h, fc, ec, lw=0.7, ls="-", r=0.6, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def arrow(ax, p0, p1, color=INK, lw=0.8, ls="-", rad=0.0, z=5, ms=7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def line(ax, xs, ys, color=INK, lw=0.8, ls="-", z=5):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")


def vec(ax, x, y, color, n=4, cw=1.6, ch=1.25, ec=None):
    """A token drawn as a column vector of n stacked cells; (x, y) = bottom-left."""
    for k in range(n):
        ax.add_patch(Rectangle((x, y + k * ch), cw, ch, fc=color, ec=ec or "white", lw=0.5, zorder=6))
    return x + cw / 2, y + n * ch


def token_row(ax, x, y, color, labels, n=4, cw=1.6, ch=1.25, gap=1.3):
    """Row of node tokens h_1 h_2 ... h_N (with an ellipsis before the last)."""
    xs = []
    for i, lab in enumerate(labels):
        if lab == "…":
            ax.text(x + cw / 2, y + n * ch / 2, "…", ha="center", va="center", fontsize=6.5, color=color)
        else:
            vec(ax, x, y, color, n, cw, ch)
            ax.text(x + cw / 2, y - 0.5, lab, ha="center", va="top", fontsize=5.2, color=color)
            xs.append(x + cw / 2)
        x += cw + gap
    return x - gap


def trapezoid(ax, x0, x1, yl, yr, fc, ec):
    """yl / yr: (bottom, top) of the left / right edge."""
    ax.add_patch(Polygon([(x0, yl[0]), (x0, yl[1]), (x1, yr[1]), (x1, yr[0])], closed=True,
                         fc=fc, ec=ec, lw=0.9, zorder=1, joinstyle="round"))


def grid_glyph(ax, x, y, w, cols, dot, edges=None):
    n = cols.shape[0]; s = w / (n - 1)
    xs, ys = x + np.arange(n) * s, y + w - np.arange(n) * s
    if edges:
        for k in range(n):
            line(ax, [xs[0], xs[-1]], [ys[k]] * 2, color=edges, lw=0.4, z=2)
            line(ax, [xs[k]] * 2, [ys[0], ys[-1]], color=edges, lw=0.4, z=2)
    X, Y = np.meshgrid(xs, ys)
    ax.scatter(X.ravel(), Y.ravel(), s=dot, c=cols.reshape(-1, 3), ec="none", zorder=3)


def layer(ax, x, y, w, h, text, ec, fs=5.8, ls="-", fc="white"):
    box(ax, x, y, w, h, fc, ec, lw=0.6, ls=ls, r=0.5, z=3)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4, linespacing=1.12)


def node_grid(ax, x, y, w, color, n=6, dot=5.0, edge="#9aa9bb"):
    """A uniform n x n grid graph (one token per node)."""
    grid_glyph(ax, x, y, w, np.tile(np.array(matplotlib.colors.to_rgb(color)), (n, n, 1)), dot, edges=edge)


def mask_stencil(ax, x, y, w):
    """Encoder attention mask: query cell (dark), 4 neighbours (light); CLS reads all nodes."""
    n = 5; c = w / n
    for r in range(n):
        for k in range(n):
            d = abs(r - 2) + abs(k - 2)
            fc = BLUE if d == 0 else (BLUE_L if d == 1 else "white")
            ax.add_patch(Rectangle((x + k * c, y + r * c), c, c, fc=fc, ec="#9aa9bb", lw=0.3, zorder=3))
    cx = x + w + 2.4
    line(ax, [x + w + 0.3] * 2, [y + 0.2, y + w - 0.2], color=RED, lw=0.5, z=3)
    arrow(ax, (x + w + 0.3, y + w / 2), (cx, y + w / 2), color=RED, lw=0.5, z=3, ms=5)
    ax.add_patch(Rectangle((cx, y + w / 2 - c / 2), c, c, fc=RED, ec="none", zorder=3))
    ax.text(cx + c / 2, y + w / 2 - c / 2 - 0.4, "CLS", ha="center", va="top", fontsize=4.6, color=RED)


def main(npz, out):
    d = np.load(npz)
    ramp = LinearSegmentedColormap.from_list("panck", ["#141414", "#E8C34C"])
    v_in = -d["z"][PC]
    lo, hi = np.percentile(v_in, 2), np.percentile(v_in, 98)
    to_cols = lambda v: ramp(np.clip((v - lo) / (hi - lo), 0, 1))[..., :3]
    cols_in = to_cols(v_in)
    cols_out = to_cols(-d["xhat"][PC]) if "xhat" in d else cols_in

    plt.rcParams.update({"font.family": "serif", "font.size": 7, "mathtext.fontset": "dejavuserif"})
    XMIN, XMAX, YMIN, YMAX = -4.5, 161, -6, 61
    fig = plt.figure(figsize=(6.3, 6.3 * (YMAX - YMIN) / (XMAX - XMIN)))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX); ax.axis("off")
    ax.set_aspect("equal")
    yc, yg = 20, 50          # centre of the node path and of the global path

    # ---- input ------------------------------------------------------------------------------
    grid_glyph(ax, 0, yc - 6, 12, cols_in, dot=4.6)
    ax.text(6, yc + 7.4, "grid graph", ha="center", va="bottom", fontsize=6.8, weight="bold")
    ax.text(6, yc - 7.6, r"$X\in\mathbb{R}^{N\times128}$", ha="center", va="top", fontsize=6.4)
    ax.text(6, yc - 10.6, r"$N=256\ (16\times16)$", ha="center", va="top", fontsize=5.6)

    # ---- encoder: container with its layers, bottom to top ----------------------------------
    ex, ey, ew, eh = 16, 2, 27, 54
    box(ax, ex, ey, ew, eh, BLUE_F, BLUE, lw=0.9, r=1.2, z=1)
    ax.text(ex + ew / 2, ey + eh - 2.0, "Encoder", ha="center", va="top", fontsize=7.5, weight="bold", color=BLUE)
    lx, lw_ = ex + 2, ew - 4
    layer(ax, lx, 5.0, lw_, 4.6, "Linear 128→512\n+ prepend CLS token", BLUE, fs=5.2)
    layer(ax, lx, 11.8, lw_, 3.8, "Linear + SiLU", BLUE)
    layer(ax, lx, 17.8, lw_, 6.2, "Transformer block ×4\nlocal attention, no PE", BLUE, fs=5.3)
    layer(ax, lx, 26.2, lw_, 3.8, "LayerNorm", BLUE)
    for y0, y1 in [(9.6, 11.8), (15.6, 17.8), (24.0, 26.2)]:
        arrow(ax, (lx + lw_ / 2, y0), (lx + lw_ / 2, y1), lw=0.6, ms=5)
    mask_stencil(ax, ex + 8.0, 40.5, 6.0)
    ax.text(ex + ew / 2, 38.6, "attention mask\nnode: 4 neighbours + self\nCLS: reads all nodes", ha="center",
            va="top", fontsize=4.8, color=BLUE, linespacing=1.12)
    arrow(ax, (12.8, yc), (lx, 7.3))

    # outputs: CLS token c (red) and node tokens H on the grid (blue)
    o = (ex + ew + 1.8, 28.1)
    line(ax, [lx + lw_, o[0]], [o[1], o[1]])
    ax.scatter([o[0]], [o[1]], s=4, c=INK, zorder=6)
    cx0 = o[0] + 3.0
    vec(ax, cx0, 33.0, RED)
    arrow(ax, o, (cx0 - 0.2, 35.5), color=RED, lw=0.7, rad=-0.2)
    ax.text(cx0 + 2.3, 35.5, r"$c\in\mathbb{R}^{512}$", ha="left", va="center", fontsize=5.8, color=RED)
    hx, hw = cx0, 8.0
    node_grid(ax, hx, yc - hw / 2, hw, BLUE, dot=5.5)
    arrow(ax, o, (hx - 0.2, yc + 2.0), color=BLUE, lw=0.7, rad=0.25)
    ax.text(hx + hw / 2, yc - hw / 2 - 1.2, r"$H\in\mathbb{R}^{N\times512}$", ha="center", va="top", fontsize=5.8, color=BLUE)

    # ---- global readout -> z_global -> downstream ------------------------------------------
    gx, gw, gh = ex + ew + 3.0, 32, 10
    box(ax, gx, yg - gh / 2, gw, gh, "white", RED, lw=0.8)
    ax.text(gx + gw / 2, yg + 2.9, "global readout", ha="center", va="center", fontsize=6.0, color=RED, zorder=4)
    ax.text(gx + gw / 2, yg - 0.1, r"$\mathrm{LN}(c + W\,s(H))$", ha="center", va="center", fontsize=6.8, zorder=4)
    ax.text(gx + gw / 2, yg - 3.1, r"$s(H)=[\mathrm{mean};\mathrm{var};\max]$ over nodes" + "\nLN without learned parameters",
            ha="center", va="center", fontsize=4.8, color="#555555", zorder=4, linespacing=1.1)
    arrow(ax, (cx0 + 0.8, 38.0), (cx0 + 0.8, yg - gh / 2), color=RED)
    sx = hx + hw + 4.0                            # s(H): pooled statistics over the node grid
    line(ax, [hx + hw + 0.4, sx], [yc + hw / 2 - 0.5, yc + hw / 2 - 0.5], color=BLUE, lw=0.55, ls=(0, (2, 1.5)))
    arrow(ax, (sx, yc + hw / 2 - 0.5), (sx, yg - gh / 2), color=BLUE, lw=0.55, ls=(0, (2, 1.5)))
    ax.text(sx + 0.8, 38.5, r"$s(H)$", fontsize=5.6, color=BLUE, ha="left", va="center")

    zx = gx + gw + 4.5
    vec(ax, zx, yg - 3.4, RED, n=5, cw=2.0, ch=1.36)
    ax.text(zx + 3.0, yg + 1.4, r"$z_{\mathrm{global}}$", ha="left", va="center", fontsize=7.6, color=RED)
    ax.text(zx + 3.0, yg - 2.2, r"$\mathbb{R}^{512}$, invariant", ha="left", va="center", fontsize=5.4, color=RED)
    ax.text(zx + 1.0, yg + 6.4, r"$z_{\mathrm{global}}(gX)=z_{\mathrm{global}}(X)$", ha="center", va="center", fontsize=5.4, color=RED)
    arrow(ax, (gx + gw, yg), (zx - 0.3, yg), color=RED)
    dx = zx + 26
    box(ax, dx, yg - 2.6, 14, 5.2, "white", GREY)
    ax.text(dx + 7, yg, "downstream\ntasks", ha="center", va="center", fontsize=6.0, zorder=4)
    arrow(ax, (zx + 17.0, yg), (dx, yg), color=RED, lw=1.0)

    # ---- node path: VAE head (mu, sigma) -> sample -> z_nodes on the grid ------------------
    vx = sx + 4.5
    my, sy = yc + 6.4, yc - 6.4
    for lab, yy in [(r"$\mu$", my), (r"$\sigma$", sy)]:
        box(ax, vx, yy - 2.0, 5.2, 4.0, "white", BLUE, ls=DASH, lw=0.7)
        ax.text(vx + 2.6, yy, lab, ha="center", va="center", fontsize=7.2, zorder=4)
    ax.text(vx + 2.6, sy - 2.8, "VAE head", ha="center", va="top", fontsize=5.6, color=BLUE)
    arrow(ax, (hx + hw + 0.4, yc + 0.8), (vx, my - 0.6), color=BLUE, lw=0.6, rad=-0.2)
    arrow(ax, (hx + hw + 0.4, yc - 0.8), (vx, sy + 0.6), color=BLUE, lw=0.6, rad=0.2)
    cxs = vx + 10.0                              # the sampling node
    line(ax, [vx + 5.2, cxs, cxs], [my, my, yc + 2.4], lw=0.7)
    arrow(ax, (cxs, yc + 3.0), (cxs, yc + 2.4), lw=0.7, ms=5)
    line(ax, [vx + 5.2, cxs, cxs], [sy, sy, yc - 2.4], lw=0.7)
    arrow(ax, (cxs, yc - 3.0), (cxs, yc - 2.4), lw=0.7, ms=5)
    ax.text(vx + 6.0, my + 0.6, "sample", fontsize=4.8, ha="left", va="bottom", color="#555555")
    ax.add_patch(plt.Circle((cxs, yc), 2.4, fc="white", ec=INK, lw=0.6, zorder=6))
    ax.text(cxs, yc, "~", ha="center", va="center", fontsize=7, zorder=7)
    znx, znw = cxs + 4.5, 7.0
    node_grid(ax, znx, yc - znw / 2, znw, BLUE_L, dot=4.2, edge="#b9c9dc")
    arrow(ax, (cxs + 2.4, yc), (znx - 0.4, yc), lw=0.6, ms=5)
    ax.text(znx + znw / 2, yc + znw / 2 + 1.2, r"$z_{\mathrm{nodes}}\in\mathbb{R}^{N\times32}$", ha="center", va="bottom",
            fontsize=5.8, color=BLUE)
    ax.text(vx - 1.0, 5.2, r"$z=\mu+\sigma\odot\epsilon,\ \epsilon\sim\mathcal{N}(0,I)$;  $z=\mu$ at evaluation",
            ha="left", va="center", fontsize=5.0, color="#555555")
    ax.text(vx - 1.0, 2.2, r"equivariant: $\mu(gX)=P_g\,\mu(X)$", ha="left", va="center", fontsize=5.2, color=BLUE)

    # ---- permuter (dormant) -----------------------------------------------------------------
    px = znx + znw + 3.0
    box(ax, px, yc - 2.6, 11.5, 5.2, "#f4f4f4", GREY, ls=DASH)
    ax.text(px + 5.75, yc, "Permuter", ha="center", va="center", fontsize=5.8, color="#555555", zorder=4)
    arrow(ax, (znx + znw + 0.4, yc), (px, yc), color=BLUE, lw=0.7)

    # ---- decoder: container with its layers, left to right ----------------------------------
    dcx, dcy, dcw, dch = px + 14.5, yc - 10, 39.5, 20
    box(ax, dcx, dcy, dcw, dch, GOLD_F, GOLD, lw=0.9, r=1.2, z=1)
    ax.text(dcx + 1.5, dcy + dch - 1.3, "Decoder", ha="left", va="top", fontsize=7.5, weight="bold", color=GOLD)
    ly, lh = dcy + 2.0, dch - 7.0
    m_x, m_w = dcx + 1.8, 11.2
    t_x, t_w = m_x + m_w + 2.0, 14.0
    o_x, o_w = t_x + t_w + 2.0, 6.7
    layer(ax, m_x, ly, m_w, lh, "MLP\n32→512\n+ absolute\n2D PE\n(row, col)", GOLD, fs=5.4)
    layer(ax, t_x, ly, t_w, lh, "Transformer\nblock ×4\nfull attention", GOLD, fs=5.7)
    layer(ax, o_x, ly, o_w, lh, "Linear\n512\n→128", GOLD, fs=5.4)
    arrow(ax, (m_x + m_w, ly + lh / 2), (t_x, ly + lh / 2), lw=0.6, ms=5)
    arrow(ax, (t_x + t_w, ly + lh / 2), (o_x, ly + lh / 2), lw=0.6, ms=5)
    arrow(ax, (px + 11.5, yc), (m_x, yc), color=BLUE, lw=0.7)

    # FiLM conditioning from z_global (optional): generator -> (gamma_l, beta_l) into every block
    fx = t_x + t_w / 2; fgw = 11.0; fgy = 41.0
    line(ax, [zx + 1.0, zx + 1.0, fx - fgw / 2], [yg - 3.4, fgy, fgy], color=RED, ls=DASH)
    box(ax, fx - fgw / 2, fgy - 2.3, fgw, 4.6, "white", RED, ls=DASH)
    ax.text(fx, fgy, "FiLM gen.", ha="center", va="center", fontsize=5.5, color=RED, zorder=4)
    arrow(ax, (fx, fgy - 2.3), (fx, ly + lh), color=RED, ls=DASH, ms=6)
    ax.text(fx - 1.0, 34.6, r"$(1+\gamma_\ell)\,\mathrm{norm}(x)+\beta_\ell$" + "\n" +
            r"both pre-norms, $\ell=1..4$", ha="right", va="center", fontsize=5.0, color=RED, linespacing=1.15)

    # ---- reconstruction ---------------------------------------------------------------------
    rx = dcx + dcw + 3.0
    grid_glyph(ax, rx, yc - 5, 10, cols_out, dot=3.2)
    ax.text(rx + 5, yc - 6.6, r"$\hat X\in\mathbb{R}^{N\times128}$", ha="center", va="top", fontsize=6.4)
    arrow(ax, (o_x + o_w, yc), (rx - 0.6, yc))
    ax.text(dcx + dcw / 2 + 3, -2.6, r"$\mathcal{L}=\mathcal{L}_{\mathrm{rec}}(\hat X, X)\ +\ \lambda\,"
            r"\mathrm{KL}(q(z_{\mathrm{nodes}}\mid H)\,\|\,\mathcal{N}(0,I))$" + "\n" +
            r"$\mathcal{L}_{\mathrm{rec}}$ over all $N$ nodes and all 128 features", ha="center", va="center",
            fontsize=5.2, linespacing=1.3)

    # ---- legend -----------------------------------------------------------------------------
    lgx, lgy = 16, -4.0
    line(ax, [lgx, lgx + 4], [lgy, lgy]); ax.text(lgx + 5, lgy, "active", fontsize=5.2, va="center")
    line(ax, [lgx + 12, lgx + 16], [lgy, lgy], color=RED, ls=DASH)
    ax.text(lgx + 17, lgy, "optional (on in FiLM runs)", fontsize=5.2, va="center")
    ax.add_patch(Rectangle((lgx + 43, lgy - 1.0), 4, 2.0, fc="#f4f4f4", ec=GREY, lw=0.6, ls=DASH))
    ax.text(lgx + 48, lgy, "dormant in all IMC runs", fontsize=5.2, va="center")

    print("right edge:", rx + 10, "of", XMAX)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="figures/_cache/pipeline_7775_384_256.npz")
    ap.add_argument("--out", default="figures/fig_architecture")
    a = ap.parse_args(); main(a.npz, a.out)
