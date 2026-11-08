"""Thesis figure (Sec. 5.x, open item 13): training curves of the reference experiment
vae16_fb0p0_film (wandb run 81zasevr, "vae16-fb0.0-s0.01-FILM-start5-40ep").

(a) reconstruction term by optimiser step (log), train every 250 steps, val once per epoch
(b) reconstruction term vs validation KL per epoch: the exchange along the alpha ramp
(c) KL annealing: the weight alpha and the KL term as it enters the loss (0.01 * alpha * KL)
(d) Huber component, (e) per-node cosine similarity (= 1 - beta_recon / 0.1),
(f) reconstruction SNR in dB (logged for monitoring, not optimised).
Reconstruction term = alpha_recon + beta_recon + gamma_recon (Huber + 0.1*cosine + 0.001*gradient).
All epoch panels shade the alpha ramp. Data: wandb history -> thesis_figures/_cache/history_81zasevr.csv
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

BLUE, RED, GREY, INK, GOLD = "#2f6db5", "#c8323e", "#9a9a9a", "#2b2b2b", "#a87f10"
plt.rcParams.update({"font.family": "serif", "font.size": 7.2, "mathtext.fontset": "dejavuserif",
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})
KL_SCALE, SPE = 0.01, 5000          # kld_loss_scale of the run; optimiser steps per epoch
EXP, RUN, NAME = "vae16_fb0p0_film", "81zasevr", "vae16-fb0.0-s0.01-FILM-start5-40ep"


def load():
    df = pd.read_csv(f"thesis_figures/_cache/history_{RUN}.csv")
    v = df.dropna(subset=["val_alpha_recon"]).set_index("epoch").sort_index()
    t = df.dropna(subset=["alpha_recon"]).copy()
    for d, p in [(v, "val_"), (t, "")]:
        d["rec"] = d[p + "alpha_recon"] + d[p + "beta_recon"] + d[p + "gamma_recon"]
        d["huber"] = d[p + "alpha_recon"]
        d["cos"] = 1 - d[p + "beta_recon"] / 0.1
        d["snr"] = d[p + "signal_to_noise_ratio_loss"]
        d["kl"] = d[p + "kld_loss"]
    te = t.groupby("epoch")[["rec", "huber", "cos", "snr", "kl"]].mean()
    return v, t, te


def ramp(ax, v):
    """Shade the KL weight alpha (0 -> 1 over the panel height) behind an epoch panel."""
    ep = v.index.values + 1
    tf = blended_transform_factory(ax.transData, ax.transAxes)
    ylim = ax.get_ylim()                                   # shading must not change the data limits
    ax.fill_between(ep, 0, v["val_latent/kld_alpha"].values, transform=tf, color=GREY, alpha=0.12, lw=0, zorder=0)
    ax.set_ylim(ylim); ax.set_xlim(1, 40); ax.set_xticks([1, 10, 20, 30, 40])


def train_val(ax, v, te, key, ylabel, title, mark="min"):
    ep = v.index.values + 1
    ax.plot(te.index.values + 1, te[key].values, color=BLUE, lw=0.9, ls=(0, (3, 1.5)), alpha=0.6)
    ax.plot(ep, v[key].values, color=BLUE, lw=1.4)
    k = int(np.argmin(v[key].values) if mark == "min" else np.argmax(v[key].values))
    ax.scatter([ep[k]], [v[key].values[k]], s=14, color=BLUE, zorder=5, ec="white", lw=0.5)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=7.3, pad=3, loc="left")
    ax.set_xlabel("epoch")


def main(out):
    v, t, te = load()
    ep = v.index.values + 1
    fig, axs = plt.subplots(2, 3, figsize=(6.3, 4.5))
    (a, b, c), (d, e, f) = axs

    # (a) by optimiser step
    step = t["trainer/global_step"].values
    a.axvspan(step.min(), SPE, color=BLUE, alpha=0.07, lw=0)
    a.plot(step, t.rec.values, color=BLUE, lw=0.8, ls=(0, (3, 1.5)), alpha=0.6)
    a.plot(ep * SPE, v.rec.values, color=BLUE, lw=1.2, marker="o", ms=1.8)
    for e_, lab in [(6, r"$\alpha$ ramp"), (26, r"$\alpha=1$")]:
        a.axvline(e_ * SPE, color=GREY, lw=0.7, ls="--")
        a.text(e_ * SPE * 1.08, 0.068, lab, fontsize=5.8, color="#6f6f6f", va="top", rotation=90)
    a.text(np.sqrt(step.min() * SPE), 0.068, "epoch 1", ha="center", va="top", fontsize=6.0, color=BLUE)
    a.set_xscale("log"); a.set_xlim(step.min(), step.max()); a.set_ylim(0, 0.07)
    a.set_xlabel("optimiser step (log scale)"); a.set_ylabel("reconstruction term")
    a.set_title("(a) by optimiser step", fontsize=7.3, pad=3, loc="left")
    r2 = 1 - 0.0299 / 0.3233
    a.text(0.985, 0.50, rf"val. $R^2={r2:.2f}$" + "\n(MSE 0.030\nvs 0.323\nfor the mean)", transform=a.transAxes,
           ha="right", va="center", fontsize=5.8, linespacing=1.12, zorder=6,
           bbox=dict(fc="white", ec="#cccccc", lw=0.4, pad=1.8, alpha=0.95))

    # (b) the exchange: reconstruction term vs KL
    train_val(b, v, te, "rec", "reconstruction term", "(b) reconstruction vs KL")
    k = int(np.argmin(v.rec.values))
    b.annotate(rf"min. at $\alpha$={v['val_latent/kld_alpha'].values[k]:.2f}", (ep[k], v.rec.values[k]), xytext=(0.36, 0.05),
               textcoords="axes fraction", fontsize=5.8, color=BLUE,
               arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.5, shrinkA=0, shrinkB=2))
    b.set_ylim(0.0105, 0.0175); ramp(b, v)
    kx = b.twinx(); kx.spines["right"].set_visible(True); kx.spines["top"].set_visible(False)
    kx.plot(ep, v.kl.values, color=RED, lw=1.4); kx.set_yscale("log"); kx.set_ylim(0.4, 15)
    kx.set_ylabel("val. KL (log)", color=RED, labelpad=1); kx.tick_params(axis="y", colors=RED, which="both", labelsize=6)

    # (c) KL annealing
    c.plot(ep, v["val_latent/kld_alpha"].values, color=GREY, lw=1.4, ls="--")
    c.fill_between(ep, 0, v["val_latent/kld_alpha"].values, color=GREY, alpha=0.12, lw=0)
    c.set_ylim(0, 1.05); c.set_xlim(1, 40); c.set_xticks([1, 10, 20, 30, 40])
    c.set_ylabel(r"KL weight $\alpha$", color="#6f6f6f"); c.set_xlabel("epoch")
    c.set_title("(c) KL annealing", fontsize=7.3, pad=3, loc="left")
    cx = c.twinx(); cx.spines["right"].set_visible(True); cx.spines["top"].set_visible(False)
    w = KL_SCALE * v["val_latent/kld_alpha"].values * v.kl.values
    cx.plot(ep, w, color=RED, lw=1.4)
    cx.set_ylabel(r"KL term in loss, $0.01\,\alpha\,\mathrm{KL}$", color=RED, labelpad=1)
    cx.tick_params(axis="y", colors=RED, labelsize=6)

    # (d)-(f) the components and the monitor metric
    train_val(d, v, te, "huber", "Huber term", "(d) Huber component"); ramp(d, v)
    train_val(e, v, te, "cos", "mean cosine similarity", "(e) per-node cosine similarity", mark="max"); ramp(e, v)
    train_val(f, v, te, "snr", "SNR (dB)", "(f) SNR (monitor only)", mark="max"); ramp(f, v)

    handles = [plt.Line2D([], [], color=BLUE, lw=0.9, ls=(0, (3, 1.5)), alpha=0.6),
               plt.Line2D([], [], color=BLUE, lw=1.4), plt.Line2D([], [], color=RED, lw=1.4),
               plt.Rectangle((0, 0), 1, 1, fc=GREY, alpha=0.25, ec="none")]
    fig.legend(handles, ["train", "validation", "KL", r"KL weight $\alpha$ (shaded: 0 $\to$ 1)"], loc="lower center",
               ncol=4, frameon=False, fontsize=6.6, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{EXP}  (wandb run {RUN}, {NAME})", fontsize=7.6, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.975), h_pad=1.4, w_pad=1.3)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)


if __name__ == "__main__":
    main("thesis_figures/fig_training_curves")
