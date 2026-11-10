"""Training curves of the reference experiment (experiment=pigvae, wandb run 81zasevr).

(a) reconstruction term by optimiser step (log); train every 250 steps, validation once per epoch
(b) reconstruction term vs validation KL per epoch: the exchange along the alpha ramp
(c) KL annealing: the weight alpha and the KL term as it enters the loss (0.01 * alpha * KL)
(d) Huber component, (e) per-node cosine similarity (= 1 - beta_recon / 0.1),
(f) reconstruction SNR in dB (logged for monitoring, not optimised).
Reconstruction term = alpha_recon + beta_recon + gamma_recon (Huber + 0.1 cosine + 0.001 gradient).
Epoch panels shade the alpha ramp. Data: wandb history in ``figures/_cache/history_81zasevr.csv``.

Usage:
    uv run python scripts/thesis_figures/fig_training_curves.py
"""

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

from figlib import io, paths, style

GREY = "#9a9a9a"  # slightly lighter than style.GREY for the shaded ramp
NOTE = "#6f6f6f"
KL_SCALE = 0.01  # kld_loss_scale of the run
STEPS_PER_EPOCH = 5000
RUN_NAME = "vae16-fb0.0-s0.01-FILM-start5-40ep"
TRAIN_LS = (0, (3, 1.5))
ALPHA = "val_latent/kld_alpha"


def load_history() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(validation per epoch, train per log step, train averaged per epoch) with derived terms."""
    df = pd.read_csv(paths.CACHE / f"history_{paths.REF_WANDB_ID}.csv")
    val = df.dropna(subset=["val_alpha_recon"]).set_index("epoch").sort_index()
    train = df.dropna(subset=["alpha_recon"]).copy()
    for d, prefix in [(val, "val_"), (train, "")]:
        d["rec"] = d[prefix + "alpha_recon"] + d[prefix + "beta_recon"] + d[prefix + "gamma_recon"]
        d["huber"] = d[prefix + "alpha_recon"]
        d["cos"] = 1 - d[prefix + "beta_recon"] / 0.1
        d["snr"] = d[prefix + "signal_to_noise_ratio_loss"]
        d["kl"] = d[prefix + "kld_loss"]
    train_epoch = train.groupby("epoch")[["rec", "huber", "cos", "snr", "kl"]].mean()
    return val, train, train_epoch


def shade_ramp(ax, val) -> None:
    """Shade the KL weight alpha (0 -> 1 over the panel height) behind an epoch panel."""
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    ylim = ax.get_ylim()  # shading must not change the data limits
    ax.fill_between(
        val.index.values + 1,
        0,
        val[ALPHA].values,
        transform=transform,
        color=GREY,
        alpha=0.12,
        lw=0,
        zorder=0,
    )
    ax.set_ylim(ylim)
    ax.set_xlim(1, 40)
    ax.set_xticks([1, 10, 20, 30, 40])


def twin_right(ax):
    twin = ax.twinx()
    twin.spines["right"].set_visible(True)
    twin.spines["top"].set_visible(False)
    return twin


def plot_train_val(
    ax, val, train_epoch, key: str, ylabel: str, title: str, best: str = "min"
) -> None:
    epochs = val.index.values + 1
    ax.plot(
        train_epoch.index.values + 1,
        train_epoch[key].values,
        color=style.BLUE,
        lw=0.9,
        ls=TRAIN_LS,
        alpha=0.6,
    )
    ax.plot(epochs, val[key].values, color=style.BLUE, lw=1.4)
    k = int(np.argmin(val[key].values) if best == "min" else np.argmax(val[key].values))
    ax.scatter(
        [epochs[k]], [val[key].values[k]], s=14, color=style.BLUE, zorder=5, ec="white", lw=0.5
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=7.3, pad=3, loc="left")
    ax.set_xlabel("epoch")


def plot_by_step(ax, val, train) -> None:
    """(a) reconstruction term against the optimiser step."""
    step = train["trainer/global_step"].values
    ax.axvspan(step.min(), STEPS_PER_EPOCH, color=style.BLUE, alpha=0.07, lw=0)
    ax.plot(step, train.rec.values, color=style.BLUE, lw=0.8, ls=TRAIN_LS, alpha=0.6)
    ax.plot(
        (val.index.values + 1) * STEPS_PER_EPOCH,
        val.rec.values,
        color=style.BLUE,
        lw=1.2,
        marker="o",
        ms=1.8,
    )
    for epoch, label in [(6, r"$\alpha$ ramp"), (26, r"$\alpha=1$")]:
        ax.axvline(epoch * STEPS_PER_EPOCH, color=GREY, lw=0.7, ls="--")
        ax.text(
            epoch * STEPS_PER_EPOCH * 1.08,
            0.068,
            label,
            fontsize=5.8,
            color=NOTE,
            va="top",
            rotation=90,
        )
    ax.text(
        np.sqrt(step.min() * STEPS_PER_EPOCH),
        0.068,
        "epoch 1",
        ha="center",
        va="top",
        fontsize=6.0,
        color=style.BLUE,
    )
    ax.set_xscale("log")
    ax.set_xlim(step.min(), step.max())
    ax.set_ylim(0, 0.07)
    ax.set_xlabel("optimiser step (log scale)")
    ax.set_ylabel("reconstruction term")
    ax.set_title("(a) by optimiser step", fontsize=7.3, pad=3, loc="left")
    r2 = 1 - 0.0299 / 0.3233
    ax.text(
        0.985,
        0.50,
        rf"val. $R^2={r2:.2f}$" + "\n(MSE 0.030\nvs 0.323\nfor the mean)",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=5.8,
        linespacing=1.12,
        zorder=6,
        bbox={"fc": "white", "ec": "#cccccc", "lw": 0.4, "pad": 1.8, "alpha": 0.95},
    )


def plot_exchange(ax, val, train_epoch) -> None:
    """(b) reconstruction term against the validation KL."""
    epochs = val.index.values + 1
    plot_train_val(ax, val, train_epoch, "rec", "reconstruction term", "(b) reconstruction vs KL")
    k = int(np.argmin(val.rec.values))
    ax.annotate(
        rf"min. at $\alpha$={val[ALPHA].values[k]:.2f}",
        (epochs[k], val.rec.values[k]),
        xytext=(0.36, 0.05),
        textcoords="axes fraction",
        fontsize=5.8,
        color=style.BLUE,
        arrowprops={"arrowstyle": "-", "color": style.BLUE, "lw": 0.5, "shrinkA": 0, "shrinkB": 2},
    )
    ax.set_ylim(0.0105, 0.0175)
    shade_ramp(ax, val)
    kl_axis = twin_right(ax)
    kl_axis.plot(epochs, val.kl.values, color=style.RED, lw=1.4)
    kl_axis.set_yscale("log")
    kl_axis.set_ylim(0.4, 15)
    kl_axis.set_ylabel("val. KL (log)", color=style.RED, labelpad=1)
    kl_axis.tick_params(axis="y", colors=style.RED, which="both", labelsize=6)


def plot_annealing(ax, val) -> None:
    """(c) the KL weight alpha and the KL term as it enters the loss."""
    epochs = val.index.values + 1
    ax.plot(epochs, val[ALPHA].values, color=GREY, lw=1.4, ls="--")
    ax.fill_between(epochs, 0, val[ALPHA].values, color=GREY, alpha=0.12, lw=0)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, 40)
    ax.set_xticks([1, 10, 20, 30, 40])
    ax.set_ylabel(r"KL weight $\alpha$", color=NOTE)
    ax.set_xlabel("epoch")
    ax.set_title("(c) KL annealing", fontsize=7.3, pad=3, loc="left")
    loss_axis = twin_right(ax)
    loss_axis.plot(epochs, KL_SCALE * val[ALPHA].values * val.kl.values, color=style.RED, lw=1.4)
    loss_axis.set_ylabel(
        r"KL term in loss, $0.01\,\alpha\,\mathrm{KL}$", color=style.RED, labelpad=1
    )
    loss_axis.tick_params(axis="y", colors=style.RED, labelsize=6)


def main(out: str) -> None:
    style.use_serif(7.2, thin_axes=True, open_axes=True)
    val, train, train_epoch = load_history()
    fig, axs = plt.subplots(2, 3, figsize=(6.3, 4.5))
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axs

    plot_by_step(ax_a, val, train)
    plot_exchange(ax_b, val, train_epoch)
    plot_annealing(ax_c, val)
    plot_train_val(ax_d, val, train_epoch, "huber", "Huber term", "(d) Huber component")
    shade_ramp(ax_d, val)
    plot_train_val(
        ax_e,
        val,
        train_epoch,
        "cos",
        "mean cosine similarity",
        "(e) per-node cosine similarity",
        best="max",
    )
    shade_ramp(ax_e, val)
    plot_train_val(ax_f, val, train_epoch, "snr", "SNR (dB)", "(f) SNR (monitor only)", best="max")
    shade_ramp(ax_f, val)

    handles = [
        plt.Line2D([], [], color=style.BLUE, lw=0.9, ls=TRAIN_LS, alpha=0.6),
        plt.Line2D([], [], color=style.BLUE, lw=1.4),
        plt.Line2D([], [], color=style.RED, lw=1.4),
        plt.Rectangle((0, 0), 1, 1, fc=GREY, alpha=0.25, ec="none"),
    ]
    fig.legend(
        handles,
        ["train", "validation", "KL", r"KL weight $\alpha$ (shaded: 0 $\to$ 1)"],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=6.6,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        f"{paths.REF_EXPERIMENT}  (wandb run {paths.REF_WANDB_ID}, {RUN_NAME})",
        fontsize=7.6,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.975), h_pad=1.4, w_pad=1.3)
    io.save(fig, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(paths.figure("fig_training_curves")))
    main(ap.parse_args().out)
