"""The AE-VAE axis on the 16x16 grid.

(a) validation reconstruction MSE against the KL scale (log), AE (no VAE head) at the left end, with
    and without FiLM; the right axis gives the same values as R^2 against predicting the mean.
(b) rate-distortion: validation KL (rate) against validation MSE (distortion), labelled by scale.
All runs share the preprocessing (PCA-128 on the 16x16 grid, no whitening, no clip), free bits off.
Data: final wandb summaries in ``figures/_cache/ae_vae_axis_runs.csv``.

Usage:
    uv run python scripts/thesis_figures/fig_ae_vae_axis.py
"""

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figlib import io, paths, style

TARGET_POWER = 0.3233  # mean squared target = MSE of predicting the PCA mean (validation)
AE_X = 3e-4  # where the AE (no KL term) is drawn on the log axis
NOTE = "#555555"


def load_runs() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
    """(AE run, VAE runs without FiLM, VAE runs with FiLM, final model)."""
    df = pd.read_csv(paths.CACHE / "ae_vae_axis_runs.csv")
    sweep = df[(df.fb == 0.0) & (df.z == 32)]
    ae = sweep[~sweep.vae.astype(bool)].iloc[0]
    vae = sweep[sweep.vae.astype(bool)]
    no_film = vae[~vae.film.astype(bool)].sort_values("scale")
    film = vae[vae.film.astype(bool)].sort_values("scale")
    final = film[film.id == paths.REF_WANDB_ID].iloc[0]
    return ae, no_film, film, final


def plot_distortion(ax, ae, no_film, film, final) -> None:
    ax.plot(
        np.r_[AE_X, no_film.scale.values],
        np.r_[ae.vmse, no_film.vmse.values],
        color=style.BLUE,
        lw=1.2,
        marker="o",
        ms=4,
        label="without FiLM",
        zorder=3,
    )
    ax.plot(
        film.scale,
        film.vmse,
        color=style.GOLD,
        lw=1.2,
        marker="s",
        ms=4,
        label="with FiLM",
        zorder=4,
    )
    ax.scatter(
        [final.scale],
        [final.vmse],
        s=110,
        marker="*",
        color=style.RED,
        zorder=5,
        ec="white",
        lw=0.5,
        label="final model",
    )
    ax.set_xscale("log")
    ax.set_xticks([AE_X, 1e-3, 1e-2, 3e-2])
    ax.set_xticklabels(["AE\n(no KL)", "0.001", "0.01", "0.03"])
    ax.set_xlim(2e-4, 4.5e-2)
    ax.axvspan(2e-4, 6e-4, color=style.GREY, alpha=0.08, lw=0)
    ax.set_xlabel("KL scale (log)  →  more variational")
    ax.set_ylabel("validation reconstruction MSE")
    ax.set_ylim(0, 0.085)

    r2_axis = ax.twinx()
    r2_axis.spines["right"].set_visible(True)
    r2_axis.spines["top"].set_visible(False)
    r2_axis.set_ylim(ax.get_ylim())
    r2 = np.array([0.95, 0.9, 0.85, 0.8, 0.75])
    r2_axis.set_yticks(TARGET_POWER * (1 - r2))
    r2_axis.set_yticklabels([f"{v:.2f}" for v in r2])
    r2_axis.set_ylabel(r"$R^2$ (vs predicting the mean)", color=NOTE)
    r2_axis.tick_params(axis="y", colors=NOTE, labelsize=6.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.set_title("(a) reconstruction along the AE–VAE axis", fontsize=7.5, loc="left", pad=4)


def plot_rate_distortion(ax, ae, no_film, film, final) -> None:
    ax.plot(no_film.vkl, no_film.vmse, color=style.BLUE, lw=1.2, marker="o", ms=4)
    ax.plot(film.vkl, film.vmse, color=style.GOLD, lw=1.2, marker="s", ms=4)
    ax.scatter(
        [final.vkl], [final.vmse], s=110, marker="*", color=style.RED, zorder=5, ec="white", lw=0.5
    )
    for _, run in no_film.iterrows():
        ax.annotate(
            f"{run.scale:g}",
            (run.vkl, run.vmse),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=6.3,
            color=style.BLUE,
        )
    ax.annotate(
        "0.01, FiLM\n(final model)",
        (final.vkl, final.vmse),
        xytext=(10, -26),
        textcoords="offset points",
        fontsize=6.3,
        color=style.RED,
    )
    ax.axhline(ae.vmse, color=style.GREY, lw=0.8, ls="--")
    ax.text(
        2.45,
        ae.vmse - 0.0025,
        "AE (no KL term)",
        ha="right",
        va="top",
        fontsize=6.3,
        color="#666666",
    )
    ax.set_xlabel("validation KL per latent dimension (rate)")
    ax.set_ylabel("validation MSE (distortion)")
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 0.085)
    ax.set_title("(b) rate–distortion", fontsize=7.5, loc="left", pad=4)


def main(out: str) -> None:
    style.use_serif(7.3, thin_axes=True, open_axes=True)
    runs = load_runs()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(6.3, 2.6), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    plot_distortion(ax_a, *runs)
    plot_rate_distortion(ax_b, *runs)
    fig.tight_layout(w_pad=2.2)
    io.save(fig, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(paths.figure("fig_ae_vae_axis")))
    main(ap.parse_args().out)
