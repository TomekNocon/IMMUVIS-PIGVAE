"""Thesis figure (open item 14): the AE-VAE axis on the 16x16 grid.

(a) validation reconstruction MSE against the KL scale (log), AE (no VAE head) at the left end,
    with and without FiLM; right axis gives the same values as R^2 against predicting the mean.
(b) rate-distortion: validation KL (rate) against validation MSE (distortion), points labelled by scale.
All runs share the preprocessing (PCA-128 on the 16x16 grid, no whitening, no clip), free bits off.
Data: final wandb summaries -> thesis_figures/_cache/ae_vae_axis_runs.csv.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, GOLD, GREY, INK, RED = "#2f6db5", "#a87f10", "#8a8a8a", "#2b2b2b", "#c8323e"
plt.rcParams.update({"font.family": "serif", "font.size": 7.3, "mathtext.fontset": "dejavuserif",
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})
TGT_POW = 0.3233          # mean squared target = MSE of predicting the PCA mean (validation)
AE_X = 3e-4               # where the AE (no KL term) is drawn on the log axis


def main(out):
    df = pd.read_csv("thesis_figures/_cache/ae_vae_axis_runs.csv")
    sweep = df[(df.fb == 0.0) & (df.z == 32)]
    ae = sweep[sweep.vae == False].iloc[0]
    vae = sweep[sweep.vae == True]
    nofilm, film = vae[vae.film == False].sort_values("scale"), vae[vae.film == True].sort_values("scale")

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.6), gridspec_kw=dict(width_ratios=[1.15, 1]))

    # (a) distortion vs KL scale
    xa = np.r_[AE_X, nofilm.scale.values]; ya = np.r_[ae.vmse, nofilm.vmse.values]
    a.plot(xa, ya, color=BLUE, lw=1.2, marker="o", ms=4, label="without FiLM", zorder=3)
    a.plot(film.scale, film.vmse, color=GOLD, lw=1.2, marker="s", ms=4, label="with FiLM", zorder=4)
    fin = film[film.id == "81zasevr"].iloc[0]
    a.scatter([fin.scale], [fin.vmse], s=110, marker="*", color=RED, zorder=5, ec="white", lw=0.5, label="final model")
    a.set_xscale("log")
    a.set_xticks([AE_X, 1e-3, 1e-2, 3e-2]); a.set_xticklabels(["AE\n(no KL)", "0.001", "0.01", "0.03"])
    a.set_xlim(2e-4, 4.5e-2)
    a.axvspan(2e-4, 6e-4, color=GREY, alpha=0.08, lw=0)
    a.set_xlabel("KL scale (log)  →  more variational"); a.set_ylabel("validation reconstruction MSE")
    a.set_ylim(0, 0.085)
    ra = a.twinx(); ra.spines["right"].set_visible(True); ra.spines["top"].set_visible(False)
    ra.set_ylim(a.get_ylim())
    r2 = np.array([0.95, 0.9, 0.85, 0.8, 0.75])
    ra.set_yticks(TGT_POW * (1 - r2)); ra.set_yticklabels([f"{v:.2f}" for v in r2])
    ra.set_ylabel(r"$R^2$ (vs predicting the mean)", color="#555555"); ra.tick_params(axis="y", colors="#555555", labelsize=6.5)
    a.legend(frameon=False, fontsize=6.5, loc="upper left")
    a.set_title("(a) reconstruction along the AE–VAE axis", fontsize=7.5, loc="left", pad=4)

    # (b) rate-distortion
    b.plot(nofilm.vkl, nofilm.vmse, color=BLUE, lw=1.2, marker="o", ms=4)
    b.plot(film.vkl, film.vmse, color=GOLD, lw=1.2, marker="s", ms=4)
    b.scatter([fin.vkl], [fin.vmse], s=110, marker="*", color=RED, zorder=5, ec="white", lw=0.5)
    for _, r in nofilm.iterrows():
        b.annotate(f"{r.scale:g}", (r.vkl, r.vmse), xytext=(5, 4), textcoords="offset points", fontsize=6.3, color=BLUE)
    b.annotate("0.01, FiLM\n(final model)", (fin.vkl, fin.vmse), xytext=(10, -26), textcoords="offset points", fontsize=6.3, color=RED)
    b.axhline(ae.vmse, color=GREY, lw=0.8, ls="--")
    b.text(2.45, ae.vmse - 0.0025, "AE (no KL term)", ha="right", va="top", fontsize=6.3, color="#666666")
    b.set_xlabel("validation KL per latent dimension (rate)"); b.set_ylabel("validation MSE (distortion)")
    b.set_xlim(0, 2.5); b.set_ylim(0, 0.085)
    b.set_title("(b) rate–distortion", fontsize=7.5, loc="left", pad=4)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)


if __name__ == "__main__":
    main("thesis_figures/fig_ae_vae_axis")
