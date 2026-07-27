# src/utils/inspection/latent.py
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.utils.inspection.stats import energy_rank, participation_ratio


def latent_diagnostics(
    z_nodes: torch.Tensor,
    mu: torch.Tensor | None = None,
    logvar: torch.Tensor | None = None,
) -> dict:
    """Bottleneck health for per-node latent z_nodes [B, N, D].

    When the encoder ``mu``/``logvar`` are supplied (VAE), also locate the run on
    the AE<->VAE spectrum (``spectrum`` block) via the aggregate-posterior
    decomposition — see :func:`ae_vae_spectrum`.
    """
    z = z_nodes.detach().float()
    b, n, d = z.shape
    flat = z.reshape(b * n, d)

    var_per_dim = flat.var(dim=0, unbiased=False)
    max_var = var_per_dim.max().clamp_min(1e-12)
    active = int((var_per_dim > 0.01 * max_var).sum().item())

    centered = flat - flat.mean(0, keepdim=True)
    cov = (centered.T @ centered) / flat.shape[0]
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    eff_rank = participation_ratio(eig)
    # Honest "how narrow can z go": smallest #dims holding 90%/99% of the latent variance.
    # eig are variances (energy); energy_rank squares its input, so pass sqrt(eig).
    er = energy_rank(eig.sqrt())

    node_norms = z.norm(dim=-1)
    zn = F.normalize(z, dim=-1)
    sim = torch.matmul(zn, zn.transpose(1, 2))
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    off_diag = sim[:, ~eye]

    out = {
        "z_dim": int(d),
        "active_dims": active,
        "effective_rank": eff_rank,
        "rank_ratio": eff_rank / d,
        "energy_rank90": er["rank90"],
        "energy_rank99": er["rank99"],
        "energy_rank90_ratio": er["rank90"] / d,
        "node_norm_mean": node_norms.mean().item(),
        "node_norm_std": node_norms.std(unbiased=False).item(),
        "node_norm_min": node_norms.min().item(),
        "node_norm_max": node_norms.max().item(),
        "inter_node_cossim_mean": off_diag.mean().item(),
    }
    if mu is not None and logvar is not None:
        out["spectrum"] = ae_vae_spectrum(mu, logvar)
        out["ae_vae_position"] = out["spectrum"]["verdict"]
    return out


def ae_vae_spectrum(mu: torch.Tensor, logvar: torch.Tensor) -> dict:
    """Locate the latent on the AE<->VAE spectrum from encoder ``mu``/``logvar``.

    The dial is the per-sample encoder noise sigma vs. the spread of the codes
    (``mu``). The principled summary is the diagonal of the **aggregate posterior**
    covariance, which the law of total variance splits per dim into

        Var_x(mu_d)      "signal / rate"   — how far apart codes for different inputs sit
        E_x[sigma_d^2]   "noise"           — how wide each posterior blob is

    and whose sum should be ~1.0 per dim if the aggregate posterior matches the
    N(0,1) prior. From these:

    - ``overlap_ratio`` = sigma_rms / spacing_rms  (the wiki's sigma/spacing).
        << 1  -> tight, well-separated blobs = AE-ward: sharp recon, *holes* in the
                 latent (prior samples land where no posterior is -> bad generation).
        ~ 1  -> blobs tile the prior = balanced.
        >> 1  -> blobs overlap = over-regularised / posterior-collapse-leaning:
                 the decoder can't tell inputs apart -> blurry, info-poor latent.
    - ``noise_fraction`` = E[sigma^2] / aggregate_var  in [0, 1]; 0 = deterministic
        AE, 1 = full collapse. A scale-free coordinate on the same axis.

    Cheap (encoder-side only) and consistent with the live wandb ``std_mean`` /
    ``mu_std`` so offline numbers line up with the training curves.
    """
    mu = mu.detach().float().reshape(-1, mu.shape[-1])
    logvar = logvar.detach().float().reshape(-1, logvar.shape[-1])
    sigma = (0.5 * logvar).exp()
    dim = mu.shape[-1]

    mu_var = mu.var(dim=0, unbiased=False)        # between-code variance (signal/rate)
    sigma_sq = sigma.pow(2).mean(dim=0)           # within-code variance (noise)
    agg_var = mu_var + sigma_sq                   # aggregate-posterior diagonal (prior -> 1)
    kld_per_dim = 0.5 * (mu.pow(2) + sigma.pow(2) - 1.0 - logvar).mean(dim=0)

    sigma_rms = sigma_sq.mean().sqrt()
    spacing_rms = mu_var.mean().clamp_min(1e-12).sqrt()
    overlap = (sigma_rms / spacing_rms).item()
    noise_fraction = (sigma_sq.mean() / agg_var.mean().clamp_min(1e-12)).item()

    if overlap < 0.3:
        verdict = (
            f"AE-ward (overlap {overlap:.2f}): near-deterministic codes — sharp recon, "
            "latent holes, prior-sampling unreliable"
        )
    elif overlap <= 1.0:
        verdict = (
            f"balanced (overlap {overlap:.2f}): sigma comparable to code spacing — "
            "usable for both recon and prior-sampling"
        )
    else:
        verdict = (
            f"over-regularised (overlap {overlap:.2f}): posteriors overlap heavily — "
            "blurry / info-poor, collapse-leaning"
        )

    return {
        "verdict": verdict,
        "overlap_ratio_sigma_over_spacing": round(overlap, 4),
        "noise_fraction_0AE_1collapse": round(noise_fraction, 4),
        "sigma_rms": round(sigma_rms.item(), 5),
        "code_spacing_rms": round(spacing_rms.item(), 5),
        "aggregate_var_mean_prior1": round(agg_var.mean().item(), 4),
        "aggregate_mean_abs_prior0": round(mu.mean(0).abs().mean().item(), 5),
        "prior_match_gap": round((agg_var - 1.0).abs().mean().item(), 4),
        "mu_var_mean_signal": round(mu_var.mean().item(), 5),
        "sigma_sq_mean_noise": round(sigma_sq.mean().item(), 5),
        "kld_per_node": round(kld_per_dim.sum().item(), 4),
        "active_units_mu_var>0.01": int((mu_var > 0.01).sum().item()),
        "collapsed_dims_kld<0.01": int((kld_per_dim < 0.01).sum().item()),
        "dims_underfilled_aggvar<0.5": int((agg_var < 0.5).sum().item()),
        "dims_prior_matched_0.5to1.5": int(((agg_var >= 0.5) & (agg_var <= 1.5)).sum().item()),
        "dims_overspread_aggvar>1.5": int((agg_var > 1.5).sum().item()),
        "total_dims": int(dim),
    }


@torch.no_grad()
def film_diagnostics(decoder, z_global: torch.Tensor) -> dict:
    """FiLM gamma/beta magnitudes per layer — confirms whether tanh-bounding fired.

    With ``film_bound`` on, gamma/beta are tanh-squashed so |gamma|,|beta| <= 1. If the
    readout shows values > 1 the bound is NOT active; if <= 1 it is (or the projections
    are still small). Returns {} when the decoder has no FiLM.
    """
    film = getattr(decoder, "film", None)
    if film is None:
        return {}
    params = film(z_global.detach().float())
    per_layer = [
        {
            "layer": i,
            "gamma_absmax": float(g.abs().max()),
            "gamma_absmean": float(g.abs().mean()),
            "beta_absmax": float(b.abs().max()),
            "beta_absmean": float(b.abs().mean()),
        }
        for i, (g, b) in enumerate(params)
    ]
    return {
        "bound": bool(getattr(film, "bound", False)),
        "gamma_absmax_overall": max(p["gamma_absmax"] for p in per_layer),
        "beta_absmax_overall": max(p["beta_absmax"] for p in per_layer),
        "per_layer": per_layer,
    }
