# src/utils/inspection/latent.py
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.utils.inspection.stats import energy_rank, participation_ratio


def latent_diagnostics(z_nodes: torch.Tensor) -> dict:
    """Bottleneck health for per-node latent z_nodes [B, N, D]."""
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

    return {
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
