# src/utils/inspection/latent.py
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.utils.inspection.stats import participation_ratio


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
        "node_norm_mean": node_norms.mean().item(),
        "node_norm_std": node_norms.std(unbiased=False).item(),
        "node_norm_min": node_norms.min().item(),
        "node_norm_max": node_norms.max().item(),
        "inter_node_cossim_mean": off_diag.mean().item(),
    }
