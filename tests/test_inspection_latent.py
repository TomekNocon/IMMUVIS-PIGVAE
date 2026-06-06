# tests/test_inspection_latent.py
import torch
from src.utils.inspection.latent import latent_diagnostics


def test_latent_full_rank_random():
    torch.manual_seed(0)
    z = torch.randn(8, 36, 32)
    out = latent_diagnostics(z)
    assert out["z_dim"] == 32
    assert out["active_dims"] >= 30          # random -> nearly all dims active
    assert out["rank_ratio"] > 0.8           # random -> near full rank
    assert out["node_norm_std"] > 0          # magnitude varies


def test_latent_collapsed():
    z = torch.zeros(8, 36, 32)
    z[..., 0] = torch.randn(8, 36)           # only one dim carries variance
    out = latent_diagnostics(z)
    assert out["active_dims"] == 1
    assert out["rank_ratio"] < 0.1
