"""AE<->VAE spectrum diagnostic — synthetic-latent sanity checks.

Builds per-node mu/logvar with known overlap and asserts ae_vae_spectrum() places
them on the right part of the axis (AE-ward / balanced / over-regularised), and that
the law-of-total-variance decomposition holds.
"""
import math

import torch

from src.utils.inspection.latent import ae_vae_spectrum, latent_diagnostics


def _make(spacing: float, sigma: float, dim: int = 256, n: int = 4000):
    """mu ~ N(0, spacing^2) per dim; constant sigma -> known overlap = sigma/spacing."""
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(n, dim, generator=g) * spacing
    logvar = torch.full((n, dim), 2.0 * math.log(sigma))
    return mu, logvar


def test_ae_ward_tight_blobs_well_separated():
    s = ae_vae_spectrum(*_make(spacing=1.0, sigma=0.05))
    assert s["overlap_ratio_sigma_over_spacing"] < 0.3
    assert s["noise_fraction_0AE_1collapse"] < 0.05
    assert s["verdict"].startswith("AE-ward")


def test_balanced_blobs_tile_prior():
    s = ae_vae_spectrum(*_make(spacing=0.7, sigma=0.5))
    assert 0.3 <= s["overlap_ratio_sigma_over_spacing"] <= 1.0
    assert s["verdict"].startswith("balanced")


def test_over_regularised_overlapping_blobs():
    s = ae_vae_spectrum(*_make(spacing=0.1, sigma=1.0))
    assert s["overlap_ratio_sigma_over_spacing"] > 1.0
    assert s["noise_fraction_0AE_1collapse"] > 0.6
    assert s["verdict"].startswith("over-regularised")


def test_law_of_total_variance_identity():
    s = ae_vae_spectrum(*_make(spacing=0.8, sigma=0.4))
    agg = s["aggregate_var_mean_prior1"]
    parts = s["mu_var_mean_signal"] + s["sigma_sq_mean_noise"]
    assert abs(agg - parts) < 1e-2
    assert 0.0 <= s["noise_fraction_0AE_1collapse"] <= 1.0


def test_overlap_is_monotonic_in_sigma():
    lo = ae_vae_spectrum(*_make(spacing=0.7, sigma=0.1))["overlap_ratio_sigma_over_spacing"]
    mid = ae_vae_spectrum(*_make(spacing=0.7, sigma=0.5))["overlap_ratio_sigma_over_spacing"]
    hi = ae_vae_spectrum(*_make(spacing=0.7, sigma=1.5))["overlap_ratio_sigma_over_spacing"]
    assert lo < mid < hi


def test_latent_diagnostics_attaches_spectrum_when_mu_logvar_given():
    mu, logvar = _make(spacing=0.7, sigma=0.5)
    z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
    z_nodes = z.reshape(40, 100, 256)  # [B, N, D]
    out = latent_diagnostics(z_nodes, mu=z_nodes, logvar=logvar.reshape(40, 100, 256))
    assert "spectrum" in out and "ae_vae_position" in out
    # Backward-compat: no mu/logvar -> no spectrum block, no crash.
    assert "spectrum" not in latent_diagnostics(z_nodes)
