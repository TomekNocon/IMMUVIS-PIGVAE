# tests/test_inspection_stats.py
import torch

from src.utils.inspection.stats import linear_spectral, participation_ratio, tensor_stats


def test_tensor_stats_basic():
    t = torch.arange(0, 100, dtype=torch.float32)
    s = tensor_stats(t)
    assert s["count"] == 100
    assert abs(s["mean"] - 49.5) < 1e-3
    assert s["max"] == 99.0
    assert s["max_abs"] == 99.0
    assert "p50" in s and "kurtosis" in s and "frac_dead" in s


def test_tensor_stats_empty():
    assert tensor_stats(torch.empty(0))["count"] == 0


def test_participation_ratio_uniform_vs_spiked():
    uniform = torch.ones(10)
    assert abs(participation_ratio(uniform) - 10.0) < 1e-4   # all equal -> full PR
    spiked = torch.tensor([100.0, 1e-6, 1e-6])
    assert participation_ratio(spiked) < 1.1                 # dominated by one -> ~1


def test_linear_spectral_identity():
    w = torch.eye(8)
    out = linear_spectral(w)
    assert abs(out["spectral_norm"] - 1.0) < 1e-4
    assert abs(out["effective_rank"] - 8.0) < 1e-3
    assert abs(out["rank_ratio"] - 1.0) < 1e-3
