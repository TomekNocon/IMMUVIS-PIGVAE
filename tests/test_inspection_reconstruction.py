# tests/test_inspection_reconstruction.py
import torch
from src.utils.inspection.reconstruction import reconstruction_diagnostics


def test_perfect_reconstruction():
    true = torch.randn(8, 36, 16)
    out = reconstruction_diagnostics(true.clone(), true, num_views=8)
    assert out["overall_mse"] < 1e-10
    assert min(out["per_channel_r2"]) > 0.999
    assert len(out["per_channel_mse"]) == 16
    assert len(out["per_position_mse"]) == 36
    assert len(out["per_view_mse"]) == 8


def test_one_bad_channel_is_flagged_worst():
    torch.manual_seed(0)
    true = torch.randn(8, 36, 16)
    pred = true.clone()
    pred[..., 5] = 0.0                       # destroy channel 5
    out = reconstruction_diagnostics(pred, true, num_views=8)
    assert out["worst_channels"][0] == 5
    assert out["per_channel_r2"][5] < 0.1
