# tests/test_inspection_weights.py
import torch.nn as nn

from src.utils.inspection.weights import weight_diagnostics


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8))
        self.decoder = nn.Linear(8, 4)


def test_weight_diagnostics_keys_and_sections():
    out = weight_diagnostics(Tiny())
    # one Linear in encoder, one LayerNorm in encoder, one Linear in decoder
    lin_keys = [k for k, v in out.items() if "spectral_norm" in v]
    norm_keys = [k for k, v in out.items() if v.get("type") == "LayerNorm"]
    assert len(lin_keys) == 2
    assert len(norm_keys) == 1
    assert out[lin_keys[0]]["section"] in {"encoder", "decoder"}
    assert any(v["section"] == "decoder" for v in out.values())
    # LayerNorm gain init = 1 -> gain_rms ~ 1
    assert abs(out[norm_keys[0]]["gain_rms"] - 1.0) < 1e-4
