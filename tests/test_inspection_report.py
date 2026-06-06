# tests/test_inspection_report.py
import json
from src.utils.inspection.report import build_flags, write_report


def test_build_flags_detects_anomalies():
    results = {
        "weights": {"decoder.norm": {"type": "RMSNorm", "gain_rms": 0.3, "section": "decoder"}},
        "activations": {"encoder.graph_transformer": {"max_abs": 260.0}},
        "latent": {"z_dim": 32, "active_dims": 5, "rank_ratio": 0.3},
        "reconstruction": {"per_channel_r2": [0.9, 0.1, 0.8], "worst_channels": [1]},
    }
    flags = build_flags(results)
    text = " ".join(flags)
    assert any("max_abs" in f for f in flags)
    assert any("gain" in f.lower() for f in flags)
    assert any("rank" in f.lower() or "active" in f.lower() for f in flags)


def test_write_report_creates_artifacts(tmp_path):
    results = {
        "meta": {"run": "test"},
        "weights": {}, "activations": {}, "attention": {},
        "latent": {"z_dim": 32, "active_dims": 30, "rank_ratio": 0.9},
        "reconstruction": {"per_channel_r2": [0.9], "per_channel_mse": [0.1],
                           "per_position_mse": [0.1], "worst_channels": [0],
                           "per_view_mse": None, "error_by_magnitude": []},
    }
    write_report(results, tmp_path)
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
    loaded = json.loads((tmp_path / "report.json").read_text())
    assert loaded["meta"]["run"] == "test"
