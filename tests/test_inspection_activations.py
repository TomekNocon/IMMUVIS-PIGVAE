# tests/test_inspection_activations.py
import torch
import torch.nn as nn

from src.utils.inspection.activations import collect_activation_stats


class Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x):
        return self.b(self.a(x))


def test_collect_activation_stats_per_module_and_token():
    model = Seq()
    x = torch.randn(4, 36, 8)
    stats = collect_activation_stats(model, lambda: model(x),
                                     module_filter=lambda n, m: isinstance(m, nn.Linear))
    assert "a" in stats and "b" in stats
    assert "max_abs" in stats["a"]
    # 3D output -> per-token norms recorded with length N=36
    assert len(stats["a"]["per_token_norm"]) == 36
    assert "top_dims_by_mean_abs" in stats["a"]
