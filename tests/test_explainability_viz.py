import torch

from src.explainability.viz import node_importance_to_grid


def test_node_importance_to_grid_rowmajor():
    imp = torch.arange(256).float()
    g = node_importance_to_grid(imp, 16)
    assert g.shape == (16, 16)
    # row-major: node i -> (i // 16, i % 16)
    assert g[0, 0] == 0 and g[0, 1] == 1 and g[1, 0] == 16
