import torch

from src.models.components.losses import CosineSimilarityLoss


def test_cosine_loss_zero_for_identical():
    x = torch.randn(2, 5, 8)
    assert CosineSimilarityLoss()(x.clone(), x).item() < 1e-5


def test_cosine_loss_is_per_node_not_global():
    # node 0: matched direction, large magnitude (cos 1); node 1: orthogonal, small magnitude (cos 0).
    target = torch.tensor([[[3.0, 0, 0, 0], [1.0, 0, 0, 0]]])
    pred = torch.tensor([[[3.0, 0, 0, 0], [0.0, 1, 0, 0]]])
    loss = CosineSimilarityLoss()(pred, target).item()
    # per-node: mean(cos) = (1 + 0)/2 = 0.5 -> 1 - 0.5 = 0.5
    # (the old global/flattened version would be ~0.1, dominated by the large node 0)
    assert abs(loss - 0.5) < 1e-5
