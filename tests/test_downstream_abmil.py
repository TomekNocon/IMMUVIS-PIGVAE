import torch


def test_gated_abmil_shapes_and_masking():
    from src.downstream.abmil.model import GatedABMIL

    m = GatedABMIL(emb_dim=16, hidden_dim=8, num_heads=2, num_classes=2)
    x = torch.randn(3, 5, 16)
    mask = torch.zeros(3, 5, dtype=torch.bool)
    mask[:, 4] = True
    logits, pooled = m(x, mask=mask)
    assert logits.shape == (3, 1) and pooled.shape == (3, 2 * 16)
