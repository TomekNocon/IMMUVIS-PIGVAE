import torch


def test_gated_abmil_shapes_and_masking():
    from src.downstream.abmil.model import GatedABMIL

    m = GatedABMIL(emb_dim=16, hidden_dim=8, num_heads=2, num_classes=2)
    x = torch.randn(3, 5, 16)
    mask = torch.zeros(3, 5, dtype=torch.bool)
    mask[:, 4] = True
    logits, pooled = m(x, mask=mask)
    assert logits.shape == (3, 1) and pooled.shape == (3, 2 * 16)

    # Regression check: `mask` uses the True == padding convention, so
    # perturbing only the masked (padded) instances must leave the pooled
    # representation and logits unchanged. This fails if the mask sense is
    # ever inverted (e.g. True treated as "valid" or masked_fill(~mask, ...)).
    m.eval()
    with torch.no_grad():
        assert (~mask).any(), "at least one instance must remain unmasked"
        logits1, pooled1 = m(x, mask=mask)

        x_perturbed = x.clone()
        x_perturbed[mask] = x_perturbed[mask] + 1000.0

        logits2, pooled2 = m(x_perturbed, mask=mask)

        assert torch.allclose(pooled1, pooled2, atol=1e-5)
        assert torch.allclose(logits1, logits2, atol=1e-5)
