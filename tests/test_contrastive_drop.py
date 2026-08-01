import torch
from src.data.components.graphs_datamodules import DenseGraphBatch


def _full_batch(B=4, N=256, D=8):
    return DenseGraphBatch(
        node_features=torch.randn(B, N, D),
        edge_features=torch.empty(0),
        mask=torch.ones(B, N, dtype=torch.bool),
    )


def test_drop_views_masks_fraction_shares_features_and_differs():
    from src.models.components.contrastive import drop_views
    batch = _full_batch()
    g = torch.Generator().manual_seed(0)
    a, b = drop_views(batch, p=0.2, n=2, generator=g)
    # ~20% dropped -> ~80% kept
    assert 0.70 < a.mask.float().mean().item() < 0.90
    # node_features untouched and shared (same object)
    assert torch.equal(a.node_features, batch.node_features)
    # the two views differ
    assert not torch.equal(a.mask, b.mask)
    # >=1 valid node per row
    assert a.mask.any(dim=1).all() and b.mask.any(dim=1).all()


def test_drop_views_never_reactivates_padded_nodes():
    from src.models.components.contrastive import drop_views
    batch = _full_batch()
    batch.mask[:, 200:] = False          # simulate padding
    a, = drop_views(batch, p=0.5, n=1, generator=torch.Generator().manual_seed(1))
    # dropped mask is a subset of the original valid nodes
    assert (a.mask & ~batch.mask).sum() == 0
