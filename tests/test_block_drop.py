import math
import torch
from src.data.components.graphs_datamodules import DenseGraphBatch
from src.models.components.contrastive import block_drop


def _grid_batch(B=4, grid=16, D=8):
    N = grid * grid
    return DenseGraphBatch(
        node_features=torch.randn(B, N, D),
        edge_features=torch.empty(0),
        mask=torch.ones(B, N, dtype=torch.bool),
    )


def test_block_drop_masks_and_zeroes_a_contiguous_block():
    grid = 16
    B, D = 4, 8
    batch = _grid_batch(B=B, grid=grid, D=D)
    out = block_drop(batch, frac=0.5, grid_size=grid, generator=torch.Generator().manual_seed(0))
    dropped = ~out.mask                                    # [B, N] True where removed
    frac_removed = dropped.float().mean().item()
    assert 0.4 <= frac_removed <= 0.6                      # ≈ 50%
    # dropped node_features are exactly zero; kept ones are unchanged
    assert torch.count_nonzero(out.node_features[dropped]) == 0
    assert torch.equal(out.node_features[out.mask], batch.node_features[out.mask])
    # contiguity: per row, dropped indices form a rectangle in (row, col) grid coords
    for b in range(B):
        rc = dropped[b].reshape(grid, grid)
        rows = rc.any(dim=1).nonzero().flatten()
        cols = rc.any(dim=0).nonzero().flatten()
        # rows and cols are each a contiguous run, and the block is their full product
        assert torch.equal(rows, torch.arange(rows.min(), rows.max() + 1))
        assert torch.equal(cols, torch.arange(cols.min(), cols.max() + 1))
        assert rc[rows.min():rows.max() + 1, cols.min():cols.max() + 1].all()


def test_block_drop_does_not_mutate_input_features():
    batch = _grid_batch()
    before = batch.node_features.clone()
    _ = block_drop(batch, frac=0.5, grid_size=16, generator=torch.Generator().manual_seed(1))
    assert torch.equal(batch.node_features, before)        # clean target untouched


def test_block_drop_keeps_at_least_one_node_per_row():
    batch = _grid_batch()
    out = block_drop(batch, frac=0.5, grid_size=16, generator=torch.Generator().manual_seed(2))
    assert out.mask.any(dim=1).all()


def test_block_drop_reproducible_with_same_seed():
    batch = _grid_batch()
    a = block_drop(batch, frac=0.5, grid_size=16, generator=torch.Generator().manual_seed(7))
    b = block_drop(batch, frac=0.5, grid_size=16, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a.mask, b.mask)
    assert torch.equal(a.node_features, b.node_features)


def test_block_drop_infers_grid_size_from_num_nodes():
    batch = _grid_batch(grid=16)
    out = block_drop(batch, frac=0.5, generator=torch.Generator().manual_seed(3))  # grid_size=None
    assert (~out.mask).float().mean().item() >= 0.4
