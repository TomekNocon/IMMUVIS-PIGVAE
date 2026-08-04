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


def test_block_drop_restored_nodes_have_real_features():
    """Padded input: only a few valid nodes per row, all inside the block.

    After fallback restores a node's mask, its features must be non-zero (not zeroed
    by the block). This catches the bug where a row is emptied by the block, the
    fallback restores mask=True, but features remain all-zero — model treats it as
    real content when it is empty.
    """
    grid = 16
    D = 8
    N = grid * grid

    # Padded batch: only nodes [0:4] valid per row (rest are padding)
    node_features = torch.randn(1, N, D)
    mask = torch.zeros(1, N, dtype=torch.bool)
    mask[0, 0:4] = True  # only 4 valid nodes per row (grid positions (0,0), (0,1), (0,2), (0,3))

    batch = DenseGraphBatch(
        node_features=node_features,
        edge_features=torch.empty(0),
        mask=mask,
    )

    # Use high frac (0.8) to get a large block (14x15) that will cover all 4 valid nodes at the
    # top-left. With seed=7, block position is (top=0, left=0), so it covers rows [0,14), cols [0,15),
    # which includes all valid nodes [0:4]. This forces the row to be emptied, triggering the fallback
    # to restore the first valid node (node 0), which is also in the block and will have zeroed features.
    out = block_drop(batch, frac=0.8, grid_size=grid, generator=torch.Generator().manual_seed(7))

    # Every row with mask=True must have at least one non-zero feature
    for b in range(1):
        for n in range(N):
            if out.mask[b, n]:  # if this node is marked valid
                # All features at this position must not all be zero
                assert torch.any(out.node_features[b, n] != 0.0), \
                    f"Restored node at [{b}, {n}] has mask=True but all-zero features (bug!)"
