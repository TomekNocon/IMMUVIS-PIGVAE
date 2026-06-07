import torch

from src.models.components.llama_graph_transformer import _create_neighborhood_mask


def _content_mask(num_nodes: int, radius: int) -> torch.Tensor:
    """Content-content block of the encoder mask (strip the CLS row/col at index 0)."""
    m = _create_neighborhood_mask(num_nodes, is_encoder=True, device="cpu", radius=radius)
    return m[1:, 1:]


def test_radius1_is_grid_clipped_von_neumann():
    """radius=1 = orthogonal neighbours clipped to the grid: count depends on position."""
    m = _content_mask(37, radius=1)  # 6x6 grid + CLS
    assert int(m[14].sum()) == 5   # interior (2,2): self + 4
    assert int(m[1].sum()) == 4    # top edge (0,1): self + left + right + down
    assert int(m[0].sum()) == 3    # corner (0,0): self + right + down
    # corner connectivity, Manhattan: right (idx1) and down (idx6); no diagonal / distance-2
    assert m[0, 1] and m[0, 6]
    assert not m[0, 7]             # (1,1) diagonal excluded (Manhattan)
    assert not m[0, 2]             # (0,2) distance 2 excluded at radius 1


def test_radius2_grows_the_ball():
    m = _content_mask(37, radius=2)
    assert int(m[14].sum()) == 13  # interior: self + 4 (d1) + 8 (d2)
    assert m[0, 7]                 # (0,0)->(1,1) Manhattan 2 now included
    assert m[0, 2]                 # (0,0)->(0,2) Manhattan 2 now included
    assert not m[0, 3]             # (0,3) Manhattan 3 still excluded


def test_d4_equivariance_of_content_mask():
    """The Manhattan ball must be invariant under D4 grid symmetries: P M Pᵀ == M."""
    m = _content_mask(37, radius=2).float()
    n = 6
    idx = torch.arange(n * n).reshape(n, n)
    for perm in (torch.rot90(idx, 1).reshape(-1), idx.flip(1).reshape(-1)):
        p = torch.eye(n * n)[perm]
        assert torch.allclose(p @ m @ p.t(), m)


def test_cls_asymmetry_preserved_at_any_radius():
    m = _create_neighborhood_mask(37, is_encoder=True, device="cpu", radius=3)
    assert bool(m[0].all())          # CLS attends to everything
    assert not bool(m[1:, 0].any())  # content cannot attend to CLS
