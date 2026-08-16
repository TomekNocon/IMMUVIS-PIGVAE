import pytest
import torch

from src.data.components.graphs_datamodules import DenseGraphBatch


def _gae_and_batch(B=2, N=256, D=128, seed=0):
    from src.downstream.encode import _build_pl_module
    torch.manual_seed(seed)
    gae = _build_pl_module("vae16_fb0p0_film").graph_ae.eval()
    batch = DenseGraphBatch(
        node_features=torch.randn(B, N, D),
        edge_features=torch.empty(0),
        mask=torch.ones(B, N, dtype=torch.bool),
    )
    return gae, batch


def test_cls_plus_stats_equals_full_graph_emb():
    gae, batch = _gae_and_batch()
    with torch.no_grad():
        full, _nodes, cls, stats = gae.encoder(
            node_features=batch.node_features,
            edge_features=batch.edge_features,
            mask=batch.mask,
            return_parts=True,
        )
    assert torch.allclose(cls + stats, full, atol=1e-6)


def test_full_view_matches_untouched_encode_zglobal():
    gae, batch = _gae_and_batch()
    with torch.no_grad():
        _, z_global, *_ = gae.encode(batch, sample=False)
        parts = gae.encode_zglobal_parts(batch)
    assert torch.allclose(parts["full"], z_global, atol=1e-6)


def test_parts_shape_and_finite_and_distinct():
    gae, batch = _gae_and_batch()
    with torch.no_grad():
        parts = gae.encode_zglobal_parts(batch)
    for k in ("full", "cls", "stats"):
        assert parts[k].shape == (2, 512)
        assert torch.isfinite(parts[k]).all()
    assert not torch.allclose(parts["cls"], parts["stats"], atol=1e-4)


def test_return_parts_raises_if_structural_correction_present():
    gae, batch = _gae_and_batch()
    gae.encoder.structural_correction = torch.nn.Identity()  # non-None sentinel
    with pytest.raises(ValueError):
        gae.encoder(
            node_features=batch.node_features,
            edge_features=batch.edge_features,
            mask=batch.mask,
            return_parts=True,
        )


def test_default_forward_unchanged_arity():
    gae, batch = _gae_and_batch()
    with torch.no_grad():
        out = gae.encoder(
            node_features=batch.node_features,
            edge_features=batch.edge_features,
            mask=batch.mask,
        )
    assert len(out) == 2  # (graph_emb, node_features)
