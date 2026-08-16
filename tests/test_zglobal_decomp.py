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
    # The stats_correction is zero-initialized; reinitialize to get non-zero stats contribution
    torch.nn.init.normal_(gae.encoder.stats_correction.proj.weight, mean=0.0, std=0.01)
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
    # The stats_correction is zero-initialized; reinitialize to get non-zero stats contribution
    torch.nn.init.normal_(gae.encoder.stats_correction.proj.weight, mean=0.0, std=0.01)
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


def test_source_subdir_maps_decomposition_views():
    from src.downstream.encode import source_subdir
    assert source_subdir("zglobal_cls") == "zglobal_cls"
    assert source_subdir("zglobal_stats") == "zglobal_stats"


def test_encode_patches_cls_stats_shapes_and_distinct():
    import numpy as np
    from src.downstream.encode import encode_patches
    gae, _ = _gae_and_batch()
    # The stats_correction is zero-initialized; reinitialize to get non-zero stats contribution
    torch.nn.init.normal_(gae.encoder.stats_correction.proj.weight, mean=0.0, std=0.01)
    pca = lambda x: x[..., :128]  # [B,256,768] -> [B,256,128] stub
    patches = np.random.randn(2, 768, 16, 16).astype("float32")
    cls = encode_patches(gae, pca, patches, "cpu", feature_source="zglobal_cls")
    stats = encode_patches(gae, pca, patches, "cpu", feature_source="zglobal_stats")
    full = encode_patches(gae, pca, patches, "cpu", feature_source="zglobal")
    assert cls.shape == (2, 512) and stats.shape == (2, 512)
    assert np.isfinite(cls).all() and np.isfinite(stats).all()
    assert not np.allclose(cls, stats, atol=1e-4)
    assert not np.allclose(cls, full, atol=1e-4)
