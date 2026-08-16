import torch

from src.data.components.graphs_datamodules import DenseGraphBatch
from src.downstream.encode import _build_pl_module


def test_default_builds_stats_correction():
    gae = _build_pl_module("vae16_fb0p0_film_b64").graph_ae
    assert gae.encoder.stats_correction is not None


def test_nostats_config_disables_stats_correction():
    gae = _build_pl_module("vae16_fb0p0_film_nostats").graph_ae
    assert gae.encoder.stats_correction is None


def test_forward_skips_stats_when_disabled():
    # With a non-zero stats weight, removing the correction must change graph_emb
    # (proving the add is gated), and the gated forward must run finite [B, 512].
    torch.manual_seed(0)
    gae = _build_pl_module("vae16_fb0p0_film_b64").graph_ae.eval()
    torch.nn.init.normal_(gae.encoder.stats_correction.proj.weight, std=0.01)
    batch = DenseGraphBatch(
        node_features=torch.randn(2, 256, 128),
        edge_features=torch.empty(0),
        mask=torch.ones(2, 256, dtype=torch.bool),
    )
    with torch.no_grad():
        emb_on, _ = gae.encoder(
            node_features=batch.node_features,
            edge_features=batch.edge_features,
            mask=batch.mask,
        )
        gae.encoder.stats_correction = None
        emb_off, _ = gae.encoder(
            node_features=batch.node_features,
            edge_features=batch.edge_features,
            mask=batch.mask,
        )
    assert not torch.allclose(emb_on, emb_off)
    assert emb_off.shape == (2, 512)
    assert torch.isfinite(emb_off).all()
