import torch

from src.downstream.encode import _build_pl_module
from src.explainability.rollout import attention_rollout, capture_attention


def test_capture_attention_collects_encoder_layers():
    gae = _build_pl_module("vae16_fb0p0_film").graph_ae.eval()
    from src.data.components.graphs_datamodules import DenseGraphBatch

    nf = torch.randn(1, 256, 128)
    mask = torch.ones(1, 256, dtype=torch.bool)
    store: list = []
    with capture_attention(store):
        gae.encode(DenseGraphBatch(node_features=nf, edge_features=torch.empty(0), mask=mask), sample=False)
    assert len(store) == 4  # 4 encoder transformer layers, one SDPA call each
    assert store[0].shape[-1] == 257  # T = 256 content + 1 CLS


def test_attention_rollout_shape_and_valid():
    torch.manual_seed(0)
    gae = _build_pl_module("vae16_fb0p0_film").graph_ae.eval()
    nf = torch.randn(2, 256, 128)
    imp = attention_rollout(gae, nf)
    assert imp.shape == (2, 256)  # per content-node importance
    assert torch.isfinite(imp).all()
    assert (imp >= 0).all()  # rollout is a product of row-stochastic matrices
