import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from src.models.pigvae_auto_module import PLGraphAE

def _build_tiny_graph_ae(tmp_path):
    """Build a graph_ae from the real model config so encode() is exercised, save a ckpt."""
    from src.downstream.encode import _build_pl_module  # helper Task A1 exposes
    pl = _build_pl_module(experiment="vae16_fb0p0", paths_name="szary")
    ckpt = tmp_path / "tiny.ckpt"
    torch.save({"state_dict": pl.state_dict()}, ckpt)
    return str(ckpt)

def test_load_frozen_pigvae_returns_eval_graph_ae(tmp_path):
    from src.downstream.encode import load_frozen_pigvae
    ckpt = _build_tiny_graph_ae(tmp_path)
    gae = load_frozen_pigvae(ckpt, experiment="vae16_fb0p0")
    assert not gae.training                                        # eval mode
    assert all(not p.requires_grad for p in gae.parameters())      # weights frozen
    assert any(p.numel() > 0 for p in gae.parameters())            # weights present
    # encode runs: 2 graphs, 256 nodes, 128 pca dims -> z_global [2, D]
    nf = torch.randn(2, 256, 128)
    from src.data.components.graphs_datamodules import DenseGraphBatch
    # A full 16x16 grid has no padding, i.e. an all-True mask (mask=None crashes:
    # GraphEncoder.add_emb_node_and_feature does F.pad(mask, ...) unconditionally
    # in CLS mode / use_pma=False, which this experiment config uses).
    mask = torch.ones(2, 256, dtype=torch.bool)
    batch = DenseGraphBatch(node_features=nf, edge_features=torch.empty(0), mask=mask)
    with torch.no_grad():
        z_nodes, z_global, *_ = gae.encode(batch, sample=False)
    assert z_global.shape[0] == 2 and z_global.dim() == 2
