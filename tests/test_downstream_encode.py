import torch

def _build_tiny_graph_ae(tmp_path):
    """Build a graph_ae from the real model config so encode() is exercised, save a ckpt.

    Stamps a known, distinctive value into one specific `graph_ae` parameter
    before saving, then mutates the live parameter afterwards, so a test that
    later observes the marker value on a freshly-loaded module can only be
    explained by `load_frozen_pigvae` actually reading weights back off disk
    (as opposed to e.g. silently keeping a randomly-initialized module).
    """
    from src.downstream.encode import _build_pl_module  # helper Task A1 exposes
    pl = _build_pl_module(experiment="vae16_fb0p0", paths_name="szary")
    param_name, param = next(iter(pl.graph_ae.named_parameters()))
    with torch.no_grad():
        param.fill_(12345.0)
    marker_value = param.detach().clone()
    ckpt = tmp_path / "tiny.ckpt"
    torch.save({"state_dict": pl.state_dict()}, ckpt)
    # Mutate the in-memory parameter after saving: the checkpoint on disk keeps
    # the marker value regardless of what happens to `pl` afterwards.
    with torch.no_grad():
        param.fill_(-1.0)
    return str(ckpt), param_name, marker_value

def test_load_frozen_pigvae_returns_eval_graph_ae(tmp_path):
    from src.downstream.encode import load_frozen_pigvae
    ckpt, param_name, marker_value = _build_tiny_graph_ae(tmp_path)
    gae = load_frozen_pigvae(ckpt, experiment="vae16_fb0p0")
    assert not gae.training                                        # eval mode
    assert all(not p.requires_grad for p in gae.parameters())      # weights frozen
    assert any(p.numel() > 0 for p in gae.parameters())            # weights present

    # Core promise of load_frozen_pigvae: weights were actually loaded from
    # ckpt_path, not just a freshly/randomly-initialized module of the right shape.
    loaded_param = dict(gae.named_parameters())[param_name]
    assert torch.equal(loaded_param, marker_value)
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


def test_patch_to_nodes_order_and_pca_shape():
    import numpy as np, torch
    from src.downstream.encode import patch_to_nodes
    patch = np.arange(768 * 16 * 16, dtype=np.float32).reshape(768, 16, 16)
    nodes = patch_to_nodes(patch)                 # (256, 768)
    assert nodes.shape == (256, 768)
    # node n = grid cell (row=n//16, col=n%16), all 768 channels of that cell
    assert torch.allclose(nodes[0], torch.from_numpy(patch[:, 0, 0]))
    assert torch.allclose(nodes[17], torch.from_numpy(patch[:, 1, 1]))
