import torch
from omegaconf import OmegaConf

from src.models.components.llama_graph_transformer import RelativePositionBias2D, Transformer
from src.models.components.modules import GraphDecoder


def test_alibi_shape_and_distance_decay():
    pb = RelativePositionBias2D("alibi", grid_size=6, n_head=8)
    b = pb()
    assert b.shape == (8, 36, 36)
    assert torch.allclose(b.diagonal(dim1=1, dim2=2), torch.zeros(8, 36))  # self-bias 0
    # closer (idx0=(0,0) → idx1=(0,1), dist 1) must score higher than far (idx 35=(5,5), dist 10)
    assert (b[:, 0, 1] > b[:, 0, 35]).all()
    # bias is symmetric (Manhattan distance is symmetric)
    assert torch.allclose(b, b.transpose(1, 2))


def test_swin_is_learnable_and_correct_shape():
    pb = RelativePositionBias2D("swin", grid_size=6, n_head=8)
    b = pb()
    assert b.shape == (8, 36, 36)
    # one learnable table per head over the (2g-1)² relative offsets
    assert pb.table.shape == (8, 11 * 11)
    assert pb.table.requires_grad
    # equal relative offsets share a bias entry: (0,0)->(0,1) and (1,0)->(1,1) are both Δ=(0,+1)
    assert torch.allclose(b[:, 0, 1], b[:, 6, 7])


def test_transformer_pos_bias_changes_output():
    torch.manual_seed(0)
    x = torch.randn(2, 36, 64)
    mask = torch.ones(2, 36, dtype=torch.bool)
    base = Transformer(64, 4, 128, 2, dropout=0.0, qk_norm=True)
    alibi = Transformer(64, 4, 128, 2, dropout=0.0, qk_norm=True, pos_bias="alibi", grid_size=6)
    out_base = base(x, is_encoder=False, mask=mask)
    out_alibi = alibi(x, is_encoder=False, mask=mask)
    assert out_alibi.shape == out_base.shape
    assert torch.isfinite(out_alibi).all()
    assert not torch.allclose(out_base, out_alibi)  # bias actually affects attention


def _decoder(pos_bias: str) -> GraphDecoder:
    h = OmegaConf.create({
        "graph_decoder_hidden_dim": 64, "graph_decoder_num_heads": 4,
        "graph_decoder_ppf_hidden_dim": 128, "graph_decoder_num_layers": 2,
        "dropout": 0.0, "head_dim": 16, "qk_norm": True, "use_film": False,
        "use_rope": False, "pos_bias": pos_bias, "node_z_dim": 32, "project": True,
        "num_node_features": 128, "grid_size": 6, "encoder_hidden_dim": 64,
    })
    return GraphDecoder(h)


def test_decoder_runs_with_each_pos_bias_and_drops_pe():
    z_nodes = torch.randn(8, 36, 32)
    z_global = torch.randn(8, 64)
    mask = torch.ones(8, 36, dtype=torch.bool)
    for mode, expect_pe in [("none", True), ("alibi", False), ("swin", False)]:
        dec = _decoder(mode)
        assert dec.use_pos_emb is expect_pe  # absolute PE only when no relative bias
        out, _ = dec(z_nodes, z_global, mask)
        assert out.shape == (8, 36, 128)
        assert torch.isfinite(out).all()
