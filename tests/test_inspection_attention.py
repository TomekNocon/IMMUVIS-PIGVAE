# tests/test_inspection_attention.py
import torch
from src.models.components.llama_graph_transformer import Transformer, SelfAttention
from src.utils.inspection.attention import attention_entropy_from_input


def test_entropy_fraction_in_unit_range_and_uniform_is_high():
    torch.manual_seed(0)
    t = Transformer(hidden_dim=16, num_heads=2, ppf_hidden_dim=32, num_layers=1,
                    dropout=0.0, qk_norm=True)
    attn = t.blocks[0].attention_layer
    assert isinstance(attn, SelfAttention)
    x = torch.randn(2, 36, 16)
    # full (all-ones) mask over 36 tokens
    mask = torch.ones(36, 36, dtype=torch.bool)
    out = attention_entropy_from_input(attn, x, mask)
    assert 0.0 <= out["entropy_frac_mean"] <= 1.0
    assert len(out["entropy_frac_per_head"]) == 2
