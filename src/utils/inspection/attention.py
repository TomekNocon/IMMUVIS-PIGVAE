# src/utils/inspection/attention.py
from __future__ import annotations

import math

import torch


@torch.no_grad()
def attention_entropy_from_input(attn_module, x: torch.Tensor, attn_mask: torch.Tensor) -> dict:
    """Recompute attention weights for one SelfAttention and report entropy as a fraction
    of log(n_keys). Mirrors SelfAttention.forward up to the softmax.

    x:         [B, N, D] input to the attention module (post pre-norm).
    attn_mask: [N, N] bool, True = keep (same convention as get_*_mask).
    """
    b, n = x.shape[0], x.shape[1]
    q = attn_module.q_proj(x).view(b, n, attn_module.n_head, -1).transpose(1, 2)
    k = attn_module.k_proj(x).view(b, n, attn_module.n_head, -1).transpose(1, 2)
    if getattr(attn_module, "qk_norm", False):
        q = attn_module.q_norm(q)
        k = attn_module.k_norm(k)
    if attn_module.rope:
        q = attn_module.rope.rotate_queries_or_keys(q)
        k = attn_module.rope.rotate_queries_or_keys(k)

    scale = 1.0 / math.sqrt(q.shape[-1])
    logits = (q @ k.transpose(-2, -1)) * scale                      # [B, H, N, N]
    logits = logits.masked_fill(~attn_mask, float("-inf"))
    w = torch.softmax(logits, dim=-1)                               # [B, H, N, N]

    n_keys = attn_mask.sum(dim=-1).clamp_min(2).float()             # [N]
    ent = -(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(-1)  # [B, H, N]
    norm_ent = ent / torch.log(n_keys).view(1, 1, n)
    return {
        "entropy_frac_mean": norm_ent.mean().item(),
        "entropy_frac_per_head": norm_ent.mean(dim=(0, 2)).tolist(),
        "frac_mass_on_token0": w[..., 0].mean().item(),
    }
