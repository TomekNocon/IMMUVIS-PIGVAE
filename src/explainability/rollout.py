"""Attention-rollout baseline for the encoder (Abnar & Zuidema, 2020).

A cheap, gradient-free importance map: aggregate the raw self-attention across the encoder's
layers (per-layer head-mean, add the residual identity, then matrix-product across depth) and
read the CLS row to get per-node importance for ``z_global``.

The encoder attends via fused ``F.scaled_dot_product_attention``, which returns no weights, so
``capture_attention`` temporarily wraps SDPA to recompute the softmax on the side (the layer
output is left exact — the original SDPA still produces it). Rollout is only a baseline; it is
less faithful than the Integrated-Gradients paths (no value magnitudes, no gradients), so use
it as a sanity check against ``encoder_attribution``, validated with ``faithfulness``.
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F

from src.data.components.graphs_datamodules import DenseGraphBatch


@contextlib.contextmanager
def capture_attention(store: list):
    """Patch SDPA to append each call's attention weights ``[B, H, Lq, Lk]`` to ``store``.

    The real SDPA still computes the (exact) output; we only recompute the softmax to observe
    the weights, replicating whatever ``scale``/``attn_mask`` the caller passed.
    """
    orig = F.scaled_dot_product_attention

    def patched(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        s = (1.0 / math.sqrt(query.shape[-1])) if scale is None else scale
        logits = (query @ key.transpose(-2, -1)) * s
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                logits = logits.masked_fill(~attn_mask, float("-inf"))
            else:
                logits = logits + attn_mask
        attn = torch.softmax(logits, dim=-1)
        store.append(torch.nan_to_num(attn.detach()))  # all-masked rows -> 0
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)

    F.scaled_dot_product_attention = patched
    try:
        yield
    finally:
        F.scaled_dot_product_attention = orig


@torch.no_grad()
def attention_rollout(gae: torch.nn.Module, node_features: torch.Tensor) -> torch.Tensor:
    """CLS-row attention rollout over the encoder -> per content-node importance ``[B, N]``.

    Runs one encoder forward capturing per-layer attention, averages over heads, adds the
    residual identity (``0.5·A + 0.5·I``, row-renormalised), matrix-multiplies across layers,
    and returns the CLS (row 0) attention to the ``N`` content nodes.
    """
    gae.eval()
    batch_size, num_nodes, _ = node_features.shape
    mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=node_features.device)
    graph = DenseGraphBatch(node_features=node_features, edge_features=torch.empty(0, device=node_features.device), mask=mask)

    store: list = []
    with capture_attention(store):
        gae.encode(graph, sample=False)
    if not store:
        raise RuntimeError("no attention captured — encoder did not call scaled_dot_product_attention")

    # Each layer: [B, H, T, T] with T = N+1 (CLS at 0). Head-mean + residual identity.
    tokens = store[0].shape[-1]
    eye = torch.eye(tokens, device=store[0].device)
    rollout = eye.unsqueeze(0).expand(batch_size, tokens, tokens).clone()
    for attn in store:
        a = attn.mean(dim=1)  # [B, T, T]
        a = 0.5 * a + 0.5 * eye
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        rollout = a @ rollout
    return rollout[:, 0, 1:]  # CLS row, content nodes only
