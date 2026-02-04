import math
from functools import lru_cache

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import for optimized attention backends
try:
    from torch.nn.attention import SDPBackend

    SDPA_AVAILABLE = True
except ImportError:
    SDPA_AVAILABLE = False

# from torch.nn.attention import SDPBackend

from src.models.components.custom_pytorch_functions import RMSNorm
from src.models.components.rotary_embedding import BaseRotaryEmbedding

"""
adapted from https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ppf_hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        rope: BaseRotaryEmbedding | None = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.ppf_hidden_dim = ppf_hidden_dim  # TBDeleted
        # self.embedding_layer = EmbeddingLayer(
        #     config.vocab_size, config.d_model, config.max_len
        # )
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout, num_layers, rope)
            for _ in range(num_layers)
        ])

        self.rope = rope
        # Final norm (LLaMA-style) stabilizes the residual stream scale.
        self.final_norm = RMSNorm(hidden_dim=hidden_dim, eps=1e-5)
        # Apply LLaMA-style depth-scaled initialization for stability.
        # This affects *fresh* training runs; checkpoint loading will override weights.
        self.reset_parameters()

        # self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self, x: torch.Tensor, is_encoder: bool, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # output = self.embedding_layer(input_ids)

        for block in self.blocks:
            x = block(x, is_encoder, mask)

        output = self.final_norm(x)
        # output = self.head(output)
        return output

    @property
    def is_rope(self) -> bool:
        return self.rope is not None

    def reset_parameters(self) -> None:
        self.final_norm.reset_parameters()
        for block in self.blocks:
            if hasattr(block, "init_weights"):
                block.init_weights()


class TransformerBlock(nn.Module):
    """TransformerBlock Module.

    Args:
        layer_id (int): Identifier for the layer.
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_head: int,
        dropout: float,
        num_layers: int,
        rope: BaseRotaryEmbedding | None = None,
    ):
        super().__init__()
        self.attention_layer = SelfAttention(n_head, hidden_dim, dropout, rope)
        self.feed_forward_layer = FeedForward(
            hidden_dim=hidden_dim,
            ffn_hidden_dim=hidden_dim,  # I put the same since this is computed in feed forward layer
            multiple_of=32,  # fine tune that
            dropout=dropout,
            ffn_dim_multiplier=None,
        )

        self.attention_norm = RMSNorm(hidden_dim=hidden_dim, eps=1e-5)
        self.ffn_norm = RMSNorm(hidden_dim=hidden_dim, eps=1e-5)
        self.num_layers = num_layers

        # if model_args.depth_init:
        #     self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        # else:
        #     self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5

        self.weight_init_std = 0.02 / (2 * float(self.num_layers)) ** 0.5

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     freqs_cis: torch.Tensor,
    # ):
    #     """Perform a forward pass through the TransformerBlock.

    #     Args:
    #         x (torch.Tensor): Input tensor.
    #         freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    #     Returns:
    #         torch.Tensor: Output tensor after applying attention and feedforward layers.

    #     """
    #     h = x + self.attention(self.attention_norm(x), freqs_cis)
    #     return h + self.feed_forward(self.ffn_norm(h))

    def forward(
        self,
        x: torch.Tensor,
        is_encoder: bool,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_attention = self.attention_layer(self.attention_norm(x), is_encoder, attention_mask)
        x = x + out_attention

        out_feed_forward = self.feed_forward_layer(self.ffn_norm(x))
        x = x + out_feed_forward
        return x

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        # Depth-scaled residual init (matches common LLaMA practice):
        # - Attention out proj and FFN down proj are scaled by 1/sqrt(2L)
        self.attention_layer.init_weights(self.weight_init_std)
        self.feed_forward_layer.init_weights(self.weight_init_std)


class FeedForward(nn.Module):
    """FeedForward module.

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_hidden_dim: int,
        multiple_of: int,
        dropout: float | None = 0.0,
        ffn_dim_multiplier: float | None = None,
    ):
        super().__init__()
        ffn_hidden_dim = int(2 * ffn_hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            ffn_hidden_dim = int(ffn_dim_multiplier * ffn_hidden_dim)
        ffn_hidden_dim = multiple_of * ((ffn_hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(hidden_dim, ffn_hidden_dim, bias=False)
        self.w2 = nn.Linear(ffn_hidden_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, ffn_hidden_dim, bias=False)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        x = self.dropout(x)
        return x

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        n_head: int,
        hidden_dim: int,
        dropout: float,
        rope: BaseRotaryEmbedding | None = None,
    ):
        super().__init__()

        self.n_head = n_head
        self.hidden_dim = hidden_dim

        # self.input_projection = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = rope

    def forward(
        self,
        x: torch.Tensor,
        is_encoder: bool = True,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: b x nn x nn x dv
        batch_size, num_nodes = x.size(0), x.size(1)
        # projected = self.input_projection(x)

        device = x.device

        # q_chunk, k_chunk, v_chunk = torch.chunk(projected, chunks=3, dim=-1)
        # query = q_chunk.view(batch_size, num_nodes, self.n_head, -1).transpose(1, 2)
        # key = k_chunk.view(batch_size, num_nodes, self.n_head, -1).transpose(1, 2)
        # value = v_chunk.view(batch_size, num_nodes, self.n_head, -1).transpose(1, 2)
        query = self.q_proj(x).view(batch_size, num_nodes, self.n_head, -1).transpose(1, 2)
        key = self.k_proj(x).view(batch_size, num_nodes, self.n_head, -1).transpose(1, 2)
        value = self.v_proj(x).view(batch_size, num_nodes, self.n_head, -1).transpose(1, 2)
        if self.rope:
            query = self.rope.rotate_queries_or_keys(query)
            key = self.rope.rotate_queries_or_keys(key)

        if mask is None:
            attn_mask = get_neighborhood_mask(num_nodes, is_encoder, device)
        else:
            attn_mask = get_full_mask(mask, is_encoder, device)
        # Prefer optimized SDPA backends when available, but always produce an output.
        if SDPA_AVAILABLE:
            try:
                # Try optimized backends in order of preference
                with torch.nn.attention.sdpa_kernel([
                    SDPBackend.FLASH_ATTENTION,  # Most memory efficient
                    SDPBackend.EFFICIENT_ATTENTION,  # Good memory efficiency
                    SDPBackend.MATH,  # Fallback (standard implementation)
                ]):
                    attention_output = F.scaled_dot_product_attention(
                        query=query,
                        key=key,
                        value=value,
                        attn_mask=attn_mask,
                        is_causal=False,
                    )
            except Exception:
                # Fallback: let PyTorch pick a working backend (e.g., CPU math)
                attention_output = F.scaled_dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=attn_mask,
                    is_causal=False,
                )
        else:
            attention_output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                is_causal=False,
            )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))
        output = self.dropout(output)
        return output

    def init_weights(self, init_std: float) -> None:
        # LLaMA-style: q/k/v use base std, out proj uses depth-scaled std.
        nn.init.trunc_normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.v_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.output_projection.weight, mean=0.0, std=init_std)


# Cache masks to avoid recomputation
@lru_cache(maxsize=32)
def _create_neighborhood_mask(num_nodes: int, is_encoder: bool):
    """Create neighborhood mask and cache it."""
    if is_encoder:
        n = num_nodes - 1
    else:
        n = num_nodes
    n = int(math.sqrt(n))
    graph = nx.grid_2d_graph(n, n)
    adjacency_matrix = torch.tensor(nx.to_numpy_array(graph), dtype=torch.bool)
    mask = adjacency_matrix | torch.eye(adjacency_matrix.shape[0], dtype=torch.bool)
    if is_encoder:
        mask = F.pad(mask, (1, 0, 1, 0), value=True)
    return mask


def get_neighborhood_mask(num_nodes: int, is_encoder: bool, device: torch.device = None):
    """Get neighborhood mask, creating on the correct device."""
    mask = _create_neighborhood_mask(num_nodes, is_encoder)
    if device is not None:
        mask = mask.to(device)
    return mask


def get_full_mask(mask: torch.Tensor, is_encoder: bool, device: torch.device = None):
    """Create a padding-aware attention mask.

    PyTorch SDPA uses boolean masks where True means "keep/allow attention" and False
    means "mask out" (will be treated as -inf in attention logits).

    Supported inputs:
    - mask: (B, N) padding mask (1/True = valid node, 0/False = padded/invalid)
    - mask: (B, N, N) precomputed boolean attention mask per sample

    Returns:
    - attn_mask: (B, 1, L, S) boolean mask (broadcastable over heads)
      where L/S include the CLS token if is_encoder=True.
    """
    if mask.dim() == 2:  # (B, N) valid-node mask
        valid = mask.to(dtype=torch.bool)
        if is_encoder:
            # Prepend CLS token as always valid. Input x already has CLS at position 0.
            valid = F.pad(valid, (1, 0), value=True)  # (B, N+1)

        # Allow attention only between valid query/key positions.
        # Shape: (B, 1, L, S)
        attn_mask = valid[:, None, :, None] & valid[:, None, None, :]

    elif mask.dim() == 3:  # (B, N, N) per-sample attention mask
        attn_mask = mask.to(dtype=torch.bool)
        if is_encoder:
            # Pad rows/cols for CLS token with True (CLS attends to all valid tokens).
            attn_mask = F.pad(attn_mask, (1, 0, 1, 0), value=True)  # (B, N+1, N+1)
        attn_mask = attn_mask[:, None, :, :]  # (B, 1, L, S)

    else:
        raise ValueError(f"Mask should be 2D or 3D, got shape {tuple(mask.shape)}")

    if device is not None:
        attn_mask = attn_mask.to(device)
    return attn_mask
