import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import for optimized attention backends
try:
    from torch.nn.attention import SDPBackend

    SDPA_AVAILABLE = True
except ImportError:
    SDPA_AVAILABLE = False

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
        use_final_norm: bool = True,
        output_init_std: float | None = None,
        qk_norm: bool = False,
        neighborhood_radius: int = 1,
        pos_bias: str = "none",
        grid_size: int = 0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.ppf_hidden_dim = ppf_hidden_dim
        weight_init_std = output_init_std if output_init_std is not None else 0.02 / (2 * float(num_layers)) ** 0.5
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim, num_heads, ppf_hidden_dim, dropout, weight_init_std,
                rope, qk_norm, neighborhood_radius,
                # per-layer 2D relative-position bias (Swin tables are per-layer)
                pos_bias=(
                    RelativePositionBias2D(pos_bias, grid_size, num_heads)
                    if pos_bias != "none" else None
                ),
            )
            for _ in range(num_layers)
        ])

        self.rope = rope
        self.final_norm = (
            RMSNorm(hidden_dim, eps=1e-5) if use_final_norm else nn.Identity()
        )

        self.init_weights()  # called automatically on construction

    def init_weights(self):
        for block in self.blocks:
            block.init_weights()

    def forward(
        self,
        x: torch.Tensor,
        is_encoder: bool,
        mask: torch.Tensor | None = None,
        film_params: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            film = film_params[i] if film_params is not None else None
            x = block(x, is_encoder, mask, film=film)

        output = self.final_norm(x)
        return output

    @property
    def is_rope(self) -> bool:
        return self.rope is not None


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
        attention_norm (LayerNorm): Layer normalization for attention output.
        ffn_norm (LayerNorm): Layer normalization for feedforward output.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_head: int,
        ppf_hidden_dim: int,
        dropout: float,
        weight_init_std: float,
        rope: BaseRotaryEmbedding | None = None,
        qk_norm: bool = False,
        neighborhood_radius: int = 1,
        pos_bias: nn.Module | None = None,
    ):
        super().__init__()
        self.attention_layer = SelfAttention(
            n_head, hidden_dim, dropout, rope, qk_norm, neighborhood_radius, pos_bias
        )
        self.feed_forward_layer = FeedForward(
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ppf_hidden_dim,
            multiple_of=32,
            ffn_dim_multiplier=None,
            dropout=dropout,
        )

        self.attention_norm = RMSNorm(hidden_dim, eps=1e-5)
        self.ffn_norm = RMSNorm(hidden_dim, eps=1e-5)
        self.weight_init_std = weight_init_std

    def forward(
        self,
        x: torch.Tensor,
        is_encoder: bool,
        attention_mask: torch.Tensor | None = None,
        film: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if film is not None:
            gamma, beta = film  # [B, D] each
            # (1 + gamma) * norm(x) + beta: identity at init when gamma=beta=0
            attn_in = (1 + gamma.unsqueeze(1)) * self.attention_norm(x) + beta.unsqueeze(1)
        else:
            attn_in = self.attention_norm(x)
        out_attention = self.attention_layer(attn_in, is_encoder, attention_mask)
        x = x + out_attention

        if film is not None:
            ffn_in = (1 + gamma.unsqueeze(1)) * self.ffn_norm(x) + beta.unsqueeze(1)
        else:
            ffn_in = self.ffn_norm(x)
        out_feed_forward = self.feed_forward_layer(ffn_in)
        x = x + out_feed_forward
        return x

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
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
        qk_norm: bool = False,
        neighborhood_radius: int = 1,
        pos_bias: nn.Module | None = None,
    ):
        super().__init__()

        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.neighborhood_radius = neighborhood_radius
        self.pos_bias = pos_bias

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = rope
        self.qk_norm = qk_norm
        if qk_norm:
            head_dim = hidden_dim // n_head
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        is_encoder: bool = True,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_nodes = x.size(0), x.size(1)

        device = x.device

        query = self.q_proj(x).view(batch_size, num_nodes, self.n_head, -1).transpose(1, 2)
        key = self.k_proj(x).view(batch_size, num_nodes, self.n_head, -1).transpose(1, 2)
        value = self.v_proj(x).view(batch_size, num_nodes, self.n_head, -1).transpose(1, 2)
        # QK-norm (Gemma2 / ViT-22B style): learnable RMSNorm over head_dim, applied
        # BEFORE RoPE. Unlike L2 normalisation, RMSNorm does not force unit length, so
        # it preserves dynamic range — SDPA's default 1/sqrt(head_dim) scale stays valid
        # and attention logits are not crushed toward a uniform average.
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        if self.rope:
            query = self.rope.rotate_queries_or_keys(query)
            key = self.rope.rotate_queries_or_keys(key)

        if mask is None:
            attn_mask = get_neighborhood_mask(num_nodes, is_encoder, device, self.neighborhood_radius)
        else:
            attn_mask = get_full_mask(mask, is_encoder, device)
        if self.pos_bias is not None:
            # Add a per-head 2D relative-position bias to the logits (soft locality).
            # Bool attn_mask → float: keep the bias, set masked pairs to -inf.
            bias = self.pos_bias().to(query.dtype)  # [H, N, N]
            attn_mask = bias.masked_fill(~attn_mask, float("-inf"))
        try:
            with torch.nn.attention.sdpa_kernel([
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]):
                attention_output = F.scaled_dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=attn_mask,
                    is_causal=False,
                )
        except (RuntimeError, ImportError) as e:
            print(f"Falling back to standard attention: {e}")
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

    def init_weights(self, init_std: float):
        """Initialize attention projection weights."""
        for linear in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.output_projection.weight, mean=0.0, std=init_std)


@lru_cache(maxsize=64)
def _create_neighborhood_mask(num_nodes: int, is_encoder: bool, device: str, radius: int = 1):
    """Create a dilated grid-neighborhood mask, cached per (size, role, device, radius).

    A content node attends to every node within Manhattan grid distance <= radius
    (radius=1 reproduces the original 4-neighbour + self mask). The Manhattan ball is
    invariant under the grid's D4 automorphisms, so the encoder stays D4-equivariant
    at any radius.
    """
    n_content = num_nodes - 1 if is_encoder else num_nodes
    n = int(math.sqrt(n_content))
    rows = torch.arange(n_content) // n
    cols = torch.arange(n_content) % n
    dist = (rows[:, None] - rows[None, :]).abs() + (cols[:, None] - cols[None, :]).abs()
    mask = dist <= radius
    if is_encoder:
        # CLS is at position 0.
        # Col 0 (CLS as key)  = False: content nodes cannot attend to CLS.
        # Row 0 (CLS as query) = True: CLS attends to all content nodes.
        # Without this asymmetry the CLS acts as a global bus — every content node
        # reads the global average through CLS and all nodes collapse to the same repr.
        mask = F.pad(mask, (1, 0, 0, 0), value=False)  # left col: content→CLS = False
        mask = F.pad(mask, (0, 0, 1, 0), value=True)   # top row:  CLS→all   = True
    return mask.to(device)


def get_neighborhood_mask(
    num_nodes: int, is_encoder: bool, device: torch.device = None, radius: int = 1
):
    """Get dilated neighborhood mask, cached per device."""
    device_str = str(device) if device is not None else "cpu"
    return _create_neighborhood_mask(num_nodes, is_encoder, device_str, radius)


def get_full_mask(mask: torch.Tensor, is_encoder: bool, device: torch.device = None):
    """Create full attention mask where all nodes can attend to all nodes.

    The caller is responsible for padding the mask if a hub/CLS token was
    prepended to the sequence — this function uses mask.size(1) as-is.
    """
    if mask.dim() == 2:
        num_nodes = mask.size(1)
    elif mask.dim() == 3:
        num_nodes = mask.size(1)
    else:
        raise ValueError(f"Mask should be 2D or 3D, got shape {mask.shape}")

    attn_mask = torch.ones(num_nodes, num_nodes, dtype=torch.bool)

    if device is not None:
        attn_mask = attn_mask.to(device)

    return attn_mask


class RelativePositionBias2D(nn.Module):
    """Additive per-head attention bias from 2D grid relative position.

    Soft locality that keeps global reach (no hard mask). Returns `[n_head, N, N]` to add
    to the attention logits, where `N = grid_size²` (decoder grid has no CLS token).

    - ``mode="alibi"``: parameter-free. `bias = -slope_h · Manhattan_distance(i, j)`, with
      geometric per-head slopes (closer nodes biased up, far nodes down but not masked out).
    - ``mode="swin"``: a learned bias table indexed by the relative offset `(Δrow, Δcol)`,
      one table per head (Swin-Transformer style).
    """

    def __init__(self, mode: str, grid_size: int, n_head: int):
        super().__init__()
        self.mode = mode
        self.n_head = n_head
        g = grid_size
        n = g * g
        rows = torch.arange(n) // g
        cols = torch.arange(n) % g
        drow = rows[:, None] - rows[None, :]  # [N, N]
        dcol = cols[:, None] - cols[None, :]
        if mode == "alibi":
            dist = (drow.abs() + dcol.abs()).float()  # Manhattan distance
            self.register_buffer("dist", dist)
            slopes = 2.0 ** (-8.0 * torch.arange(1, n_head + 1) / n_head)
            self.register_buffer("slopes", slopes.float())
        elif mode == "swin":
            rel_index = (drow + (g - 1)) * (2 * g - 1) + (dcol + (g - 1))  # [N, N] -> table idx
            self.register_buffer("rel_index", rel_index.long())
            self.table = nn.Parameter(torch.zeros(n_head, (2 * g - 1) * (2 * g - 1)))
            nn.init.trunc_normal_(self.table, std=0.02)
        else:
            raise ValueError(f"unknown pos_bias mode: {mode!r} (expected 'alibi' or 'swin')")

    def forward(self) -> torch.Tensor:
        if self.mode == "alibi":
            return -self.slopes[:, None, None] * self.dist[None, :, :]  # [H, N, N]
        return self.table[:, self.rel_index]  # [H, N, N]