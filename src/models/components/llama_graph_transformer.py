"""Pre-norm LLaMA-style graph transformer with grid-local attention masks and FiLM hooks.

The encoder attends within a Manhattan neighbourhood on the node grid (plus a one-way CLS
readout); the decoder attends globally, optionally with a 2D relative-position bias.
"""

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.models.components.custom_pytorch_functions import RMSNorm
from src.models.components.rotary_embedding import BaseRotaryEmbedding

SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class Transformer(nn.Module):
    """Stack of :class:`TransformerBlock` with an optional final RMSNorm."""

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
        weight_init_std = (
            output_init_std if output_init_std is not None else 0.02 / (2 * num_layers) ** 0.5
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim,
                num_heads,
                ppf_hidden_dim,
                dropout,
                weight_init_std,
                rope,
                qk_norm,
                neighborhood_radius,
                # One 2D relative-position bias per layer (Swin tables are per-layer).
                pos_bias=(
                    RelativePositionBias2D(pos_bias, grid_size, num_heads)
                    if pos_bias != "none"
                    else None
                ),
            )
            for _ in range(num_layers)
        ])

        self.rope = rope
        self.final_norm = RMSNorm(hidden_dim, eps=1e-5) if use_final_norm else nn.Identity()
        self.init_weights()

    def init_weights(self) -> None:
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
        return self.final_norm(x)

    @property
    def is_rope(self) -> bool:
        return self.rope is not None


class TransformerBlock(nn.Module):
    """Pre-norm block: ``x + Attn(norm(x))`` then ``x + FFN(norm(x))``.

    With ``film=(gamma, beta)`` both pre-norms are modulated as ``(1 + gamma) * norm(x) + beta``,
    which is the identity at initialisation (``gamma = beta = 0``).
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
        x = x + self.attention_layer(attn_in, is_encoder, attention_mask)

        if film is not None:
            ffn_in = (1 + gamma.unsqueeze(1)) * self.ffn_norm(x) + beta.unsqueeze(1)
        else:
            ffn_in = self.ffn_norm(x)
        return x + self.feed_forward_layer(ffn_in)

    def init_weights(self) -> None:
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention_layer.init_weights(self.weight_init_std)
        self.feed_forward_layer.init_weights(self.weight_init_std)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network (LLaMA): ``w2(silu(w1 x) * w3 x)``.

    Args:
        hidden_dim: Model width.
        ffn_hidden_dim: Nominal FFN width; scaled by 2/3 (SwiGLU) and rounded up to ``multiple_of``.
        multiple_of: Rounding granularity of the FFN width.
        dropout: Dropout on the output.
        ffn_dim_multiplier: Optional extra multiplier on the FFN width.
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
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

    def init_weights(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class SelfAttention(torch.nn.Module):
    """Multi-head self-attention with optional QK-norm, RoPE and 2D relative-position bias."""

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
        # QK-norm (Gemma 2 / ViT-22B): learnable RMSNorm over head_dim, applied before RoPE.
        # Unlike L2 normalisation it keeps the dynamic range, so SDPA's 1/sqrt(head_dim)
        # scale stays valid and the logits are not crushed toward a uniform average.
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        if self.rope:
            query = self.rope.rotate_queries_or_keys(query)
            key = self.rope.rotate_queries_or_keys(key)

        if mask is None:
            attn_mask = get_neighborhood_mask(
                num_nodes, is_encoder, device, self.neighborhood_radius
            )
        else:
            attn_mask = get_full_mask(mask, is_encoder, device)
        if self.pos_bias is not None:
            # Per-head 2D relative-position bias on the logits (soft locality); the boolean
            # mask becomes a float mask that keeps the bias and sets masked pairs to -inf.
            bias = self.pos_bias().to(query.dtype)  # [H, N, N]
            attn_mask = bias.masked_fill(~attn_mask, float("-inf"))
        with sdpa_kernel(SDPA_BACKENDS):
            attention_output = F.scaled_dot_product_attention(
                query=query, key=key, value=value, attn_mask=attn_mask, is_causal=False
            )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))
        return self.dropout(output)

    def init_weights(self, init_std: float) -> None:
        """Initialize attention projection weights."""
        for linear in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.output_projection.weight, mean=0.0, std=init_std)


@lru_cache(maxsize=64)
def _create_neighborhood_mask(
    num_nodes: int, is_encoder: bool, device: str, radius: int = 1
) -> torch.Tensor:
    """Create a dilated grid-neighborhood mask, cached per (size, role, device, radius).

    A content node attends to every node within Manhattan grid distance <= radius,
    clipped to the grid, so the neighbour count is position-dependent (radius=1:
    interior 4, edge 3, corner 2, plus self).
    The Manhattan ball is invariant under the grid's D4 automorphisms, so the encoder
    stays D4-equivariant at any radius.
    """
    n_content = num_nodes - 1 if is_encoder else num_nodes
    n = int(math.sqrt(n_content))
    rows = torch.arange(n_content) // n
    cols = torch.arange(n_content) % n
    dist = (rows[:, None] - rows[None, :]).abs() + (cols[:, None] - cols[None, :]).abs()
    mask = dist <= radius
    if is_encoder:
        # CLS sits at position 0. Column 0 (CLS as key) is False: content nodes cannot read
        # CLS. Row 0 (CLS as query) is True: CLS reads every node. Without this asymmetry CLS
        # acts as a global bus and all node representations collapse to the same average.
        mask = F.pad(mask, (1, 0, 0, 0), value=False)  # content -> CLS blocked
        mask = F.pad(mask, (0, 0, 1, 0), value=True)  # CLS -> all allowed
    return mask.to(device)


def get_neighborhood_mask(
    num_nodes: int, is_encoder: bool, device: torch.device | None = None, radius: int = 1
) -> torch.Tensor:
    """Get dilated neighborhood mask, cached per device."""
    device_str = str(device) if device is not None else "cpu"
    return _create_neighborhood_mask(num_nodes, is_encoder, device_str, radius)


def get_full_mask(
    mask: torch.Tensor, is_encoder: bool, device: torch.device | None = None
) -> torch.Tensor:
    """All-True ``[N, N]`` attention mask (full attention) sized from ``mask.size(1)``.

    Grids are never padded, so the node mask only provides ``N``. The caller pads ``mask``
    if a CLS token was prepended.
    """
    if mask.dim() not in (2, 3):
        raise ValueError(f"Mask should be 2D or 3D, got shape {mask.shape}")
    num_nodes = mask.size(1)
    return torch.ones(num_nodes, num_nodes, dtype=torch.bool, device=device)


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
