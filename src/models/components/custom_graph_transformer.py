import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# from torch.nn.attention import SDPBackend
from collections import OrderedDict

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
        # rope: Optional[BaseRotaryEmbedding] = None,
        # perm: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.ppf_hidden_dim = ppf_hidden_dim  # TBDeleted
        # self.embedding_layer = EmbeddingLayer(
        #     config.vocab_size, config.d_model, config.max_len
        # )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim,
                    num_heads,
                    dropout,
                    #  rope, perm
                )
                for _ in range(num_layers)
            ]
        )

        # self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # output = self.embedding_layer(input_ids)

        for block in self.blocks:
            x = block(x, mask)

        output = x
        # output = self.head(output)
        return output

    def init_weights(self) -> None:
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: torch.Tensor,
        n_head: torch.Tensor,
        dropout: float,
        num_layers: Optional[int] = None,
        # rope: Optional[BaseRotaryEmbedding] = None,
        # perm: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.attention_layer = SelfAttention(
            n_head,
            hidden_dim,
            dropout,
            #  rope, perm
        )
        self.feed_forward_layer = FeedForward(hidden_dim, dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out_attention = self.attention_layer(x, attention_mask)
        x = x + out_attention

        out_feed_forward = self.feed_forward_layer(x)
        x = x + out_feed_forward
        return x


def FeedForward(hidden_size: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            [
                ("ff_layernorm", nn.LayerNorm(hidden_size)),
                (
                    "pre_relu",
                    nn.Linear(
                        hidden_size,
                        4 * hidden_size,
                        bias=True,
                    ),
                ),
                ("relu", nn.ReLU()),
                (
                    "post_relu",
                    nn.Linear(
                        4 * hidden_size,
                        hidden_size,
                        bias=True,
                    ),
                ),
                ("dropout", nn.Dropout(dropout)),
            ]
        )
    )


class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        n_head: torch.Tensor,
        hidden_dim: torch.Tensor,
        dropout: float,
        # rope: Optional[BaseRotaryEmbedding] = None,
        # perm: Optional[torch.Tensor] = None
    ):
        super().__init__()

        self.n_head = n_head
        self.hidden_dim = hidden_dim

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.input_projection = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        # self.rope = rope
        # self.perm = perm

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: b x nn x nn x dv
        batch_size, num_nodes = x.size(0), x.size(1)
        x = self.layer_norm(x)
        projected = self.input_projection(x)

        device = x.device

        q_chunk, k_chunk, v_chunk = torch.chunk(projected, chunks=3, dim=-1)
        query = q_chunk.view(batch_size, num_nodes, self.n_head, -1).transpose(1, 2)
        key = k_chunk.view(batch_size, num_nodes, self.n_head, -1).transpose(1, 2)
        value = v_chunk.view(batch_size, num_nodes, self.n_head, -1).transpose(1, 2)
        # if self.rope:
        #     query = self.rope.rotate_queries_or_keys(query)
        #     key = self.rope.rotate_queries_or_keys(key)

        attn_mask = mask.to(device)

        attn_mask = attn_mask.unsqueeze(1).unsqueeze(
            2
        )  # Shape: (batch_size, 1, 1, num_nodes)
        attn_mask = attn_mask.expand(-1, self.n_head, num_nodes, -1)

        # with torch.nn.attention.sdpa_kernel(
        #     [
        #         SDPBackend.FLASH_ATTENTION,
        #         SDPBackend.EFFICIENT_ATTENTION,
        #         SDPBackend.MATH,
        #     ]
        # ):

        attention_output = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            is_causal=False,
        )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))
        # if self.rope and self.perm:
        #     output = torch.matmul(self.perm, output)
        output = self.dropout(output)
        return output


# Efficient implementation equivalent to the following:
# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(
        (*query.shape[:-2], L, S), dtype=query.dtype, device=query.device
    )
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask[0].logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
