import numpy as np
import torch
import math
import torch.nn as nn

# from torch.nn.attention import SDPBackend
from collections import OrderedDict

"""
adapted from https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""


class Transformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ppf_hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.ppf_hidden_dim = ppf_hidden_dim  # TBDeleted
        # self.embedding_layer = EmbeddingLayer(
        #     config.vocab_size, config.d_model, config.max_len
        # )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        # self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, x, mask):
        # output = self.embedding_layer(input_ids)

        for block in self.blocks:
            x = block(x, mask)

        output = x
        # output = self.head(output)
        return output

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_head, dropout):
        super().__init__()
        self.attention_layer = SelfAttention(n_head, hidden_dim, dropout)
        self.feed_forward_layer = FeedForward(hidden_dim, dropout)

    def forward(self, x, attention_mask):
        out_attention = self.attention_layer(x, attention_mask)
        x = x + out_attention

        out_feed_forward = self.feed_forward_layer(x)
        x = x + out_feed_forward
        return x


def FeedForward(hidden_size, dropout=0.1):
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
    def __init__(self, n_head, hidden_dim, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.hidden_dim = hidden_dim

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.input_projection = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: b x nn x nn x dv
        batch_size, num_nodes = x.size(0), x.size(1)
        x = self.layer_norm(x)
        projected = self.input_projection(x)

        device = x.device

        q_chunk, k_chunk, v_chunk = torch.chunk(projected, chunks=3, dim=-1)
        query = q_chunk.view(batch_size, num_nodes, num_nodes, self.n_head, -1).permute(
            0, 3, 1, 2, 4
        )
        key = k_chunk.view(batch_size, num_nodes, num_nodes, self.n_head, -1).permute(
            0, 3, 2, 1, 4
        )
        value = v_chunk.view(batch_size, num_nodes, num_nodes, self.n_head, -1).permute(
            0, 3, 2, 1, 4
        )

        attn_mask = mask.masked_fill(
            torch.eye(num_nodes, num_nodes, device=device).bool(), 0
        )
        attn_mask = attn_mask.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        attn_mask = attn_mask * (
            torch.eye(num_nodes, num_nodes, device=device) == 0
        ).bool().unsqueeze(0).unsqueeze(-2).expand(-1, -1, num_nodes, -1)

        # with torch.nn.attention.sdpa_kernel(
        #     [
        #         SDPBackend.FLASH_ATTENTION,
        #         SDPBackend.EFFICIENT_ATTENTION,
        #         SDPBackend.MATH,
        #     ]
        # ):

        attention_output = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            is_causal=False,
        )
        attention_output = attention_output.permute(0, 2, 3, 1, 4).contiguous()
        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))
        output = self.dropout(output)
        return output


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, batch_size, num_nodes):
        x = self.pos_table[:, :num_nodes].clone().detach()
        x = x.expand(batch_size, -1, -1)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        positions = (
            torch.arange(seq_len, dtype=torch.long, device=x.device)
            .unsqueeze(0)
            .expand_as(x)
        )
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)
        embeddings = token_embeddings + position_embeddings
        return embeddings


# Efficient implementation equivalent to the following:
# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
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
