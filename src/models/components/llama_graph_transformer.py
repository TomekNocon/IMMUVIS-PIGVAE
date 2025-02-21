import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.nn.attention import SDPBackend
from typing import Optional
from src.models.components.custom_pytorch_functions import scaled_dot_product_attention
from src.models.components.custom_pytorch_functions import RMSNorm


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
            [
                TransformerBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, x, mask):
        # output = self.embedding_layer(input_ids)

        for block in self.blocks:
            x = block(x, mask)

        output = x
        # output = self.head(output)
        return output


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

    def __init__(self, hidden_dim, n_head, dropout):
        super().__init__()
        self.attention_layer = SelfAttention(n_head, hidden_dim, dropout)
        self.feed_forward_layer = FeedForward(
            hidden_dim=hidden_dim,
            ffn_hidden_dim=4 * hidden_dim,
            multiple_of=32,
            ffn_dim_multiplier=None,
        )

        self.attention_norm = RMSNorm(hidden_dim=hidden_dim, eps=1e-5)
        self.ffn_norm = RMSNorm(hidden_dim=hidden_dim, eps=1e-5)
        self.num_layers = 4  # change that so you can pass by params

        # if model_args.depth_init:
        #     self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        # else:
        #     self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5

        self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5

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

    def forward(self, x, attention_mask):
        out_attention = self.attention_layer(self.attention_norm(x), attention_mask)
        x = x + out_attention

        out_feed_forward = self.feed_forward_layer(self.ffn_norm(x))
        x = x + out_feed_forward
        return x

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


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
        dropout: Optional[float] = 0.0,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(hidden_dim, ffn_hidden_dim, bias=False)
        self.w2 = nn.Linear(ffn_hidden_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, ffn_hidden_dim, bias=False)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        x = self.dropout(x)
        return x

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class SelfAttention(torch.nn.Module):
    def __init__(self, n_head, hidden_dim, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: b x nn x nn x dv
        batch_size, num_nodes = x.size(0), x.size(1)
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
