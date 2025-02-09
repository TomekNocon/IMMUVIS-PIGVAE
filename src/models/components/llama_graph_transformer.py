import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.attention import SDPBackend
from typing import Optional

"""
adapted from https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""

class Transformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ppf_hidden_dim, num_layers, dropout = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.ppf_hidden_dim = ppf_hidden_dim #TBDeleted
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
        hidden_dim,
        n_head,
        dropout
    ):
        super().__init__()
        self.attention_layer = SelfAttention(n_head, hidden_dim, dropout)
        self.feed_forward_layer = FeedForward(
            hidden_dim = hidden_dim,
            ffn_hidden_dim = 4 * hidden_dim,
            multiple_of=32,
            ffn_dim_multiplier=None,
        )

        self.attention_norm = RMSNorm(hidden_dim=hidden_dim, eps=1e-5)
        self.ffn_norm = RMSNorm(hidden_dim=hidden_dim, eps=1e-5)
        self.num_layers = 4 # change that so you can pass by params 

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
        query = q_chunk.view(batch_size, num_nodes, num_nodes, self.n_head, -1).permute(0, 3, 1, 2, 4)
        key = k_chunk.view(batch_size, num_nodes, num_nodes, self.n_head, -1).permute(0, 3, 2, 1, 4)
        value = v_chunk.view(batch_size, num_nodes, num_nodes, self.n_head, -1).permute(0, 3, 2, 1, 4)
        
        attn_mask = mask.masked_fill(torch.eye(num_nodes, num_nodes, device=device).bool(), 0)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        attn_mask = attn_mask * (torch.eye(
            num_nodes, num_nodes, device=device) == 0).bool().unsqueeze(0).unsqueeze(-2).expand(-1, -1, num_nodes, -1
                                                                                                )
        
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
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
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
    
class RMSNorm(nn.Module):
    """Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore
    

# Efficient implementation equivalent to the following:
# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros((*query.shape[:-2],  L, S), dtype=query.dtype, device=query.device)
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
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
    
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
