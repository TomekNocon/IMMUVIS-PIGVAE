import math

import torch
import torch.nn as nn


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_hid: int, n_position: int = 200):
        super().__init__()

        # Not a parameter
        self.register_buffer(
            "pos_table",
            self._get_sinusoid_encoding_table(n_position, d_hid),
        )

    def _get_sinusoid_encoding_table(
        self,
        n_position: int,
        d_hid: int,
    ) -> torch.FloatTensor:
        position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid),
        )
        table = torch.zeros(n_position, d_hid)
        table[:, 0::2] = torch.sin(position * div_term)
        table[:, 1::2] = torch.cos(position * div_term)
        return table.unsqueeze(0)

    def forward(self, batch_size: int, num_nodes: int) -> torch.FloatTensor:
        x = self.pos_table[:, :num_nodes].clone().detach()
        x = x.expand(batch_size, -1, -1)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_len: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        positions = (
            torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand_as(x)
        )
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)
        embeddings = token_embeddings + position_embeddings
        return embeddings
