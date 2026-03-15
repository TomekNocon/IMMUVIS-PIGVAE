import math

import torch
import torch.nn as nn


class PositionalEncoding(torch.nn.Module):
    """2D sinusoidal positional encoding for grid-structured data.

    Splits d_hid in half: first half encodes the row, second half encodes the
    column.  For non-square grids, pass grid_h and grid_w explicitly.
    Falls back to 1D encoding when grid_size is not set (grid_size=0).
    """

    def __init__(self, d_hid: int, grid_size: int = 7, n_position: int = 200):
        super().__init__()
        self.d_hid = d_hid
        self.grid_size = grid_size

        if grid_size > 0:
            # Build a 2D table: (1, grid_size*grid_size, d_hid)
            self.register_buffer(
                "pos_table",
                self._build_2d_table(grid_size, d_hid),
            )
        else:
            # Fallback: classic 1D sinusoidal
            self.register_buffer(
                "pos_table",
                self._build_1d_table(n_position, d_hid),
            )

    # ---- 2D encoding -------------------------------------------------------
    @staticmethod
    def _build_2d_table(grid_size: int, d_hid: int) -> torch.Tensor:
        half = d_hid // 2
        # Row frequencies
        row_pos = torch.arange(grid_size, dtype=torch.float).unsqueeze(1)  # (H, 1)
        row_div = torch.exp(
            torch.arange(0, half, 2, dtype=torch.float) * (-math.log(10000.0) / half)
        )
        row_enc = torch.zeros(grid_size, half)
        row_enc[:, 0::2] = torch.sin(row_pos * row_div)
        row_enc[:, 1::2] = torch.cos(row_pos * row_div)

        # Column frequencies
        col_pos = torch.arange(grid_size, dtype=torch.float).unsqueeze(1)
        col_div = torch.exp(
            torch.arange(0, d_hid - half, 2, dtype=torch.float)
            * (-math.log(10000.0) / (d_hid - half))
        )
        col_enc = torch.zeros(grid_size, d_hid - half)
        col_enc[:, 0::2] = torch.sin(col_pos * col_div)
        col_enc[:, 1::2] = torch.cos(col_pos * col_div)

        # Combine: for each (r, c) concatenate row_enc[r] and col_enc[c]
        # Result shape: (grid_size*grid_size, d_hid)
        table = torch.cat(
            [
                row_enc.unsqueeze(1).expand(-1, grid_size, -1).reshape(-1, half),
                col_enc.unsqueeze(0).expand(grid_size, -1, -1).reshape(-1, d_hid - half),
            ],
            dim=-1,
        )
        return table.unsqueeze(0)  # (1, N, d_hid)

    # ---- 1D fallback --------------------------------------------------------
    @staticmethod
    def _build_1d_table(n_position: int, d_hid: int) -> torch.Tensor:
        position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))
        table = torch.zeros(n_position, d_hid)
        table[:, 0::2] = torch.sin(position * div_term)
        table[:, 1::2] = torch.cos(position * div_term)
        return table.unsqueeze(0)

    def forward(self, batch_size: int, num_nodes: int) -> torch.FloatTensor:
        x = self.pos_table[:, :num_nodes].clone()
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
