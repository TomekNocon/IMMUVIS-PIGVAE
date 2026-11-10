"""Gated attention-based multiple-instance learning (Ilse et al., 2018)."""

import torch
import torch.nn as nn


class GatedABMIL(nn.Module):
    """Gated-attention MIL pooling followed by a linear bag classifier.

    Mask convention: ``mask`` is True at *padding* positions (the opposite of the PIGVAE
    encoder mask, where True = valid node); padding is excluded from the attention softmax.
    Binary problems use a single logit.
    """

    def __init__(self, emb_dim: int, hidden_dim: int, num_heads: int = 1, num_classes: int = 2):
        super().__init__()
        self.V = nn.Linear(emb_dim, hidden_dim)
        self.U = nn.Linear(emb_dim, hidden_dim)
        self.W = nn.Linear(hidden_dim, num_heads)
        self.num_heads = num_heads
        out_dim = num_classes - 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(num_heads * emb_dim, out_dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(logits [B, C'], pooled [B, num_heads * D])`` for bags ``x [B, S, D]``."""
        v = torch.tanh(self.V(x))
        u = torch.sigmoid(self.U(x))
        a = self.W(v * u)  # B, S, H
        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(2), -1e9)
        a = torch.softmax(a, dim=1).transpose(1, 2)  # B, H, S
        pooled = torch.bmm(a, x).reshape(x.size(0), self.num_heads * x.size(2))
        return self.classifier(pooled), pooled
