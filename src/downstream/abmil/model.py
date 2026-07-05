import torch
import torch.nn as nn


class GatedABMIL(nn.Module):
    """Gated attention-based multiple-instance-learning (MIL) bag classifier.

    Ported from the reference implementation at
    `/home/tnocon/mil/gated_abmil/models/ABMIL/gated_abmil.py` (Ilse et al.,
    "Attention-based Deep Multiple Instance Learning"), with the classifier
    head folded directly into this module (the original
    `GatedABMILClassifierWithValidation` wrapper is dropped; a Lightning
    module takes over that role in Task B4).

    Mask convention: `mask` is `True` at PADDING positions (opposite of the
    PIGVAE encoder's mask, where `True` = valid node). Padding positions are
    masked out of the attention softmax via `masked_fill(..., -1e9)`.
    """

    def __init__(self, emb_dim, hidden_dim, num_heads=1, num_classes=2):
        super().__init__()
        self.V = nn.Linear(emb_dim, hidden_dim)
        self.U = nn.Linear(emb_dim, hidden_dim)
        self.W = nn.Linear(hidden_dim, num_heads)
        self.num_heads = num_heads
        out_dim = num_classes - 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(num_heads * emb_dim, out_dim)

    def forward(self, x, mask=None):
        """
        x: bag of instance embeddings, shape [B, S, D]
        mask: bool tensor [B, S], True = padding (masked out), or None
        Returns: (logits [B, C'], pooled [B, num_heads * D])
        """
        v = torch.tanh(self.V(x))
        u = torch.sigmoid(self.U(x))
        a = self.W(v * u)  # B, S, H
        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(2), -1e9)
        a = torch.softmax(a, dim=1).transpose(1, 2)  # B, H, S
        pooled = torch.bmm(a, x).reshape(x.size(0), self.num_heads * x.size(2))
        return self.classifier(pooled), pooled
