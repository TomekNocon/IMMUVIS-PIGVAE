"""Decision-level explainability for the downstream gated-ABMIL classifier.

The clinical decision is made by ``GatedABMIL`` over a bag of frozen ``z_global`` crop
embeddings, so *this* is where "which tissue regions drive the decision" is answered.

Two complementary views:
- ``abmil_attention``: the gated attention weight per crop (what the pooling up-weights) —
  cheap, but raw attention is only a weak explanation (validate with ``faithfulness``).
- ``abmil_ig``: Captum Integrated Gradients of the class logit w.r.t. the crop embeddings —
  signed, class-discriminative, and more faithful than raw attention.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def abmil_attention(model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Recompute the gated-ABMIL attention weights over crops.

    Mirrors ``GatedABMIL.forward`` up to the softmax (deterministic from ``V/U/W``), so no
    forward hook is needed.

    Args:
        model: a ``GatedABMIL``.
        x: crop-bag embeddings ``[B, S, D]``.
        mask: ``[B, S]`` bool, ``True`` = padding (ABMIL convention), or None.

    Returns:
        attention ``[B, S, H]``, softmaxed over the S crops (per head).
    """
    model.eval()
    v = torch.tanh(model.V(x))
    u = torch.sigmoid(model.U(x))
    a = model.W(v * u)  # [B, S, H]
    if mask is not None:
        a = a.masked_fill(mask.unsqueeze(2), -1e9)
    return torch.softmax(a, dim=1)  # softmax over crops


def abmil_ig(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: int = 0,
    mask: torch.Tensor | None = None,
    baselines: torch.Tensor | None = None,
    n_steps: int = 50,
    return_convergence_delta: bool = False,
):
    """Integrated Gradients of the ABMIL class logit w.r.t. the crop embeddings.

    Args:
        model: a ``GatedABMIL`` (``forward`` returns ``(logits, pooled)``).
        x: crop-bag embeddings ``[B, S, D]``.
        target: class-logit column to explain (binary head is 1-D -> use 0).
        mask: ``[B, S]`` padding mask (passed through as a fixed forward arg).
        baselines: reference bag ``[B, S, D]``; defaults to zeros.
        n_steps: IG Riemann steps.
        return_convergence_delta: also return the IG completeness gap.

    Returns:
        attributions ``[B, S, D]``, or ``(attributions, delta)``.
    """
    from captum.attr import IntegratedGradients

    model.eval()
    x = x.detach().clone().float().requires_grad_(True)
    if baselines is None:
        baselines = torch.zeros_like(x)

    def forward_logits(inp: torch.Tensor, m: torch.Tensor | None) -> torch.Tensor:
        logits, _pooled = model(inp, m)
        return logits

    ig = IntegratedGradients(forward_logits)
    out = ig.attribute(
        x,
        baselines=baselines,
        target=target,
        additional_forward_args=(mask,),
        n_steps=n_steps,
        return_convergence_delta=return_convergence_delta,
    )
    if return_convergence_delta:
        attributions, delta = out
        return attributions.detach(), delta.detach()
    return out.detach()


def crop_importance(attr: torch.Tensor, signed: bool = True) -> torch.Tensor:
    """Reduce a per-crop attribution ``[B, S, D]`` to a per-crop score ``[B, S]``.

    ``signed=True`` sums over D (direction toward the target class); ``signed=False`` uses
    the absolute contribution magnitude.
    """
    return attr.sum(dim=-1) if signed else attr.abs().sum(dim=-1)
