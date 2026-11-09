"""Deletion / insertion faithfulness checks for an importance map over a bag.

Raw attention (and even gradient attributions) can be unfaithful — the literature shows
some attention explanations don't beat random. These curves are the standard sanity check:
if an importance ranking is faithful, removing the top-ranked instances (deletion) should
drop the score fast, and revealing them from a baseline (insertion) should raise it fast.

Operates on a single bag ``x`` ``[S, D]`` with per-instance ``importance`` ``[S]`` and a
``forward_fn`` mapping a bag ``[S, D]`` to the scalar target score (e.g. the ABMIL logit).
Instances are neutralised by replacement with ``baseline`` (default zeros), matching the IG
baseline so the two analyses are consistent.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def _curve(forward_fn, x: torch.Tensor, order: list[int], baseline: torch.Tensor, mode: str, steps: int | None) -> torch.Tensor:
    """Score at ``steps+1`` checkpoints as instances are removed/revealed in ``order``.

    ``steps=None`` records every instance (``S+1`` points, per-instance granularity); a smaller
    ``steps`` groups the ranking into that many chunks — far fewer ``forward_fn`` calls, which
    matters when ``forward_fn`` is a full encoder pass and ``S`` is 256 nodes.
    """
    s = len(order)
    steps = s if steps is None else min(int(steps), s)
    ks = [round(i * s / steps) for i in range(steps + 1)]  # 0 .. S removed/revealed
    scores: list[float] = []
    for k in ks:
        if mode == "deletion":
            cur = x.clone()
            for i in order[:k]:
                cur[i] = baseline[i]
        elif mode == "insertion":
            cur = baseline.clone()
            for i in order[:k]:
                cur[i] = x[i]
        else:
            raise ValueError(f"mode must be 'deletion' or 'insertion', got {mode!r}")
        scores.append(float(forward_fn(cur)))
    return torch.tensor(scores)


def deletion_curve(forward_fn, x: torch.Tensor, importance: torch.Tensor, baseline: torch.Tensor | None = None, descending: bool = True, steps: int | None = None) -> torch.Tensor:
    """Score as top-``importance`` instances are progressively removed. Returns ``[steps+1]`` (``[S+1]`` if steps=None)."""
    if baseline is None:
        baseline = torch.zeros_like(x)
    order = torch.argsort(importance, descending=descending).tolist()
    return _curve(forward_fn, x, order, baseline, "deletion", steps)


def insertion_curve(forward_fn, x: torch.Tensor, importance: torch.Tensor, baseline: torch.Tensor | None = None, descending: bool = True, steps: int | None = None) -> torch.Tensor:
    """Score as top-``importance`` instances are progressively revealed. Returns ``[steps+1]`` (``[S+1]`` if steps=None)."""
    if baseline is None:
        baseline = torch.zeros_like(x)
    order = torch.argsort(importance, descending=descending).tolist()
    return _curve(forward_fn, x, order, baseline, "insertion", steps)


def curve_auc(scores: torch.Tensor) -> float:
    """Normalised area under a deletion/insertion curve (x-axis in [0, 1])."""
    n = scores.numel()
    return float(torch.trapz(scores, dx=1.0 / (n - 1)))


def faithfulness_score(forward_fn, x: torch.Tensor, importance: torch.Tensor, baseline: torch.Tensor | None = None, steps: int | None = None) -> dict[str, float]:
    """Summary: insertion AUC minus deletion AUC. Higher = more faithful importance map."""
    dele = deletion_curve(forward_fn, x, importance, baseline, steps=steps)
    ins = insertion_curve(forward_fn, x, importance, baseline, steps=steps)
    del_auc, ins_auc = curve_auc(dele), curve_auc(ins)
    return {"deletion_auc": del_auc, "insertion_auc": ins_auc, "faithfulness": ins_auc - del_auc}
