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
def _scores_along_order(forward_fn, x: torch.Tensor, order: list[int], baseline: torch.Tensor, mode: str) -> torch.Tensor:
    scores: list[float] = []
    if mode == "deletion":
        cur = x.clone()
        scores.append(float(forward_fn(cur)))
        for i in order:
            cur = cur.clone()
            cur[i] = baseline[i]
            scores.append(float(forward_fn(cur)))
    elif mode == "insertion":
        cur = baseline.clone()
        scores.append(float(forward_fn(cur)))
        for i in order:
            cur = cur.clone()
            cur[i] = x[i]
            scores.append(float(forward_fn(cur)))
    else:
        raise ValueError(f"mode must be 'deletion' or 'insertion', got {mode!r}")
    return torch.tensor(scores)


def deletion_curve(forward_fn, x: torch.Tensor, importance: torch.Tensor, baseline: torch.Tensor | None = None, descending: bool = True) -> torch.Tensor:
    """Score as top-``importance`` instances are progressively removed. Returns ``[S+1]``."""
    if baseline is None:
        baseline = torch.zeros_like(x)
    order = torch.argsort(importance, descending=descending).tolist()
    return _scores_along_order(forward_fn, x, order, baseline, "deletion")


def insertion_curve(forward_fn, x: torch.Tensor, importance: torch.Tensor, baseline: torch.Tensor | None = None, descending: bool = True) -> torch.Tensor:
    """Score as top-``importance`` instances are progressively revealed. Returns ``[S+1]``."""
    if baseline is None:
        baseline = torch.zeros_like(x)
    order = torch.argsort(importance, descending=descending).tolist()
    return _scores_along_order(forward_fn, x, order, baseline, "insertion")


def curve_auc(scores: torch.Tensor) -> float:
    """Normalised area under a deletion/insertion curve (x-axis in [0, 1])."""
    n = scores.numel()
    return float(torch.trapz(scores, dx=1.0 / (n - 1)))


def faithfulness_score(forward_fn, x: torch.Tensor, importance: torch.Tensor, baseline: torch.Tensor | None = None) -> dict[str, float]:
    """Summary: insertion AUC minus deletion AUC. Higher = more faithful importance map."""
    dele = deletion_curve(forward_fn, x, importance, baseline)
    ins = insertion_curve(forward_fn, x, importance, baseline)
    del_auc, ins_auc = curve_auc(dele), curve_auc(ins)
    return {"deletion_auc": del_auc, "insertion_auc": ins_auc, "faithfulness": ins_auc - del_auc}
