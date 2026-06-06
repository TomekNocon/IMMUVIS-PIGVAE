# src/utils/inspection/stats.py
from __future__ import annotations

import torch

_MAX_QUANTILE_ELEMS = 1_000_000


def tensor_stats(t: torch.Tensor) -> dict:
    """Summary distribution stats for a tensor (flattened, fp32)."""
    t = t.detach().float().flatten()
    n = int(t.numel())
    if n == 0:
        return {"count": 0}
    tq = t if n <= _MAX_QUANTILE_ELEMS else t[torch.randperm(n)[:_MAX_QUANTILE_ELEMS]]
    q = torch.tensor([0.001, 0.01, 0.5, 0.99, 0.999], device=t.device)
    pcts = torch.quantile(tq, q).tolist()
    mean = t.mean()
    std = t.std(unbiased=False)
    kurt = ((((t - mean) / std) ** 4).mean().item() - 3.0) if std > 0 else 0.0
    return {
        "count": n,
        "mean": mean.item(),
        "std": std.item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "max_abs": t.abs().max().item(),
        "p0.1": pcts[0], "p1": pcts[1], "p50": pcts[2], "p99": pcts[3], "p99.9": pcts[4],
        "kurtosis": kurt,
        "frac_dead": (t.abs() < 1e-6).float().mean().item(),
    }


def participation_ratio(values: torch.Tensor) -> float:
    """Participation ratio of a non-negative spectrum: (Σv)² / Σv². 1≈spiked, len≈uniform."""
    v = values.detach().float().clamp_min(0)
    s = v.sum()
    if s <= 0:
        return 0.0
    return (s * s / (v * v).sum()).item()


def linear_spectral(weight: torch.Tensor) -> dict:
    """Spectral norm and effective rank (PR over singular values²) of a 2D weight."""
    w = weight.detach().float()
    if w.ndim > 2:
        w = w.flatten(1)
    elif w.ndim < 2:
        w = w.unsqueeze(0)
    sv = torch.linalg.svdvals(w)
    pr = participation_ratio(sv ** 2)
    return {
        "spectral_norm": sv.max().item(),
        "effective_rank": pr,
        "rank_ratio": pr / min(w.shape),
    }
