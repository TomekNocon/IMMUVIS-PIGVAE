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
    tq = t if n <= _MAX_QUANTILE_ELEMS else t[torch.randperm(n, device=t.device)[:_MAX_QUANTILE_ELEMS]]
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


def energy_rank(singular_values: torch.Tensor, fractions=(0.9, 0.99)) -> dict:
    """Smallest number of singular values capturing each energy fraction of Σσ².

    This is the practical "how narrow can I go" measure: rank90=k means the top k
    singular directions hold 90% of the weight's Frobenius energy. Unlike the
    participation ratio it is not skewed by the squared-spectrum tail.
    """
    sv = singular_values.detach().float().clamp_min(0)
    energy = torch.sort(sv, descending=True).values ** 2
    total = energy.sum()
    out: dict = {}
    if total <= 0:
        for f in fractions:
            out[f"rank{round(f * 100)}"] = 0
        return out
    cum = torch.cumsum(energy, dim=0) / total
    for f in fractions:
        k = int((cum >= f).to(torch.int).argmax().item()) + 1  # first index reaching f, 1-indexed
        out[f"rank{round(f * 100)}"] = k
    return out


def linear_spectral(weight: torch.Tensor) -> dict:
    """Spectral norm, effective rank (PR over σ²), and 90/99%-energy rank of a 2D weight."""
    w = weight.detach().float()
    if w.ndim > 2:
        w = w.flatten(1)
    elif w.ndim < 2:
        w = w.unsqueeze(0)
    sv = torch.linalg.svdvals(w)
    pr = participation_ratio(sv ** 2)
    dim = min(w.shape)
    er = energy_rank(sv)
    return {
        "spectral_norm": sv.max().item(),
        "effective_rank": pr,
        "rank_ratio": pr / dim,
        "rank90": er["rank90"],
        "rank99": er["rank99"],
        "rank90_ratio": er["rank90"] / dim,
        "rank99_ratio": er["rank99"] / dim,
    }
