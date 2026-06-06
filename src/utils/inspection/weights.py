# src/utils/inspection/weights.py
from __future__ import annotations

import torch.nn as nn

from src.utils.inspection.stats import linear_spectral, tensor_stats

_SECTIONS = ("encoder", "decoder", "node_bottleneck", "permuter")


def _section_of(name: str) -> str:
    head = name.split(".")[0]
    for sec in _SECTIONS:
        if head == sec or name.startswith(sec + ".") or f".{sec}." in name:
            return sec
    return "other"


def weight_diagnostics(model: nn.Module) -> dict:
    """Per-Linear and per-Norm weight stats, keyed by module name, grouped by section."""
    out: dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            entry = tensor_stats(module.weight)
            entry["l2_norm"] = module.weight.detach().float().norm().item()
            entry.update(linear_spectral(module.weight))
            entry["section"] = _section_of(name)
            entry["type"] = "Linear"
            out[name] = entry
        elif module.__class__.__name__ in ("RMSNorm", "LayerNorm"):
            w = getattr(module, "weight", None)
            if w is None:
                continue
            wd = w.detach().float()
            out[name] = {
                "gain_rms": wd.pow(2).mean().sqrt().item(),
                "gain_min": wd.min().item(),
                "gain_max": wd.max().item(),
                "section": _section_of(name),
                "type": module.__class__.__name__,
            }
    return out
