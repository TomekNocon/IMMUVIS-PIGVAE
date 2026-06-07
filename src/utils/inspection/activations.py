# src/utils/inspection/activations.py
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from src.utils.inspection.stats import tensor_stats


def collect_activation_stats(
    model: nn.Module,
    run_forward: Callable[[], object],
    module_filter: Callable[[str, nn.Module], bool],
    top_k_dims: int = 10,
) -> dict:
    """Register forward hooks on modules matching `module_filter`, run `run_forward`,
    and return per-module output stats. For 3D [B, N, D] outputs also record per-token
    L2 norm (mean over batch) and the top-k hidden dims by mean |activation|."""
    out: dict = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inp, output):
            t = output[0] if isinstance(output, (tuple, list)) and output else output
            if not isinstance(t, torch.Tensor):
                return
            entry = tensor_stats(t)
            if t.dim() == 3:
                tf = t.detach().float()
                entry["per_token_norm"] = tf.norm(dim=-1).mean(dim=0).tolist()
                dim_mag = tf.abs().mean(dim=(0, 1))
                k = min(top_k_dims, dim_mag.numel())
                vals, idx = torch.topk(dim_mag, k)
                entry["top_dims_by_mean_abs"] = list(
                    zip(idx.tolist(), vals.tolist(), strict=True)
                )
            out[name] = entry
        return hook

    for name, module in model.named_modules():
        if module_filter(name, module):
            handles.append(module.register_forward_hook(make_hook(name)))
    try:
        with torch.no_grad():
            run_forward()
    finally:
        for h in handles:
            h.remove()
    return out
