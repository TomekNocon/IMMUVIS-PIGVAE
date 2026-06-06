# Model Inspection Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add offline, checkpoint-level model inspection (weights, activations, attention, latent bottleneck, reconstruction) producing a machine-readable `report.json` + `report.md` + key plots, driven by a `--inspect` flag on `scripts/diagnose_model.py`.

**Architecture:** A focused package `src/utils/inspection/` of small, independently testable functions that operate on plain tensors / `nn.Module`s. `diagnose_model.py` orchestrates them under `--inspect` and writes artifacts. All units are CPU-testable with tiny synthetic tensors / a small `Transformer`; only the final end-to-end run needs szary (GPU + data + checkpoint).

**Tech Stack:** PyTorch, NumPy, matplotlib (all already in `pyproject.toml`), pytest.

---

## File structure

- Create `src/utils/inspection/__init__.py` — exports the public functions.
- Create `src/utils/inspection/stats.py` — pure tensor-stat helpers (`tensor_stats`, `participation_ratio`, `linear_spectral`).
- Create `src/utils/inspection/weights.py` — `weight_diagnostics(model)`.
- Create `src/utils/inspection/latent.py` — `latent_diagnostics(z_nodes)`.
- Create `src/utils/inspection/reconstruction.py` — `reconstruction_diagnostics(pred, true, num_views)`.
- Create `src/utils/inspection/activations.py` — `collect_activation_stats(model, run_forward)` via hooks.
- Create `src/utils/inspection/attention.py` — `attention_diagnostics(model, batch)` (recompute entropy; SDPA is fused).
- Create `src/utils/inspection/report.py` — `build_flags(results)`, `write_report(results, out_dir)`.
- Modify `scripts/diagnose_model.py` — add `--inspect` / `--out-dir`, orchestrate, call `write_report`.
- Tests: `tests/test_inspection_stats.py`, `tests/test_inspection_weights.py`, `tests/test_inspection_latent.py`, `tests/test_inspection_reconstruction.py`, `tests/test_inspection_activations.py`, `tests/test_inspection_attention.py`, `tests/test_inspection_report.py`.

Tests run with `uv run pytest <path> -v`.

---

## Task 1: stats helpers

**Files:**
- Create: `src/utils/inspection/__init__.py`
- Create: `src/utils/inspection/stats.py`
- Test: `tests/test_inspection_stats.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inspection_stats.py
import math
import torch
from src.utils.inspection.stats import tensor_stats, participation_ratio, linear_spectral


def test_tensor_stats_basic():
    t = torch.arange(0, 100, dtype=torch.float32)
    s = tensor_stats(t)
    assert s["count"] == 100
    assert abs(s["mean"] - 49.5) < 1e-3
    assert s["max"] == 99.0
    assert s["max_abs"] == 99.0
    assert "p50" in s and "kurtosis" in s and "frac_dead" in s


def test_tensor_stats_empty():
    assert tensor_stats(torch.empty(0))["count"] == 0


def test_participation_ratio_uniform_vs_spiked():
    uniform = torch.ones(10)
    assert abs(participation_ratio(uniform) - 10.0) < 1e-4   # all equal -> full PR
    spiked = torch.tensor([100.0, 1e-6, 1e-6])
    assert participation_ratio(spiked) < 1.1                 # dominated by one -> ~1


def test_linear_spectral_identity():
    W = torch.eye(8)
    out = linear_spectral(W)
    assert abs(out["spectral_norm"] - 1.0) < 1e-4
    assert abs(out["effective_rank"] - 8.0) < 1e-3
    assert abs(out["rank_ratio"] - 1.0) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inspection_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: src.utils.inspection.stats`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/utils/inspection/__init__.py
"""Offline model-inspection diagnostics (weights, activations, attention, latent, reconstruction)."""
```

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_inspection_stats.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/utils/inspection/__init__.py src/utils/inspection/stats.py tests/test_inspection_stats.py
git commit -m "feat(inspection): tensor stat helpers (stats, participation ratio, spectral)"
```

---

## Task 2: weight diagnostics

**Files:**
- Create: `src/utils/inspection/weights.py`
- Test: `tests/test_inspection_weights.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inspection_weights.py
import torch
import torch.nn as nn
from src.utils.inspection.weights import weight_diagnostics


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8))
        self.decoder = nn.Linear(8, 4)


def test_weight_diagnostics_keys_and_sections():
    out = weight_diagnostics(Tiny())
    # one Linear in encoder, one LayerNorm in encoder, one Linear in decoder
    lin_keys = [k for k, v in out.items() if "spectral_norm" in v]
    norm_keys = [k for k, v in out.items() if v.get("type") == "LayerNorm"]
    assert len(lin_keys) == 2
    assert len(norm_keys) == 1
    assert out[lin_keys[0]]["section"] in {"encoder", "decoder"}
    assert any(v["section"] == "decoder" for v in out.values())
    # LayerNorm gain init = 1 -> gain_rms ~ 1
    assert abs(out[norm_keys[0]]["gain_rms"] - 1.0) < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inspection_weights.py -v`
Expected: FAIL with `ModuleNotFoundError: src.utils.inspection.weights`.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_inspection_weights.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/inspection/weights.py tests/test_inspection_weights.py
git commit -m "feat(inspection): per-layer weight norms, spectral norm, effective rank, norm gains"
```

---

## Task 3: latent diagnostics

**Files:**
- Create: `src/utils/inspection/latent.py`
- Test: `tests/test_inspection_latent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inspection_latent.py
import torch
from src.utils.inspection.latent import latent_diagnostics


def test_latent_full_rank_random():
    torch.manual_seed(0)
    z = torch.randn(8, 36, 32)
    out = latent_diagnostics(z)
    assert out["z_dim"] == 32
    assert out["active_dims"] >= 30          # random -> nearly all dims active
    assert out["rank_ratio"] > 0.8           # random -> near full rank
    assert out["node_norm_std"] > 0          # magnitude varies


def test_latent_collapsed():
    z = torch.zeros(8, 36, 32)
    z[..., 0] = torch.randn(8, 36)           # only one dim carries variance
    out = latent_diagnostics(z)
    assert out["active_dims"] == 1
    assert out["rank_ratio"] < 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inspection_latent.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/utils/inspection/latent.py
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.utils.inspection.stats import participation_ratio


def latent_diagnostics(z_nodes: torch.Tensor) -> dict:
    """Bottleneck health for per-node latent z_nodes [B, N, D]."""
    z = z_nodes.detach().float()
    b, n, d = z.shape
    flat = z.reshape(b * n, d)

    var_per_dim = flat.var(dim=0, unbiased=False)
    max_var = var_per_dim.max().clamp_min(1e-12)
    active = int((var_per_dim > 0.01 * max_var).sum().item())

    centered = flat - flat.mean(0, keepdim=True)
    cov = (centered.T @ centered) / flat.shape[0]
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    eff_rank = participation_ratio(eig)

    node_norms = z.norm(dim=-1)
    zn = F.normalize(z, dim=-1)
    sim = torch.matmul(zn, zn.transpose(1, 2))
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    off_diag = sim[:, ~eye]

    return {
        "z_dim": int(d),
        "active_dims": active,
        "effective_rank": eff_rank,
        "rank_ratio": eff_rank / d,
        "node_norm_mean": node_norms.mean().item(),
        "node_norm_std": node_norms.std(unbiased=False).item(),
        "node_norm_min": node_norms.min().item(),
        "node_norm_max": node_norms.max().item(),
        "inter_node_cossim_mean": off_diag.mean().item(),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_inspection_latent.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/inspection/latent.py tests/test_inspection_latent.py
git commit -m "feat(inspection): latent bottleneck diagnostics (active dims, effective rank, node collapse)"
```

---

## Task 4: reconstruction diagnostics

**Files:**
- Create: `src/utils/inspection/reconstruction.py`
- Test: `tests/test_inspection_reconstruction.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inspection_reconstruction.py
import torch
from src.utils.inspection.reconstruction import reconstruction_diagnostics


def test_perfect_reconstruction():
    true = torch.randn(8, 36, 16)
    out = reconstruction_diagnostics(true.clone(), true, num_views=8)
    assert out["overall_mse"] < 1e-10
    assert min(out["per_channel_r2"]) > 0.999
    assert len(out["per_channel_mse"]) == 16
    assert len(out["per_position_mse"]) == 36
    assert len(out["per_view_mse"]) == 8


def test_one_bad_channel_is_flagged_worst():
    torch.manual_seed(0)
    true = torch.randn(8, 36, 16)
    pred = true.clone()
    pred[..., 5] = 0.0                       # destroy channel 5
    out = reconstruction_diagnostics(pred, true, num_views=8)
    assert out["worst_channels"][0] == 5
    assert out["per_channel_r2"][5] < 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inspection_reconstruction.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/utils/inspection/reconstruction.py
from __future__ import annotations

import torch


def reconstruction_diagnostics(
    pred: torch.Tensor, true: torch.Tensor, num_views: int = 8
) -> dict:
    """Where reconstruction struggles: per-channel R²/MSE, per-position, per-view, vs magnitude.

    pred, true: [B, N, C].
    """
    pred = pred.detach().float()
    true = true.detach().float()
    b, n, c = true.shape
    se = (pred - true).pow(2)

    mse_ch = se.mean(dim=(0, 1))                              # [C]
    var_ch = true.var(dim=(0, 1), unbiased=False).clamp_min(1e-12)
    r2_ch = 1.0 - mse_ch / var_ch
    mse_pos = se.mean(dim=(0, 2))                             # [N]

    per_view = None
    if b % num_views == 0 and num_views > 0:
        bv = b // num_views
        per_view = se.view(num_views, bv, n, c).mean(dim=(1, 2, 3)).tolist()

    node_mag = true.norm(dim=-1).reshape(-1)                 # [B*N]
    node_err = se.mean(dim=-1).reshape(-1)
    edges = torch.quantile(node_mag, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    bins = []
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        m = (node_mag >= lo) & (node_mag <= hi) if i == 3 else (node_mag >= lo) & (node_mag < hi)
        bins.append({
            "mag_lo": lo.item(), "mag_hi": hi.item(),
            "mean_err": node_err[m].mean().item() if m.any() else 0.0,
            "count": int(m.sum().item()),
        })

    return {
        "overall_mse": se.mean().item(),
        "per_channel_mse": mse_ch.tolist(),
        "per_channel_r2": r2_ch.tolist(),
        "worst_channels": torch.topk(mse_ch, k=min(10, c)).indices.tolist(),
        "per_position_mse": mse_pos.tolist(),
        "per_view_mse": per_view,
        "error_by_magnitude": bins,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_inspection_reconstruction.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/inspection/reconstruction.py tests/test_inspection_reconstruction.py
git commit -m "feat(inspection): reconstruction breakdown (per-channel R2, per-position, per-view, vs magnitude)"
```

---

## Task 5: activation collection (hooks)

**Files:**
- Create: `src/utils/inspection/activations.py`
- Test: `tests/test_inspection_activations.py`

Captures, for every hooked module, output `tensor_stats`; and for any module whose output is a
3D `[B, N, D]` sequence, also per-token L2 norm (mean over batch) and per-dim mean-abs (top-k).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inspection_activations.py
import torch
import torch.nn as nn
from src.utils.inspection.activations import collect_activation_stats


class Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x):
        return self.b(self.a(x))


def test_collect_activation_stats_per_module_and_token():
    model = Seq()
    x = torch.randn(4, 36, 8)
    stats = collect_activation_stats(model, lambda: model(x),
                                     module_filter=lambda n, m: isinstance(m, nn.Linear))
    assert "a" in stats and "b" in stats
    assert "max_abs" in stats["a"]
    # 3D output -> per-token norms recorded with length N=36
    assert len(stats["a"]["per_token_norm"]) == 36
    assert "top_dims_by_mean_abs" in stats["a"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inspection_activations.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
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
                entry["top_dims_by_mean_abs"] = list(zip(idx.tolist(), vals.tolist()))
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_inspection_activations.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/inspection/activations.py tests/test_inspection_activations.py
git commit -m "feat(inspection): hook-based activation stats incl. per-token norm + top massive-activation dims"
```

---

## Task 6: attention entropy

**Files:**
- Create: `src/utils/inspection/attention.py`
- Test: `tests/test_inspection_attention.py`

`F.scaled_dot_product_attention` is fused and does not expose weights, so we recompute
`softmax(QKᵀ·scale + mask)` from a captured `SelfAttention` input, mirroring its forward
(q/k proj → qk_norm → RoPE → mask). Entropy is reported as a fraction of `log(n_keys)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inspection_attention.py
import torch
from src.models.components.llama_graph_transformer import Transformer, SelfAttention
from src.utils.inspection.attention import attention_entropy_from_input


def test_entropy_fraction_in_unit_range_and_uniform_is_high():
    torch.manual_seed(0)
    t = Transformer(hidden_dim=16, num_heads=2, ppf_hidden_dim=32, num_layers=1,
                    dropout=0.0, qk_norm=True)
    attn = t.blocks[0].attention_layer
    assert isinstance(attn, SelfAttention)
    x = torch.randn(2, 36, 16)
    # full (all-ones) mask over 36 tokens
    mask = torch.ones(36, 36, dtype=torch.bool)
    out = attention_entropy_from_input(attn, x, mask)
    assert 0.0 <= out["entropy_frac_mean"] <= 1.0
    assert len(out["entropy_frac_per_head"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inspection_attention.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/utils/inspection/attention.py
from __future__ import annotations

import math

import torch


@torch.no_grad()
def attention_entropy_from_input(attn_module, x: torch.Tensor, attn_mask: torch.Tensor) -> dict:
    """Recompute attention weights for one SelfAttention and report entropy as a fraction
    of log(n_keys). Mirrors SelfAttention.forward up to the softmax.

    x:         [B, N, D] input to the attention module (post pre-norm).
    attn_mask: [N, N] bool, True = keep (same convention as get_*_mask).
    """
    b, n = x.shape[0], x.shape[1]
    q = attn_module.q_proj(x).view(b, n, attn_module.n_head, -1).transpose(1, 2)
    k = attn_module.k_proj(x).view(b, n, attn_module.n_head, -1).transpose(1, 2)
    if getattr(attn_module, "qk_norm", False):
        q = attn_module.q_norm(q)
        k = attn_module.k_norm(k)
    if attn_module.rope:
        q = attn_module.rope.rotate_queries_or_keys(q)
        k = attn_module.rope.rotate_queries_or_keys(k)

    scale = 1.0 / math.sqrt(q.shape[-1])
    logits = (q @ k.transpose(-2, -1)) * scale                      # [B, H, N, N]
    logits = logits.masked_fill(~attn_mask, float("-inf"))
    w = torch.softmax(logits, dim=-1)                               # [B, H, N, N]

    n_keys = attn_mask.sum(dim=-1).clamp_min(2).float()             # [N]
    ent = -(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(-1)  # [B, H, N]
    norm_ent = ent / torch.log(n_keys).view(1, 1, n)
    return {
        "entropy_frac_mean": norm_ent.mean().item(),
        "entropy_frac_per_head": norm_ent.mean(dim=(0, 2)).tolist(),
        "frac_mass_on_token0": w[..., 0].mean().item(),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_inspection_attention.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/inspection/attention.py tests/test_inspection_attention.py
git commit -m "feat(inspection): attention entropy (recomputed from QK, since SDPA is fused)"
```

---

## Task 7: report assembly (flags, JSON, markdown, plots)

**Files:**
- Create: `src/utils/inspection/report.py`
- Test: `tests/test_inspection_report.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inspection_report.py
import json
from src.utils.inspection.report import build_flags, write_report


def test_build_flags_detects_anomalies():
    results = {
        "weights": {"decoder.norm": {"type": "RMSNorm", "gain_rms": 0.3, "section": "decoder"}},
        "activations": {"encoder.graph_transformer": {"max_abs": 260.0}},
        "latent": {"z_dim": 32, "active_dims": 5, "rank_ratio": 0.3},
        "reconstruction": {"per_channel_r2": [0.9, 0.1, 0.8], "worst_channels": [1]},
    }
    flags = build_flags(results)
    text = " ".join(flags)
    assert any("max_abs" in f for f in flags)
    assert any("gain" in f.lower() for f in flags)
    assert any("rank" in f.lower() or "active" in f.lower() for f in flags)


def test_write_report_creates_artifacts(tmp_path):
    results = {
        "meta": {"run": "test"},
        "weights": {}, "activations": {}, "attention": {},
        "latent": {"z_dim": 32, "active_dims": 30, "rank_ratio": 0.9},
        "reconstruction": {"per_channel_r2": [0.9], "per_channel_mse": [0.1],
                           "per_position_mse": [0.1], "worst_channels": [0],
                           "per_view_mse": None, "error_by_magnitude": []},
    }
    write_report(results, tmp_path)
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
    loaded = json.loads((tmp_path / "report.json").read_text())
    assert loaded["meta"]["run"] == "test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inspection_report.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/utils/inspection/report.py
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Flag thresholds (tune here).
MAX_ABS_FLAG = 50.0
GAIN_RMS_LO, GAIN_RMS_HI = 0.5, 2.0
RANK_RATIO_FLAG = 0.5
R2_FLAG = 0.3


def build_flags(results: dict) -> list[str]:
    flags: list[str] = []
    for name, w in results.get("weights", {}).items():
        if "gain_rms" in w and not (GAIN_RMS_LO <= w["gain_rms"] <= GAIN_RMS_HI):
            flags.append(f"[weights] {name}: gain_rms={w['gain_rms']:.3f} outside [{GAIN_RMS_LO},{GAIN_RMS_HI}]")
        if "rank_ratio" in w and w["rank_ratio"] < RANK_RATIO_FLAG:
            flags.append(f"[weights] {name}: low rank_ratio={w['rank_ratio']:.2f}")
    for name, a in results.get("activations", {}).items():
        if a.get("max_abs", 0) > MAX_ABS_FLAG:
            flags.append(f"[activations] {name}: max_abs={a['max_abs']:.1f} > {MAX_ABS_FLAG}")
    lat = results.get("latent", {})
    if lat:
        if lat.get("rank_ratio", 1.0) < RANK_RATIO_FLAG:
            flags.append(f"[latent] effective rank_ratio={lat['rank_ratio']:.2f} (bottleneck under-used)")
        if lat.get("active_dims", lat.get("z_dim", 0)) < 0.5 * lat.get("z_dim", 1):
            flags.append(f"[latent] only {lat['active_dims']}/{lat['z_dim']} active dims")
    rec = results.get("reconstruction", {})
    bad = [i for i, r2 in enumerate(rec.get("per_channel_r2", [])) if r2 < R2_FLAG]
    if bad:
        flags.append(f"[reconstruction] {len(bad)} channels with R2<{R2_FLAG}: {bad[:15]}")
    return flags


def _save_plots(results: dict, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    rec = results.get("reconstruction", {})
    if rec.get("per_channel_r2"):
        fig, ax = plt.subplots()
        ax.plot(rec["per_channel_r2"])
        ax.set(title="Per-channel R²", xlabel="channel", ylabel="R²")
        fig.savefig(plot_dir / "per_channel_r2.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
    acts = results.get("activations", {})
    maxes = {k: v.get("max_abs", 0) for k, v in acts.items() if "block" in k.lower()}
    if maxes:
        fig, ax = plt.subplots()
        ax.bar(range(len(maxes)), list(maxes.values()))
        ax.set_xticks(range(len(maxes)))
        ax.set_xticklabels(list(maxes.keys()), rotation=90, fontsize=6)
        ax.set(title="Per-block max_abs")
        fig.savefig(plot_dir / "per_block_max_abs.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


def _render_md(results: dict, flags: list[str]) -> str:
    lines = ["# Model inspection report", ""]
    meta = results.get("meta", {})
    if meta:
        lines += ["## Meta", "```json", json.dumps(meta, indent=2), "```", ""]
    lines += ["## Flags", ""]
    lines += [f"- {f}" for f in flags] if flags else ["- (none)"]
    lat = results.get("latent", {})
    if lat:
        lines += ["", "## Latent",
                  f"- active_dims: {lat.get('active_dims')}/{lat.get('z_dim')}",
                  f"- effective rank_ratio: {lat.get('rank_ratio'):.3f}"]
    rec = results.get("reconstruction", {})
    if rec:
        lines += ["", "## Reconstruction",
                  f"- overall_mse: {rec.get('overall_mse')}",
                  f"- worst channels: {rec.get('worst_channels')}"]
    return "\n".join(lines) + "\n"


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def write_report(results: dict, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flags = build_flags(results)
    results = dict(results)
    results["flags"] = flags
    (out_dir / "report.json").write_text(json.dumps(_to_jsonable(results), indent=2))
    (out_dir / "report.md").write_text(_render_md(results, flags))
    _save_plots(results, out_dir / "plots")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_inspection_report.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/inspection/report.py tests/test_inspection_report.py
git commit -m "feat(inspection): report assembly — flags, report.json, report.md, key plots"
```

---

## Task 8: wire `--inspect` into diagnose_model.py and run on szary

**Files:**
- Modify: `scripts/diagnose_model.py` (argparse in `main` ~`scripts/diagnose_model.py:449`, and after `load_model_and_data`)
- Modify: `src/utils/inspection/__init__.py` (export public API)

This task has no unit test (it's orchestration of already-tested units against a real
checkpoint, which requires szary). It is verified by the end-to-end run at the end.

- [ ] **Step 1: Export the public API**

```python
# src/utils/inspection/__init__.py  (append)
from src.utils.inspection.activations import collect_activation_stats
from src.utils.inspection.attention import attention_entropy_from_input
from src.utils.inspection.latent import latent_diagnostics
from src.utils.inspection.reconstruction import reconstruction_diagnostics
from src.utils.inspection.report import write_report
from src.utils.inspection.weights import weight_diagnostics

__all__ = [
    "collect_activation_stats", "attention_entropy_from_input", "latent_diagnostics",
    "reconstruction_diagnostics", "write_report", "weight_diagnostics",
]
```

- [ ] **Step 2: Add an `inspect_model` orchestrator to diagnose_model.py**

Add near the other diagnostics (after `run_diagnostics`). `model` is the `GraphAE`
(`pl_module.graph_ae`); it exposes `.encode(graph)`, `.decode(...)`, `.encoder`, `.decoder`,
`.node_bottleneck`. `batch` is a `DenseGraphBatch` already on-device.

```python
import torch.nn as nn
from src.models.components.llama_graph_transformer import (
    SelfAttention, TransformerBlock, get_neighborhood_mask, get_full_mask,
)
from src.utils.inspection import (
    collect_activation_stats, attention_entropy_from_input, latent_diagnostics,
    reconstruction_diagnostics, weight_diagnostics, write_report,
)


def inspect_model(model, batch, out_dir, meta: dict) -> dict:
    model.eval()
    results = {"meta": meta}

    # A. Weights (no data)
    results["weights"] = weight_diagnostics(model)

    # B. Activations: hook transformer blocks + the two transformers + node_fc_out + projection_in
    def act_filter(name, m):
        return isinstance(m, TransformerBlock) or name.endswith("graph_transformer") \
            or isinstance(m, nn.Linear) and ("fc_out" in name or "projection_in" in name)
    with torch.no_grad():
        results["activations"] = collect_activation_stats(
            model, lambda: model.encode(batch), act_filter
        )

    # C. Attention entropy per SelfAttention. Capture each module input via pre-hook,
    #    then recompute with the correct mask (encoder=neighborhood, decoder=full).
    captured: dict = {}
    handles = []
    for name, m in model.named_modules():
        if isinstance(m, SelfAttention):
            def pre_hook(_mod, args, _name=name):
                captured[_name] = args[0].detach()
            handles.append(m.register_forward_pre_hook(pre_hook))
    with torch.no_grad():
        z_nodes, z_global, node_features, mu, logvar = model.encode(batch)
        graph_pred = model.decode(z_nodes, z_global, batch.mask)
    for h in handles:
        h.remove()
    attn = {}
    for name, m in model.named_modules():
        if isinstance(m, SelfAttention) and name in captured:
            x = captured[name]
            n = x.shape[1]
            is_encoder = name.startswith("encoder")
            mask = get_neighborhood_mask(n, is_encoder, x.device) if is_encoder \
                else get_full_mask(batch.mask, False, x.device)
            attn[name] = attention_entropy_from_input(m, x, mask)
    results["attention"] = attn

    # D. Latent
    results["latent"] = latent_diagnostics(z_nodes)

    # E. Reconstruction
    num_views = getattr(model.permuter, "num_permutations", 8)
    results["reconstruction"] = reconstruction_diagnostics(
        graph_pred.node_features, batch.node_features, num_views
    )

    write_report(results, out_dir)
    return results
```

- [ ] **Step 3: Add CLI flags and call it in `main`**

In `main` (`scripts/diagnose_model.py:449`), after the argparser and after
`load_model_and_data(...)` returns `(model, batch, tau, ...)`, add:

```python
    parser.add_argument("--inspect", action="store_true",
                        help="Run full weight/activation/attention/latent/reconstruction inspection")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir for inspection artifacts (default logs/diagnostics/<ckpt-stem>)")
```

and after the model/batch are loaded:

```python
    if args.inspect:
        from pathlib import Path
        out_dir = args.out_dir or f"logs/diagnostics/{Path(args.ckpt).stem}"
        meta = {"ckpt": args.ckpt, "split": args.split, "tau": tau,
                "run": Path(args.ckpt).parent.name}
        inspect_model(model.graph_ae if hasattr(model, "graph_ae") else model, batch, out_dir, meta)
        print(f"[inspect] wrote artifacts to {out_dir}")
```

(If `load_model_and_data` returns the LightningModule, `model.graph_ae` is the `GraphAE`; if a batch
generator is returned, take the first batch. Match the existing return signature at
`scripts/diagnose_model.py:378`.)

- [ ] **Step 4: Run the full unit suite (local, CPU)**

Run: `uv run pytest tests/test_inspection_*.py -v`
Expected: PASS (all inspection tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/diagnose_model.py src/utils/inspection/__init__.py
git commit -m "feat(inspection): --inspect flag on diagnose_model.py wiring all sections + report"
```

- [ ] **Step 6: End-to-end run on szary (manual verification)**

On szary, with a finished-run checkpoint (e.g. under `logs/.../checkpoints/last.ckpt`):

```bash
uv run python scripts/diagnose_model.py --ckpt <path/to/last.ckpt> --paths szary --split val --inspect
```
Expected: writes `logs/diagnostics/<ckpt-stem>/report.json`, `report.md`, `plots/*.png`.
Verify `report.md` Flags section is populated and `report.json` loads with `json.load`.

---

## Self-review notes

- **Spec coverage:** A weights → Task 2; B activations (per-block/per-token/per-dim) → Task 5 + Task 8 block hooks; C attention entropy → Task 6 + Task 8; D latent → Task 3; E reconstruction → Task 4; artifacts (json/md/flags/plots) → Task 7. All spec sections mapped.
- **Type consistency:** `weight_diagnostics`, `latent_diagnostics`, `reconstruction_diagnostics`, `collect_activation_stats`, `attention_entropy_from_input`, `write_report` names are identical across tasks and `__init__` exports.
- **Known integration risk (Task 8):** the exact return signature of `load_model_and_data` (`scripts/diagnose_model.py:378`) and whether `encode` returns 5 values must be confirmed against the file during implementation; the orchestrator assumes `GraphAE.encode → (z_nodes, z_global, node_features, mu, logvar)` and `GraphAE.decode(z_nodes, z_global, mask)`, which matches `src/models/components/modules.py`. Adjust the unpacking if the script wraps the model differently.
