# Design — `z_global` stats-vs-CLS decomposition (diagnostic)

**Date:** 2026-08-07
**Branch:** `exp/zglobal-stats-decomp` (off `imc-pigvae-contrastive-drop` @ b714891)
**Status:** approved design → implementation plan next

## Context & motivation

Both drop-family experiments failed to beat the FiLM baseline on downstream transfer:
drop-contrastive (`az9dkslj`) was **vacuous** (loss→0), drop-MAE (`u7v7at4p`) was **non-vacuous but
neutral** (best `mae_last` ties FiLM on DX/Grade, below on Stage/Relapse). A simpler explanation than
"the objective is wrong" is emerging from the data: **`z_global` may be dominated by a pooled input
statistic**, capping how much *any* learned objective can add.

The free evidence already in hand: the `raw` baseline — a plain spatial mean-pool of the 768-d patch,
**no encoder at all** — reaches DX **0.917**, while the fully-trained `z_global` reaches **0.928**. The
entire learned encoder adds only **~+0.011 AUC** over a pooled statistic of the input.

This diagnostic asks the sharper question directly: **within the trained `z_global`, does the label
signal live in the learned CLS token or in the pooled mean/var/max stats correction?**

## What `z_global` is (today)

In `GraphEncoder.forward` (`src/models/components/modules.py:383`):

```
graph_emb = CLS_token_output                          # learned, from the transformer
graph_emb = graph_emb + stats_correction(nodes, mask) # residual: proj([mean, var, max]) over nodes
# structural_correction is None for these checkpoints (use_hadamard / use_mlp_edges both False)
z_global  = LayerNorm(graph_emb)                      # GraphAE.encode, modules.py:241
```

`stats_correction` is a `NodeStatsProjection` (`modules.py:18`): a learned linear projection of the
concatenated masked mean / var / max of the **post-transformer** node features. It is added as a
**residual** to the CLS token, then the sum is LayerNorm'd.

## Scope

**Decomposition only — attribution, no training.** Run on the *existing* checkpoints. Escalation to a
no-stats retrain (the causal "does the component help the architecture" answer) is **out of scope**
here; it is a deferred follow-up, warranted only if the decomposition is ambiguous.

This is an **attribution**, not a counterfactual: because the model was trained *with* the stats
correction, the CLS learned to be complementary to it. The decomposition tells us where the signal
*ended up* in the trained code, not whether removing the component would hurt. That caveat is recorded
in the interpretation.

## The two new views

Add two **read-only sibling views** of `z_global`, each LayerNorm'd on its own so all three sit on the
same scale the ABMIL probe consumes:

- **`zglobal_cls`**   = `LayerNorm(CLS)`                       — the learned token alone
- **`zglobal_stats`** = `LayerNorm(stats_correction(nodes))`  — the pooled statistic alone
- **`zglobal`** (existing) = `LayerNorm(CLS + stats)`         — full

## Components (isolated, unit-testable)

1. **`GraphEncoder.forward(..., return_parts: bool = False)`** — `src/models/components/modules.py`.
   When `return_parts=True`, compute `cls` (the transformer readout, pre-stats) and
   `stats = stats_correction(node_features, mask)` separately and return
   `(graph_emb, node_features, cls, stats)` where `graph_emb == cls + stats`. When `False` (default),
   behavior is **bit-identical** to today: returns `(graph_emb, node_features)`. Assert
   `structural_correction is None` when `return_parts=True` (holds for these checkpoints; makes the
   two-way split exact — fail loudly rather than silently drop a third term).

2. **`GraphAE.encode_zglobal_parts(graph) -> dict[str, Tensor]`** — `src/models/components/modules.py`.
   Runs the encoder with `return_parts=True`, applies `LayerNorm` (over the last dim, matching
   `encode()`) to each of `cls`, `stats`, and `cls+stats`, and returns
   `{"full": ..., "cls": ..., "stats": ...}`, each `[B, D]`. The trained `encode()` path is untouched.

3. **`encode.py`** — `src/downstream/encode.py`.
   - `encode_patches`: when `feature_source in {"zglobal_cls", "zglobal_stats"}`, obtain the view via
     `gae.encode_zglobal_parts(batch)["cls" | "stats"]` (deterministic, `sample=False` semantics — the
     parts are functions of `mu`/CLS, no reparameterisation noise).
   - `source_subdir`: map `"zglobal_cls" -> "zglobal_cls"`, `"zglobal_stats" -> "zglobal_stats"`.
   - Existing sources (`raw`, `zglobal`, `node`) unchanged.

## Testing (TDD)

On a tiny real `_build_pl_module` instance (not mocks), following `tests/test_contrastive_drop.py`:

- **Exactness:** with `return_parts=True`, `cls + stats` equals the `graph_emb` returned by the default
  path, to floating-point tolerance (same input, same seed).
- **No-regression:** `LayerNorm(full)` from `encode_zglobal_parts` is **bit-identical** to the
  `z_global` returned by the untouched `encode()` on the same input.
- **Clean-mask sanity:** all-True mask → each view has shape `[B, 512]` and is finite.
- **Structural guard:** `return_parts=True` raises if `structural_correction is not None`.

## Run (no training)

Sweep the two new sources on **`mae_last`** (`u7v7at4p` → `last-v1.ckpt`, `mae_enabled=True`) and
**`ctl_last`** (`x4i39e0z` → `last.ckpt`, `mae_enabled=False`), both under
`/raid_encrypted/.../logs/train/runs/2026-08-06_21-22-30/checkpoints`. `raw` and `zglobal` (full) are
already computed for both tags from the prior sweep — reuse them. 512-d sources ⇒ no OOM (unlike the
8192-d `node_flatten` that OOM'd). Collect one table per checkpoint:
**raw / stats_only / cls_only / full** × {DX.name, Grade, Stage, Relapse}, AUC.

## Success criterion (what the diagnostic decides)

Not "beat the bar" — this is a **decision gate**:

- **stats_only ≈ full** (within ~0.01–0.02 CV noise) **and cls_only ≲ stats_only** ⇒ pooled statistics
  carry the transfer signal ⇒ objective-shaping (MAE ②, SupCon ③) is capped; the real lever is richer
  inputs (more PCA components) or the node/permuter path.
- **cls_only ≫ stats_only** ⇒ the learned CLS carries the signal ⇒ shaping it (② / ③) is worth pursuing.

Either outcome informs whether ② and ③ are worth their GPU slots — that is the point of running it first.

## Constraints & notes

- SLURM: single node `szary`, ~2 concurrent jobs, submit via sbatch (login node has no raid). Encode +
  10-fold CV only; no training.
- Attribution ≠ counterfactual (see Scope). The clean causal answer is the deferred no-stats retrain.
- Permuter dormant throughout.
- Out of scope: the no-stats retrain; any change to the trained forward path; new PCA components.
