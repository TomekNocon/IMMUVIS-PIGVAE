# Design — No-stats retrain (causal disentangler for the z_global attribution)

**Date:** 2026-08-07
**Branch:** `exp/nostats-retrain` (off `imc-pigvae-contrastive-drop` @ b714891)
**Status:** approved design → implementation plan next

## Context & motivation

The decomposition diagnostic (workstream ①, `exp/zglobal-stats-decomp`) showed that in the trained
models, `z_global`'s downstream signal reads out of the **pooled mean/var/max stats correction**, not the
learned **CLS token**: on both `mae_last` and `ctl_last`, `stats_only ≈ full` and `cls_only < stats_only`
on almost every target (e.g. control DX.name: stats 0.920 = full 0.920, cls 0.912).

But that is **attribution on a model trained *with* stats**, and it cannot distinguish two explanations:

- **H1 — the CLS is intrinsically weak:** attention-pooling into one token is a worse global summary than
  explicit moments. → shaping the CLS (drop-MAE, contrastive) is capped.
- **H2 — the CLS was crowded out:** the stats residual is a cheap always-available path, so during
  training the model offloads the global-summary job onto stats and the CLS specialises elsewhere; alone
  it looks weak only because it was never forced to be the sole global carrier. → removing stats could
  unlock the CLS, and shaping it becomes worthwhile.

This experiment is the **causal disentangler**: retrain the current-best recipe with the stats
correction **disabled**, forcing the CLS to be the only global path, and measure whether the CLS-only
`z_global` rises to the stats-with level.

## Scope — change ONE variable

Retrain **only** the plain FiLM control recipe (`vae16_fb0p0_film_b64`) with `stats_correction` off.
**Not** the drop or MAE variants — adding a second change would confound the result. The drop/MAE
objectives on a no-stats substrate are a *conditional follow-up*, warranted only if this run shows the
CLS can carry the signal.

## The gate

`GraphEncoder` currently builds `self.stats_correction = NodeStatsProjection(...)` unconditionally
(`modules.py:304`) and adds it in `forward` (`graph_emb = graph_emb + self.stats_correction(...)`,
`modules.py:~383`). Add a config flag, mirroring the existing `use_hadamard` pattern
(`getattr(hparams, "use_hadamard", False)`):

- `use_stats_correction: bool` — read via `getattr(hparams, "use_stats_correction", True)` in
  `GraphEncoder.__init__`. When `True` (default), build `NodeStatsProjection` and add it — **bit-identical
  to today**, and existing checkpoints (trained with stats) load unchanged. When `False`, set
  `self.stats_correction = None` and skip the add in `forward` → `graph_emb` is the pure CLS readout.

## Recipe

New config `configs/experiment/vae16_fb0p0_film_nostats.yaml` — a clone of `vae16_fb0p0_film_b64.yaml`
(VAE, FiLM, balanced KL `s0.01`, grid16, `node_z_dim 32`, batch 64, 40 ep) with a single change:
`model.graph_ae.hparams.encoder.use_stats_correction: false`, plus a distinct wandb name/tags
(`nostats`). One training run.

## Success criterion (what this decides)

Compared against the on-record `ctl_last` (stats-trained) numbers — stats_only ≈ 0.920 DX / 0.781 Grade /
0.681 Relapse / 0.625 Stage; cls_only (crowded) ≈ 0.912 / 0.769 / 0.653 / 0.619:

- **No-stats CLS-`z_global` rises to ~the stats-with level** (≈0.92 DX) ⇒ **H2**: the CLS was crowded out
  → shaping it (drop-MAE ②, SupCon ③) is back in play, on a no-stats substrate.
- **No-stats CLS-`z_global` stays near the crowded ~0.91 / cannot reach the stats level** ⇒ **H1**: the
  pooled moment is genuinely the better global summary → CLS-shaping is capped → pivot to input/pooling
  levers (more PCA components).

Deciding metric is downstream transfer on the same 4-way sweep; recon health is a sanity check only.

## Testing

- Unit (CPU, real `_build_pl_module`): default builds `stats_correction` (not None, bit-identical
  default); `vae16_fb0p0_film_nostats` config → `encoder.stats_correction is None`; `forward` with stats
  disabled runs finite `[B, 512]` and differs from the stats-added output (with a non-zero stats weight),
  proving the add is genuinely gated.
- CPU fast-dev-run probe (szary, before the full run): the nostats config composes, batch 64, one
  train+val step runs finite. Mirror the drop-MAE probe harness (rootutils + multiply/divide resolvers +
  `wandb.init(mode="disabled")`).

## Recipe & ops

- Train `vae16_fb0p0_film_nostats` (40 ep, batch 64, ~1.75h) via sbatch on szary.
- Encode the resulting checkpoint's `z_global` (which is now pure CLS) and run the 4-way downstream sweep;
  compare `zglobal` against the `ctl_last` cls_only / stats_only / full numbers.

## Constraints & notes

- Default `use_stats_correction=True` keeps every existing config and checkpoint bit-identical.
- SLURM: single node szary, ~2 concurrent, submit via sbatch (login node has no raid).
- Permuter dormant. Out of scope: the drop/MAE-on-no-stats follow-up (conditional on this result), more
  PCA components.
