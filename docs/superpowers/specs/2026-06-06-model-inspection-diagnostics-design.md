# Model Inspection Diagnostics — design

**Date:** 2026-06-06
**Status:** approved (pending spec review)
**Scope:** extend `scripts/diagnose_model.py` with offline, checkpoint-level inspection of
weights, activations, attention, the latent bottleneck, and reconstruction error.

## Context & motivation

`actmon` (the `ActivationMonitorCallback`) only logs *aggregate* per-transformer std/max during
training. It cannot say **which block / token / dim** is responsible. Concretely, run `cuogzab1`
(latent-norm + encoder `final_norm` removed) showed encoder `max_abs` ≈ 100–260 with no way to
localize it. We need a per-layer / per-block / per-token / per-dim view, plus weight-level and
reconstruction-level breakdowns, to find **where the model struggles**.

The **primary consumer of the output is an LLM agent analyzing it later**, so the main artifact is
a machine-readable `report.json`. Human-readable summary and plots are secondary.

`scripts/diagnose_model.py` already loads a checkpoint + data (`load_model_and_data`), runs val
batches, registers forward hooks (`make_activation_hook`), and prints a report
(`run_diagnostics` → `print_report`). We extend it; we do not start over.

## Goals / success criteria

- A single command on szary produces `report.json` + `report.md` + `plots/*.png` for a checkpoint.
- `report.json` is self-describing (every metric keyed by module/section), loadable without the model.
- `report.md` includes a **Flags** section auto-derived from heuristics (anomalies surfaced, not buried).
- Covers sections A–E below. Reuses existing loading + hook machinery. Runs headless, eval mode, `no_grad`.

## Non-goals (YAGNI)

- No training-time logging (offline only; live grad norms already covered by `permutation_diagnostics`).
- No backward/gradient inspection.
- No interactive UI / dashboard.
- No new model code; inspection only reads the model + one forward pass.

## Output artifacts

Written to `logs/diagnostics/<ckpt-stem>/` (override with `--out-dir`):

- **`report.json`** — nested dict: `{meta, weights, activations, attention, latent, reconstruction}`.
  `meta` = ckpt path, run name, config snapshot (z_dim, qk_norm, use_film/use_rope, num_layers, grid),
  split, num_batches, tau, timestamp.
- **`report.md`** — headline table + `## Flags` (heuristic anomalies, see below) + per-section summaries.
- **`plots/*.png`** — key figures only: per-module weight & activation histograms, residual-norm-by-block
  bar, attention-entropy-by-layer, per-channel recon error, z singular-value spectrum.

### Flag heuristics (examples, tunable constants in one place)
- activation `max_abs` > 50 on any module → flag (localize block/token).
- RMSNorm/LayerNorm gain RMS < 0.5 or > 2.0 → flag.
- Linear effective-rank / dim < 0.5 → flag (low-rank collapse).
- latent active-dim count < 0.5·z_dim, or z effective-rank < 0.5·z_dim → flag (bottleneck collapse).
- per-channel recon R² < 0.3 → list channels; per-position error > 1.5× median → list positions.
- attention entropy < 0.2·log(n_keys) (over-peaked) or > 0.95·log(n_keys) (uniform) → flag.

## Diagnostic sections

### A. Weights (no data; iterate `model.named_parameters()` / `named_modules()`)
Per weight tensor (grouped by section: encoder / decoder / bottleneck / permuter):
- L2 norm; mean, std, min, max, |max|; percentiles p0.1/1/50/99/99.9; kurtosis; fraction |w|<1e-6 (dead).
- For each `nn.Linear` weight: **spectral norm** (top singular value) and **effective rank** =
  participation ratio `(Σσ²)² / Σσ⁴` over singular values (normalized by `min(shape)`).
- For each RMSNorm/LayerNorm: gain (`weight`) vector RMS, min, max (LayerNorm `bias` too if present).

### B. Activations (one forward pass; reuse `make_activation_hook`, hook every named submodule of interest)
- Per-module output: mean, std, max_abs, percentiles, kurtosis, fraction-saturated (|a|>3·std).
- **Per-block residual-stream norm**: hook each `TransformerBlock` output → mean token L2 norm per block,
  encoder and decoder separately (localizes growth across depth).
- **Per-token norm**: for encoder/decoder transformer output, L2 norm per sequence position (incl. CLS at
  pos 0 for encoder) → identifies sink/CLS token blow-up.
- **Per-dim magnitude**: mean |activation| per hidden dim (over batch×tokens) for the transformer outputs →
  top-k "massive activation" dims.

### C. Attention health (recompute weights — SDPA is fused and does not expose them)
For the inspection forward pass, hook each `SelfAttention` to capture post-(qk-norm, RoPE) Q,K and the mask,
then recompute `softmax(QKᵀ·scale + mask)`:
- per-layer, per-head **entropy** of the attention distribution (mean over query positions), reported as a
  fraction of `log(n_keys)`.
- mean attention mass placed on the CLS/sink token (encoder) and on self vs others.

### D. Latent / bottleneck (`z_nodes` from `encode`, before decode)
- per-dim variance over (batch×nodes) → **active-dim count** (var > 1% of max-dim var).
- per-node z L2-norm distribution (mean/std/min/max) → is the freed per-node magnitude used (std > 0)?
- **z effective rank**: participation ratio of eigenvalues of the `[z_dim×z_dim]` covariance of z (pooled
  over batch×nodes), normalized by z_dim → is the bottleneck genuinely used or collapsed?
- mean pairwise cosine similarity between the 36 nodes' z within a sample → spatial collapse check.

### E. Reconstruction — "where it struggles" (pred vs `graph_true.node_features`, [B,36,128])
- **per-channel** (128): MSE and R² (`1 − SS_res/SS_tot`) per feature dim → which PCA components fail
  (expect low-variance tail to be worst).
- **per-position** (36→6×6): mean error per node, reshaped to the grid → edges/corners vs center.
- per-D4-view error (8 views) → orientation consistency.
- error vs target magnitude: bin nodes by `‖target‖`, report mean error per bin → does it fail on
  high-intensity patches?

## Mechanics

- New `--inspect` flag on `diagnose_model.py` (default off; existing behavior unchanged). `--out-dir`
  override; `--num-batches` reused (accumulate stats across batches where meaningful, e.g. recon/latent).
- Add `weight_diagnostics(model)`, `attention_diagnostics(...)`, `latent_diagnostics(...)`,
  `reconstruction_diagnostics(...)`, and extend the activation collection in `run_diagnostics`.
- Add a `write_report(results, out_dir)` that dumps JSON, renders `report.md` (incl. Flags), and saves plots.
- Eval mode, `torch.no_grad()`, runs on szary via the existing `--paths szary` loader.
- Dependencies: matplotlib (already used in the repo) for plots; numpy/torch only otherwise.

## Assumptions

- Grid is square (`N = grid²`); reuse `math.isqrt` as `losses.py` does.
- Checkpoints live on szary (`/raid_encrypted/...`); the script is run there, not on the dev box (no GPU/data locally).
- One forward pass on a few batches is representative for distribution/outlier stats.
