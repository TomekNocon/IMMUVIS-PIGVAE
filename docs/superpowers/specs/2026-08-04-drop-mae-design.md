# Design — Drop-MAE (denoising) objective on z_global

**Date:** 2026-08-04
**Branch:** `imc-pigvae-contrastive-drop` (continues the contrastive line; drop-MAE replaces the drop-contrastive objective)
**Status:** approved design → implementation plan next

## Context & motivation

The drop-*contrastive* experiment failed and, crucially, told us *why*
(wiki `exp-2026-08-04-drop-contrastive-zglobal`, wandb `az9dkslj`):

- A drop-invariance NT-Xent on `z_global` regressed downstream transfer on all 4 cords targets vs
  FiLM (DX.name 0.903 vs 0.928) and sat at/below the raw baseline.
- `contrastive_loss` collapsed to ~0 by epoch 12. **Mechanism:** `z_global` is a pooled global
  statistic (CLS + stats-correction over 256 nodes) and is *already* near-invariant to a 20% node
  drop, so the objective had nothing to teach — it was vacuous.

**The pivot.** Invariance-to-drop is *free*, but *reconstructing the full crop from a corrupted view*
is not: it forces `z_global` to **carry** the global structure needed to fill the missing region.
That is a real, non-vanishing gradient in exactly the thesis-relevant direction (a global code that
encodes global structure). This is the MAE / denoising-autoencoder branch that was deferred on the
original contrastive spec (`2026-08-01-contrastive-drop-design.md`); the negative result is the
empirical justification to promote it.

## Approach (chosen: B — pure denoising/MAE)

Considered three structures:
- **A (add-on):** clean forward (recon+KL, unchanged) + a second encode(dropped)→decode→MSE branch.
  2 enc + 2 dec passes/step. Rejected: heaviest, and the extra clean branch is redundant once the
  recon target is the clean crop anyway.
- **B (pure denoising/MAE) — CHOSEN:** a single forward whose *input is the corrupted (block-dropped)
  view* and whose *target is the clean crop*. 1 enc + 1 dec passes/step — same cost as the FiLM
  baseline. The reconstruction task itself becomes "rebuild the whole crop from a corrupted view."
- **C (masked-decoder MAE):** as B but held-out nodes are mask-tokens on the *decoder* side, loss only
  on them. The "true" MAE, forces global-code reliance hardest, but needs a decoder mask-token path.
  Held as the fallback if B's signal proves too easy (local inpainting).

B is chosen because it captures most of C's benefit with almost no new code, is cheap enough to also
run the confound control, and — because the loss target is the *clean* crop with the *unchanged* loss
function — keeps val_mse directly comparable to the FiLM baseline. The **block** drop (below) is what
prevents B from degenerating into trivial local inpainting.

## Architecture: training step

Single forward, batch **B = 64** single-view (same as the drop-contrastive run):

```
x_dropped = block_drop(x_clean, mask, frac=0.5, generator)   # encoder-side corruption, TRAIN ONLY
z_nodes, z_global, ... = encode(x_dropped, sample=True)       # dropped nodes absent from the encoder
x_hat = decode(z_nodes, z_global)                             # reconstruct all 256 positions
loss  = recon(x_hat, x_clean) + kld_scale · kld_alpha · kld
```

- **Target = the clean crop `x_clean`.** The reconstruction loss is the **exact same
  Huber+Cosine+Gradient function** used by the FiLM baseline — not forked. `val_mse` therefore means
  "clean reconstruction" and is comparable to `81zasevr` / the FiLM-only control.
- **Loss over all 256 nodes** (not held-out-only). Kept nodes are in the input and reconstruct to the
  floor, so the informative gradient concentrates on the masked block automatically; keeping the loss
  over all nodes is what preserves the identical-loss comparability. (Held-out-only weighting remains a
  future knob if the signal looks too weak.)
- **No projection head, no contrastive term, no second pass.** The contrastive machinery
  (`drop_views`, `ProjectionHead`, `ContrastiveLoss`, the `_drop_contrastive` gate) stays in the repo,
  gated off (`contrastive_loss_scale: 0.0`).
- `z_global` is deterministic given the (dropped) input (independent of VAE node-latent sampling); the
  only training stochasticity in the input is the block mask.

**Encoder mask-awareness is already built and load-bearing here.** The drop-contrastive work made the
encoder attention "neighborhood AND padding" mask-aware and `NodeStatsProjection` masked (mean/var/max),
with the self-key `| eye` fix and `nan_to_num` guard — all exact no-ops when the mask is all-True. That
is precisely what makes a block-dropped input *visible* to `z_global` while leaving the clean-input
(all-True) path bit-identical. No re-work needed; drop-MAE consumes it.

## Corruption: block-drop (train-only)

`block_drop(batch, frac=0.5, generator=None) -> DenseGraphBatch` — a new pure function beside
`drop_views` in `src/models/components/contrastive.py`.

- Masks a **single contiguous rectangular block** covering **~50%** of the 16×16 grid (≈128 of 256
  nodes), at a **random position** (random block height/width with area ≈ frac, random top-left,
  clamped to the grid). Contiguity is the point: removing a node *and its neighbors* denies the
  decoder a local-inpainting shortcut, forcing reliance on `z_global`.
- Combines with the existing padding mask: `new_mask = mask AND ~block`. Guarantees ≥1 kept node
  (trivially true at 50%). `node_features` unchanged (only the mask changes), matching the `drop_views`
  contract. Device/seed-deterministic.
- **Grid geometry:** the function needs the grid side (16). Derive from `sqrt(num_nodes)` or thread the
  configured `grid_size`; the block is indexed in 2-D grid coordinates then flattened to the node axis
  (row-major, matching the grid node ordering used elsewhere).
- **Training only.** Validation and the downstream **encode use the clean full grid** — mask all-True,
  the no-op path — so the encoder is evaluated and transferred exactly as before.

## Recipe & the confound control

Two runs:

1. **Drop-MAE run** — balanced `s0.01` + FiLM (`use_film`, `film_bound`, `film_weight_decay 0.05`),
   grid16, `node_z_dim 32`, KL `scale 0.01` + `fb 0.0`, α 0→1 ep5–25, 40 ep, **batch 64 single-view**
   (`num_aug_per_sample: 1`). New config `configs/experiment/vae16_fb0p0_film_mae.yaml`, cloned from
   `vae16_fb0p0_film_drop.yaml` with the contrastive scale off and a new `mae_block_frac: 0.5` (+
   `mae_enabled: true`) knob wired into the active module's `training_step`.
2. **Batch-64 FiLM-only control** — clean, no drop, otherwise identical to run 1. This is the missing
   like-for-like baseline that retires the batch-64-vs-8 reconstruction confound (the batch-8 FiLM run
   `81zasevr` reached val_mse 0.0254 but at ~6× more optimizer steps). Config
   `configs/experiment/vae16_fb0p0_film_b64.yaml` (or reuse `vae16_fb0p0_film` with a batch-64 +
   single-view override).

## Components (isolated, unit-testable)

- `block_drop(batch, frac, generator)` — pure function, as specified above. Tests: block is contiguous
  in grid coords; masked fraction ≈ frac; ≥1 kept node; `node_features` untouched; all-True in → a
  smaller-but-valid mask out; device/seed determinism; a **visibility guard** (a block-dropped view
  yields a *different* `z_global` than clean — the analogue of the drop-contrastive degeneracy guard,
  proving the corruption is actually seen).
- Active LightningModule (`pigvae_auto_module.py`, `PLGraphAE`) — `training_step` gains an
  `mae_enabled`/`mae_block_frac` gate: when on, replace the clean encoder input with
  `block_drop(...)` while keeping the clean crop as the recon target; `validation_step` unchanged
  (clean). Mirror the `_drop_contrastive` gating style so the feature is default-off and the FiLM
  baselines are untouched.
- Config `vae16_fb0p0_film_mae.yaml` + the batch-64 FiLM-only control config.

## Success criterion

The retrained `z_global` **beats FiLM-`z_global`** on the existing 4-way downstream sweep
(bar: DX.name 0.928 / Grade 0.792 / Stage 0.660), with reconstruction judged against the **batch-64
FiLM-only control** (not the batch-8 FiLM run). Deciding metric is downstream transfer, not recon. A
non-vacuous training signal is a necessary precondition: unlike the drop-contrastive run, the MAE recon
loss on the masked block must carry a real, non-collapsing gradient (sanity-check the train-loss curve
is not at the clean floor).

## Testing

- Unit tests for `block_drop` (above), added to `tests/test_contrastive_drop.py` or a sibling
  `tests/test_block_drop.py`.
- Reuse the existing gate-on/gate-off tests pattern: a test that `training_step` with `mae_enabled`
  corrupts the encoder input but not the target, and with it off is bit-identical to the FiLM baseline.
- CPU shape-probe gate before spending a GPU slot (the established practice): confirm the drop-MAE
  config yields batch 64 → 64 graphs and a finite loss on a tiny CPU run.

## Constraints & notes

- SLURM: single node `szary` (RTX 5000 Ada 32 GB), ~2 concurrent jobs; submit via
  `sbatch --time=HH:MM:SS --export=ALL,EXPERIMENT=<name> scripts/ablation_slurm.sh` (login node has no
  raid mount). 1 enc + 1 dec/step ⇒ expect ~2–2.5 h like the drop run, not the 3-pass budget.
- Permuter dormant throughout; this remains substrate transfer, not invariance.
- Out of scope: Approach C (decoder mask-tokens), held-out-only loss weighting, jitter/bag views,
  more PCA components. Each is a separate follow-up if drop-MAE proves out.
