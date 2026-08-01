# Design — Drop-invariance contrastive objective on z_global

**Date:** 2026-08-01
**Branch:** `imc-pigvae-contrastive-drop` (off `imc-pigvae-film-multi`)
**Status:** approved design → implementation plan next

## Context & motivation

The downstream transfer eval established:
- Without FiLM, `z_global` (the 512-d CLS) gets zero reconstruction gradient and does not beat a
  raw-768 mean-pool baseline.
- With FiLM (route `z_global` into the decoder), `z_global` beats raw on most cords targets
  (balanced `s0.01`+FiLM best: DX.name 0.928 / Grade 0.792 / Stage 0.660). See wiki
  `exp-2026-08-01-film-zglobal-transfer`.

FiLM adds a *reconstruction* objective on the CLS. This branch adds a *discriminability* objective:
a SimCLR-style contrastive loss that makes `z_global` invariant to **node dropout** (a random subset
of tissue missing). This is the first of three planned augmentation families (drop → jitter → bag);
drop is first because its positives are two on-the-fly dropped views of the *same crop*, needing **no
P-K sampler and no image-id dataloader change** (those are only for the "bag" view).

Permutation views are NOT usable positives: the encoder is permutation-invariant by construction, so
permuted copies map to the *same* `z_global` (zero gradient). Dropping nodes is *visible* to the
encoder, so it produces genuinely different `z_global` — a real learning signal.

## Success criterion

The retrained `z_global` **beats FiLM-`z_global`** on the existing 4-way downstream sweep
(bar: DX.name 0.928 / Grade 0.792 / Stage 0.660), with **no reconstruction regression** (val_mse
in line with the FiLM run `81zasevr`). Deciding metric is downstream transfer, not recon.

## Architecture: training step (Approach A)

Per training step, batch size **B = 96** (chosen by GPU memory probe — see Constraints):

1. **Clean forward (1× per crop):** `encode(clean, sample=True)` → `decode` → reconstruction loss
   (Huber+Cosine+Gradient) + KLD. **Identical to the FiLM run — unchanged.** Decoder runs here only.
2. **Drop views (2× per crop):** `encode(drop_view, sample=True)` → `z_global` → projection head →
   L2-normalize → stack `[view_a(all B); view_b(all B)]` → NT-Xent. **Encoder-only, no decoder.**
3. **Total loss:** `recon + kld_scale·kld_alpha·kld + λ·contrastive`.

Encoder passes/step = 3B; decoder passes/step = B. Drop views are generated **in the training step,
on-GPU**, from the clean batch — the dataloader is untouched.

`z_global = F.layer_norm(graph_emb)` is derived from the encoder CLS/summary node and is independent
of VAE node-latent sampling, so it is deterministic given the (dropped) input; the only stochasticity
in a drop view is the node mask.

## Components (isolated, unit-testable)

### `drop_views(batch, p=0.20, n=2)` — pure function
Given a `DenseGraphBatch` (node_features `[B,256,128]`, mask `[B,256]` bool) returns `n` new batches,
each with a fresh random `p`-fraction of nodes masked out (mask AND ~drop), `node_features` shared
(unchanged), guaranteeing ≥1 kept node per row. No model state; deterministic under a seed.

### `ProjectionHead` — `nn.Module`
2-layer MLP `512 → 512 → 128` (Linear, ReLU, Linear). Applied to `z_global` **only for the contrastive
loss**. Discarded at inference: the encode/downstream pipeline reads raw `z_global`, so no encode-side
change and no downstream-pipeline change.

### `ContrastiveLoss` — existing (`losses.py:478`)
NT-Xent already implemented and correct (groups views per image, pos/neg `logsumexp`). Instantiate with
`temperature = 0.2`, view count = 2. Un-comment the `Critic` hook (`model.py:32,71,84`) and wire it.

### Wiring
- `Critic.forward` gains a `contrastive` term added to the loss dict, scaled by λ.
- The active LightningModule's `training_step` calls `drop_views` + `ProjectionHead` and passes the
  projected features to the critic. (Identify the active module — `pigvae_auto_module.py` vs manual —
  during planning; wire only the one `src/train.py` uses.)

## Hyperparameters

| Param | Value | Notes |
|---|---|---|
| drop fraction `p` | 0.20 (fixed) | random subset per view; sweep 0.1/0.2/0.3 later |
| temperature | 0.2 | NT-Xent |
| batch size `B` | 96 | 190 negatives/anchor; from memory probe |
| `λ` (contrastive_loss_scale) | ~0.05, short warmup | **primary knob.** NT-Xent magnitude ~O(log N)≈5 vs recon ~0.03–0.3, so λ must be small or it swamps recon. Warmup so recon settles first. Watch val_mse for drift. |
| base | `vae16_fb0p0_film` | s0.01 + FiLM(`film_bound`), 40 ep, from scratch |

## Config & branch

- Branch `imc-pigvae-contrastive-drop` off `imc-pigvae-film-multi`.
- New experiment config **`configs/experiment/vae16_fb0p0_film_drop.yaml`** — clone of
  `vae16_fb0p0_film.yaml` + `contrastive_loss_scale`, `drop_p`, `contrastive_temperature`,
  `batch_size: 96`. **The contrastive path is gated by `contrastive_loss_scale > 0`** (training-step /
  critic side), NOT the existing `is_contrastive` datamodule flag — that flag drives the old
  permutation-view dataloader path, which is unused here (drop views are generated in the training step).
  Leave `is_contrastive` off.
- Downstream: **reuse the existing 4-way sweep unchanged** (`sbatch_downstream_sweep.sh`); new run_tag
  `vae16_fb0p0_film_drop_<date>`. z_global is read raw, so the encode switch needs no change.

## Constraints (measured)

GPU memory probe (RTX 5000 Ada, 32 GB, full Approach-A step incl. backward + Adam):
B=64 → 15.3 GB, B=96 → 22.8 GB, B=128 → 30.2 GB (tight), B=192 → OOM. **B=96 chosen** for ~9 GB
headroom (dataloader, heavier real recon loss, fragmentation). No MoCo memory queue needed — 190
negatives/anchor is ample at this scale.

## Testing

- `drop_views`: exact fraction masked (± rounding), two views differ, `node_features` preserved,
  ≥1 node kept per row, reproducible under seed.
- `ProjectionHead`: output shape `[N,128]`, gradients flow.
- **Degeneracy guard** (encodes the core insight): contrastive gradient to the encoder is **non-zero**
  for drop views but **≈zero** for identical/permuted views.
- Integration smoke test: one training step on `vae16_fb0p0_film_drop` runs, total loss finite,
  contrastive term > 0, gradients reach encoder params; recon path unchanged vs FiLM step.

## Out of scope (later)

- Jitter and bag augmentation families (bag needs the P-K sampler + image-id dataloader).
- The AE-ward (`s0.001`) contrastive run — add only if drop proves out on balanced.
- λ / p sweeps — after the first run gives a yes/no.
- Fine-tuning-from-checkpoint variant — a cheaper second iteration once drop is validated.
