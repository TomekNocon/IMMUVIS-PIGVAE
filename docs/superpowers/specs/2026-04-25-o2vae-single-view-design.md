# Design: Single-View D4 Alignment Loss (O2-VAE-style)

**Date:** 2026-04-25
**Branch:** imc-pigvae-o2
**Status:** Approved

## Motivation

The encoder is architecturally permutation-invariant (no positional encodings, CLS token
aggregation). This means all 8 D4-augmented views of the same sample produce the **exact
same latent z**. The current Sinkhorn permuter is therefore redundant on the encoder side,
and the 8-view batch is pure computational waste.

The O2-VAE insight applied here: replace the learned permuter with a **logsumexp
reconstruction loss** that tries all 8 D4 orientations of the decoder output and picks the
best match analytically. This eliminates the permuter, its curriculum, shadow mode, entropy
scheduler, and temperature scheduler — while preserving and strengthening orientation
invariance.

## Architecture After Changes

```
Training:
  x  →  encoder  →  z [B, D]
      →  decoder(z, identity_perm)  →  x_hat [B, N, D]
      →  D4AlignmentLoss(x_hat, x)   (logsumexp over 8 D4 transforms, per-sample)
      →  KLDLoss(mu, logvar)
      →  total = D4AlignmentLoss + kld_scale * kld_alpha * KLD

Inference:
  x  →  encoder  →  z  (orientation-invariant by design)
      →  decoder(z, identity_perm)  →  x_hat  (canonical orientation)
```

---

## Section 1: Data Pipeline

**Goal:** Switch from 8-view batch `[8*B, N, D]` to single-view batch `[B, N, D]`.

### `src/data/components/graphs_datamodules.py`

**Add `SingleViewTransform`** — replaces **both `IMCBaseDictTransform` and
`PatchAugmentations`** entirely. Takes the raw single `[C, H, W]` embedding directly from
`PickleDataset` and handles everything in one pass:
- Training: randomly picks one key from `IMC_GRAPH_VIEW_KEYS` (`randint(0, 7)`), calls
  `_spatial_view(emb, key)` (already in the file) to apply one D4 transform on demand
- Validation (`is_validation=True`): always uses `_spatial_view(emb, "r0_nf")` (identity —
  no rotation, no flip) → stable, deterministic
- Applies all preprocessing to that **one view**: center-crop, clip, normalise (same params
  as the current `IMCBaseDictTransform`: `center_crop_size`, `normalize`, `clip_percentiles`,
  etc.)
- Reshapes `[C, H, W]` → `[N, C]`
- Returns `(tensor [1, N, C], argsort [1, N], perm [0])` — same interface as
  `PatchAugmentations`, single-element leading dim so `from_sparse_graph_list` is unchanged

**`PickleDataset`:** Add `single_view: bool = False`. When `True`, set `generate_views=False`
internally so `make_views()` is never called and the raw `[C, H, W]` embedding is returned
directly. No 8-view array is ever allocated.

**`DenseGraphBatch.from_sparse_graph_list`:** No changes needed — `augmented_embedding[perm]`
with `perm=[0]` and shape `[1, N, D]` produces `[N, D]`; the stack+flatten logic produces
`[B, N, D]` naturally. The `factor = batch_size / batch_size_mask` becomes 1 (no-op repeat).

### `src/data/imc_datamodule.py`

- Add `single_view: bool` to hparams (default `True` for new training runs)
- When `single_view=True`: use `SingleViewTransform` in `DualOutputTransform` instead of
  `PatchAugmentations`
- Update `num_aug_per_sample` default to `1` in config

---

## Section 2: D4AlignmentLoss

**New class in `src/models/components/losses.py`.**

Precomputes the 8 D4 permutation matrices (same construction as `SimplePermuter`), registers
them as a buffer. Computes the logsumexp reconstruction loss **per sample** — critical because
different samples have different best orientations.

```
for k in 0..7:
    x_hat_k = perm_matrices[k] @ x_hat          # reorder nodes: [B, N, D]
    reshape to grid: [B, D, H, W]

    huber_k  = smooth_l1(x_hat_k, x).mean([-2,-1])     # [B]
    cosine_k = 1 - cosine_sim(x_hat_k.flat, x.flat)    # [B]
    grad_k   = gradient_l1_per_sample(x_hat_k, x)      # [B]

    loss_k = alpha*huber_k + beta*cosine_k + gamma*grad_k   # [B]

losses = stack([loss_0 .. loss_7], dim=1)   # [B, 8]
lse    = -logsumexp(-losses, dim=1) + log(8)   # [B]  — soft-min + uniform prior normalisation
return {"loss": lse.mean(), "d4_alignment_loss": lse.mean()}  # dict, consistent with GraphReconstructionLoss
```

`+ log(8)` normalises by the uniform prior over orientations (equivalent to the partition
function in the ELBO for a discrete symmetry group with 8 elements).

**Constructor params:** `grid_size: int`, `huber_beta: float`, `alpha: float`, `beta: float`,
`gamma: float` — same weights as the current `GraphReconstructionLoss`.

**Helper functions** (private, in the same class):
- `_huber_per_sample(x_hat_k, x, beta)` → `[B]`
- `_cosine_per_sample(x_hat_k, x)` → `[B]`
- `_gradient_per_sample(x_hat_k_grid, x_grid)` → `[B]`

MAE, MSE, SNR losses in `Critic` are batch-averaged metrics logged for monitoring — they are
**not** inside the logsumexp loop, computed once on the raw (unpermuted) `x_hat`.

---

## Section 3: GraphAE and Critic

### `src/models/components/modules.py` — `GraphAE`

- Remove `self.permuter = SimplePermuter(hparams.permuter)`
- `forward()`: construct identity permutation `eye = torch.eye(N).unsqueeze(0).expand(B,-1,-1)`
  and pass to decoder. The decoder's `pos_emb = perm @ pos_emb` becomes a no-op.
- `forward()` return: `(graph_emb, graph_pred, mu, logvar)` — drop `soft_probs` and `perm`
- Keep `SimplePermuter` class in file (used by callbacks; don't delete)

### `src/models/components/modules.py` — `BottleNeckEncoder` (batch-size fix)

Current code shares epsilon across all 8 views so they sample the same z. With single-view
this is unnecessary. Simplify to standard VAE reparameterization:

```python
# Before (8-view epsilon sharing):
batch_size = x.shape[0] // self.num_permutations
batch_std  = std[:batch_size]
batch_eps  = torch.randn_like(batch_std)
eps = batch_eps.unsqueeze(0).repeat(self.num_permutations, 1, 1).view(-1, d)
x = mu + eps * std

# After (standard):
eps = torch.randn_like(std)
x   = mu + eps * std
```

Remove `self.num_permutations` field from `BottleNeckEncoder`.

### `src/models/components/model.py` — `Critic`

- Replace `GraphReconstructionLoss + PermutationLoss` with `D4AlignmentLoss`
- Remove `soft_probs`, `perm`, `beta` (entropy weight) params from `forward()` / `evaluate()`
- KLD, MAE, MSE, SNR losses unchanged
- `forward()` loss dict: `{"loss": d4_loss + kld_scale*kld_alpha*kld, "d4_alignment_loss": ..., "kld_loss": ..., "mae_loss": ..., ...}`

---

## Section 4: PLGraphAE Training Module

### Remove entirely from `pigvae_auto_module.py`:
- `_apply_curriculum()`
- `self.temperature_scheduler`, `self.entropy_weight_scheduler` (constructor + usage)
- `tau`, `beta` (perm entropy weight) variables in `training_step` / `validation_step`
- `self.perms` list and all perm accumulation
- `permuter` component from `configure_gradient_clipping`
- `soft_probs` from all forward calls and outputs dict

### Keep:
- `self.kld_alpha_scheduler` + alpha annealing (posterior collapse prevention)
- All latent diagnostic logging (`_log_latent_stats`)

### `on_validation_epoch_end` — simplified visualization:
Replace the entire 8-view layout with:
- Take first validation batch, show `min(n_examples, batch_size)` samples
- Log 3 W&B image panels: **Predictions**, **Ground Truth**, **Diff**
- Log **PCA** scatter (latent space coloured by label) — keep existing `pL.plot_pca`
- Remove: permutation counters, MSE-per-transform bar chart, 8-view grid plots

### `configure_gradient_clipping`:
Remove `ae.permuter` from the per-component clip loop.

---

## Section 5: Batch Size Audit

All locations that assume `batch_size = total // 8` or produce `8*B` output:

| File | Line(s) | Issue | Fix |
|---|---|---|---|
| `modules.py` | 630–643 | `BottleNeckEncoder` epsilon sharing (8-view) | Simplify to standard reparameterization (covered in §3) |
| `pigvae_auto_module.py` | 276 | `predictions.shape[0] // 8` | `predictions.shape[0]` |
| `pigvae_auto_module.py` | 278 | `R.batch_augmented_indices(batch_size, num_permutations=8, ...)` | `np.arange(min(n_examples, batch_size))` |
| `pigvae_auto_module.py` | 398, 400 | same in `on_test_epoch_end` | same fix |
| `pigvae_auto_module.py` | 265 | `argsort_augmented_features` restore step | Remove restore — identity argsort, no reorder needed |
| `graphs_datamodules.py` | `from_sparse_graph_list` factor | `factor = 8*B / B = 8` → `repeat_interleave(8)` | With single-view factor=1 — **no change needed**, already correct |
| `metrics/recontructions.py` | all | `batch_augmented_indices`, `mse_per_transform` assume 8-view layout | No longer called; functions kept but unused |
| Configs | `num_aug_per_sample` | `8` | `1` |
| Configs | `permuter:` block | present | remove |
| Configs | `perm_loss_scale`, `temperature_scheduler`, `entropy_weight_scheduler` | present | remove |

---

## Files Changed

| File | Change type |
|---|---|
| `src/data/components/graphs_datamodules.py` | Add `SingleViewTransform` (replaces `IMCBaseDictTransform + PatchAugmentations`) |
| `src/data/imc_datamodule.py` | Add `single_view` param; wire `SingleViewTransform` + `single_view=True` in `PickleDataset` |
| `src/models/components/losses.py` | Add `D4AlignmentLoss`; remove `PermutationLoss` usage |
| `src/models/components/modules.py` | Remove permuter from `GraphAE`; simplify `BottleNeckEncoder` |
| `src/models/components/model.py` | Replace recon+perm losses with `D4AlignmentLoss` in `Critic` |
| `src/models/pigvae_auto_module.py` | Remove curriculum/schedulers; fix batch-size refs; simplify viz |
| `configs/model/model.yaml` | Remove permuter block; add `d4_alignment_loss` section |
| `configs/train.yaml` (and related) | Remove temperature/entropy schedulers; set `num_aug_per_sample: 1` |

## What Is NOT Changed

- Encoder architecture (no positional encodings, CLS token — invariance already holds)
- Decoder architecture (GraphDecoder, BottleNeckDecoder unchanged)
- KLD annealing schedule (`kld_alpha_scheduler`)
- AdamW + cosine warmup LR schedule
- Latent diagnostics (`_log_latent_stats`, active dims, per-dim KLD)
- `SimplePermuter` class (kept for potential ablation use via callbacks)
- `PatchAugmentations` class (kept, used in other paths)
