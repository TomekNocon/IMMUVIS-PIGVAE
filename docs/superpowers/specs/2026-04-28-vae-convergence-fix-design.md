# VAE Convergence Fix — Design Spec

**Date:** 2026-04-28
**Branch:** imc-pigvae
**Run diagnosed:** dainty-donkey-87 (W&B run `01aii0fe`)

## Problem

Run 87 showed severe posterior collapse and stagnant reconstruction loss:

- `latent/active_dims` crashed from 703 → 106 within the first ~23 epochs
- `latent/kld_per_dim_median` dropped from 0.399 → 0.0005 (essentially zero)
- `latent/std_mean` approached 1.0 (posterior converging to prior)
- Total loss oscillated 0.24–0.28 for 113 epochs with no downward trend

Root causes identified:
1. `emb_dim = 768` gives the model ~6.8× more latent capacity than it can use on 40k samples, creating enormous KLD pressure to collapse spare dimensions
2. `kld_free_bits = 0` means every dimension receives collapse pressure with no protection
3. KLD annealing starts at epoch 1 and ramps over only 66 epochs — reconstruction never gets a head start

## Chosen Approach

Option A: reduce latent dim to 128 + free bits + slower/delayed KLD schedule.

128 is chosen because it matches `num_pca_components = 128` (the input feature dimension), making it the theoretically correct information bottleneck for this data.

## Changes — `configs/model/model.yaml` only

| Parameter | Old | New | Reason |
|---|---|---|---|
| `model.graph_ae.hparams.emb_dim` | 768 | 128 | Matches PCA input dim; forces real compression |
| `model.graph_ae.hparams.input_size` | 768 | 128 | Transformer hidden dim must match emb_dim flow |
| `critic.hparams.kld_free_bits` | 0.0 | 1.0 | Protects each dim from zero-gradient collapse |
| `kld_alpha_scheduler.hparams.start_epoch` | 1 | 10 | Reconstruction establishes gradients before KLD pressure |
| `kld_alpha_scheduler.hparams.num_epochs` | `${divide:${trainer.max_epochs},3}` | `${divide:${trainer.max_epochs},2}` | Slower KLD ramp (~100 epochs instead of ~66) |

## Architectural side-effects (all safe)

- `num_heads: 4` still divides `input_size: 128` → head_dim = 32 ✓
- `ppf_hidden_dim = multiply:128,4 = 512` (was 3072) — reasonable FFN ratio
- Model shrinks from ~60M → ~6M parameters — appropriate for 40k samples
- Encoder, decoder, and permuter all use `${model.graph_ae.hparams.input_size}` via Hydra interpolation — all updated automatically

## What is NOT changed

- `kld_loss_scale: 0.05` — scale is fine once dims and free_bits are correct
- `num_heads: 4`, `num_layers: 3`, `dropout: 0.15` — architecture depth/structure unchanged
- Optimizer (AdamW lr=0.0003) — not the root cause
- Permuter config — permuter issues are downstream of latent collapse; no independent fix needed
- Reconstruction loss scales (alpha/beta/gamma) — balanced, not root cause

## No Python changes

`KLDLoss` already supports `free_bits` (implemented in `src/models/components/losses.py:405`).
`KLDAlphaScheduler` already supports `start_epoch` and `num_epochs`.
Only the config file needs to change.

## Fresh run required

The architectural change (emb_dim 768→128) is incompatible with existing checkpoints. This must be a fresh training run.

## Expected outcome

- `latent/active_dims` should stabilise at a high fraction of 128 (target: >80)
- `latent/kld_per_dim_median` should stay above 0.05 throughout training
- Reconstruction loss (MSE, MAE) should show consistent downward trend
- `latent/std_mean` should stay well below 0.95 during training
