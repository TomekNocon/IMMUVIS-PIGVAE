# Diagnostic Run Design: AE + Oracle Permuter + FiLM

**Date:** 2026-05-10
**Branch:** imc-pigvae-film
**Status:** Approved

## Context

Run 94 (FiLM branch) suffered severe posterior collapse: latent dims dropped from 768 to ~116 by epoch 29, `kld_per_dim_median` fell to 0.003, and `latent/std_mean` climbed to 0.88 (≈ prior). The collapse was driven by three config issues unrelated to FiLM itself:

1. `kld_free_bits=0` — no floor on per-dim KLD, so the optimizer zeroed unused dims
2. `emb_dim=768` — 6× larger latent than run 93, far more dimensions to collapse
3. KLD annealing started at epoch 1 with a fast ramp — KLD pressure arrived before reconstruction could establish a useful latent structure

Because z collapsed, FiLM was modulating decoder layers with random Gaussian noise rather than meaningful signal, making it impossible to evaluate whether FiLM helps.

Run 93 (base, `emb_dim=128`) plateaued at `val_mse≈0.81` with a healthy latent, largely due to the permuter not converging (`mean_confidence=0.452` — barely above chance).

## Goal

Establish a clean lower bound: can the FiLM encoder-decoder reconstruct well when given everything it needs — no VAE stochasticity, no permuter prediction error?

If this works, subsequent runs add one piece of complexity at a time:
1. This run: AE + oracle permuter + FiLM
2. Next: AE + learned permuter + FiLM
3. Then: VAE + learned permuter + FiLM (with `kld_free_bits=1`, delayed schedule)

## Architecture

| Parameter | Value | Rationale |
|---|---|---|
| `emb_dim` | 128 | Matches run 93 — proven stable, no collapse risk |
| `input_size` | 256 | Matches run 93 |
| `num_layers` | 3 | Unchanged |
| `num_heads` | 4 | Unchanged |
| `vae` | false | Pure AE — no reparameterization, no KLD |
| `use_film` | true | This is what we are testing |

Approximate parameter count: ~6.8M (same as run 93).

## Permuter

`turn_off: true` — the permuter returns the ground-truth inverse transform for each augmentation (the 8 D4 symmetries: 4 rotations × 2 flips). The decoder always receives the correctly canonicalized positional embeddings.

With `perm_loss_scale: 0.0` the permuter shadow-mode training is also disabled — permuter weights are not updated at all this run.

## Loss Function

Active terms only:
- `alpha_recon` (node feature MSE, weight 1.0)
- `beta_recon` (edge feature, weight 0.1)
- `gamma_recon` (edge preservation, weight 0.001)
- `signal_to_noise_ratio_loss` (implicit via SNR term)

Disabled:
- `kld_loss` — auto-disabled by `vae=false`
- `permutation_loss` — `perm_loss_scale: 0.0`
- `contrastive_loss` — `contrastive_loss_scale: 0.0`

This gives the cleanest possible gradient signal for reconstruction quality.

## Config Changes from Current FiLM Branch

All changes are in `configs/model/model.yaml`:

| Field | Before (run 94) | After (this run) |
|---|---|---|
| `graph_ae.hparams.emb_dim` | 768 | 128 |
| `graph_ae.hparams.input_size` | 768 | 256 |
| `graph_ae.hparams.vae` | true | false |
| `permuter.turn_off` | false | true |
| `critic.perm_loss_scale` | 0.05 | 0.0 |
| `critic.contrastive_loss_scale` | 0.01 | 0.0 |
| `use_film` (decoder) | true | true (unchanged) |
| `kld_free_bits` | 0.0 | 0.0 (irrelevant, vae=false) |

Schedulers (`kld_alpha_scheduler`, `temperature_scheduler`) are left in config but become no-ops: KLD is not computed and the oracle permuter ignores tau.

## Success Criteria

- `val_mse_loss` clearly below 0.80 within 30–50 epochs (run 93 achieved 0.81 with more noise sources)
- Decoder activations (`actmon/decoder/std`) grow over time — FiLM gamma/beta are being used
- `latent/mu_std` stays meaningfully above 0 — encoder is writing into z (no KLD to prevent it)
- No NaN / inf in activations

## What Failure Would Mean

If `val_mse` plateaus at or above run 93's 0.81 with oracle permutations and no VAE noise, the reconstruction architecture itself (encoder, FiLM decoder, bottleneck) has a capacity or connectivity issue that needs to be fixed before re-introducing complexity.
