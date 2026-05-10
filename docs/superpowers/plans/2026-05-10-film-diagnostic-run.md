# FiLM Diagnostic Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconfigure the FiLM branch to run the simplest possible diagnostic — AE (no VAE) with oracle permutations — to isolate whether the FiLM encoder-decoder pipeline can reconstruct well before adding complexity back.

**Architecture:** Single config file change (`configs/model/model.yaml`). No Python source changes. The oracle permuter path (`turn_off=True`) and AE path (`vae=False`) are already fully implemented in the codebase — this plan only adjusts the six config values that enable them.

**Tech Stack:** Hydra (config), PyTorch Lightning, pytest

---

## File Map

| File | Action | What changes |
|---|---|---|
| `configs/model/model.yaml` | Modify | 6 field changes (emb_dim, input_size, vae, turn_off, perm_loss_scale, contrastive_loss_scale) |

No new files. No Python source edits.

---

### Task 1: Apply config changes

**Files:**
- Modify: `configs/model/model.yaml`

- [ ] **Step 1: Change `emb_dim` and `input_size` to match run 93**

In `configs/model/model.yaml`, under `graph_ae.hparams`, update:

```yaml
    emb_dim: 128       # was 768
```

and:

```yaml
    input_size: 256    # was 768
```

Both cascade automatically via Hydra interpolation to all sub-modules that reference `${model.graph_ae.hparams.emb_dim}` and `${model.graph_ae.hparams.input_size}` (encoder, decoder, permuter, bottleneck, property predictor).

- [ ] **Step 2: Switch off VAE**

Under `graph_ae.hparams`, update:

```yaml
    vae: false         # was true
```

This cascades via `${model.graph_ae.hparams.vae}` to `bottle_neck_encoder.vae` and `critic.hparams.vae`, disabling reparameterization and KLD loss everywhere automatically.

- [ ] **Step 3: Enable oracle permuter**

Under `graph_ae.hparams.permuter`, update:

```yaml
      turn_off: True         # was False
```

With `turn_off: True`, the permuter's `forward()` returns the ground-truth inverse transform for each of the 8 D4 augmentations. The decoder always receives a correctly canonicalized positional embedding. The permuter's learned weights are not used for the decoder path.

- [ ] **Step 4: Zero out permutation and contrastive loss scales**

Under `critic.hparams`, update:

```yaml
    perm_loss_scale: 0.0          # was 0.05
    contrastive_loss_scale: 0.0   # was 0.01
```

With `turn_off: True` the permuter runs in shadow mode by default (learns in background). Setting `perm_loss_scale: 0.0` disables even that — permuter weights are frozen this run. `contrastive_loss_scale: 0.0` strips the contrastive term so the total loss is purely reconstruction.

- [ ] **Step 5: Verify the full resulting config block**

After edits, the top of `configs/model/model.yaml` should look like this (confirm visually):

```yaml
graph_ae:
  _target_: "src.models.components.modules.GraphAE"
  hparams:
    input_size: 256
    num_heads: 4
    num_layers: 3
    emb_dim: 128
    vae: false
    dropout: 0.15
    ...
    decoder:
      ...
      use_film: true       # unchanged — this is what we are testing
    ...
    permuter:
      ...
      turn_off: True
      ...

critic:
  _target_: "src.models.components.model.Critic"
  hparams:
    kld_free_bits: 0.0             # irrelevant (vae=false), leave as-is
    perm_loss_scale: 0.0
    contrastive_loss_scale: 0.0
    ...
```

`kld_alpha_scheduler` and `temperature_scheduler` blocks can remain unchanged — they become no-ops (`vae=false` means KLD is never computed; `turn_off=True` means tau never affects the decoder path).

---

### Task 2: Verify config resolves and model instantiates

**Files:**
- Test: `tests/test_configs.py::test_train_config`

- [ ] **Step 1: Run the config instantiation test**

```bash
pytest tests/test_configs.py::test_train_config -v
```

Expected output:
```
PASSED tests/test_configs.py::test_train_config
```

This test calls `hydra.utils.instantiate(cfg.model)` which will:
- Resolve all `${...}` interpolations
- Construct `GraphAE`, `Critic`, all schedulers
- Raise immediately if any shape mismatch or missing key exists

If it fails with a shape error, check that `bottle_neck_decoder.num_nodes: 36` is consistent with `grid_size: 6` (36 = 6²) — this does not change.

- [ ] **Step 2: Run the fast-dev-run training test**

```bash
pytest tests/test_train.py::test_train_fast_dev_run -v
```

Expected output:
```
PASSED tests/test_train.py::test_train_fast_dev_run
```

This runs exactly one train step + one val step end-to-end on CPU. It exercises:
- Encoder forward pass
- Oracle permuter path (`turn_off=True`)
- FiLM conditioner projecting z into decoder layers
- Reconstruction loss computation (alpha + beta + gamma terms)
- Backward pass + optimizer step

If it fails, the error message will point to the exact tensor shape or missing attribute.

---

### Task 3: Commit

**Files:**
- `configs/model/model.yaml`

- [ ] **Step 1: Stage and commit**

```bash
git add configs/model/model.yaml
git commit -m "feat: configure diagnostic AE + oracle permuter + FiLM run

- emb_dim 768->128, input_size 768->256 (match run 93 stable size)
- vae=false: pure autoencoder, no KLD loss
- permuter.turn_off=True: oracle ground-truth permutation
- perm_loss_scale=0.0, contrastive_loss_scale=0.0: pure reconstruction loss

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## What to Watch in the First Run

Once training starts, the key signals within the first 20–30 epochs:

| Metric | Healthy | Concerning |
|---|---|---|
| `val_mse_loss` | Drops below 0.80 | Flat at or above run 93's 0.81 |
| `actmon/decoder/std` | Grows over time | Stays near zero (FiLM not activating) |
| `latent/mu_std` | Stays > 0.3 (encoder writing to z) | Collapses toward 0 |
| `actmon/*/has_nan` | Always 0 | Any 1 → stop immediately |

## Incremental Steps After This Run Succeeds

1. `permuter.turn_off: False` — switch to learned permuter (still AE)
2. `vae: true` + `critic.kld_free_bits: 1.0` + `kld_alpha_scheduler.start_epoch: 15` — add VAE back with safe schedule
3. `critic.contrastive_loss_scale: 0.01` — restore contrastive term
