# VAE Convergence Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix posterior collapse in the IMC PIGVAE by reducing the latent dimension from 768 to 128, adding KLD free bits, and delaying/slowing the KLD annealing schedule.

**Architecture:** All changes are in `configs/model/model.yaml` only. No Python changes are needed — `KLDLoss` already supports `free_bits` and `KLDAlphaScheduler` already supports `start_epoch`/`num_epochs`. The bottleneck encoder maps `input_size → emb_dim`; reducing both to 128 forces real compression and shrinks the transformer hidden dim in encoder, decoder, and permuter simultaneously via Hydra interpolation.

**Tech Stack:** Hydra config (YAML), PyTorch Lightning, pytest

---

## Files

- Modify: `configs/model/model.yaml` — 5 parameter changes
- Modify: `tests/test_configs.py` — add latent shape assertion

---

### Task 1: Add latent-shape test (write failing test first)

**Files:**
- Modify: `tests/test_configs.py`

- [ ] **Step 1: Add the failing test**

Open `tests/test_configs.py` and append this test after the existing `test_eval_config`:

```python
def test_model_latent_dim(cfg_train: DictConfig) -> None:
    """Verify the model instantiates with emb_dim=128 and free_bits=1.0."""
    import torch
    from hydra.utils import instantiate

    model_cfg = cfg_train.model
    model = instantiate(model_cfg)

    # emb_dim drives the bottleneck output size
    assert model.graph_ae.bottle_neck_encoder.d_out == 128, (
        f"Expected emb_dim=128, got {model.graph_ae.bottle_neck_encoder.d_out}"
    )
    # free_bits should be 1.0, not 0
    assert model.critic.kld_loss.free_bits == 1.0, (
        f"Expected free_bits=1.0, got {model.critic.kld_loss.free_bits}"
    )
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE
pytest tests/test_configs.py::test_model_latent_dim -v
```

Expected: FAIL — `AssertionError: Expected emb_dim=128, got 768`

---

### Task 2: Apply config changes

**Files:**
- Modify: `configs/model/model.yaml`

- [ ] **Step 1: Change `emb_dim` and `input_size` to 128**

In `configs/model/model.yaml`, under `graph_ae.hparams`, change:

```yaml
# Before
    input_size: 768
    ...
    emb_dim: 768
```

```yaml
# After
    input_size: 128
    ...
    emb_dim: 128
```

- [ ] **Step 2: Change `kld_free_bits` to 1.0**

Under `critic.hparams`, change:

```yaml
# Before
    kld_free_bits: 0.0  # disabled: free_bits creates a dead zone with zero gradient below threshold
```

```yaml
# After
    kld_free_bits: 1.0  # protects each dim from zero-gradient collapse
```

- [ ] **Step 3: Delay KLD annealing start to epoch 10**

Under `kld_alpha_scheduler.hparams`, change:

```yaml
# Before
    start_epoch: 1
```

```yaml
# After
    start_epoch: 10
```

- [ ] **Step 4: Slow KLD annealing duration to max_epochs/2**

Under `kld_alpha_scheduler.hparams`, change:

```yaml
# Before
    num_epochs: ${divide:${trainer.max_epochs},3}
```

```yaml
# After
    num_epochs: ${divide:${trainer.max_epochs},2}
```

- [ ] **Step 5: Verify the final state of `kld_alpha_scheduler` section looks like this**

```yaml
kld_alpha_scheduler:
  _target_: "src.models.components.schedulers.KLDAlphaScheduler"
  hparams:
    initial_alpha: 0.0
    final_alpha: 1.0
    mode: linear
    num_epochs: ${divide:${trainer.max_epochs},2}
    start_epoch: 10
```

And `critic` section contains:

```yaml
    kld_loss_scale: 0.05
    kld_free_bits: 1.0
```

And top of `graph_ae.hparams` contains:

```yaml
    input_size: 128
    num_heads: 4
    num_layers: 3
    emb_dim: 128
```

---

### Task 3: Run test and verify

**Files:**
- Test: `tests/test_configs.py`

- [ ] **Step 1: Run the shape test**

```bash
cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE
pytest tests/test_configs.py::test_model_latent_dim -v
```

Expected output:
```
PASSED tests/test_configs.py::test_model_latent_dim
```

- [ ] **Step 2: Run the full config test suite**

```bash
pytest tests/test_configs.py -v
```

Expected: all tests PASS. If `test_train_config` or `test_eval_config` fail, check that all Hydra interpolations resolve (e.g. `head_dim = divide:128,4 = 32`).

- [ ] **Step 3: Sanity-check model param count**

```bash
cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE
python -c "
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import open_dict

with initialize(version_base='1.3', config_path='configs'):
    cfg = compose(config_name='train.yaml')

with open_dict(cfg):
    cfg.paths.root_dir = '.'

model = instantiate(cfg.model)
total = sum(p.numel() for p in model.parameters())
print(f'Total params: {total:,}')
assert total < 10_000_000, f'Model unexpectedly large: {total:,} params'
print('OK — model is appropriately sized for 40k samples')
"
```

Expected: `Total params: ~5,000,000–8,000,000` (was 60M), printed `OK`.

---

### Task 4: Commit

- [ ] **Step 1: Stage and commit**

```bash
cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE
git add configs/model/model.yaml tests/test_configs.py
git commit -m "$(cat <<'EOF'
fix: reduce latent dim 768→128, add free_bits, delay KLD schedule

Addresses posterior collapse diagnosed in run 87 (dainty-donkey-87):
- emb_dim + input_size: 768 → 128 (matches PCA input dim, forces compression)
- kld_free_bits: 0.0 → 1.0 (protects each dim from zero-gradient collapse)
- kld_alpha_scheduler start_epoch: 1 → 10 (reconstruction gets a head start)
- kld_alpha_scheduler num_epochs: max/3 → max/2 (slower KLD ramp)

Model shrinks from ~60M → ~6M params. Fresh run required (arch change).
Spec: docs/superpowers/specs/2026-04-28-vae-convergence-fix-design.md

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Expected outcomes after next training run

Monitor these in W&B to confirm the fix worked:

| Metric | Old (run 87) | Target |
|---|---|---|
| `latent/active_dims` (final) | 106 | > 100 (out of 128) |
| `latent/kld_per_dim_median` (final) | 0.0005 | > 0.05 |
| `latent/std_mean` (final) | 0.92 | < 0.90 |
| `loss` trend over 50 epochs | flat ~0.26 | consistent decrease |
