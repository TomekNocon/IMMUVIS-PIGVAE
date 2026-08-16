# No-stats retrain — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a `use_stats_correction` gate to `GraphEncoder` (default True = bit-identical) and a `vae16_fb0p0_film_nostats` config that disables it, so we can retrain the FiLM control without the pooled-stats correction and test whether the CLS token can carry the transfer signal on its own.

**Architecture:** `GraphEncoder` builds and adds `stats_correction` unconditionally today. Gate both on `getattr(hparams, "use_stats_correction", True)` (mirroring the existing `use_hadamard` pattern). When False, `stats_correction` is None and the forward skips the add, leaving `graph_emb` as the pure CLS readout. A cloned experiment config sets the flag false.

**Tech Stack:** PyTorch, Lightning, Hydra/OmegaConf, pytest, `uv`, SLURM.

## Global Constraints

- Default `use_stats_correction=True` MUST be bit-identical to today (build `NodeStatsProjection`, add it). Existing checkpoints (trained with stats) must load unchanged.
- The flag lives in the **encoder** hparams (same block as `use_hadamard`), read via `getattr(hparams, "use_stats_correction", True)`.
- Only the plain FiLM recipe gets a nostats config — NOT the drop/MAE variants.
- Tests use a real `_build_pl_module(...)` instance (not mocks). Run with `uv run pytest`.
- `git add` only the named files; never `git add -A`.

---

### Task 1: `use_stats_correction` gate + nostats config

**Files:**
- Modify: `src/models/components/modules.py` — `GraphEncoder.__init__` (~line 304) and `GraphEncoder.forward` (the `graph_emb = graph_emb + self.stats_correction(...)` line, ~383).
- Create: `configs/experiment/vae16_fb0p0_film_nostats.yaml`.
- Test: `tests/test_nostats_gate.py` (create).

**Interfaces:**
- Produces: `GraphEncoder` with `self.stats_correction` = `NodeStatsProjection(...)` when `use_stats_correction` is True/absent, else `None`; `forward` adds the correction only when it is not None.
- Produces: experiment `vae16_fb0p0_film_nostats` = `vae16_fb0p0_film_b64` with `model.graph_ae.hparams.encoder.use_stats_correction: false`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nostats_gate.py`:

```python
import torch

from src.data.components.graphs_datamodules import DenseGraphBatch
from src.downstream.encode import _build_pl_module


def test_default_builds_stats_correction():
    gae = _build_pl_module("vae16_fb0p0_film_b64").graph_ae
    assert gae.encoder.stats_correction is not None


def test_nostats_config_disables_stats_correction():
    gae = _build_pl_module("vae16_fb0p0_film_nostats").graph_ae
    assert gae.encoder.stats_correction is None


def test_forward_skips_stats_when_disabled():
    # With a non-zero stats weight, removing the correction must change graph_emb
    # (proving the add is gated), and the gated forward must run finite [B, 512].
    torch.manual_seed(0)
    gae = _build_pl_module("vae16_fb0p0_film_b64").graph_ae.eval()
    torch.nn.init.normal_(gae.encoder.stats_correction.proj.weight, std=0.01)
    batch = DenseGraphBatch(
        node_features=torch.randn(2, 256, 128),
        edge_features=torch.empty(0),
        mask=torch.ones(2, 256, dtype=torch.bool),
    )
    with torch.no_grad():
        emb_on, _ = gae.encoder(
            node_features=batch.node_features,
            edge_features=batch.edge_features,
            mask=batch.mask,
        )
        gae.encoder.stats_correction = None
        emb_off, _ = gae.encoder(
            node_features=batch.node_features,
            edge_features=batch.edge_features,
            mask=batch.mask,
        )
    assert not torch.allclose(emb_on, emb_off)
    assert emb_off.shape == (2, 512)
    assert torch.isfinite(emb_off).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_nostats_gate.py -v`
Expected: FAIL — `test_nostats_config_disables_stats_correction` errors (no such config), and `test_forward_skips_stats_when_disabled` fails (forward unconditionally adds stats, so `emb_on == emb_off` after setting None would actually raise `AttributeError` on `None(...)` — the gate does not exist yet).

- [ ] **Step 3: Add the gate to `GraphEncoder.__init__`**

In `src/models/components/modules.py`, replace the unconditional line (~304):

```python
        self.stats_correction = NodeStatsProjection(hparams.graph_encoder_hidden_dim)
```

with:

```python
        self.use_stats_correction = getattr(hparams, "use_stats_correction", True)
        self.stats_correction = (
            NodeStatsProjection(hparams.graph_encoder_hidden_dim)
            if self.use_stats_correction
            else None
        )
```

- [ ] **Step 4: Gate the add in `GraphEncoder.forward`**

In the same file, replace (~383):

```python
        graph_emb = graph_emb + self.stats_correction(node_features, mask)
```

with:

```python
        if self.stats_correction is not None:
            graph_emb = graph_emb + self.stats_correction(node_features, mask)
```

(The `structural_correction` block just below is unchanged.)

- [ ] **Step 5: Create the nostats config**

Create `configs/experiment/vae16_fb0p0_film_nostats.yaml` — copy `configs/experiment/vae16_fb0p0_film_b64.yaml` verbatim, then: (a) under `model.graph_ae.hparams.encoder`, add `use_stats_correction: false` alongside `grid_size: 16`; (b) in `tags`, replace `"control"`, `"b64"` with `"nostats"`, `"ablation"`; (c) set the header comment and `logger.wandb.name` to `"vae16-fb0.0-s0.01-FILM-NOSTATS-40ep"`. Keep everything else (KL, FiLM, batch 64, 40 ep, `num_aug_per_sample: 1`) identical.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_nostats_gate.py -v`
Expected: PASS (all 3).

- [ ] **Step 7: Run the existing suite to confirm no regression**

Run: `uv run pytest tests/test_contrastive_drop.py tests/test_block_drop.py -q`
Expected: PASS (default path unchanged).

- [ ] **Step 8: Commit**

```bash
git add src/models/components/modules.py configs/experiment/vae16_fb0p0_film_nostats.yaml tests/test_nostats_gate.py
git commit -m "feat(encoder): use_stats_correction gate + vae16_fb0p0_film_nostats config"
```

---

## Post-implementation ops (controller-run)

1. **CPU fast-dev-run probe** (szary, before the full run): compose `experiment=vae16_fb0p0_film_nostats trainer=gpu paths=szary`, assert batch 64 and `encoder.stats_correction is None`, run `Trainer(fast_dev_run=1).fit`. Mirror the drop-MAE probe harness (rootutils + multiply/divide resolvers + `wandb.init(mode="disabled")`).
2. **Train** `vae16_fb0p0_film_nostats` (40 ep, batch 64, ~1.75h) via sbatch on szary.
3. **Encode + 4-way sweep** on the resulting checkpoint (`zglobal` is now pure CLS); collect the table and compare `zglobal` against the `ctl_last` cls_only / stats_only / full numbers to decide H1 vs H2.

---

## Self-Review

- **Spec coverage:** gate with default-True bit-identity (Steps 3-4 + `test_default_builds_stats_correction` + Step 7) ✓; config disables it (Step 5 + `test_nostats_config_disables_stats_correction`) ✓; forward gating proven (`test_forward_skips_stats_when_disabled`) ✓; only plain FiLM recipe gets nostats ✓; probe + train + sweep (ops) ✓.
- **Placeholder scan:** none — real code and commands throughout.
- **Type consistency:** `use_stats_correction` read the same way in `__init__`; `stats_correction` is `None`-or-module and the forward guards on `is not None`; config key path `model.graph_ae.hparams.encoder.use_stats_correction` matches where `getattr(hparams, ...)` reads (encoder hparams, same block as `use_hadamard`).
