# Drop-invariance Contrastive Objective — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a SimCLR-style drop-invariance contrastive loss on `z_global` to the FiLM training run, so the CLS embedding is trained to be invariant to random node dropout.

**Architecture:** Approach A — each training step runs the existing clean forward (recon + KL, unchanged) plus two on-GPU node-dropout views that are encoded (encoder-only), pushed through a discarded projection head, and scored with NT-Xent. The contrastive term is added to the loss in the LightningModule's `training_step`, gated by `contrastive_loss_scale > 0`.

**Tech Stack:** PyTorch, Lightning (`PLGraphAE` = `src/models/pigvae_auto_module.py`), Hydra configs, pytest, `uv`, SLURM.

## Global Constraints

- Base architecture: `vae16_fb0p0_film` (grid 16, `node_z_dim 32`, KL `scale 0.01` / `free_bits 0.0`, FiLM on with `film_bound: true`, `film_weight_decay: 0.05`).
- `z_global` width (projection head input) = `input_size` = **512**.
- Batch size **B = 96** (memory-probed ceiling with headroom on the 32 GB RTX 5000 Ada).
- Drop fraction `p = 0.20` fixed; NT-Xent temperature = **0.2**; 2 views per crop.
- `λ` (`contrastive_loss_scale`) starts at **0.05** with a linear warmup over the first **5** epochs; the contrastive term must not regress recon (watch val_mse).
- Contrastive path is gated by `contrastive_loss_scale > 0` at the `PLGraphAE` level — NOT the legacy `is_contrastive` datamodule flag (leave that off).
- `z_global` is read raw at inference; the encode/downstream pipeline is unchanged. Projection head is training-only.
- Commit after every task. Never `git add -A` — stage only the files named in the task.
- Branch: `imc-pigvae-contrastive-drop` (already created).

---

## File Structure

- **Create** `src/models/components/contrastive.py` — `drop_views()` (pure fn) + `ProjectionHead` (nn.Module). One responsibility: contrastive augmentation + head.
- **Create** `tests/test_contrastive_drop.py` — all unit + integration tests for this feature.
- **Modify** `src/models/pigvae_auto_module.py` — `PLGraphAE.__init__` (new kwargs + conditional projection head / NT-Xent) and `training_step` (drop-view contrastive term + λ warmup).
- **Modify** `configs/model/model.yaml` — pass the new `PLGraphAE` kwargs (defaults keep contrastive OFF).
- **Create** `configs/experiment/vae16_fb0p0_film_drop.yaml` — clone of `vae16_fb0p0_film.yaml` + `contrastive_loss_scale: 0.05` + `batch_size: 96`.

Existing pieces reused unchanged: `ContrastiveLoss` (`src/models/components/losses.py:478`), `DenseGraphBatch` (`src/data/components/graphs_datamodules.py`), `GraphAE.encode` (returns `z_nodes, z_global, node_features, mu, logvar`), the 4-way downstream sweep.

---

### Task 1: `drop_views` — node-dropout augmentation

**Files:**
- Create: `src/models/components/contrastive.py`
- Test: `tests/test_contrastive_drop.py`

**Interfaces:**
- Consumes: `DenseGraphBatch(node_features, edge_features, mask)` from `src.data.components.graphs_datamodules`.
- Produces: `drop_views(batch, p=0.20, n=2, generator=None) -> list[DenseGraphBatch]` — `n` views, each with a fresh random `p`-fraction of currently-valid nodes masked out; `node_features`/`edge_features` shared; ≥1 valid node per row.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contrastive_drop.py
import torch
from src.data.components.graphs_datamodules import DenseGraphBatch


def _full_batch(B=4, N=256, D=8):
    return DenseGraphBatch(
        node_features=torch.randn(B, N, D),
        edge_features=torch.empty(0),
        mask=torch.ones(B, N, dtype=torch.bool),
    )


def test_drop_views_masks_fraction_shares_features_and_differs():
    from src.models.components.contrastive import drop_views
    batch = _full_batch()
    g = torch.Generator().manual_seed(0)
    a, b = drop_views(batch, p=0.2, n=2, generator=g)
    # ~20% dropped -> ~80% kept
    assert 0.70 < a.mask.float().mean().item() < 0.90
    # node_features untouched and shared (same object)
    assert torch.equal(a.node_features, batch.node_features)
    # the two views differ
    assert not torch.equal(a.mask, b.mask)
    # >=1 valid node per row
    assert a.mask.any(dim=1).all() and b.mask.any(dim=1).all()


def test_drop_views_never_reactivates_padded_nodes():
    from src.models.components.contrastive import drop_views
    batch = _full_batch()
    batch.mask[:, 200:] = False          # simulate padding
    a, = drop_views(batch, p=0.5, n=1, generator=torch.Generator().manual_seed(1))
    # dropped mask is a subset of the original valid nodes
    assert (a.mask & ~batch.mask).sum() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_contrastive_drop.py -k drop_views -v`
Expected: FAIL — `cannot import name 'drop_views'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/models/components/contrastive.py
import torch

from src.data.components.graphs_datamodules import DenseGraphBatch


def drop_views(
    batch: DenseGraphBatch,
    p: float = 0.20,
    n: int = 2,
    generator: torch.Generator | None = None,
) -> list[DenseGraphBatch]:
    """Return `n` node-dropout views of `batch`.

    Each view masks out a fresh random `p`-fraction of the currently-valid nodes
    (mask AND ~drop), never re-activating padded nodes, guaranteeing >=1 valid node
    per row. `node_features`/`edge_features` are shared (unchanged) — only `mask`
    differs, so this is a cheap on-GPU augmentation. Pure function (no model state).
    """
    B, N = batch.mask.shape
    device = batch.mask.device
    first_valid = batch.mask.float().argmax(dim=1)  # first True index per row
    views: list[DenseGraphBatch] = []
    for _ in range(n):
        rand = torch.rand(B, N, device=device, generator=generator)
        new_mask = batch.mask & (rand >= p)
        empty = ~new_mask.any(dim=1)
        if empty.any():
            new_mask[empty, first_valid[empty]] = True
        views.append(
            DenseGraphBatch(
                node_features=batch.node_features,
                edge_features=batch.edge_features,
                mask=new_mask,
            )
        )
    return views
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_contrastive_drop.py -k drop_views -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/models/components/contrastive.py tests/test_contrastive_drop.py
git commit -m "feat(contrastive): drop_views node-dropout augmentation"
```

---

### Task 2: `ProjectionHead`

**Files:**
- Modify: `src/models/components/contrastive.py`
- Test: `tests/test_contrastive_drop.py`

**Interfaces:**
- Produces: `ProjectionHead(in_dim=512, hidden_dim=512, out_dim=128)` — `nn.Module`, `forward(x: [N, in_dim]) -> [N, out_dim]`.

- [ ] **Step 1: Write the failing test**

```python
def test_projection_head_shape_and_grad():
    import torch
    from src.models.components.contrastive import ProjectionHead
    head = ProjectionHead(in_dim=512, hidden_dim=512, out_dim=128)
    x = torch.randn(6, 512, requires_grad=True)
    y = head(x)
    assert y.shape == (6, 128)
    y.sum().backward()
    assert x.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contrastive_drop.py -k projection_head -v`
Expected: FAIL — `cannot import name 'ProjectionHead'`.

- [ ] **Step 3: Write minimal implementation** (append to `contrastive.py`)

```python
import torch.nn as nn


class ProjectionHead(nn.Module):
    """2-layer MLP applied to z_global for the contrastive loss only.

    Discarded at inference — downstream reads raw z_global, so the encode/probe
    pipeline is unaffected.
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
```

Add `import torch.nn as nn` to the top of the file if not present.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_contrastive_drop.py -k projection_head -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/components/contrastive.py tests/test_contrastive_drop.py
git commit -m "feat(contrastive): ProjectionHead MLP for z_global"
```

---

### Task 3: Wire contrastive into `PLGraphAE`

**Files:**
- Modify: `src/models/pigvae_auto_module.py` (`__init__` ~line 58-87; `training_step` ~line 131-158)
- Modify: `configs/model/model.yaml` (top-level `PLGraphAE` kwargs)
- Test: `tests/test_contrastive_drop.py`

**Interfaces:**
- Consumes: `drop_views`, `ProjectionHead` (Task 1/2); `ContrastiveLoss(temperature, num_aug_per_sample)` from `src.models.components.losses`; `GraphAE.encode(graph, sample) -> (z_nodes, z_global, node_features, mu, logvar)`.
- Produces: `PLGraphAE.__init__` gains kwargs `contrastive_loss_scale=0.0, drop_p=0.2, contrastive_temperature=0.2, contrastive_warmup_epochs=5, projection_in_dim=512, projection_hidden_dim=512, projection_dim=128`. When `contrastive_loss_scale > 0`, `self.projection_head` and `self.contrastive_loss` exist and `training_step` adds a `"contrastive_loss"` entry plus `eff_scale * contrastive` to `loss["loss"]`.

- [ ] **Step 1: Write the failing tests**

```python
def _zg_dim(gae, batch):
    with torch.no_grad():
        _, zg, *_ = gae.encode(batch, sample=False)
    return zg.shape[-1]


def test_dropped_zglobal_differs_but_identical_inputs_match():
    # The core insight, as a test: identical inputs -> identical z_global (permutation
    # views are degenerate); dropped inputs -> genuinely different z_global.
    from src.downstream.encode import _build_pl_module
    from src.models.components.contrastive import drop_views
    pl = _build_pl_module("vae16_fb0p0_film")
    gae = pl.graph_ae.eval()
    batch = DenseGraphBatch(
        node_features=torch.randn(3, 256, 128),
        edge_features=torch.empty(0),
        mask=torch.ones(3, 256, dtype=torch.bool),
    )
    with torch.no_grad():
        _, zg1, *_ = gae.encode(batch, sample=False)
        _, zg2, *_ = gae.encode(batch, sample=False)
        assert torch.allclose(zg1, zg2, atol=1e-5)          # deterministic & identical
        a, b = drop_views(batch, p=0.2, n=2, generator=torch.Generator().manual_seed(0))
        _, zga, *_ = gae.encode(a, sample=False)
        _, zgb, *_ = gae.encode(b, sample=False)
        assert not torch.allclose(zga, zgb, atol=1e-3)      # genuinely different


def test_contrastive_path_finite_loss_and_encoder_gradient():
    from src.downstream.encode import _build_pl_module
    from src.models.components.contrastive import drop_views, ProjectionHead
    from src.models.components.losses import ContrastiveLoss
    pl = _build_pl_module("vae16_fb0p0_film")
    gae = pl.graph_ae.train()
    batch = DenseGraphBatch(
        node_features=torch.randn(4, 256, 128),
        edge_features=torch.empty(0),
        mask=torch.ones(4, 256, dtype=torch.bool),
    )
    head = ProjectionHead(in_dim=_zg_dim(gae, batch), hidden_dim=512, out_dim=128)
    ntx = ContrastiveLoss(temperature=0.2, num_aug_per_sample=2)
    feats = []
    for v in drop_views(batch, p=0.2, n=2, generator=torch.Generator().manual_seed(0)):
        _, zg, *_ = gae.encode(v, sample=True)
        feats.append(head(zg))
    loss = ntx(torch.cat(feats, dim=0))                     # [2B, 128]
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    enc_grad = sum(
        p.grad.abs().sum().item()
        for n, p in gae.named_parameters()
        if "encoder" in n and p.grad is not None
    )
    assert enc_grad > 0.0
```

- [ ] **Step 2: Run tests to verify they fail/pass appropriately**

Run: `uv run pytest tests/test_contrastive_drop.py -k "zglobal or contrastive_path" -v`
Expected: These two use only Task 1/2 code + existing model, so they should PASS already. They lock in the behavior the wiring depends on. If either fails, fix before proceeding (e.g. `_build_pl_module` import path).

- [ ] **Step 3: Modify `PLGraphAE.__init__`**

Add the new kwargs to the signature (after `compile: bool`) and register the head/loss conditionally. Insert imports at top of file:

```python
from src.models.components.contrastive import ProjectionHead, drop_views
from src.models.components.losses import ContrastiveLoss
```

Signature + body additions:

```python
    def __init__(
        self,
        graph_ae: torch.nn.Module,
        critic: torch.nn.Module,
        temperature_scheduler: torch.nn.Module,
        entropy_weight_scheduler: torch.nn.Module,
        kld_alpha_scheduler: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        compile: bool,
        contrastive_loss_scale: float = 0.0,
        drop_p: float = 0.2,
        contrastive_temperature: float = 0.2,
        contrastive_warmup_epochs: int = 5,
        projection_in_dim: int = 512,
        projection_hidden_dim: int = 512,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        # ... existing body unchanged up to self.perms = [] ...
        self.contrastive_loss_scale = float(contrastive_loss_scale)
        self.drop_p = float(drop_p)
        self.contrastive_warmup_epochs = int(contrastive_warmup_epochs)
        if self.contrastive_loss_scale > 0.0:
            self.projection_head = ProjectionHead(
                in_dim=int(projection_in_dim),
                hidden_dim=int(projection_hidden_dim),
                out_dim=int(projection_dim),
            )
            self.contrastive_loss = ContrastiveLoss(
                temperature=float(contrastive_temperature), num_aug_per_sample=2
            )
```

Leave `save_hyperparameters(ignore=[...])` as-is — the new scalars are saved as hparams, which is fine.

- [ ] **Step 4: Modify `training_step`** — insert the contrastive block between the `self.critic(...)` call and `self.log_dict(loss)`:

```python
        loss = self.critic(
            graph_emb=graph_emb,
            graph_true=graph,
            graph_pred=graph_pred,
            soft_probs=soft_probs,
            perm=perm,
            beta=beta,
            kld_alpha=alpha,
            mu=mu,
            logvar=logvar,
        )
        if self.contrastive_loss_scale > 0.0:
            bs = graph.node_features.shape[0]
            feats = []
            for view in drop_views(graph, p=self.drop_p, n=2):
                _, z_global, _, _, _ = self.graph_ae.encode(view, sample=True)
                feats.append(self.projection_head(z_global))
            contrastive = self.contrastive_loss(torch.cat(feats, dim=0))
            warmup = min(1.0, (self.current_epoch + 1) / max(1, self.contrastive_warmup_epochs))
            eff_scale = self.contrastive_loss_scale * warmup
            loss["contrastive_loss"] = contrastive
            loss["loss"] = loss["loss"] + eff_scale * contrastive
            self.log("contrastive/scale", eff_scale, batch_size=bs)
        self.log_dict(loss)
```

- [ ] **Step 5: Add the kwargs to `configs/model/model.yaml`** — under the top-level `PLGraphAE` block (same level as `graph_ae:`, `critic:`), add:

```yaml
contrastive_loss_scale: 0.0
drop_p: 0.2
contrastive_temperature: 0.2
contrastive_warmup_epochs: 5
projection_in_dim: ${model.graph_ae.hparams.input_size}
projection_hidden_dim: 512
projection_dim: 128
```

(The pre-existing `critic.hparams.contrastive_loss_scale` is unused — the active knob is this top-level one.)

- [ ] **Step 6: Run the full feature test file + a compose check**

Run: `uv run pytest tests/test_contrastive_drop.py -v`
Expected: PASS (all tests).
Run: `uv run python src/train.py experiment=vae16_fb0p0_film paths=szary trainer=default --cfg job --resolve 2>&1 | grep -E "contrastive_loss_scale|projection_in_dim"`
Expected: `contrastive_loss_scale: 0.0`, `projection_in_dim: 512` — proves the plumbing resolves and default keeps contrastive OFF (existing runs unaffected).

- [ ] **Step 7: Commit**

```bash
git add src/models/pigvae_auto_module.py configs/model/model.yaml tests/test_contrastive_drop.py
git commit -m "feat(contrastive): wire drop-view NT-Xent into PLGraphAE training_step"
```

---

### Task 4: Experiment config + runnable verification

**Files:**
- Create: `configs/experiment/vae16_fb0p0_film_drop.yaml`

**Interfaces:**
- Consumes: the `PLGraphAE` contrastive kwargs (Task 3) and the FiLM decoder settings.
- Produces: a Hydra experiment `vae16_fb0p0_film_drop` runnable via `sbatch --export=ALL,EXPERIMENT=vae16_fb0p0_film_drop scripts/ablation_slurm.sh`.

- [ ] **Step 1: Create the config** — clone of `configs/experiment/vae16_fb0p0_film.yaml` with contrastive on and B=96:

```yaml
# @package _global_
# DROP-CONTRASTIVE, balanced+FiLM. Clone of vae16_fb0p0_film + drop-invariance contrastive
# loss on z_global. Approach A: clean forward -> recon+KL (unchanged); 2 node-dropout views
# -> projection head -> NT-Xent. Success = beat FiLM-z_global on the 4-way downstream sweep
# with no recon regression. See docs/superpowers/specs/2026-08-01-contrastive-drop-design.md.
#
# Run:
#   sbatch --time=12:00:00 --export=ALL,EXPERIMENT=vae16_fb0p0_film_drop scripts/ablation_slurm.sh

defaults:
  - override /data: mnist
  - override /model: model
  - override /trainer: default
  - _self_

tags: ["full_res", "grid16", "pernode_z", "vae", "low_kl", "free_bits", "fb_sweep", "vanilla_vae",
       "kl_anneal", "no_permuter", "qk_rmsnorm", "no_rope", "film", "film_bound",
       "contrastive", "drop", "z32", "pca128"]

seed: 42

trainer:
  max_epochs: 40
  min_epochs: 1

data:
  hparams:
    grid_size: 16
    size: 16
    center_crop_size: 16
    batch_size: 96          # memory-probed ceiling with headroom (3B encoder passes)

model:
  contrastive_loss_scale: 0.05   # primary knob; linear warmup over 5 ep (PLGraphAE default)
  drop_p: 0.2
  contrastive_temperature: 0.2
  graph_ae:
    hparams:
      vae: true
      node_z_dim: 32
      encoder:
        grid_size: 16
      decoder:
        grid_size: 16
        use_film: true
        film_bound: true
        film_weight_decay: 0.05
      permuter:
        grid_size: 16
  critic:
    hparams:
      kld_loss_scale: 0.01
      kld_free_bits: 0.0
  kld_alpha_scheduler:
    hparams:
      start_epoch: 5
      num_epochs: ${divide:${trainer.max_epochs},2}

logger:
  wandb:
    group: "full_res"
    tags: ${tags}
    name: "vae16-fb0.0-s0.01-FILM-DROP-p0.2-l0.05-40ep"
```

- [ ] **Step 2: Verify it composes and turns contrastive ON**

Run: `uv run python src/train.py experiment=vae16_fb0p0_film_drop paths=szary trainer=default --cfg job --resolve 2>&1 | grep -E "contrastive_loss_scale|drop_p|batch_size|use_film"`
Expected: `contrastive_loss_scale: 0.05`, `drop_p: 0.2`, `batch_size: 96`, `use_film: true`.

- [ ] **Step 3: Commit**

```bash
git add configs/experiment/vae16_fb0p0_film_drop.yaml
git commit -m "feat(experiment): vae16_fb0p0_film_drop (balanced+FiLM + drop-contrastive)"
```

---

## Launch (after all tasks pass — not a plan step)

```bash
sbatch --time=12:00:00 --export=ALL,EXPERIMENT=vae16_fb0p0_film_drop scripts/ablation_slurm.sh
```

Then the existing 4-way downstream sweep on the resulting checkpoint (new run_tag `vae16_fb0p0_film_drop_<date>`), and compare `zglobal` AUC against the FiLM bar (DX.name 0.928 / Grade 0.792 / Stage 0.660). Watch val_mse for recon drift and `contrastive/scale` for the warmup ramp.

## Self-Review

- **Spec coverage:** Approach-A step (Task 3), drop_views fixed p=0.20 (Task 1), projection head discarded at inference (Task 2), B=96 (Task 4), NT-Xent temp 0.2 (Task 3), λ=0.05 + warmup (Task 3/4), gated by `contrastive_loss_scale>0` not `is_contrastive` (Task 3 Step 3/4), base `vae16_fb0p0_film` from scratch (Task 4), downstream reuse unchanged (Launch), tests incl. degeneracy guard (Task 3 Step 1). All covered.
- **Placeholders:** none — every step has real code or a concrete command.
- **Type consistency:** `drop_views(batch, p, n, generator) -> list[DenseGraphBatch]`, `ProjectionHead(in_dim, hidden_dim, out_dim)`, `encode(...) -> (_, z_global, _, _, _)`, `ContrastiveLoss(temperature, num_aug_per_sample=2)` used identically across Tasks 1-4.
