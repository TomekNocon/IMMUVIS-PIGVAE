# z_global stats-vs-CLS decomposition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two read-only sibling views of `z_global` — `zglobal_cls` (learned CLS token) and `zglobal_stats` (pooled mean/var/max correction) — so the downstream sweep can attribute where the transfer signal lives, without touching the trained forward path.

**Architecture:** `z_global = LayerNorm(CLS + stats_correction(nodes))`. Task 1 exposes the two summands via an opt-in `return_parts` path in `GraphEncoder.forward` plus a thin `GraphAE.encode_zglobal_parts` wrapper that LayerNorms each. Task 2 wires two new `feature_source` values into the encode pipeline. Then a no-training sweep on the existing `mae_last`/`ctl_last` checkpoints produces the attribution table.

**Tech Stack:** PyTorch, Lightning, Hydra/OmegaConf, pytest, SLURM (szary), `uv`.

## Global Constraints

- Trained forward path (`GraphEncoder.forward` default, `GraphAE.encode`) MUST stay **bit-identical** — the decomposition is opt-in only.
- Decomposition is a **2-way CLS+stats split**; it requires `structural_correction is None` (true for these checkpoints: `use_hadamard`/`use_mlp_edges`/`use_spectrum` all False). Fail loudly otherwise.
- Each view is LayerNorm'd over the last dim (`F.layer_norm(t, t.shape[-1:])`), matching `GraphAE.encode`.
- No training. No new PCA components. Permuter dormant.
- Tests use a real `_build_pl_module("vae16_fb0p0_film")` instance (not mocks), per `tests/test_contrastive_drop.py`.
- Run tests with `uv run pytest`. SLURM jobs via sbatch (login node has no raid).

---

### Task 1: Expose `z_global`'s CLS and stats components

**Files:**
- Modify: `src/models/components/modules.py` — `GraphEncoder.forward` (ends ~line 386) and `GraphAE.encode` region (add method ~after line 242).
- Test: `tests/test_zglobal_decomp.py` (create).

**Interfaces:**
- Produces: `GraphEncoder.forward(node_features, edge_features, mask, return_parts=False)` — when `return_parts=True` returns `(graph_emb, node_features, cls, stats)` with `graph_emb == cls + stats`; when `False` returns `(graph_emb, node_features)` exactly as before.
- Produces: `GraphAE.encode_zglobal_parts(graph: DenseGraphBatch) -> dict[str, torch.Tensor]` with keys `"full"`, `"cls"`, `"stats"`, each `[B, D]` (D=512), LayerNorm'd. `"full"` is bit-identical to `encode()`'s `z_global`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_zglobal_decomp.py`:

```python
import pytest
import torch

from src.data.components.graphs_datamodules import DenseGraphBatch


def _gae_and_batch(B=2, N=256, D=128, seed=0):
    from src.downstream.encode import _build_pl_module
    torch.manual_seed(seed)
    gae = _build_pl_module("vae16_fb0p0_film").graph_ae.eval()
    batch = DenseGraphBatch(
        node_features=torch.randn(B, N, D),
        edge_features=torch.empty(0),
        mask=torch.ones(B, N, dtype=torch.bool),
    )
    return gae, batch


def test_cls_plus_stats_equals_full_graph_emb():
    gae, batch = _gae_and_batch()
    with torch.no_grad():
        full, _nodes, cls, stats = gae.encoder(
            node_features=batch.node_features,
            edge_features=batch.edge_features,
            mask=batch.mask,
            return_parts=True,
        )
    assert torch.allclose(cls + stats, full, atol=1e-6)


def test_full_view_matches_untouched_encode_zglobal():
    gae, batch = _gae_and_batch()
    with torch.no_grad():
        _, z_global, *_ = gae.encode(batch, sample=False)
        parts = gae.encode_zglobal_parts(batch)
    assert torch.allclose(parts["full"], z_global, atol=1e-6)


def test_parts_shape_and_finite_and_distinct():
    gae, batch = _gae_and_batch()
    with torch.no_grad():
        parts = gae.encode_zglobal_parts(batch)
    for k in ("full", "cls", "stats"):
        assert parts[k].shape == (2, 512)
        assert torch.isfinite(parts[k]).all()
    assert not torch.allclose(parts["cls"], parts["stats"], atol=1e-4)


def test_return_parts_raises_if_structural_correction_present():
    gae, batch = _gae_and_batch()
    gae.encoder.structural_correction = torch.nn.Identity()  # non-None sentinel
    with pytest.raises(ValueError):
        gae.encoder(
            node_features=batch.node_features,
            edge_features=batch.edge_features,
            mask=batch.mask,
            return_parts=True,
        )


def test_default_forward_unchanged_arity():
    gae, batch = _gae_and_batch()
    with torch.no_grad():
        out = gae.encoder(
            node_features=batch.node_features,
            edge_features=batch.edge_features,
            mask=batch.mask,
        )
    assert len(out) == 2  # (graph_emb, node_features)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_zglobal_decomp.py -v`
Expected: FAIL — `GraphEncoder.forward` has no `return_parts`; `encode_zglobal_parts` undefined.

- [ ] **Step 3: Add `return_parts` to `GraphEncoder.forward`**

In `src/models/components/modules.py`, change the `forward` signature to add `return_parts: bool = False`, and replace the tail (currently):

```python
        graph_emb, node_features = self.read_out_message_matrix(x)
        graph_emb = graph_emb + self.stats_correction(node_features, mask)
        if self.structural_correction is not None:
            graph_emb = graph_emb + self.structural_correction(node_features)
        return graph_emb, node_features
```

with (note the PMA branch sets `graph_emb` via `self.pma(...)`; this tail runs after both branches):

```python
        graph_emb, node_features = self.read_out_message_matrix(x)
        cls = graph_emb
        stats = self.stats_correction(node_features, mask)
        if return_parts and self.structural_correction is not None:
            raise ValueError(
                "return_parts=True requires structural_correction=None "
                "(the decomposition is a 2-way CLS+stats split)"
            )
        graph_emb = cls + stats
        if self.structural_correction is not None:
            graph_emb = graph_emb + self.structural_correction(node_features)
        if return_parts:
            return graph_emb, node_features, cls, stats
        return graph_emb, node_features
```

Note: the PMA path (`use_pma=True`) sets `graph_emb` earlier; `read_out_message_matrix` is the CLS path. Only the CLS-mode encoder (`use_pma=False`, our checkpoints) is decomposed here — the tail is shared, so `cls` is whichever readout produced `graph_emb`. That is correct for our models (CLS mode).

- [ ] **Step 4: Add `encode_zglobal_parts` to `GraphAE`**

In `src/models/components/modules.py`, immediately after `GraphAE.encode` (which ends `return z_nodes, z_global, node_features, mu, logvar`), add:

```python
    def encode_zglobal_parts(self, graph: DenseGraphBatch) -> dict[str, torch.Tensor]:
        """Decompose z_global into its CLS and stats-correction views (read-only diagnostic).

        Returns LayerNorm'd views {"full", "cls", "stats"}, each [B, D]. The trained
        encode() path is untouched; "full" is bit-identical to encode()'s z_global.
        """
        graph_emb, _node_features, cls, stats = self.encoder(
            node_features=graph.node_features,
            edge_features=graph.edge_features,
            mask=graph.mask,
            return_parts=True,
        )
        def _ln(t: torch.Tensor) -> torch.Tensor:
            return F.layer_norm(t, t.shape[-1:])
        return {"full": _ln(graph_emb), "cls": _ln(cls), "stats": _ln(stats)}
```

(`F` is already `import torch.nn.functional as F` in this module — it is used by `encode`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_zglobal_decomp.py -v`
Expected: PASS (all 5).

- [ ] **Step 6: Run the existing suite to confirm no regression**

Run: `uv run pytest tests/test_contrastive_drop.py tests/test_block_drop.py -q`
Expected: PASS (unchanged — the default forward path is byte-identical).

- [ ] **Step 7: Commit**

```bash
git add src/models/components/modules.py tests/test_zglobal_decomp.py
git commit -m "feat(encoder): return_parts + encode_zglobal_parts (CLS/stats decomposition)"
```

---

### Task 2: Wire `zglobal_cls` / `zglobal_stats` into the encode pipeline

**Files:**
- Modify: `src/downstream/encode.py` — `source_subdir` (~line 90) and `encode_patches` (~line 176 dispatch).
- Test: `tests/test_zglobal_decomp.py` (append).

**Interfaces:**
- Consumes: `GraphAE.encode_zglobal_parts` from Task 1.
- Produces: `source_subdir("zglobal_cls") == "zglobal_cls"`, `source_subdir("zglobal_stats") == "zglobal_stats"`; `encode_patches(..., feature_source="zglobal_cls"|"zglobal_stats")` returns `(B, 512)` numpy.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_zglobal_decomp.py`:

```python
def test_source_subdir_maps_decomposition_views():
    from src.downstream.encode import source_subdir
    assert source_subdir("zglobal_cls") == "zglobal_cls"
    assert source_subdir("zglobal_stats") == "zglobal_stats"


def test_encode_patches_cls_stats_shapes_and_distinct():
    import numpy as np
    from src.downstream.encode import encode_patches
    gae, _ = _gae_and_batch()
    pca = lambda x: x[..., :128]  # [B,256,768] -> [B,256,128] stub
    patches = np.random.randn(2, 768, 16, 16).astype("float32")
    cls = encode_patches(gae, pca, patches, "cpu", feature_source="zglobal_cls")
    stats = encode_patches(gae, pca, patches, "cpu", feature_source="zglobal_stats")
    full = encode_patches(gae, pca, patches, "cpu", feature_source="zglobal")
    assert cls.shape == (2, 512) and stats.shape == (2, 512)
    assert np.isfinite(cls).all() and np.isfinite(stats).all()
    assert not np.allclose(cls, stats, atol=1e-4)
    assert not np.allclose(cls, full, atol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_zglobal_decomp.py -k "source_subdir or encode_patches" -v`
Expected: FAIL — `source_subdir` raises `ValueError` for the new names; `encode_patches` raises for the new `feature_source`.

- [ ] **Step 3: Extend `source_subdir`**

In `src/downstream/encode.py`, in `source_subdir`, add before the final `raise`:

```python
    if feature_source in ("zglobal_cls", "zglobal_stats"):
        return feature_source
```

- [ ] **Step 4: Extend `encode_patches` dispatch**

In `src/downstream/encode.py`, in the dispatch block (after `if feature_source == "zglobal": out = z_global`), add a branch. The `z_nodes, z_global, *_ = gae.encode(...)` call above is left as-is (used by `zglobal`/`node`); the new sources call `encode_zglobal_parts` on the same `batch`:

```python
    elif feature_source in ("zglobal_cls", "zglobal_stats"):
        parts = gae.encode_zglobal_parts(batch)
        out = parts["cls"] if feature_source == "zglobal_cls" else parts["stats"]
```

(A second encoder forward for these two diagnostic sources is acceptable — this path is a one-off sweep, and keeping the existing `zglobal`/`node` code untouched preserves their behavior exactly.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_zglobal_decomp.py -v`
Expected: PASS (all 7).

- [ ] **Step 6: Commit**

```bash
git add src/downstream/encode.py tests/test_zglobal_decomp.py
git commit -m "feat(downstream): zglobal_cls / zglobal_stats feature sources"
```

---

## Post-implementation ops (controller-run, not TDD tasks)

These run on szary via sbatch (login node has no raid). Not part of the task/review loop.

1. **CPU smoke gate** (before spending the sweep slot): a tiny sbatch that composes `experiment=vae16_fb0p0_film_mae`, loads one real batch, and asserts `encode_patches(..., "zglobal_cls")` and `"zglobal_stats"` return `(B, 512)` finite arrays on real data. Mirror the drop-MAE shape-probe harness (rootutils + multiply/divide resolvers + `wandb.init(mode="disabled")`).

2. **Decomposition sweep** (no training): for each checkpoint tag `mae_last` (`last-v1.ckpt`) and `ctl_last` (`last.ckpt`) under `/raid_encrypted/.../logs/train/runs/2026-08-06_21-22-30/checkpoints`, run `encode_mil_embeddings.py` + `build_abmil_meta.py` + `run_abmil.py` for `source=zglobal_cls` and `zglobal_stats` (model_experiment matches the tag). `raw` + `zglobal` (full) already exist from the prior sweep — reuse them. 512-d sources ⇒ no OOM.

3. **Collect** the attribution table per checkpoint: **raw / stats_only / cls_only / full** × {DX.name, Grade, Stage, Relapse}, AUC. Apply the interpretation criterion from the spec (stats_only ≈ full ⇒ pooled statistics dominate; cls_only ≫ stats_only ⇒ learned CLS carries the signal).

---

## Self-Review

- **Spec coverage:** two new views (Task 1 method + Task 2 wiring) ✓; bit-identical trained path (Step 6 regression + `test_full_view_matches_untouched_encode_zglobal`) ✓; structural guard (test) ✓; LayerNorm-per-view (`encode_zglobal_parts`) ✓; run on mae_last/ctl_last (ops 2) ✓; attribution table + criterion (ops 3) ✓; no training ✓.
- **Placeholder scan:** none — all steps carry real code and exact commands.
- **Type consistency:** `encode_zglobal_parts` returns `dict[str,Tensor]` with keys `full/cls/stats`, consumed identically in Task 2; `return_parts` 4-tuple matches wrapper unpacking.
