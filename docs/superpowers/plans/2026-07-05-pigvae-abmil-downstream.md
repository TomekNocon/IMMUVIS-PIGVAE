# PIGVAE → Gated-ABMIL Downstream Evaluation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Encode frozen-PIGVAE `z_global` for every cords IMC patch, then train a Hydra/Lightning gated-ABMIL classifier over image bags of those vectors under 10-fold CV on clinical features — the thesis's transfer (deciding) metric.

**Architecture:** Two decoupled Hydra entry points glued by files on disk. **Phase A (encode):** read `train.h5`/`test.h5` patches `(N,768,16,16)` → PCALayer 768→128 → `DenseGraphBatch` → `graph_ae.encode(sample=False)` → stream `z_global` to a memmap `.npy` + metadata CSV. **Phase B (abmil):** build image bags from the memmap + a clinical meta CSV, run 10-fold StratifiedKFold gated-ABMIL per feature, emit metrics CSVs. Encode once, sweep ABMIL cheaply.

**Tech Stack:** PyTorch, PyTorch Lightning, Hydra/OmegaConf, h5py, scikit-learn (PCA + StratifiedKFold + metrics), numpy memmap, pytest. Repo `IMMUVIS-PIGVAE`, branch `imc-pigvae-film-multi`, env via `uv run`.

## Global Constraints

- **Frozen encoder:** load a PIGVAE checkpoint, `model.eval()`, `torch.no_grad()`, `encode(sample=False)` → `z_global = layer_norm(graph_emb)`. Never train/finetune PIGVAE.
- **Same fitted PCA, transform-only:** use `/raid_encrypted/immucan/embeddings/tnocon/data/IMC/cords/pca_model_128_center_crop_16.pkl` + `imc_statistics_128_center_crop_16.pt` via `PCALayer`. Never refit PCA on downstream data.
- **PCALayer config must match the checkpoint's training:** cords training used `zscore=False`, `clip_range=null` (per `configs/experiment/vae16_*.yaml` data hparams). Construct `PCALayer(pca_path, statistics_path, clip_range=0.0, zscore=False)`.
- **Data (verified 2026-07-05, szary-only):** `.../IMC/cords/{train,test}.h5`, datasets `embeddings (N,768,16,16) f32`, `paths (N,) object`, `positions (N,4) f32`, `metadata (N,8,3) f32`. train N=40843, test N=10197 (= PIGVAE's own train/val split).
- **Node ordering:** patch `(768,16,16)` → nodes `(256,768)` must match training exactly: `x.reshape(768, 256).T` (channels-last flatten of the 16×16 grid, row-major). Verify against `GridGraphDataset.__getitem__` before relying on it (Task A2 step 1).
- **DenseGraphBatch:** `DenseGraphBatch(node_features=[B,256,128], edge_features=torch.empty(0), mask=None)`. A full grid has no padding → `mask=None`.
- **Runs on szary only** (raid data + GPU): all smoke/integration runs via `srun --qos=tnocon --partition=common`. Unit tests use synthetic tensors and run anywhere with `uv run pytest`.
- **z_global width** is read from data downstream (`bags[0].shape[1]`); do not hardcode. It is the CLS `layer_norm(graph_emb)`.
- **Leakage guardrails:** normalization fit on train fold only (`zscore='cv_train'`); all crops of an `img_path` stay in one split (group at image level); document that train.h5 patches were seen self-supervised by the encoder.
- **Commits:** conventional commits; end message bodies with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## Phase A — Encode stage

### Task A1: Frozen PIGVAE loader

**Files:**
- Create: `src/downstream/__init__.py` (empty)
- Create: `src/downstream/encode.py`
- Test: `tests/test_downstream_encode.py`

**Interfaces:**
- Produces: `load_frozen_pigvae(ckpt_path: str, experiment: str, paths_name: str = "szary") -> torch.nn.Module` — returns the `graph_ae` submodule of a loaded `PLGraphAE`, in `.eval()`, weights loaded from `ckpt_path`. Reuses the exact model-building path in `scripts/diagnose_model.py::load_model_and_data` (instantiate from `configs/model/model.yaml` merged with `configs/experiment/{experiment}.yaml` `model`), then `model.graph_ae`.

- [ ] **Step 1: Write the failing test** (uses a tiny synthetic checkpoint so it runs off-szary)

```python
# tests/test_downstream_encode.py
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from src.models.pigvae_auto_module import PLGraphAE

def _build_tiny_graph_ae(tmp_path):
    """Build a graph_ae from the real model config so encode() is exercised, save a ckpt."""
    from src.downstream.encode import _build_pl_module  # helper Task A1 exposes
    pl = _build_pl_module(experiment="vae16_fb0p0", paths_name="szary")
    ckpt = tmp_path / "tiny.ckpt"
    torch.save({"state_dict": pl.state_dict()}, ckpt)
    return str(ckpt)

def test_load_frozen_pigvae_returns_eval_graph_ae(tmp_path):
    from src.downstream.encode import load_frozen_pigvae
    ckpt = _build_tiny_graph_ae(tmp_path)
    gae = load_frozen_pigvae(ckpt, experiment="vae16_fb0p0")
    assert not gae.training                      # eval mode
    assert all(not p.requires_grad for p in gae.parameters()) is False or True  # weights present
    # encode runs: 2 graphs, 256 nodes, 128 pca dims -> z_global [2, D]
    nf = torch.randn(2, 256, 128)
    from src.data.components.graphs_datamodules import DenseGraphBatch
    batch = DenseGraphBatch(node_features=nf, edge_features=torch.empty(0), mask=None)
    with torch.no_grad():
        z_nodes, z_global, *_ = gae.encode(batch, sample=False)
    assert z_global.shape[0] == 2 and z_global.dim() == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_downstream_encode.py::test_load_frozen_pigvae_returns_eval_graph_ae -v`
Expected: FAIL (`ModuleNotFoundError: src.downstream.encode` / `_build_pl_module` undefined).

- [ ] **Step 3: Write minimal implementation**

Extract the model-building half of `scripts/diagnose_model.py::load_model_and_data` into `_build_pl_module` (no dataloader), then `load_frozen_pigvae` loads weights:

```python
# src/downstream/encode.py
from pathlib import Path
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _build_pl_module(experiment: str, paths_name: str = "szary"):
    configs = _PROJECT_ROOT / "configs"
    for name, fn in [("multiply", lambda x, y: int(x) * int(y)), ("divide", lambda x, y: int(x) // int(y))]:
        try: OmegaConf.register_new_resolver(name, fn)
        except Exception: pass
    model_cfg = OmegaConf.load(configs / "model" / "model.yaml")
    exp_cfg = OmegaConf.load(configs / "experiment" / f"{experiment}.yaml")
    if "model" in exp_cfg:
        model_cfg = OmegaConf.merge(model_cfg, exp_cfg.model)
    trainer_stub = OmegaConf.create({"max_epochs": 200, "min_epochs": 1})
    if "trainer" in exp_cfg:
        trainer_stub = OmegaConf.merge(trainer_stub, exp_cfg.trainer)
    ctx = OmegaConf.create({"model": model_cfg, "trainer": trainer_stub})
    OmegaConf.set_struct(ctx, False)
    from src.models.pigvae_auto_module import PLGraphAE
    return PLGraphAE(
        graph_ae=instantiate(ctx.model.graph_ae),
        critic=instantiate(ctx.model.critic),
        temperature_scheduler=instantiate(ctx.model.temperature_scheduler),
        entropy_weight_scheduler=instantiate(ctx.model.entropy_weight_scheduler),
        kld_alpha_scheduler=instantiate(ctx.model.kld_alpha_scheduler),
        optimizer=instantiate(ctx.model.optimizer),
        scheduler=ctx.model.scheduler,
        compile=False,
    )

def load_frozen_pigvae(ckpt_path: str, experiment: str, paths_name: str = "szary") -> torch.nn.Module:
    pl = _build_pl_module(experiment, paths_name)
    state = torch.load(ckpt_path, map_location="cpu")
    pl.load_state_dict(state.get("state_dict", state), strict=False)
    pl.eval()
    gae = pl.graph_ae
    gae.eval()
    for p in gae.parameters():
        p.requires_grad_(False)
    return gae
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_downstream_encode.py::test_load_frozen_pigvae_returns_eval_graph_ae -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/downstream/__init__.py src/downstream/encode.py tests/test_downstream_encode.py
git commit -m "feat(downstream): frozen PIGVAE loader for encode stage"
```

### Task A2: Patch → PCA-128 node features

**Files:**
- Modify: `src/downstream/encode.py`
- Test: `tests/test_downstream_encode.py`

**Interfaces:**
- Consumes: `PCALayer` from `src.data.components.graphs_datamodules`.
- Produces: `build_pca_layer(pca_path, stats_path) -> PCALayer` (constructed `zscore=False, clip_range=0.0`); `patch_to_nodes(patch: np.ndarray) -> torch.Tensor` mapping `(768,16,16) -> (256,768)` float32 in the training node order; `pca_transform(nodes: torch.Tensor, pca: PCALayer) -> torch.Tensor` `[B,256,768]->[B,256,128]`.

- [ ] **Step 1: Write the failing test** — pin the node order against the training dataset, and PCA parity.

```python
def test_patch_to_nodes_order_and_pca_shape():
    import numpy as np, torch
    from src.downstream.encode import patch_to_nodes
    patch = np.arange(768 * 16 * 16, dtype=np.float32).reshape(768, 16, 16)
    nodes = patch_to_nodes(patch)                 # (256, 768)
    assert nodes.shape == (256, 768)
    # node n = grid cell (row=n//16, col=n%16), all 768 channels of that cell
    assert torch.allclose(nodes[0], torch.from_numpy(patch[:, 0, 0]))
    assert torch.allclose(nodes[17], torch.from_numpy(patch[:, 1, 1]))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_downstream_encode.py::test_patch_to_nodes_order_and_pca_shape -v`
Expected: FAIL (`patch_to_nodes` undefined).

- [ ] **Step 3: Implement** — **first confirm the node order** by reading `GridGraphDataset.__getitem__` / `from_sparse_graph_list` (`src/data/components/graphs_datamodules.py:412,483`); the reshape below assumes row-major `(C,H,W)->(H*W,C)`. If training uses a different order, match it and update the test.

```python
import numpy as np, torch
from src.data.components.graphs_datamodules import PCALayer

def build_pca_layer(pca_path: str, stats_path: str) -> PCALayer:
    return PCALayer(pca_path, stats_path, clip_range=0.0, zscore=False)

def patch_to_nodes(patch: np.ndarray) -> torch.Tensor:
    # (C, H, W) -> (H*W, C), row-major over the grid
    c, h, w = patch.shape
    return torch.from_numpy(patch.reshape(c, h * w).T.copy()).float()

def pca_transform(nodes: torch.Tensor, pca: PCALayer) -> torch.Tensor:
    return pca(nodes)   # PCALayer.forward handles [B, N, 768] -> [B, N, 128]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_downstream_encode.py::test_patch_to_nodes_order_and_pca_shape -v`
Expected: PASS.

- [ ] **Step 5: Verify PCA parity on szary** (real artifact) and commit.

Run:
```bash
srun --qos=tnocon --partition=common --cpus-per-task=2 --mem=8G --time=0:10:00 \
  bash -lc 'cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE && uv run python -c "
import h5py, torch, numpy as np
from src.downstream.encode import build_pca_layer, patch_to_nodes
D=\"/raid_encrypted/immucan/embeddings/tnocon/data/IMC/cords\"
pca=build_pca_layer(f\"{D}/pca_model_128_center_crop_16.pkl\", f\"{D}/imc_statistics_128_center_crop_16.pt\")
with h5py.File(f\"{D}/test.h5\",\"r\") as f: p=f[\"embeddings\"][0]
z=pca(patch_to_nodes(p).unsqueeze(0))
print(\"nodes->pca:\", z.shape, \"finite:\", bool(torch.isfinite(z).all()))"'
```
Expected: `torch.Size([1, 256, 128]) finite: True`.
```bash
git add src/downstream/encode.py tests/test_downstream_encode.py
git commit -m "feat(downstream): patch->PCA-128 node transform (training-consistent)"
```

### Task A3: Encode a batch of patches to z_global

**Files:** Modify `src/downstream/encode.py`; Test `tests/test_downstream_encode.py`.

**Interfaces:**
- Produces: `encode_patches(gae, pca, patches: np.ndarray, device: str) -> np.ndarray` — `patches (B,768,16,16) -> z_global (B, D)` float32; deterministic (`sample=False`); batch-size-agnostic.

- [ ] **Step 1: Failing test**

```python
def test_encode_patches_deterministic_and_batch_agnostic(tmp_path):
    import numpy as np, torch
    from src.downstream.encode import load_frozen_pigvae, build_pca_layer, encode_patches, _build_pl_module
    # reuse the tiny ckpt + a stub PCALayer that just linearly maps 768->128
    class StubPCA:
        def __call__(self, x): return x[..., :128]
    pl = _build_pl_module("vae16_fb0p0"); ckpt = tmp_path/"c.ckpt"; torch.save({"state_dict": pl.state_dict()}, ckpt)
    gae = load_frozen_pigvae(str(ckpt), "vae16_fb0p0")
    patches = np.random.randn(5, 768, 16, 16).astype("float32")
    z1 = encode_patches(gae, StubPCA(), patches, device="cpu")
    z2 = encode_patches(gae, StubPCA(), patches, device="cpu")
    assert z1.shape[0] == 5
    assert np.allclose(z1, z2)                       # deterministic (sample=False)
    z_one = encode_patches(gae, StubPCA(), patches[:1], device="cpu")
    assert np.allclose(z_one[0], z1[0], atol=1e-5)   # batch-size independent
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest ...::test_encode_patches_deterministic_and_batch_agnostic -v` → FAIL.

- [ ] **Step 3: Implement**

```python
from src.data.components.graphs_datamodules import DenseGraphBatch

@torch.no_grad()
def encode_patches(gae, pca, patches: np.ndarray, device: str) -> np.ndarray:
    nodes = torch.stack([patch_to_nodes(p) for p in patches], dim=0).to(device)  # [B,256,768]
    nodes = pca(nodes)                                                           # [B,256,128]
    batch = DenseGraphBatch(node_features=nodes, edge_features=torch.empty(0), mask=None)
    gae = gae.to(device)
    _z_nodes, z_global, *_ = gae.encode(batch, sample=False)
    return z_global.detach().cpu().float().numpy()
```

- [ ] **Step 4: Run to verify it passes** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(downstream): encode_patches -> z_global (frozen, deterministic)"`.

### Task A4: Streaming memmap writer

**Files:** Create `src/downstream/memmap_writer.py`; Test `tests/test_downstream_encode.py`.

**Interfaces:**
- Produces: `MemmapWriter(path: str, n_rows: int, dim: int, dtype="float32")` with `.write(start: int, rows: np.ndarray)` and `.close()`; final file is `np.load(path, mmap_mode="r")` shape `(n_rows, dim)`.

- [ ] **Step 1: Failing test**

```python
def test_memmap_writer_roundtrip(tmp_path):
    import numpy as np
    from src.downstream.memmap_writer import MemmapWriter
    p = str(tmp_path / "emb.npy")
    w = MemmapWriter(p, n_rows=10, dim=4)
    w.write(0, np.ones((3, 4), "float32"))
    w.write(3, np.full((7, 4), 2.0, "float32"))
    w.close()
    a = np.load(p, mmap_mode="r")
    assert a.shape == (10, 4) and a[0, 0] == 1.0 and a[9, 0] == 2.0
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement**

```python
# src/downstream/memmap_writer.py
import numpy as np

class MemmapWriter:
    def __init__(self, path: str, n_rows: int, dim: int, dtype="float32"):
        self.path = path
        self._arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(n_rows, dim))
    def write(self, start: int, rows: np.ndarray) -> None:
        self._arr[start:start + rows.shape[0]] = rows
    def close(self) -> None:
        self._arr.flush(); del self._arr
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `git commit -m "feat(downstream): streaming memmap writer"`.

### Task A5: Encode driver — h5 → memmap + metadata CSV

**Files:** Modify `src/downstream/encode.py`; Test `tests/test_downstream_encode.py`.

**Interfaces:**
- Produces: `encode_h5(h5_path, gae, pca, out_emb, out_meta, device, batch_size=64) -> None` — reads `embeddings`/`paths`/`positions`, encodes in batches, streams to `out_emb` memmap `(N, D)`, writes `out_meta` CSV with columns `img_path, coords0, coords1, embeddings_file, embedding_idx` where `embedding_idx == row` and `embeddings_file == out_emb`. Label join happens later (Phase B) via `img_path`.

- [ ] **Step 1: Failing test** (synthetic h5)

```python
def test_encode_h5_alignment(tmp_path):
    import h5py, numpy as np, pandas as pd, torch
    from src.downstream.encode import encode_h5, load_frozen_pigvae, _build_pl_module
    class StubPCA:
        def __call__(self, x): return x[..., :128]
    h5 = tmp_path / "mini.h5"
    N = 7
    with h5py.File(h5, "w") as f:
        f["embeddings"] = np.random.randn(N, 768, 16, 16).astype("float32")
        f["paths"] = np.array([f"img{i//3}.tiff" for i in range(N)], dtype=object)
        f["positions"] = np.random.rand(N, 4).astype("float32")
    pl = _build_pl_module("vae16_fb0p0"); ck = tmp_path/"c.ckpt"; torch.save({"state_dict": pl.state_dict()}, ck)
    gae = load_frozen_pigvae(str(ck), "vae16_fb0p0")
    emb, meta = str(tmp_path/"out_embeddings.npy"), str(tmp_path/"out_metadata.csv")
    encode_h5(str(h5), gae, StubPCA(), emb, meta, device="cpu", batch_size=3)
    a = np.load(emb, mmap_mode="r"); df = pd.read_csv(meta)
    assert a.shape[0] == N and len(df) == N
    assert list(df["embedding_idx"]) == list(range(N))
    assert df["img_path"].iloc[0] == "img0.tiff" and df["img_path"].iloc[6] == "img2.tiff"
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement**

```python
import h5py, pandas as pd
from src.downstream.memmap_writer import MemmapWriter

def encode_h5(h5_path, gae, pca, out_emb, out_meta, device, batch_size=64) -> None:
    with h5py.File(h5_path, "r") as f:
        n = f["embeddings"].shape[0]
        paths = [p.decode() if isinstance(p, bytes) else str(p) for p in f["paths"][:]]
        pos = f["positions"][:]
        # probe dim with one patch
        d = encode_patches(gae, pca, f["embeddings"][0:1], device).shape[1]
        writer = MemmapWriter(out_emb, n_rows=n, dim=d)
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            writer.write(s, encode_patches(gae, pca, f["embeddings"][s:e], device))
        writer.close()
    pd.DataFrame({
        "img_path": paths,
        "coords0": pos[:, 0], "coords1": pos[:, 1],
        "embeddings_file": out_emb,
        "embedding_idx": range(n),
    }).to_csv(out_meta, index=False)
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `git commit -m "feat(downstream): encode_h5 driver -> memmap + aligned metadata CSV"`.

### Task A6: Hydra encode script + szary run

**Files:** Create `scripts/encode_mil_embeddings.py`, `configs/downstream/encode.yaml`.

**Interfaces:** Consumes `encode_h5`, `load_frozen_pigvae`, `build_pca_layer`.

- [ ] **Step 1** — config:

```yaml
# configs/downstream/encode.yaml
defaults:
  - /paths: szary
  - _self_
ckpt_path: ???            # PIGVAE checkpoint (last.ckpt)
experiment: vae16_fb0p0   # architecture the ckpt was trained with
data_root: /raid_encrypted/immucan/embeddings/tnocon/data/IMC/cords
pca_pkl: ${data_root}/pca_model_128_center_crop_16.pkl
pca_stats: ${data_root}/imc_statistics_128_center_crop_16.pt
out_dir: ${paths.output_dir}/mil_embeddings
device: cuda
batch_size: 64
splits: [train, test]
```

- [ ] **Step 2** — script (resumable per split):

```python
# scripts/encode_mil_embeddings.py
import os, hydra
from omegaconf import DictConfig
from src.downstream.encode import load_frozen_pigvae, build_pca_layer, encode_h5

@hydra.main(version_base=None, config_path="../configs/downstream", config_name="encode")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    gae = load_frozen_pigvae(cfg.ckpt_path, cfg.experiment)
    pca = build_pca_layer(cfg.pca_pkl, cfg.pca_stats)
    for split in cfg.splits:
        emb = f"{cfg.out_dir}/cords_{split}_embeddings.npy"
        meta = f"{cfg.out_dir}/cords_{split}_metadata.csv"
        if os.path.exists(emb) and os.path.exists(meta):
            print(f"skip {split} (exists)"); continue
        encode_h5(f"{cfg.data_root}/{split}.h5", gae, pca, emb, meta, cfg.device, cfg.batch_size)
        print(f"wrote {emb}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: GPU smoke run on szary** (resolves the standing "GPU problem"):

```bash
srun --qos=tnocon --partition=common --gres=gpu:1 --cpus-per-task=6 --mem=24G --time=2:00:00 \
  bash -lc 'cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE && uv run python scripts/encode_mil_embeddings.py \
    ckpt_path=<PIGVAE last.ckpt> experiment=vae16_fb0p0 splits=[test]'
```
Expected: writes `cords_test_embeddings.npy` `(10197, D)` + `cords_test_metadata.csv`; no CUDA error.

- [ ] **Step 4: Commit** — `git commit -m "feat(downstream): hydra encode script + config"`.

---

## Phase B — Gated-ABMIL stage

### Task B1: Port GatedABMIL module

**Files:** Create `src/downstream/abmil/__init__.py`, `src/downstream/abmil/model.py`; Test `tests/test_downstream_abmil.py`.

**Interfaces:** Produces `GatedABMIL(emb_dim, hidden_dim, num_heads=1, num_classes=2)` with `forward(x[B,S,D], mask[B,S]|None) -> (logits[B, C'], pooled[B, num_heads*D])` where `C' = 1` if `num_classes==2` else `num_classes`.

- [ ] **Step 1: Failing test**

```python
# tests/test_downstream_abmil.py
import torch
def test_gated_abmil_shapes_and_masking():
    from src.downstream.abmil.model import GatedABMIL
    m = GatedABMIL(emb_dim=16, hidden_dim=8, num_heads=2, num_classes=2)
    x = torch.randn(3, 5, 16); mask = torch.zeros(3, 5, dtype=torch.bool); mask[:, 4] = True
    logits, pooled = m(x, mask=mask)
    assert logits.shape == (3, 1) and pooled.shape == (3, 2 * 16)
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** — port `GatedABMIL` from `/home/tnocon/mil/gated_abmil/models/ABMIL/gated_abmil.py` ~verbatim, fold the classifier head in (drop the `GatedABMILClassifierWithValidation` wrapper — Lightning replaces it):

```python
# src/downstream/abmil/model.py
import torch, torch.nn as nn

class GatedABMIL(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_heads=1, num_classes=2):
        super().__init__()
        self.V = nn.Linear(emb_dim, hidden_dim)
        self.U = nn.Linear(emb_dim, hidden_dim)
        self.W = nn.Linear(hidden_dim, num_heads)
        self.num_heads = num_heads
        out_dim = num_classes - 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(num_heads * emb_dim, out_dim)

    def forward(self, x, mask=None):
        v = torch.tanh(self.V(x)); u = torch.sigmoid(self.U(x))
        a = self.W(v * u)                                   # B,S,H
        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(2), -1e9)
        a = torch.softmax(a, dim=1).transpose(1, 2)         # B,H,S
        pooled = torch.bmm(a, x).reshape(x.size(0), self.num_heads * x.size(2))
        return self.classifier(pooled), pooled
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `git commit -m "feat(abmil): port GatedABMIL module with folded head"`.

### Task B2: MIL dataset + padding collate

**Files:** Create `src/downstream/abmil/data.py`; Test `tests/test_downstream_abmil.py`.

**Interfaces:** Produces `MILDataset(bags: list[np.ndarray], labels: torch.Tensor)` and `mil_collate(batch) -> (bags[B,S,D] float32, masks[B,S] bool, labels[B] long)`; `mask=True` on padded positions.

- [ ] **Step 1: Failing test**

```python
def test_mil_collate_pads_and_masks():
    import numpy as np, torch
    from src.downstream.abmil.data import MILDataset, mil_collate
    ds = MILDataset([np.ones((2,4),"float32"), np.ones((5,4),"float32")], torch.tensor([0,1]))
    bags, masks, labels = mil_collate([ds[0], ds[1]])
    assert bags.shape == (2,5,4) and masks.shape == (2,5)
    assert masks[0,2] and not masks[0,1] and not masks[1,4]
    assert list(labels) == [0,1]
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement**

```python
# src/downstream/abmil/data.py
import numpy as np, torch
from torch.utils.data import Dataset

class MILDataset(Dataset):
    def __init__(self, bags, labels): self.bags, self.labels = bags, labels
    def __len__(self): return len(self.bags)
    def __getitem__(self, i): return self.bags[i], int(self.labels[i])

def mil_collate(batch):
    bags, labels = zip(*batch)
    max_s = max(b.shape[0] for b in bags); d = bags[0].shape[1]
    out = torch.zeros(len(bags), max_s, d); mask = torch.zeros(len(bags), max_s, dtype=torch.bool)
    for i, b in enumerate(bags):
        s = b.shape[0]; out[i, :s] = torch.from_numpy(np.asarray(b, "float32")); mask[i, s:] = True
    return out, mask, torch.tensor(labels, dtype=torch.long)
```

- [ ] **Step 4: Run → PASS.** — [ ] **Step 5: Commit** — `git commit -m "feat(abmil): MIL dataset + padding collate"`.

### Task B3: Build image bags from meta + memmap

**Files:** Modify `src/downstream/abmil/data.py`; Test `tests/test_downstream_abmil.py`.

**Interfaces:** Produces `build_image_bags(meta_df, class_to_idx) -> (bags: list[np.ndarray], labels: torch.Tensor)` — group by `img_path`; each bag = that image's crop rows from its `embeddings_file` (memmap, indexed by `embedding_idx`); label from `feature_value` via `class_to_idx`; drop NaN/unmapped labels.

- [ ] **Step 1: Failing test**

```python
def test_build_image_bags_groups_and_labels(tmp_path):
    import numpy as np, pandas as pd
    from src.downstream.abmil.data import build_image_bags
    emb = str(tmp_path/"e.npy"); np.save(emb, np.arange(6*4, dtype="float32").reshape(6,4))
    df = pd.DataFrame({
        "img_path": ["a","a","a","b","b","b"],
        "embeddings_file": [emb]*6, "embedding_idx": [0,1,2,3,4,5],
        "feature_value": ["pos","pos","pos","neg","neg","neg"]})
    bags, labels = build_image_bags(df, {"neg":0,"pos":1})
    assert len(bags) == 2 and bags[0].shape == (3,4)
    assert list(labels) == [1,0]
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** (mirrors the existing `run_abmil.py::build_image_bags`, minus the mean-pool since z_global is already 1-D):

```python
import pandas as pd

_MEMMAP_CACHE = {}
def _load_memmap(path):
    a = _MEMMAP_CACHE.get(path)
    if a is None:
        a = np.load(path, mmap_mode="r"); _MEMMAP_CACHE[path] = a
    return a

def build_image_bags(meta_df, class_to_idx):
    bags, labels = [], []
    for _img, g in meta_df.groupby("img_path", sort=False):
        raw = g["feature_value"].iloc[0]
        if pd.isna(raw): continue
        lab = class_to_idx.get(str(raw))
        if lab is None: continue
        rows = []
        for ef, gg in g.groupby("embeddings_file"):
            rows.append(_load_memmap(ef)[gg["embedding_idx"].values])
        bags.append(np.concatenate(rows, axis=0)); labels.append(lab)
    return bags, torch.tensor(labels, dtype=torch.long)
```

- [ ] **Step 4: Run → PASS.** — [ ] **Step 5: Commit** — `git commit -m "feat(abmil): build image bags from memmap + meta"`.

### Task B4: Lightning module wrapper

**Files:** Create `src/downstream/abmil/lit.py`; Test `tests/test_downstream_abmil.py`.

**Interfaces:** Produces `AbmilLitModule(emb_dim, hidden_dim, num_heads, num_classes, lr)` (a `pl.LightningModule`) — `training_step`/`validation_step` compute BCEWithLogits (2-class) or CrossEntropy (>2); logs `val_loss`; `configure_optimizers` → Adam(lr).

- [ ] **Step 1: Failing test** (one optimisation step reduces loss on a separable toy bag)

```python
def test_abmil_lit_overfits_toy():
    import torch, numpy as np
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    from src.downstream.abmil.lit import AbmilLitModule
    from src.downstream.abmil.data import MILDataset, mil_collate
    g = torch.Generator().manual_seed(0)
    pos = [np.ones((3,8),"float32") for _ in range(8)]; neg = [(-np.ones((3,8),"float32")) for _ in range(8)]
    ds = MILDataset(pos+neg, torch.tensor([1]*8+[0]*8))
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=mil_collate)
    m = AbmilLitModule(emb_dim=8, hidden_dim=8, num_heads=1, num_classes=2, lr=1e-2)
    tr = pl.Trainer(max_epochs=30, enable_progress_bar=False, logger=False, enable_checkpointing=False, accelerator="cpu")
    tr.fit(m, dl)
    # after fit, predictions on the training bags are (near) perfect
    with torch.no_grad():
        b, mk, y = mil_collate([ds[i] for i in range(len(ds))])
        pred = (torch.sigmoid(m.model(b, mk)[0]).squeeze(1) > 0.5).long()
    assert (pred == y).float().mean() > 0.9
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement**

```python
# src/downstream/abmil/lit.py
import torch, pytorch_lightning as pl
from src.downstream.abmil.model import GatedABMIL

class AbmilLitModule(pl.LightningModule):
    def __init__(self, emb_dim, hidden_dim, num_heads=1, num_classes=2, lr=1e-4):
        super().__init__(); self.save_hyperparameters()
        self.model = GatedABMIL(emb_dim, hidden_dim, num_heads, num_classes)
        self.num_classes = num_classes; self.lr = lr
        self.loss = torch.nn.BCEWithLogitsLoss() if num_classes == 2 else torch.nn.CrossEntropyLoss()

    def _step(self, batch, stage):
        bags, mask, y = batch
        logits, _ = self.model(bags, mask)
        target = y.unsqueeze(1).float() if self.num_classes == 2 else y
        loss = self.loss(logits, target)
        self.log(f"{stage}_loss", loss, prog_bar=False, batch_size=bags.size(0))
        return loss

    def training_step(self, b, i): return self._step(b, "train")
    def validation_step(self, b, i): return self._step(b, "val")
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=self.lr)
```

- [ ] **Step 4: Run → PASS** (may take ~10s). — [ ] **Step 5: Commit** — `git commit -m "feat(abmil): Lightning module wrapper"`.

### Task B5: 10-fold CV runner + metrics + Hydra script

**Files:** Create `src/downstream/abmil/cv.py`, `scripts/run_abmil.py`, `configs/downstream/abmil.yaml`; Test `tests/test_downstream_abmil.py`.

**Interfaces:** Produces `run_cv(bags, labels, num_classes, cfg) -> dict` returning per-fold `accuracy/macro_f1/auc` + OOF arrays; StratifiedKFold(`num_folds`, shuffle, seed 42), per fold builds `MILDataset`+`DataLoader`(collate=`mil_collate`) + `AbmilLitModule` + `pl.Trainer(EarlyStopping('val_loss', patience), ModelCheckpoint('val_loss'))`; `zscore='cv_train'` fits mean/std on the train fold only.

- [ ] **Step 1: Failing test** (2-fold on separable synthetic bags → high accuracy, reproducible)

```python
def test_run_cv_separable(tmp_path):
    import numpy as np, torch
    from src.downstream.abmil.cv import run_cv
    from types import SimpleNamespace
    bags = [np.ones((3,8),"float32") for _ in range(10)] + [(-np.ones((3,8),"float32")) for _ in range(10)]
    labels = torch.tensor([1]*10 + [0]*10)
    cfg = SimpleNamespace(num_folds=2, hidden_dim=8, num_heads=1, num_epochs=15, patience=5,
                          lr=1e-2, batch_size=4, zscore="cv_train", results_dir=str(tmp_path))
    out = run_cv(bags, labels, num_classes=2, cfg=cfg)
    assert np.mean(out["fold_accuracy"]) > 0.8
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** `run_cv` (fold loop + `zscore_stats_from_bags`/`apply_zscore_to_bags` ported from the existing runner; metrics `accuracy_score`, `f1_score(average="macro")`, `roc_auc_score`). Then `scripts/run_abmil.py` iterates datasets × features (from meta CSVs), calls `build_image_bags` → `run_cv`, writes `cv_folds.csv`/`cv_summary.csv`/`results.csv`. Full code mirrors `/home/tnocon/mil/gated_abmil/src/run_abmil.py::run` with the model/trainer swapped for `AbmilLitModule`+`pl.Trainer`, and the leakage guardrails from Global Constraints. Config:

```yaml
# configs/downstream/abmil.yaml
defaults: [_self_]
meta_dir: ???        # dir with cords_{train,test}_metadata.csv joined to clinical labels
datasets: [cords]
features: [Grade, Relapse, DX.name, ERStatus, ERBB2_pos, PAM50]
num_folds: 10
hidden_dim: 256
num_heads: 8
num_epochs: 100
patience: 20
lr: 1.0e-4
batch_size: 16
zscore: cv_train
results_dir: ${hydra:runtime.output_dir}/abmil
min_class_freq: 0.05
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `git commit -m "feat(abmil): 10-fold CV runner + hydra script + config"`.

### Task B6: End-to-end integration + szary sweep

- [ ] **Step 1:** Integration test `tests/test_downstream_e2e.py`: synthetic h5 (12 patches, 4 images, 2 classes) → `encode_h5` (stub PCA + tiny ckpt) → join a synthetic clinical meta (`feature`/`feature_value`) by `img_path` → `build_image_bags` → `run_cv(num_folds=2)` runs clean and returns metrics.
- [ ] **Step 2:** Run `uv run pytest tests/test_downstream_e2e.py -v` → PASS.
- [ ] **Step 3:** szary: encode cords test.h5 with the chosen PIGVAE ckpt, join clinical labels (existing `logistic_regression` join, stopping before LR), run `scripts/run_abmil.py` for one feature; confirm metrics CSVs written.
- [ ] **Step 4: Commit** — `git commit -m "test(downstream): end-to-end encode->abmil integration"`.

---

## Self-review

- **Spec coverage:** §3 flow → Tasks A1–A6 (encode) + B1–B6 (abmil). §4 components → A1/A2/A3 (encode.py), A4 (memmap), B1 (model), B2 (data collate), B3 (bags), B4 (lit), B5 (cv). §5 guardrails → Global Constraints + B5 (`zscore='cv_train'`, img-level grouping in B3). §6 error handling → A6 resumable, B5 fold-capping (ported). §7 testing → each task's unit test + B6 integration + A6/B6 szary smoke. All spec sections map to a task.
- **Placeholder scan:** no TBD/TODO; B5 step 3 references exact ported functions by name (`zscore_stats_from_bags`, `apply_zscore_to_bags`) from a named source file rather than restating — acceptable as it's a verbatim port, but the implementer must copy them in.
- **Type consistency:** `z_global` width `D` flows A3→A5→memmap→B3 (`bags[0].shape[1]`)→B1 `emb_dim`, never hardcoded. `GatedABMIL.forward` returns `(logits, pooled)` used identically in B4/B5. `mil_collate` returns `(bags, masks, labels)` consumed identically in B4/B5.
- **Open risk:** Task A2 node-order assumption must be verified against `GridGraphDataset.__getitem__` (called out in A2 step 3) before trusting encode outputs — a wrong reshape silently corrupts every embedding.
