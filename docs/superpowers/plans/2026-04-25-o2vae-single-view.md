# O2-VAE Single-View D4 Alignment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Sinkhorn permuter + 8-view batch with a single-view data pipeline and an analytic logsumexp D4 alignment loss.

**Architecture:** Encoder is already orientation-invariant (no positional encodings, CLS token), so all 8 D4 views of the same sample produce identical z — the 8-view batch is waste. The new pipeline feeds one randomly-augmented view per sample; the decoder sees an identity permutation; orientation alignment is handled analytically at loss time by trying all 8 D4 transforms and taking the logsumexp soft-min.

**Tech Stack:** PyTorch, PyTorch Lightning, Hydra/OmegaConf, W&B, NumPy, pytest

---

## Files Changed

| File | Change |
|---|---|
| `src/data/components/graphs_datamodules.py` | Add `SingleViewTransform`; fix `perms.append(perm.squeeze(0))` → `perms.append(perm)`; add `single_view` to `PickleDataset` |
| `src/data/imc_datamodule.py` | Wire `SingleViewTransform`; pass `single_view=True` to `PickleDataset` |
| `src/models/components/losses.py` | Add `D4AlignmentLoss` |
| `src/models/components/modules.py` | Remove permuter from `GraphAE`; simplify `BottleNeckEncoder` |
| `src/models/components/model.py` | Replace `GraphReconstructionLoss + PermutationLoss` with `D4AlignmentLoss` in `Critic` |
| `src/models/pigvae_auto_module.py` | Remove schedulers; fix batch-size refs; simplify viz |
| `configs/model/model.yaml` | Remove permuter/scheduler blocks; add `grid_size` to critic |
| `configs/data/mnist.yaml` | Set `num_aug_per_sample: 1`; add `single_view: true` |

---

## Task 1: SingleViewTransform + PickleDataset single_view + perms fix

**Files:**
- Modify: `src/data/components/graphs_datamodules.py`
- Create: `tests/test_single_view_transform.py`

### What to implement

**1a.** In `PickleDataset.__init__` (line 100), add `single_view: bool = False` parameter:

```python
def __init__(
    self,
    hdf5_path,
    transform=None,
    only_embeddings: bool = False,
    generate_views: bool = False,
    center_crop_size: int | None = None,
    single_view: bool = False,
):
    self.hdf5_path = hdf5_path
    self.transform = transform
    self.only_embeddings = only_embeddings
    self.generate_views = generate_views and not single_view
    self.center_crop_size = center_crop_size
    self.single_view = single_view
    with h5py.File(hdf5_path, "r") as f:
        self._length = len(f[next(iter(f.keys()))])
```

**1b.** Fix `from_sparse_graph_list` line 518 — change `perm.squeeze(0)` to `perm`:

```python
perms.append(perm)   # was: perm.squeeze(0)
```

**1c.** Add `SingleViewTransform` class after `PatchAugmentations` (around line 217):

```python
class SingleViewTransform(nn.Module):
    """Single-view replacement for PatchAugmentations.

    Takes a raw (C, H, W) numpy array, applies one random D4 transform
    (or identity for validation), preprocesses it, and returns the same
    interface as PatchAugmentations with a leading dim of 1.
    """

    def __init__(
        self,
        is_validation: bool = False,
        center_crop_size: int | None = None,
        normalize: bool = False,
        norm_type: str = "channel_wise",
        clip_percentiles: bool = False,
        clip_lower: float = 0.01,
        clip_upper: float = 0.99,
        clip_type: str = "channel_wise",
    ):
        super().__init__()
        self.is_validation = is_validation
        self.center_crop_size = center_crop_size
        self.normalize = normalize
        self.norm_type = norm_type
        self.clip_percentiles = clip_percentiles
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
        self.clip_type = clip_type

    def forward(self, emb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            emb: np.ndarray [C, H, W] — raw single embedding from PickleDataset

        Returns:
            aug_tensor:     [1, N, C]  float32 tensor
            argsort_tensor: [1, N]     int64 tensor (identity argsort)
            perm:           tensor([0]) int64 — single-element perm index
        """
        if self.center_crop_size is not None:
            emb = _center_crop_np(emb, self.center_crop_size)

        if self.is_validation:
            key = "r0_nf"  # identity: no rotation, no flip
        else:
            key = IMC_GRAPH_VIEW_KEYS[torch.randint(0, 8, (1,)).item()]

        emb = _spatial_view(emb, key)

        t = torch.from_numpy(emb.copy()).float()  # [C, H, W]

        if self.clip_percentiles and self.clip_type != "none":
            t = self._clip_by_percentile(t, self.clip_lower, self.clip_upper, self.clip_type)

        if self.normalize:
            if self.norm_type == "channel_wise":
                eps = 1e-6
                mean = t.mean(dim=(1, 2), keepdim=True)
                std = t.std(dim=(1, 2), keepdim=True) + eps
                t = (t - mean) / std
            elif self.norm_type == "global":
                eps = 1e-6
                t = (t - t.mean()) / (t.std() + eps)

        c = t.shape[0]
        flat = t.reshape(c, -1).T  # [N, C]
        n = flat.shape[0]

        aug_tensor = flat.unsqueeze(0)  # [1, N, C]
        argsort_tensor = torch.arange(n, dtype=torch.long).unsqueeze(0)  # [1, N]
        perm = torch.tensor([0], dtype=torch.long)  # [1]

        return aug_tensor, argsort_tensor, perm

    @staticmethod
    def _clip_by_percentile(
        x: torch.Tensor, lower: float, upper: float, mode: str
    ) -> torch.Tensor:
        if mode == "global":
            flat = x.flatten()
            return torch.clamp(x, min=torch.quantile(flat, lower), max=torch.quantile(flat, upper))
        elif mode == "channel_wise":
            c = x.shape[0]
            x_reshaped = x.reshape(c, -1)
            q_low = torch.quantile(x_reshaped, lower, dim=1, keepdim=True).reshape(c, 1, 1)
            q_high = torch.quantile(x_reshaped, upper, dim=1, keepdim=True).reshape(c, 1, 1)
            return torch.maximum(torch.minimum(x, q_high), q_low)
        return x
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_single_view_transform.py
import numpy as np
import pytest
import torch

from src.data.components.graphs_datamodules import (
    DenseGraphBatch,
    GridGraphDataset,
    SingleViewTransform,
)


def make_fake_emb(C=4, H=6, W=6):
    return np.random.randn(C, H, W).astype(np.float32)


class TestSingleViewTransform:
    def test_output_shapes(self):
        t = SingleViewTransform(is_validation=True)
        emb = make_fake_emb(C=4, H=6, W=6)
        aug, argsort, perm = t(emb)
        assert aug.shape == (1, 36, 4), f"aug shape wrong: {aug.shape}"
        assert argsort.shape == (1, 36), f"argsort shape wrong: {argsort.shape}"
        assert perm.shape == (1,), f"perm shape wrong: {perm.shape}"
        assert perm[0] == 0

    def test_validation_is_deterministic(self):
        t = SingleViewTransform(is_validation=True)
        emb = make_fake_emb()
        aug1, _, _ = t(emb)
        aug2, _, _ = t(emb)
        assert torch.allclose(aug1, aug2)

    def test_training_returns_tensor(self):
        t = SingleViewTransform(is_validation=False)
        emb = make_fake_emb()
        aug, argsort, perm = t(emb)
        assert aug.dtype == torch.float32
        assert argsort.dtype == torch.int64

    def test_center_crop(self):
        t = SingleViewTransform(is_validation=True, center_crop_size=4)
        emb = make_fake_emb(C=4, H=6, W=6)
        aug, argsort, perm = t(emb)
        assert aug.shape == (1, 16, 4)  # 4x4 = 16 nodes

    def test_perms_stack_in_collator(self):
        """Regression: perm.squeeze(0) was scalar for shape [1], crashing stack."""
        import networkx as nx

        t = SingleViewTransform(is_validation=True)
        emb = make_fake_emb(C=4, H=6, W=6)
        aug, argsort, perm = t(emb)  # [1, 36, 4], [1, 36], [1]

        g = nx.grid_graph((6, 6))
        data_list = [
            (g, aug, argsort, perm, -1,
             torch.zeros(1, dtype=torch.long),
             np.array(["path"]),
             torch.zeros(1, 2))
            for _ in range(2)
        ]
        batch = DenseGraphBatch.from_sparse_graph_list(data_list)
        assert batch.node_features.shape == (2, 36, 4)
        assert batch.perms.shape == (2, 1)
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE
python -m pytest tests/test_single_view_transform.py -v
```

Expected: `ImportError: cannot import name 'SingleViewTransform'`

- [ ] **Step 3: Implement changes in `graphs_datamodules.py`**

Apply changes 1a, 1b, 1c described above.

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_single_view_transform.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/components/graphs_datamodules.py tests/test_single_view_transform.py
git commit -m "feat: add SingleViewTransform and fix perms.squeeze(0) bug in collator"
```

---

## Task 2: Wire SingleViewTransform into IMCDataModule + update data config

**Files:**
- Modify: `src/data/imc_datamodule.py`
- Modify: `configs/data/mnist.yaml`

### What to implement

**2a.** In `imc_datamodule.py`, update the import to include `SingleViewTransform`:

```python
from src.data.components.graphs_datamodules import (
    DenseGraphDataLoader,
    DualOutputTransform,
    GridGraphDataset,
    IMCBaseDictTransform,
    PatchAugmentations,
    PCADenseGraphCollator,
    PCALayer,
    PickleDataset,
    SingleViewTransform,
    WelfordOnline,
)
```

**2b.** In `IMCDataModule.__init__`, add `single_view` handling. Replace lines 86-109:

```python
self.single_view = hparams.get("single_view", False)

if self.single_view:
    svt_kwargs = dict(
        center_crop_size=hparams.center_crop_size,
        normalize=hparams.normalize,
    )
    self.dual_transforms_train = DualOutputTransform(
        base_transforms=None,
        augmentation_transforms=SingleViewTransform(is_validation=False, **svt_kwargs),
    )
    self.dual_transforms_val = DualOutputTransform(
        base_transforms=None,
        augmentation_transforms=SingleViewTransform(is_validation=True, **svt_kwargs),
    )
else:
    self.base_transforms = IMCBaseDictTransform(
        center_crop_size=hparams.center_crop_size, normalize=hparams.normalize
    )
    self.aug_transforms_train = PatchAugmentations(
        prob=hparams.augmentation_prob,
        size=hparams.size,
        patch_size=hparams.patch_size,
    )
    self.aug_transforms_val = PatchAugmentations(
        prob=hparams.augmentation_prob,
        size=hparams.size,
        patch_size=hparams.patch_size,
        is_validation=True,
    )
    self.dual_transforms_train = DualOutputTransform(
        self.base_transforms, self.aug_transforms_train
    )
    self.dual_transforms_val = DualOutputTransform(
        self.base_transforms, self.aug_transforms_val
    )
```

**2c.** In `_load_datasets` (line 145), pass `single_view`:

```python
def _load_datasets(self, paths: list[Path], transform: DualOutputTransform) -> Dataset:
    parts = [
        PickleDataset(
            path,
            transform=transform,
            generate_views=not self.single_view,
            center_crop_size=self.center_crop_size,
            single_view=self.single_view,
        )
        for path in paths
    ]
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)
```

**2d.** In `configs/data/mnist.yaml`, change:

```yaml
  num_aug_per_sample: 1   # was: 8
  single_view: true       # new
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_single_view_datamodule.py
import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from src.data.components.graphs_datamodules import SingleViewTransform, DualOutputTransform


class TestSingleViewDataModuleWiring:
    def test_dual_output_transform_with_none_base(self):
        """DualOutputTransform with base_transforms=None passes raw emb to SingleViewTransform."""
        svt = SingleViewTransform(is_validation=True, center_crop_size=6)
        dual = DualOutputTransform(base_transforms=None, augmentation_transforms=svt)
        # Simulate what PickleDataset returns when single_view=True
        emb = np.random.randn(4, 6, 6).astype(np.float32)
        fake_item = {
            "embeddings": emb,
            "metadata": torch.zeros(1, dtype=torch.long),
            "paths": np.array(["path"]),
            "positions": torch.zeros(1, 2),
        }
        aug, argsort, perm, meta, paths, pos = dual(fake_item)
        assert aug.shape == (1, 36, 4)
        assert perm[0] == 0
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_single_view_datamodule.py -v
```

Expected: FAIL — `TypeError` from `DualOutputTransform.__call__` trying `original["embeddings"] = self.base_transforms(...)` when base_transforms is `None`. (Actually it's guarded by `if self.base_transforms is not None`, so it should pass already — this test verifies the existing guard works with the new transform.)

- [ ] **Step 3: Implement changes in `imc_datamodule.py` and `configs/data/mnist.yaml`**

Apply changes 2a–2d above.

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_single_view_datamodule.py tests/test_single_view_transform.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/imc_datamodule.py configs/data/mnist.yaml tests/test_single_view_datamodule.py
git commit -m "feat: wire SingleViewTransform into IMCDataModule, set single_view in data config"
```

---

## Task 3: Add D4AlignmentLoss to losses.py

**Files:**
- Modify: `src/models/components/losses.py`
- Create: `tests/test_d4_alignment_loss.py`

### What to implement

Add at the end of `losses.py`, after the existing classes. Requires `import math` (add to existing imports if missing):

```python
class D4AlignmentLoss(torch.nn.Module):
    """Orientation-invariant reconstruction loss over the D4 symmetry group.

    Tries all 8 D4 transforms of the decoder output and takes the
    logsumexp soft-min per sample. +log(8) normalises for the uniform
    prior over orientations (partition function of the discrete group).
    """

    def __init__(
        self,
        grid_size: int,
        huber_beta: float = 1.0,
        alpha: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.001,
    ):
        super().__init__()
        self.huber_beta = huber_beta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.grid_size = grid_size
        self.register_buffer("perm_matrices", self._precompute_d4(grid_size))

    @staticmethod
    def _precompute_d4(n: int) -> torch.Tensor:
        """Compute the 8 D4 permutation matrices for an n×n grid."""
        n_nodes = n * n
        matrices = []
        # 4 rotations
        for k in range(4):
            idx = torch.arange(n_nodes).reshape(n, n)
            for _ in range(k):
                idx = idx.rot90(-1)
            matrices.append(torch.eye(n_nodes)[idx.reshape(-1)])
        # 4 reflections (horizontal flip composed with each rotation)
        base_idx = torch.arange(n_nodes).reshape(n, n)
        reflected_idx = base_idx.flip(1).reshape(-1)
        reflection = torch.eye(n_nodes)[reflected_idx]
        for k in range(4):
            idx = torch.arange(n_nodes).reshape(n, n)
            for _ in range(k):
                idx = idx.rot90(-1)
            rot = torch.eye(n_nodes)[idx.reshape(-1)]
            matrices.append(torch.matmul(reflection, rot))
        return torch.stack(matrices, dim=0)  # [8, N, N]

    @staticmethod
    def _huber_per_sample(pred: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target, beta=beta, reduction="none").mean(dim=[-2, -1])  # [B]

    @staticmethod
    def _cosine_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=-1)  # [B]

    @staticmethod
    def _gradient_per_sample(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> torch.Tensor:
        pred_gx = pred_grid[:, :, :, 1:] - pred_grid[:, :, :, :-1]
        true_gx = target_grid[:, :, :, 1:] - target_grid[:, :, :, :-1]
        pred_gy = pred_grid[:, :, 1:, :] - pred_grid[:, :, :-1, :]
        true_gy = target_grid[:, :, 1:, :] - target_grid[:, :, :-1, :]
        return (
            (pred_gx - true_gx).abs().mean(dim=[-3, -2, -1])
            + (pred_gy - true_gy).abs().mean(dim=[-3, -2, -1])
        )  # [B]

    def forward(
        self, graph_true: "DenseGraphBatch", graph_pred: "DenseGraphBatch"
    ) -> dict[str, torch.Tensor]:
        device = graph_pred.node_features.device
        x = graph_true.node_features.to(device)      # [B, N, C]
        x_hat = graph_pred.node_features.to(device)  # [B, N, C]
        B, N, C = x_hat.shape
        H = self.grid_size

        x_grid = x.view(B, H, H, C).permute(0, 3, 1, 2)     # [B, C, H, W]

        losses_per_k = []
        for k in range(8):
            x_hat_k = torch.einsum("nm,bnd->bmd", self.perm_matrices[k], x_hat)  # [B, N, C]
            x_hat_k_grid = x_hat_k.view(B, H, H, C).permute(0, 3, 1, 2)         # [B, C, H, W]
            loss_k = (
                self.alpha * self._huber_per_sample(x_hat_k, x, self.huber_beta)
                + self.beta * self._cosine_per_sample(x_hat_k, x)
                + self.gamma * self._gradient_per_sample(x_hat_k_grid, x_grid)
            )  # [B]
            losses_per_k.append(loss_k)

        losses = torch.stack(losses_per_k, dim=1)              # [B, 8]
        lse = -torch.logsumexp(-losses, dim=1) + math.log(8)  # [B]
        total = lse.mean()
        return {"loss": total, "d4_alignment_loss": total}
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_d4_alignment_loss.py
import math
import torch
import pytest

from src.data.components.graphs_datamodules import DenseGraphBatch
from src.models.components.losses import D4AlignmentLoss


def make_batch(B: int = 4, grid_size: int = 6, C: int = 8):
    N = grid_size * grid_size
    node_features = torch.randn(B, N, C)
    return DenseGraphBatch(
        node_features=node_features,
        edge_features=torch.empty(0),
    )


class TestD4AlignmentLoss:
    def test_output_is_dict_with_loss(self):
        loss_fn = D4AlignmentLoss(grid_size=6)
        gt = make_batch()
        pred = make_batch()
        out = loss_fn(gt, pred)
        assert "loss" in out
        assert "d4_alignment_loss" in out
        assert out["loss"].ndim == 0  # scalar

    def test_loss_is_positive(self):
        loss_fn = D4AlignmentLoss(grid_size=6)
        gt = make_batch()
        pred = make_batch()
        out = loss_fn(gt, pred)
        assert out["loss"].item() >= 0

    def test_perfect_reconstruction_gives_log8_normalised_zero(self):
        """When pred == gt for all orientations, logsumexp soft-min approaches 0."""
        loss_fn = D4AlignmentLoss(grid_size=6, alpha=1.0, beta=0.0, gamma=0.0)
        gt = make_batch(B=2)
        # Use the same tensor for pred as gt — loss should be near 0 before +log(8)
        pred = DenseGraphBatch(
            node_features=gt.node_features.clone(),
            edge_features=torch.empty(0),
        )
        out = loss_fn(gt, pred)
        # huber(pred==gt) = 0 for each k, logsumexp(0,...,0) = log(8), +log(8) normalises to 0
        assert abs(out["loss"].item()) < 1e-4

    def test_perm_matrices_shape(self):
        loss_fn = D4AlignmentLoss(grid_size=6)
        assert loss_fn.perm_matrices.shape == (8, 36, 36)

    def test_perm_matrices_are_orthogonal(self):
        loss_fn = D4AlignmentLoss(grid_size=6)
        for k in range(8):
            P = loss_fn.perm_matrices[k]
            I = torch.eye(36)
            assert torch.allclose(P @ P.T, I, atol=1e-5), f"P[{k}] not orthogonal"

    def test_gradients_flow(self):
        loss_fn = D4AlignmentLoss(grid_size=6)
        gt = make_batch()
        pred_features = torch.randn(4, 36, 8, requires_grad=True)
        pred = DenseGraphBatch(node_features=pred_features, edge_features=torch.empty(0))
        out = loss_fn(gt, pred)
        out["loss"].backward()
        assert pred_features.grad is not None
        assert not torch.isnan(pred_features.grad).any()
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_d4_alignment_loss.py -v
```

Expected: `ImportError: cannot import name 'D4AlignmentLoss'`

- [ ] **Step 3: Add `D4AlignmentLoss` to `losses.py`**

Add `import math` at the top of `losses.py` if not present. Add the `D4AlignmentLoss` class at the end of the file.

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_d4_alignment_loss.py -v
```

Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/components/losses.py tests/test_d4_alignment_loss.py
git commit -m "feat: add D4AlignmentLoss with logsumexp orientation invariance"
```

---

## Task 4: Remove permuter from GraphAE, use identity perm in decoder

**Files:**
- Modify: `src/models/components/modules.py`
- Create: `tests/test_graph_ae_no_permuter.py`

### What to implement

**4a.** In `GraphAE.__init__` (line 17), remove `self.permuter = SimplePermuter(hparams.permuter)`:

```python
class GraphAE(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.input_size = hparams.input_size
        self.vae = hparams.vae
        self.encoder = GraphEncoder(hparams.encoder)
        self.bottle_neck_encoder = BottleNeckEncoder(hparams.bottle_neck_encoder)
        self.bottle_neck_decoder = BottleNeckDecoder(hparams.bottle_neck_decoder)
        self.decoder = GraphDecoder(hparams.decoder)
        # No permuter — orientation handled analytically in D4AlignmentLoss
```

**4b.** Update `GraphAE.encode` to drop `node_features` from return (permuter consumed it):

```python
def encode(
    self, graph: DenseGraphBatch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    node_features = graph.node_features
    edge_features = graph.edge_features
    mask = graph.mask
    graph_emb, _ = self.encoder(
        node_features=node_features,
        edge_features=edge_features,
        mask=mask,
    )
    graph_emb, mu, logvar = self.bottle_neck_encoder(graph_emb)
    return graph_emb, mu, logvar
```

**4c.** Update `GraphAE.forward` to use identity permutation:

```python
def forward(self, graph: DenseGraphBatch) -> tuple:
    graph_emb, mu, logvar = self.encode(graph=graph)
    B = graph_emb.shape[0]
    N = graph.mask.shape[1]
    device = graph_emb.device
    eye = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
    graph_pred = self.decode(graph_emb, eye, graph.mask)
    return graph_emb, graph_pred, mu, logvar
```

**4d.** Update `GraphAE.decode` to remove the `perm` param (keep it for backward compat with the caller — just pass the identity perm through the existing signature, no change to `decode`):

The `decode` method signature stays the same (`graph_emb`, `perm`, `mask`). No change needed there.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_ae_no_permuter.py
import torch
import pytest
from omegaconf import OmegaConf

from src.data.components.graphs_datamodules import DenseGraphBatch


def make_graph_ae_hparams():
    return OmegaConf.create({
        "input_size": 64,
        "num_heads": 2,
        "num_layers": 1,
        "emb_dim": 32,
        "vae": True,
        "dropout": 0.0,
        "encoder": {
            "graph_encoder_hidden_dim": 64,
            "num_node_features": 8,
            "num_edge_features": 0,
            "graph_encoder_num_heads": 2,
            "graph_encoder_ppf_hidden_dim": 128,
            "graph_encoder_num_layers": 1,
            "emb_dim": 32,
            "dropout": 0.0,
            "grid_size": 4,
            "project": True,
        },
        "decoder": {
            "graph_decoder_hidden_dim": 64,
            "graph_decoder_pos_emb_dim": 64,
            "graph_decoder_num_heads": 2,
            "graph_decoder_ppf_hidden_dim": 128,
            "graph_decoder_num_layers": 1,
            "dropout": 0.0,
            "head_dim": 32,
            "num_embeddings": 64,
            "num_node_features": 8,
            "num_edge_features": 0,
            "project": True,
            "grid_size": 4,
        },
        "bottle_neck_encoder": {
            "graph_encoder_hidden_dim": 64,
            "emb_dim": 32,
            "vae": True,
            "activation": "silu",
        },
        "bottle_neck_decoder": {
            "emb_dim": 32,
            "graph_decoder_hidden_dim": 64,
            "num_nodes": 16,
        },
        "property_predictor": {
            "emb_dim": 32,
            "property_predictor_hidden_dim": 64,
            "num_properties": 1,
        },
    })


def make_batch(B: int = 2, N: int = 16, C: int = 8):
    node_features = torch.randn(B, N, C)
    mask = torch.ones(B, N, dtype=torch.bool)
    return DenseGraphBatch(
        node_features=node_features,
        edge_features=torch.empty(0),
        mask=mask,
    )


class TestGraphAENoPermuter:
    def test_forward_returns_4_tuple(self):
        from src.models.components.modules import GraphAE
        hparams = make_graph_ae_hparams()
        ae = GraphAE(hparams)
        graph = make_batch(B=2, N=16, C=8)
        out = ae(graph)
        assert len(out) == 4, f"expected 4-tuple, got {len(out)}"
        graph_emb, graph_pred, mu, logvar = out
        assert graph_emb.shape[0] == 2
        assert mu.shape == (2, 32)

    def test_no_permuter_attribute(self):
        from src.models.components.modules import GraphAE
        hparams = make_graph_ae_hparams()
        ae = GraphAE(hparams)
        assert not hasattr(ae, "permuter"), "permuter should be removed"
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_graph_ae_no_permuter.py -v
```

Expected: FAIL — `forward` returns 6-tuple (current code), test expects 4-tuple. `test_no_permuter_attribute` also fails since `permuter` attribute still exists.

- [ ] **Step 3: Apply changes 4a–4d in `modules.py`**

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_graph_ae_no_permuter.py -v
```

Expected: Both tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/components/modules.py tests/test_graph_ae_no_permuter.py
git commit -m "feat: remove SimplePermuter from GraphAE, use identity perm in decoder"
```

---

## Task 5: Simplify BottleNeckEncoder reparameterization

**Files:**
- Modify: `src/models/components/modules.py`
- Create: `tests/test_bottleneck_encoder.py`

### What to implement

In `BottleNeckEncoder.__init__` (line 604), remove `self.num_permutations = hparams.num_permutations`:

```python
class BottleNeckEncoder(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.d_in = hparams.graph_encoder_hidden_dim
        self.d_out = hparams.emb_dim
        self.vae = hparams.vae
        self.activation = {
            "relu": torch.nn.ReLU(),
            "gelu": torch.nn.GELU(),
            "silu": torch.nn.SiLU(),
            "gelu2": torch.nn.GELU(approximate="tanh"),
            "leaky_relu": torch.nn.LeakyReLU(negative_slope=0.01),
        }[hparams.activation.lower()]
        self.fc_hidden = nn.Linear(self.d_in, self.d_in)
        if self.vae:
            self.w = nn.Linear(self.d_in, 2 * self.d_out)
        else:
            self.w = nn.Linear(self.d_in, self.d_out)
```

In `BottleNeckEncoder.forward` (line 624), replace the epsilon-sharing block with standard reparameterization:

```python
def forward(
    self, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    x = self.w(self.activation(self.fc_hidden(x)))
    if self.vae:
        mu = x[:, : self.d_out]
        logvar = x[:, self.d_out :]
        logvar = torch.clamp(logvar, -10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x = mu + eps * std
        return x, mu, logvar
    else:
        return x, None, None
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bottleneck_encoder.py
import torch
import pytest
from omegaconf import OmegaConf

from src.models.components.modules import BottleNeckEncoder


def make_hparams(vae=True):
    return OmegaConf.create({
        "graph_encoder_hidden_dim": 64,
        "emb_dim": 32,
        "vae": vae,
        "activation": "silu",
    })


class TestBottleNeckEncoderSingleView:
    def test_no_num_permutations_attr(self):
        enc = BottleNeckEncoder(make_hparams())
        assert not hasattr(enc, "num_permutations"), "num_permutations should be removed"

    def test_different_samples_get_different_eps(self):
        """Reparameterized samples with the same mu/logvar must differ (standard VAE)."""
        enc = BottleNeckEncoder(make_hparams())
        x = torch.randn(4, 64)
        z1, mu1, _ = enc(x)
        z2, mu2, _ = enc(x)
        # mu must be deterministic
        assert torch.allclose(mu1, mu2, atol=1e-5)
        # z must differ (different eps sampled)
        assert not torch.allclose(z1, z2), "z should differ due to different eps"

    def test_output_shapes_vae(self):
        enc = BottleNeckEncoder(make_hparams(vae=True))
        x = torch.randn(4, 64)
        z, mu, logvar = enc(x)
        assert z.shape == (4, 32)
        assert mu.shape == (4, 32)
        assert logvar.shape == (4, 32)

    def test_output_shapes_ae(self):
        enc = BottleNeckEncoder(make_hparams(vae=False))
        x = torch.randn(4, 64)
        z, mu, logvar = enc(x)
        assert z.shape == (4, 32)
        assert mu is None
        assert logvar is None
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_bottleneck_encoder.py -v
```

Expected: `test_no_num_permutations_attr` FAILS (attribute still present). `test_different_samples_get_different_eps` FAILS (old code shares eps so z1 == z2 for same input in same forward call... actually they are different forward calls so eps differs anyway — but the test for `num_permutations` will fail).

- [ ] **Step 3: Apply changes in `modules.py`**

Remove `self.num_permutations = hparams.num_permutations` from `__init__`. Replace the epsilon-sharing block in `forward` with standard reparameterization.

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_bottleneck_encoder.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/components/modules.py tests/test_bottleneck_encoder.py
git commit -m "feat: simplify BottleNeckEncoder to standard VAE reparameterization"
```

---

## Task 6: Replace reconstruction+perm losses with D4AlignmentLoss in Critic

**Files:**
- Modify: `src/models/components/model.py`
- Create: `tests/test_critic_d4.py`

### What to implement

Replace the entire `model.py` content:

```python
import os
from typing import Any

import rootutils
import torch
from omegaconf import DictConfig

from src.data.components.graphs_datamodules import DenseGraphBatch
from src.models.components.losses import (
    D4AlignmentLoss,
    KLDLoss,
    MAELoss,
    MSEGridLoss,
    SignalToNoiseRatioLoss,
)

rootutils.setup_root(os.getcwd(), indicator=".project-root", pythonpath=True)


class Critic(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.kld_scale = float(getattr(hparams, "kld_loss_scale", 1.0))
        self.vae = hparams.vae
        self.d4_alignment_loss = D4AlignmentLoss(
            grid_size=hparams.grid_size,
            huber_beta=hparams.huber_beta,
            alpha=hparams.alpha_scale,
            beta=hparams.beta_scale,
            gamma=hparams.gamma_scale,
        )
        self.kld_loss = KLDLoss(
            normalize_by_latent_dim=True, free_bits=hparams.get("kld_free_bits", 0.0)
        )
        self.mae_loss = MAELoss()
        self.signal_to_noise_ratio_loss = SignalToNoiseRatioLoss()
        self.mse_loss = MSEGridLoss()

    def forward(
        self,
        graph_emb: torch.Tensor,
        graph_true: DenseGraphBatch,
        graph_pred: DenseGraphBatch,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kld_alpha: float | None = None,
    ) -> dict[str, Any]:
        loss = {
            **self.d4_alignment_loss(graph_true=graph_true, graph_pred=graph_pred),
            "mae_loss": self.mae_loss(graph_true=graph_true, graph_pred=graph_pred),
            "mse_loss": self.mse_loss(graph_true=graph_true, graph_pred=graph_pred),
            "signal_to_noise_ratio_loss": self.signal_to_noise_ratio_loss(
                graph_true=graph_true, graph_pred=graph_pred
            ),
        }
        if self.vae:
            kld = self.kld_loss(mu, logvar)
            loss["kld_loss"] = kld
            scale = self.kld_scale * (1.0 if kld_alpha is None else float(kld_alpha))
            loss["loss"] = loss["loss"] + scale * kld
        return loss

    def evaluate(
        self,
        graph_emb: torch.Tensor,
        graph_true: DenseGraphBatch,
        graph_pred: DenseGraphBatch,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kld_alpha: float | None = None,
        prefix: str | None = None,
    ) -> dict[str, Any]:
        loss = self(
            graph_emb=graph_emb,
            graph_true=graph_true,
            graph_pred=graph_pred,
            kld_alpha=kld_alpha,
            mu=mu,
            logvar=logvar,
        )
        if prefix is not None:
            loss = {prefix + "_" + k: v for k, v in loss.items()}
        return loss


if __name__ == "__main__":
    pass
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_critic_d4.py
import torch
import pytest
from omegaconf import OmegaConf

from src.data.components.graphs_datamodules import DenseGraphBatch


def make_critic_hparams():
    return OmegaConf.create({
        "kld_loss_scale": 0.05,
        "kld_free_bits": 0.0,
        "vae": True,
        "grid_size": 4,
        "huber_beta": 2.0,
        "alpha_scale": 1.0,
        "beta_scale": 0.1,
        "gamma_scale": 0.001,
    })


def make_batch(B=2, grid_size=4, C=8):
    N = grid_size * grid_size
    return DenseGraphBatch(
        node_features=torch.randn(B, N, C),
        edge_features=torch.empty(0),
        mask=torch.ones(B, N, dtype=torch.bool),
    )


class TestCriticD4:
    def test_forward_returns_expected_keys(self):
        from src.models.components.model import Critic
        critic = Critic(make_critic_hparams())
        gt = make_batch()
        pred = make_batch()
        mu = torch.randn(2, 32)
        logvar = torch.zeros(2, 32)
        out = critic(
            graph_emb=torch.randn(2, 32),
            graph_true=gt,
            graph_pred=pred,
            mu=mu,
            logvar=logvar,
        )
        assert "loss" in out
        assert "d4_alignment_loss" in out
        assert "kld_loss" in out
        assert "mae_loss" in out
        assert "mse_loss" in out
        assert "signal_to_noise_ratio_loss" in out

    def test_no_permutation_loss_key(self):
        from src.models.components.model import Critic
        critic = Critic(make_critic_hparams())
        gt = make_batch()
        pred = make_batch()
        out = critic(
            graph_emb=torch.randn(2, 32),
            graph_true=gt,
            graph_pred=pred,
            mu=torch.randn(2, 32),
            logvar=torch.zeros(2, 32),
        )
        assert "permutation_loss" not in out

    def test_evaluate_adds_prefix(self):
        from src.models.components.model import Critic
        critic = Critic(make_critic_hparams())
        gt = make_batch()
        pred = make_batch()
        out = critic.evaluate(
            graph_emb=torch.randn(2, 32),
            graph_true=gt,
            graph_pred=pred,
            mu=torch.randn(2, 32),
            logvar=torch.zeros(2, 32),
            prefix="val",
        )
        assert "val_loss" in out
        assert "val_d4_alignment_loss" in out
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_critic_d4.py -v
```

Expected: `test_no_permutation_loss_key` FAILS (`permutation_loss` still in output). Other tests may fail due to signature mismatch.

- [ ] **Step 3: Replace `model.py` with the new implementation above**

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_critic_d4.py tests/test_d4_alignment_loss.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/components/model.py tests/test_critic_d4.py
git commit -m "feat: replace GraphReconstructionLoss+PermutationLoss with D4AlignmentLoss in Critic"
```

---

## Task 7: Update PLGraphAE — remove schedulers, fix batch-size refs, simplify viz

**Files:**
- Modify: `src/models/pigvae_auto_module.py`

### What to implement

**7a.** Update `__init__` — remove `temperature_scheduler` and `entropy_weight_scheduler`:

```python
def __init__(
    self,
    graph_ae: torch.nn.Module,
    critic: torch.nn.Module,
    kld_alpha_scheduler: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    compile: bool,
) -> None:
    super().__init__()
    self.save_hyperparameters(
        ignore=["graph_ae", "critic", "kld_alpha_scheduler"],
        logger=False,
    )
    self.graph_ae = graph_ae
    self.critic = critic
    self.kld_alpha_scheduler = kld_alpha_scheduler
    self.automatic_optimization = True
    self.validation_step_outputs: list[dict[str, Any]] = []
    self.test_step_outputs: list[dict[str, Any]] = []
```

Note: `self.perms` removed.

**7b.** Update `forward`:

```python
def forward(self, graph: DenseGraphBatch) -> tuple:
    graph_emb, graph_pred, mu, logvar = self.graph_ae(graph)
    return graph_emb, graph_pred, mu, logvar
```

**7c.** Remove `_apply_curriculum` method entirely.

**7d.** Update `training_step`:

```python
def training_step(self, graph: DenseGraphBatch, batch_idx: int) -> torch.Tensor:
    alpha = self.kld_alpha_scheduler(self.current_epoch)
    graph_emb, graph_pred, mu, logvar = self(graph=graph)
    loss = self.critic(
        graph_emb=graph_emb,
        graph_true=graph,
        graph_pred=graph_pred,
        kld_alpha=alpha,
        mu=mu,
        logvar=logvar,
    )
    self.log_dict(loss)
    if mu is not None:
        self._log_latent_stats(mu, logvar, alpha, prefix="")
    return loss["loss"]
```

**7e.** Update `validation_step`:

```python
def validation_step(self, graph: DenseGraphBatch, batch_idx: int) -> dict[str, Any]:
    alpha = self.kld_alpha_scheduler(self.current_epoch)
    graph_emb, graph_pred, mu, logvar = self(graph=graph)
    outputs = {
        "prediction": graph_pred,
        "ground_truth": graph,
        "graph_emb": graph_emb,
    }
    self.validation_step_outputs.append(outputs)
    batch_size = graph_pred.node_features.shape[0]
    metrics = self.critic.evaluate(
        graph_emb=graph_emb,
        graph_true=graph,
        graph_pred=graph_pred,
        kld_alpha=alpha,
        mu=mu,
        logvar=logvar,
        prefix="val",
    )
    metrics["alpha"] = alpha
    self.log_dict(
        metrics,
        sync_dist=True,
        on_epoch=True,
        on_step=False,
        batch_size=batch_size,
    )
    if mu is not None:
        self._log_latent_stats(mu, logvar, alpha, prefix="val_")
    return metrics
```

**7f.** Replace `on_validation_epoch_end` with simplified version:

```python
def on_validation_epoch_end(self) -> None:
    if self.trainer.is_global_zero:
        n_examples = 4
        predictions = self.validation_step_outputs[0]["prediction"].node_features
        ground_truths = self.validation_step_outputs[0]["ground_truth"].node_features
        graph_emb = self.validation_step_outputs[0]["graph_emb"]
        targets = self.validation_step_outputs[0]["ground_truth"].y

        batch_size = predictions.shape[0]
        n_show = min(n_examples, batch_size)

        pred_imgs = predictions[:n_show, :, :].detach().cpu()
        gt_imgs = ground_truths[:n_show, :, :].detach().cpu()
        diff = pred_imgs - gt_imgs

        pred_min, pred_max = pred_imgs.min().item(), pred_imgs.max().item()
        gt_min, gt_max = gt_imgs.min().item(), gt_imgs.max().item()
        vmin = min(pred_min, gt_min)
        vmax = max(pred_max, gt_max)
        diff_abs_max = diff.abs().max().item()

        fig_prediction = pL.plot_feature_map(pred_imgs, n_show, vmin=vmin, vmax=vmax)
        fig_ground_truth = pL.plot_feature_map(gt_imgs, n_show, vmin=vmin, vmax=vmax)
        fig_diff = pL.plot_feature_map(diff, n_show, vmin=-diff_abs_max, vmax=diff_abs_max)

        all_embs = torch.cat(
            [el["graph_emb"] for el in self.validation_step_outputs], dim=0
        ).detach().cpu().float().numpy()
        all_targets = torch.cat(
            [el["ground_truth"].y for el in self.validation_step_outputs]
        ).numpy()
        fig_pca = pL.plot_pca(all_embs, all_targets, n_rows=100, n_cols=8)

        wandb.log({
            "Predictions": [
                wandb.Image(fig, caption=f"Predictions {i + 1}")
                for i, fig in enumerate(fig_prediction)
            ],
            "Ground Truth": [
                wandb.Image(fig, caption=f"Ground Truth {i + 1}")
                for i, fig in enumerate(fig_ground_truth)
            ],
            "Diff": [
                wandb.Image(fig, caption=f"Diff {i + 1}")
                for i, fig in enumerate(fig_diff)
            ],
            "PCA": wandb.Image(fig_pca, caption="PCA"),
        })
        for fig in fig_prediction + fig_ground_truth + fig_diff:
            plt.close(fig)
        plt.close(fig_pca)
    self.validation_step_outputs.clear()
```

**7g.** Update `test_step`:

```python
def test_step(self, graph: DenseGraphBatch, batch_idx: int) -> None:
    graph_emb, graph_pred, _, _ = self(graph=graph)
    outputs = {
        "prediction": graph_pred,
        "ground_truth": graph,
        "graph_emb": graph_emb,
    }
    self.test_step_outputs.append(outputs)
```

**7h.** Update `on_test_epoch_end` — fix `predictions.shape[0] // 8` and remove `R.batch_augmented_indices`:

```python
def on_test_epoch_end(self) -> None:
    if self.trainer.is_global_zero:
        n_examples = 10
        predictions = self.test_step_outputs[0]["prediction"].node_features
        ground_truths = self.test_step_outputs[0]["ground_truth"].node_features
        batch_size = predictions.shape[0]
        n_show = min(n_examples, batch_size)

        graph_emb = torch.cat([el["graph_emb"] for el in self.test_step_outputs], dim=0)
        targets = np.concatenate(
            [el["ground_truth"].y.numpy() for el in self.test_step_outputs], axis=0
        )

        pred_imgs = predictions[:n_show, :, :].detach().cpu()
        gt_imgs = ground_truths[:n_show, :, :].detach().cpu()

        pca_predictions = graph_emb.detach().cpu().float().numpy()
        fig_pca = pL.plot_pca(pca_predictions, targets, n_rows=100, n_cols=8)
        fig_prediction = pL.plot_feature_map(pred_imgs, n_show)
        fig_ground_truth = pL.plot_feature_map(gt_imgs, n_show)

        wandb.log({
            "Test/Prediction": [wandb.Image(fig) for fig in fig_prediction],
            "Test/Ground Truth": [wandb.Image(fig) for fig in fig_ground_truth],
            "Test/PCA": wandb.Image(fig_pca, caption="PCA"),
        })
        for fig in fig_prediction + fig_ground_truth:
            plt.close(fig)
        plt.close(fig_pca)
    self.test_step_outputs.clear()
```

**7i.** Update `configure_gradient_clipping` — remove `ae.permuter`:

```python
def configure_gradient_clipping(
    self,
    optimizer: torch.optim.Optimizer,
    gradient_clip_val: float | None = None,
    gradient_clip_algorithm: str | None = None,
) -> None:
    ae = self.graph_ae
    component_max_norm = 5.0
    for component in (
        ae.encoder, ae.decoder,
        ae.bottle_neck_encoder, ae.bottle_neck_decoder,
    ):
        torch.nn.utils.clip_grad_norm_(component.parameters(), max_norm=component_max_norm)
    self.clip_gradients(
        optimizer,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
    )
```

**7j.** Update `predict_step`:

```python
def predict_step(self, batch: DenseGraphBatch, batch_idx: int) -> torch.Tensor:
    self.eval()
    with torch.no_grad():
        graph_emb, *_ = self(graph=batch)
        return graph_emb
```

- [ ] **Step 1: No separate test file needed** — the existing `test_train.py` smoke test (Task 9) will verify this.

- [ ] **Step 2: Apply all changes 7a–7j in `pigvae_auto_module.py`**

- [ ] **Step 3: Verify no syntax errors**

```
python -c "from src.models.pigvae_auto_module import PLGraphAE; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/models/pigvae_auto_module.py
git commit -m "feat: remove permuter schedulers from PLGraphAE, fix batch-size refs, simplify viz"
```

---

## Task 8: Update model config

**Files:**
- Modify: `configs/model/model.yaml`

### What to implement

Replace `configs/model/model.yaml` with the following (remove `permuter`, `temperature_scheduler`, `entropy_weight_scheduler` blocks; add `grid_size` to critic; remove `num_permutations` from `bottle_neck_encoder`; remove `perm_loss_scale`, `contrastive_loss_scale`, `temperature`, `num_aug_per_sample` from critic):

```yaml
_target_: "src.models.pigvae_auto_module.PLGraphAE"

graph_ae:
  _target_: "src.models.components.modules.GraphAE"
  hparams:
    input_size: 768
    num_heads: 4
    num_layers: 3
    emb_dim: 768
    vae: true
    dropout: 0.15
    encoder:
      graph_encoder_hidden_dim: ${model.graph_ae.hparams.input_size}
      num_node_features: 128
      num_edge_features: 0
      graph_encoder_num_heads: ${model.graph_ae.hparams.num_heads}
      graph_encoder_ppf_hidden_dim: ${multiply:${model.graph_ae.hparams.input_size},4}
      graph_encoder_num_layers: ${model.graph_ae.hparams.num_layers}
      emb_dim: ${model.graph_ae.hparams.emb_dim}
      dropout: ${model.graph_ae.hparams.dropout}
      grid_size: 6
      project: True

    decoder:
      graph_decoder_hidden_dim: ${model.graph_ae.hparams.input_size}
      graph_decoder_pos_emb_dim: ${model.graph_ae.hparams.input_size}
      graph_decoder_num_heads: ${model.graph_ae.hparams.num_heads}
      graph_decoder_ppf_hidden_dim: ${multiply:${model.graph_ae.hparams.input_size},4}
      graph_decoder_num_layers: ${model.graph_ae.hparams.num_layers}
      dropout: ${model.graph_ae.hparams.dropout}
      head_dim: ${divide:${model.graph_ae.hparams.input_size},${model.graph_ae.hparams.num_heads}}
      num_embeddings: 170
      num_node_features: 128
      num_edge_features: 0
      project: True
      grid_size: 6

    bottle_neck_encoder:
      graph_encoder_hidden_dim: ${model.graph_ae.hparams.input_size}
      emb_dim: ${model.graph_ae.hparams.emb_dim}
      vae: ${model.graph_ae.hparams.vae}
      activation: silu

    bottle_neck_decoder:
      emb_dim: ${model.graph_ae.hparams.emb_dim}
      graph_decoder_hidden_dim: ${model.graph_ae.hparams.input_size}
      num_nodes: 36

    property_predictor:
      emb_dim: ${model.graph_ae.hparams.emb_dim}
      property_predictor_hidden_dim: ${model.graph_ae.hparams.input_size}
      num_properties: 1

kld_alpha_scheduler:
  _target_: "src.models.components.schedulers.KLDAlphaScheduler"
  hparams:
    initial_alpha: 0.0
    final_alpha: 1.0
    mode: linear
    num_epochs: ${divide:${trainer.max_epochs},3}
    start_epoch: 1

critic:
  _target_: "src.models.components.model.Critic"
  hparams:
    kld_loss_scale: 0.05
    kld_free_bits: 0.0
    vae: ${model.graph_ae.hparams.vae}
    grid_size: 6
    huber_beta: 2.0
    alpha_scale: 1.0
    beta_scale: 0.1
    gamma_scale: 0.001

optimizer:
  _target_: torch.optim.AdamW
  _partial_: true
  lr: 0.0003
  weight_decay: 0.01

scheduler:
  batch_size: ${data.hparams.batch_size}
  type: cosine_warmup
  warmup: 0.05
  max_lr: 0.0003
  div_factor: 10
  final_div_factor: 100

compile: false
```

- [ ] **Step 1: Apply the config change above**

- [ ] **Step 2: Verify Hydra can load the config**

```
python -c "
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
with initialize(version_base='1.3', config_path='configs'):
    cfg = compose(config_name='train.yaml')
    print('model target:', cfg.model._target_)
    print('critic grid_size:', cfg.model.critic.hparams.grid_size)
    print('OK')
"
```

Expected output includes `model target: src.models.pigvae_auto_module.PLGraphAE` and `critic grid_size: 6`

- [ ] **Step 3: Commit**

```bash
git add configs/model/model.yaml configs/data/mnist.yaml
git commit -m "config: remove permuter/scheduler blocks, add grid_size to critic, single_view data"
```

---

## Task 9: End-to-end smoke test

**Files:**
- Read: existing `tests/test_train.py` (no changes needed)

### What to verify

Run the existing `test_train_fast_dev_run` test. This exercises one full training + validation step with Hydra config. It will catch any remaining wiring errors (wrong tuple sizes, missing config keys, shape mismatches).

- [ ] **Step 1: Run smoke test**

```
python -m pytest tests/test_train.py::test_train_fast_dev_run -v -s 2>&1 | head -80
```

Expected: PASS

If it fails with `KeyError` on a config key (e.g. `temperature_scheduler`), check if `train.yaml` still references removed keys and remove them.

- [ ] **Step 2: Run all new unit tests together**

```
python -m pytest tests/test_single_view_transform.py tests/test_single_view_datamodule.py tests/test_d4_alignment_loss.py tests/test_graph_ae_no_permuter.py tests/test_bottleneck_encoder.py tests/test_critic_d4.py -v
```

Expected: All PASS

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "test: verify all O2-VAE single-view components pass smoke test"
```

---

## Batch-Size Audit Reference

All `// 8` / `* 8` assumptions identified and addressed in the tasks above:

| Location | Old | Fixed in |
|---|---|---|
| `modules.py` `BottleNeckEncoder` lines 630–643 | `x.shape[0] // num_permutations` epsilon sharing | Task 5 |
| `pigvae_auto_module.py` line 276 | `predictions.shape[0] // 8` | Task 7h |
| `pigvae_auto_module.py` line 277–279 | `R.batch_augmented_indices(batch_size, num_permutations=8, ...)` | Task 7h |
| `pigvae_auto_module.py` lines 398, 400 | same in `on_test_epoch_end` | Task 7h |
| `graphs_datamodules.py` line 518 | `perm.squeeze(0)` → scalar | Task 1b |
| `configs/data/mnist.yaml` | `num_aug_per_sample: 8` | Task 2d |
| `configs/model/model.yaml` | `permuter:`, `temperature_scheduler:` etc. | Task 8 |
