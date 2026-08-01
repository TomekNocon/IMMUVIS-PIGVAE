"""Frozen PIGVAE loader for the downstream evaluation pipeline.

Reconstructs a `PLGraphAE` from its Hydra configs (mirroring the model-building
half of `scripts/diagnose_model.py::load_model_and_data`, minus the dataloader),
loads checkpoint weights, and hands back the frozen `graph_ae` submodule.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.downstream.memmap_writer import MemmapWriter

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build_pl_module(experiment: str, paths_name: str = "szary"):
    configs = _PROJECT_ROOT / "configs"
    for name, fn in [("multiply", lambda x, y: int(x) * int(y)), ("divide", lambda x, y: int(x) // int(y))]:
        try:
            OmegaConf.register_new_resolver(name, fn)
        except Exception:
            pass
    model_cfg = OmegaConf.load(configs / "model" / "model.yaml")
    exp_cfg = OmegaConf.load(configs / "experiment" / f"{experiment}.yaml")
    if "model" in exp_cfg:
        model_cfg = OmegaConf.merge(model_cfg, exp_cfg.model)
    trainer_stub = OmegaConf.create({"max_epochs": 200, "min_epochs": 1})
    if "trainer" in exp_cfg:
        trainer_stub = OmegaConf.merge(trainer_stub, exp_cfg.trainer)
    # `model.yaml` interpolates ${data.hparams.*} (num_aug_per_sample, batch_size), so a
    # `data` stub must be present even though this loader never builds a dataloader.
    # Mirrors diagnose_model.py::load_model_and_data's data_stub + experiment `data` overlay.
    data_stub = OmegaConf.create({"hparams": {"num_aug_per_sample": 8, "batch_size": 16}})
    if "data" in exp_cfg:
        data_stub = OmegaConf.merge(data_stub, exp_cfg.data)
    ctx = OmegaConf.create({"model": model_cfg, "trainer": trainer_stub, "data": data_stub})
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
    """Load a PIGVAE checkpoint and return its frozen `graph_ae` submodule.

    The encoder is put in `.eval()` with `requires_grad_(False)` on all
    parameters. Never train/finetune this module downstream.
    """
    pl = _build_pl_module(experiment, paths_name)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = pl.load_state_dict(state.get("state_dict", state), strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    pl.eval()
    gae = pl.graph_ae
    gae.eval()
    for p in gae.parameters():
        p.requires_grad_(False)
    return gae


def build_pca_layer(pca_path: str, stats_path: str):
    """Construct the frozen, fitted PCALayer used by cords training.

    `zscore=False, clip_range=0.0` matches cords training's data hparams
    (see `configs/experiment/vae16_*.yaml`); never refit PCA downstream.
    """
    from src.data.components.graphs_datamodules import PCALayer

    return PCALayer(pca_path, stats_path, clip_range=0.0, zscore=False)


def source_subdir(feature_source: str, node_agg: str = "mean") -> str:
    """Canonical per-crop-feature subdir name for the downstream bundle.

    One frozen encoder can expose several per-crop features; each lands in its
    own `mil_embeddings/<subdir>/` (and matching meta_tables/results) so the
    four are compared under an identical bag/CV protocol:
      - `raw`          : spatial mean-pool of the raw 768-d IMC patch (baseline,
                         reproduces the old gated-ABMIL input; no encoder).
      - `zglobal`      : the 512-d CLS graph embedding (the trained-encoder byproduct).
      - `node_mean`    : per-node latent mean-pooled over the 256 nodes (node_z_dim).
      - `node_flatten` : per-node latent flattened (256 * node_z_dim).
    """
    if feature_source == "raw":
        return "raw"
    if feature_source == "zglobal":
        return "zglobal"
    if feature_source == "node":
        if node_agg not in ("mean", "flatten"):
            raise ValueError(f"node_agg must be 'mean' or 'flatten', got {node_agg!r}")
        return f"node_{node_agg}"
    raise ValueError(f"feature_source must be 'raw', 'zglobal', or 'node', got {feature_source!r}")


def patch_to_nodes(patch: np.ndarray) -> torch.Tensor:
    """Flatten a raw IMC patch `(C, H, W)` into training-order nodes `(H*W, C)`.

    Row-major over the grid: node `n` is grid cell `(row=n // W, col=n % W)`,
    holding all `C` channels of that cell. This matches
    `IMCBaseDictTransform.forward`'s `embedding.reshape(c, -1).T`
    (`src/data/components/graphs_datamodules.py:290`), which is the exact
    reshape training-time node features are built from before PCA/collation.
    """
    c, h, w = patch.shape
    return torch.from_numpy(patch.reshape(c, h * w).T.copy()).float()


def pca_transform(nodes: torch.Tensor, pca) -> torch.Tensor:
    """Apply the frozen, fitted PCA: `[B, 256, 768] -> [B, 256, 128]`."""
    return pca(nodes)


@torch.no_grad()
def encode_patches(
    gae,
    pca,
    patches: np.ndarray,
    device: str,
    feature_source: str = "zglobal",
    node_agg: str = "mean",
) -> np.ndarray:
    """Encode a batch of raw IMC patches `(B, 768, 16, 16)` to a per-crop feature `(B, D)`.

    `feature_source` selects which frozen representation each crop contributes to
    the downstream MIL bag (all deterministic, `sample=False`, batch-size-agnostic):

      - ``"raw"``      : spatial mean-pool of the raw patch -> `(B, 768)`. Baseline;
                         the encoder/PCA are NOT used (``gae``/``pca`` may be None).
      - ``"zglobal"``  : the 512-d CLS graph embedding `z_global` -> `(B, 512)`.
      - ``"node"``     : the per-node latent `z_nodes` `(B, 256, node_z_dim)`, aggregated
                         by ``node_agg``: ``"mean"`` -> `(B, node_z_dim)` (the pool
                         `forward` uses for logging), ``"flatten"`` -> `(B, 256*node_z_dim)`.

    For the encoder path: builds per-patch training-order nodes, applies the frozen
    fitted PCA, then runs the frozen encoder with an all-True `[B, 256]` mask (a full
    16x16 grid has no padding). The encoder's operative attention mask is its INTERNAL
    neighbor mask (modules.py:281-284); `DenseGraphBatch.mask` is discarded by the
    transformer but must be non-None so the CLS `F.pad(mask, (1, 0))` doesn't crash.
    """
    if feature_source == "raw":
        # Baseline: mean-pool the raw (B, C, H, W) patch over the H*W grid -> (B, C).
        # Reproduces the old gated-ABMIL input (768-d ImmuVis embedding, no encoder).
        p = np.asarray(patches, dtype=np.float32)
        return p.reshape(p.shape[0], p.shape[1], -1).mean(axis=2)

    from src.data.components.graphs_datamodules import DenseGraphBatch

    gae = gae.to(device)
    if isinstance(pca, torch.nn.Module):
        pca = pca.to(device)

    nodes = torch.stack([patch_to_nodes(p) for p in patches], dim=0).to(device)  # [B,256,768]
    nodes = pca(nodes)                                                           # [B,256,128]
    mask = torch.ones(nodes.shape[0], nodes.shape[1], dtype=torch.bool, device=device)
    batch = DenseGraphBatch(node_features=nodes, edge_features=torch.empty(0), mask=mask)
    z_nodes, z_global, *_ = gae.encode(batch, sample=False)  # z_nodes [B,256,node_z_dim]

    if feature_source == "zglobal":
        out = z_global
    elif feature_source == "node":
        if node_agg == "mean":
            out = z_nodes.mean(dim=1)                     # [B, node_z_dim]
        elif node_agg == "flatten":
            out = z_nodes.reshape(z_nodes.shape[0], -1)   # [B, 256*node_z_dim]
        else:
            raise ValueError(f"node_agg must be 'mean' or 'flatten', got {node_agg!r}")
    else:
        raise ValueError(f"feature_source must be 'raw', 'zglobal', or 'node', got {feature_source!r}")
    return out.detach().cpu().float().numpy()


def encode_h5(
    h5_path,
    gae,
    pca,
    out_emb,
    out_meta,
    device,
    batch_size: int = 64,
    feature_source: str = "zglobal",
    node_agg: str = "mean",
) -> None:
    """Encode an h5 of raw IMC patches to a streaming memmap + aligned metadata CSV.

    Reads `embeddings (N,768,16,16)`, `paths (N,)`, `positions (N,4)` from
    `h5_path`, encodes in batches of `batch_size` via `encode_patches`, and
    streams the resulting `(N, D)` embeddings to `out_emb` (a `MemmapWriter`
    memmap `.npy`). `D` is probed from one encoded patch, never hardcoded.
    Writes `out_meta` with columns `img_path, coords0, coords1,
    embeddings_file, embedding_idx`, where `embedding_idx == row index`
    (0..N-1), 1:1 aligned to the memmap rows, and `embeddings_file == out_emb`.
    Label join with `img_path` happens later, in Phase B.
    """
    with h5py.File(h5_path, "r") as f:
        n = f["embeddings"].shape[0]
        paths = [p.decode() if isinstance(p, bytes) else str(p) for p in f["paths"][:]]
        pos = f["positions"][:]
        # probe dim with one patch (D depends on feature_source/node_agg, never hardcoded)
        d = encode_patches(gae, pca, f["embeddings"][0:1], device, feature_source, node_agg).shape[1]
        writer = MemmapWriter(out_emb, n_rows=n, dim=d)
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            writer.write(s, encode_patches(gae, pca, f["embeddings"][s:e], device, feature_source, node_agg))
        writer.close()
    pd.DataFrame({
        "img_path": paths,
        "coords0": pos[:, 0], "coords1": pos[:, 1],
        "embeddings_file": out_emb,
        "embedding_idx": range(n),
    }).to_csv(out_meta, index=False)
