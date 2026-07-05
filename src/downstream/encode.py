"""Frozen PIGVAE loader for the downstream evaluation pipeline.

Reconstructs a `PLGraphAE` from its Hydra configs (mirroring the model-building
half of `scripts/diagnose_model.py::load_model_and_data`, minus the dataloader),
loads checkpoint weights, and hands back the frozen `graph_ae` submodule.
"""

from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

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
