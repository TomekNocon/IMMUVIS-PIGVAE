"""Integration tests: a few PIGVAE training steps on real IMC data (skipped without the data)."""

import math
from pathlib import Path

import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train import train
from tests.helpers.run_if import RunIf


def _run(cfg: DictConfig) -> dict:
    HydraConfig().set_config(cfg)
    metrics, _ = train(cfg)
    return metrics


@RunIf(imc_data=True)
def test_train_fast_dev_run(cfg_train: DictConfig) -> None:
    """One train, val and test step on CPU."""
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
    _run(cfg_train)


@RunIf(min_gpus=1, imc_data=True)
def test_train_fast_dev_run_gpu(cfg_train: DictConfig) -> None:
    """One train, val and test step on GPU."""
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
    _run(cfg_train)


@RunIf(imc_data=True)
@pytest.mark.slow
def test_train_logs_finite_losses(cfg_train: DictConfig) -> None:
    """One epoch (validation twice) logs finite training and validation losses."""
    with open_dict(cfg_train):
        cfg_train.trainer.val_check_interval = 0.5
    metrics = _run(cfg_train)
    for key in ("loss", "val_loss", "val_mse_loss", "val_kld_loss"):
        assert key in metrics, f"{key} not logged"
        assert math.isfinite(float(metrics[key])), f"{key} = {metrics[key]}"


@RunIf(imc_data=True)
@pytest.mark.slow
def test_train_resume(tmp_path: Path, cfg_train: DictConfig) -> None:
    """Train one epoch, then resume from ``last.ckpt`` for a second one."""
    _run(cfg_train)
    last = tmp_path / "checkpoints" / "last.ckpt"
    assert last.is_file()
    assert torch.load(last, map_location="cpu", weights_only=False)["epoch"] == 0

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(last)
        cfg_train.trainer.max_epochs = 2
    _run(cfg_train)
    assert torch.load(last, map_location="cpu", weights_only=False)["epoch"] == 1
