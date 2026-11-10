"""Conditional skipping of tests that need a GPU or the IMC data."""

from importlib.util import find_spec
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]

# The IMC HDF5 files live on the cluster raid (configs/paths/szary.yaml), not in the repo.
IMC_PATHS = "szary"
IMC_DATA_DIR = Path(OmegaConf.load(ROOT / "configs" / "paths" / f"{IMC_PATHS}.yaml").root_dir)
IMC_DATA_AVAILABLE = (IMC_DATA_DIR / "data" / "IMC").is_dir()


def RunIf(min_gpus: int = 0, imc_data: bool = False, wandb: bool = False) -> pytest.MarkDecorator:  # noqa: N802
    """``skipif`` marker for tests that need GPUs, the IMC data or the ``wandb`` package.

    Example::

        @RunIf(min_gpus=1, imc_data=True)
        def test_train_gpu(cfg_train): ...
    """
    missing = []
    if min_gpus and torch.cuda.device_count() < min_gpus:
        missing.append(f"GPUs>={min_gpus}")
    if imc_data and not IMC_DATA_AVAILABLE:
        missing.append(f"IMC data under {IMC_DATA_DIR}")
    if wandb and find_spec("wandb") is None:
        missing.append("wandb")
    return pytest.mark.skipif(bool(missing), reason=f"Requires: [{' + '.join(missing)}]")
