"""Smoke tests of Hydra multirun sweeps through ``src/train.py`` (``fast_dev_run``)."""

import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import TEST_OVERRIDES
from tests.helpers.run_if import ROOT, RunIf


def _multirun(tmp_path: Path, *overrides: str) -> None:
    base = TEST_OVERRIDES
    if any(o.startswith("experiment=") for o in overrides):
        base = [o for o in base if not o.startswith("experiment=")]
    command = [
        sys.executable,
        "src/train.py",
        "-m",
        f"hydra.sweep.dir={tmp_path}",
        "++trainer.fast_dev_run=true",
        *base,
        *overrides,
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.fail(result.stderr[-5000:])


@RunIf(imc_data=True)
@pytest.mark.slow
def test_experiments(tmp_path: Path) -> None:
    """Every top-level experiment config runs."""
    _multirun(tmp_path, "experiment=glob(*)")


@RunIf(imc_data=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path: Path) -> None:
    """A basic grid sweep over the learning rate."""
    _multirun(tmp_path, "model.optimizer.lr=0.005,0.01")


@RunIf(imc_data=True)
@pytest.mark.slow
def test_optuna_sweep(tmp_path: Path) -> None:
    """The PIGVAE Optuna search (in-memory storage, a few trials)."""
    _multirun(
        tmp_path,
        "hparams_search=pigvae_optuna",
        "hydra.sweeper.storage=null",
        "hydra.sweeper.n_trials=3",
        "hydra.sweeper.sampler.n_startup_trials=2",
    )
