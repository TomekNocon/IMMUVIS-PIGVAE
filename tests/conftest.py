"""Hydra config fixtures for the integration tests (train / eval / sweeps).

The configs compose the final model (``experiment=pigvae``) on the cluster paths with a tiny data
split, CPU training and the CSV logger; each test writes into its own ``tmp_path``.
"""

from pathlib import Path

import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.resolvers import register_resolvers
from tests.helpers.run_if import IMC_PATHS

register_resolvers()

# Shared by the in-process fixtures and the subprocess sweeps (tests/test_sweeps.py).
TEST_OVERRIDES = [
    "experiment=pigvae",
    f"paths={IMC_PATHS}",
    "logger=csv",
    "data.hparams.train_val_test_split=[32,16,16,0]",
    "data.hparams.batch_size=4",
    "data.hparams.num_workers=0",
    "trainer.accelerator=cpu",
    "trainer.devices=1",
    "+trainer.limit_train_batches=2",
    "+trainer.limit_val_batches=2",
    "+trainer.limit_test_batches=2",
    "extras.print_config=false",
    "extras.enforce_tags=false",
]


def _compose(config_name: str, overrides: list[str]) -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name=config_name, return_hydra_config=True, overrides=overrides)
    with open_dict(cfg):
        cfg.trainer.max_epochs = 1
    return cfg


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """Training config shared by the package (composed once)."""
    return _compose("train.yaml", TEST_OVERRIDES)


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """Evaluation config shared by the package (``ckpt_path`` is set by each test)."""
    overrides = [o for o in TEST_OVERRIDES if "limit_train" not in o and "limit_val" not in o]
    return _compose("eval.yaml", [*overrides, "ckpt_path=."])


def _with_tmp_paths(cfg: DictConfig, tmp_path: Path) -> DictConfig:
    cfg = cfg.copy()
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
    return cfg


@pytest.fixture
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    """Training config writing to this test's ``tmp_path``."""
    yield _with_tmp_paths(cfg_train_global, tmp_path)
    GlobalHydra.instance().clear()


@pytest.fixture
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> DictConfig:
    """Evaluation config writing to this test's ``tmp_path``."""
    yield _with_tmp_paths(cfg_eval_global, tmp_path)
    GlobalHydra.instance().clear()
