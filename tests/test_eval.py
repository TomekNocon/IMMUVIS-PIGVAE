"""Integration test: train one epoch, then evaluate the checkpoint with ``eval.py``."""

from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.eval import evaluate
from src.train import train
from tests.helpers.run_if import RunIf


@RunIf(imc_data=True)
@pytest.mark.slow
def test_train_eval(tmp_path: Path, cfg_train: DictConfig, cfg_eval: DictConfig) -> None:
    """The test loss after training equals the test loss of the saved checkpoint in eval.py."""
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.test = True
    HydraConfig().set_config(cfg_train)
    train_metrics, _ = train(cfg_train)
    assert (tmp_path / "checkpoints" / "last.ckpt").is_file()

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
    HydraConfig().set_config(cfg_eval)
    test_metrics, _ = evaluate(cfg_eval)

    assert test_metrics["test_loss"] > 0.0
    assert abs(float(train_metrics["test_loss"]) - float(test_metrics["test_loss"])) < 1e-3
