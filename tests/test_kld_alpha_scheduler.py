import pytest
from omegaconf import OmegaConf

from src.models.components.schedulers import KLDAlphaScheduler


def _sched(**overrides):
    hparams = {"initial_alpha": 0.0, "final_alpha": 1.0, "mode": "linear", "num_epochs": 20}
    hparams.update(overrides)
    return KLDAlphaScheduler(OmegaConf.create(hparams))


def test_linear_default_start_is_backward_compatible():
    """start_epoch absent -> ramp begins at epoch 0 (old behaviour)."""
    s = _sched()
    assert s(0) == pytest.approx(0.0)
    assert s(10) == pytest.approx(0.5)
    assert s(20) == pytest.approx(1.0)


def test_start_epoch_holds_initial_then_ramps():
    """start_epoch=5 -> alpha stays at initial until epoch 5, then ramps over num_epochs."""
    s = _sched(start_epoch=5, num_epochs=20)
    # held at initial through the delay
    assert s(0) == pytest.approx(0.0)
    assert s(3) == pytest.approx(0.0)
    assert s(5) == pytest.approx(0.0)
    # ramp starts after start_epoch
    assert s(15) == pytest.approx(0.5)   # halfway: 5 + 20/2
    assert s(25) == pytest.approx(1.0)   # end: 5 + 20


def test_clamps_below_start_and_above_end():
    """Below start_epoch -> initial; well past the end -> final (no overshoot)."""
    s = _sched(start_epoch=5, num_epochs=20, initial_alpha=0.0, final_alpha=1.0)
    assert s(-3) == pytest.approx(0.0)
    assert s(100) == pytest.approx(1.0)


def test_constant_mode_ignores_schedule():
    s = _sched(mode="constant", final_alpha=0.3, start_epoch=5)
    assert s(0) == pytest.approx(0.3)
    assert s(50) == pytest.approx(0.3)
