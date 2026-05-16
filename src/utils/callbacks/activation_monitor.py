from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback

log = logging.getLogger(__name__)


@dataclass
class _ActStats:
    mean: float
    std: float
    max_abs: float
    has_nan: bool
    has_inf: bool


def _safe_stats(t: torch.Tensor) -> _ActStats:
    t = t.detach()
    if not torch.is_floating_point(t):
        t = t.float()
    tf = t.float()
    return _ActStats(
        mean=float(tf.mean().item()),
        std=float(tf.std(unbiased=False).item()),
        max_abs=float(tf.abs().max().item()),
        has_nan=bool(torch.isnan(tf).any().item()),
        has_inf=bool(torch.isinf(tf).any().item()),
    )


def _resolve_attr(root: Any, dotted_path: str) -> Any:
    cur = root
    for part in dotted_path.split("."):
        cur = getattr(cur, part)
    return cur


def _actmon_forward_hook(module: Any, _inp: Any, out: Any) -> None:
    """Module-level forward hook (must be picklable for checkpointing)."""
    t = out[0] if isinstance(out, (tuple, list)) and len(out) > 0 else out
    if isinstance(t, torch.Tensor):
        module._actmon_latest = _safe_stats(t)


class ActivationMonitorCallback(Callback):
    """Logs lightweight activation health metrics to W&B (or any Lightning logger).

    Focuses on detecting exploding activations / NaNs early with minimal overhead.
    """

    def __init__(
        self,
        module_paths: list[str],
        log_every_n_steps: int = 200,
        max_abs_warn: float = 1_000.0,
        max_abs_alert: float = 5_000.0,
    ) -> None:
        super().__init__()
        self.module_paths = module_paths
        self.log_every_n_steps = int(log_every_n_steps)
        self.max_abs_warn = float(max_abs_warn)
        self.max_abs_alert = float(max_abs_alert)

        self._hooks: list[Any] = []
        self._modules: dict[str, Any] = {}

    def setup(self, trainer, pl_module, stage: str) -> None:
        if stage != "fit":
            return

        # Avoid duplicating hooks on re-setup.
        self.teardown(trainer, pl_module, stage)

        for path in self.module_paths:
            try:
                mod = _resolve_attr(pl_module, path)
            except AttributeError:
                if trainer.is_global_zero:
                    log.warning("ActivationMonitorCallback: module path %r not found, skipping.", path)
                continue

            # Store module reference for reading latest stats.
            self._modules[path] = mod
            # Ensure attribute exists even before first forward.
            mod._actmon_latest = None
            # Register a picklable (module-level) hook.
            self._hooks.append(mod.register_forward_hook(_actmon_forward_hook))

    # ---- Checkpoint safety -------------------------------------------------
    # Lightning saves callback state into checkpoints via `state_dict()`.
    # Forward-hook handles capture local (non-picklable) functions; never serialize them.
    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Nothing to restore; hooks will be re-registered in `setup()`.
        return

    def teardown(self, trainer, pl_module, stage: str) -> None:
        for h in getattr(self, "_hooks", []):
            with suppress(Exception):
                h.remove()
        self._hooks = []
        self._modules = {}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        if not trainer.is_global_zero:
            return
        if self.log_every_n_steps <= 0:
            return
        if (trainer.global_step % self.log_every_n_steps) != 0:
            return

        metrics: dict[str, float] = {}
        alert = 0.0

        for name, mod in self._modules.items():
            s = getattr(mod, "_actmon_latest", None)
            if s is None:
                continue
            metrics[f"actmon/{name}/mean"] = s.mean
            metrics[f"actmon/{name}/std"] = s.std
            metrics[f"actmon/{name}/max_abs"] = s.max_abs
            metrics[f"actmon/{name}/has_nan"] = 1.0 if s.has_nan else 0.0
            metrics[f"actmon/{name}/has_inf"] = 1.0 if s.has_inf else 0.0

            if s.has_nan or s.has_inf or s.max_abs >= self.max_abs_alert:
                alert = 1.0

        # One concise, always-present health signal.
        metrics["actmon/alert"] = alert
        pl_module.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=False)

        # Optional: warn (but don't spam) in console via a single metric.
        if alert > 0:
            pl_module.log("actmon/warn", 1.0, on_step=True, on_epoch=False, prog_bar=True)
