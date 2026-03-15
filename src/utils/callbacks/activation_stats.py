from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback
from torch import nn


class ActivationStatsCallback(Callback):
    def __init__(
        self,
        log_every_n_steps: int = 100,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        include_module_types: list[str] | None = None,
        log_on_train: bool = True,
        log_on_val: bool = False,
        log_on_test: bool = False,
    ) -> None:
        super().__init__()
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []
        self.include_module_types = include_module_types or []
        self.log_on_train = bool(log_on_train)
        self.log_on_val = bool(log_on_val)
        self.log_on_test = bool(log_on_test)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._stats: dict[str, dict[str, float]] = {}

    def setup(self, trainer, pl_module, stage: str) -> None:
        if self._handles:
            return
        self._register_hooks(pl_module)

    def teardown(self, trainer, pl_module, stage: str) -> None:
        self._remove_hooks()

    def on_fit_end(self, trainer, pl_module) -> None:
        self._remove_hooks()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self.log_on_train:
            return
        self._maybe_log(trainer, pl_module, batch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self.log_on_val:
            return
        self._maybe_log(trainer, pl_module, batch)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self.log_on_test:
            return
        self._maybe_log(trainer, pl_module, batch)

    def _register_hooks(self, root_module: nn.Module) -> None:
        for name, module in root_module.named_modules():
            if not self._should_monitor(name, module):
                continue
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _should_monitor(self, name: str, module: nn.Module) -> bool:
        if self.include_patterns and not any(pat in name for pat in self.include_patterns):
            return False
        if self.exclude_patterns and any(pat in name for pat in self.exclude_patterns):
            return False
        if module.__class__.__name__ in self.include_module_types:
            return True
        return isinstance(
            module,
            (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU, nn.ELU),
        )

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = self._extract_tensor(output)
            if tensor is None:
                return
            stats = self._tensor_stats(tensor)
            if stats:
                self._stats[name] = stats

        return hook

    @staticmethod
    def _extract_tensor(output: Any) -> torch.Tensor | None:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (list, tuple)):
            for item in output:
                if torch.is_tensor(item):
                    return item
        return None

    @staticmethod
    def _tensor_stats(tensor: torch.Tensor) -> dict[str, float] | None:
        if tensor.numel() == 0:
            return None
        with torch.no_grad():
            data = tensor.detach()
            zero_frac = (data.abs() < 1e-6).float().mean().item()
            return {
                "mean": data.mean().item(),
                "std": data.std(unbiased=False).item(),
                "min": data.min().item(),
                "max": data.max().item(),
                "zero_frac": zero_frac,
            }

    def _maybe_log(self, trainer, pl_module, batch: Any) -> None:
        if not self._stats:
            return
        if trainer.global_step % self.log_every_n_steps != 0:
            self._stats = {}
            return
        log_dict = self._pop_log_dict(prefix="activations")
        if not log_dict:
            return
        batch_size = self._infer_batch_size(batch)
        pl_module.log_dict(
            log_dict,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=batch_size,
        )

    def _pop_log_dict(self, prefix: str) -> dict[str, float]:
        log_dict: dict[str, float] = {}
        for name, stats in self._stats.items():
            safe_name = name.replace(".", "/")
            for key, value in stats.items():
                log_dict[f"{prefix}/{safe_name}/{key}"] = float(value)
        self._stats = {}
        return log_dict

    @staticmethod
    def _infer_batch_size(batch: Any) -> int | None:
        if hasattr(batch, "node_features") and torch.is_tensor(batch.node_features):
            return int(batch.node_features.shape[0])
        if torch.is_tensor(batch):
            return int(batch.shape[0])
        if isinstance(batch, Iterable):
            for item in batch:
                if torch.is_tensor(item):
                    return int(item.shape[0])
        return None
