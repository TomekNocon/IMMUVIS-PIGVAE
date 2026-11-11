"""The ImmuVis encoder (``src.models.immuvis``) bound to the reference checkpoint and cords panel."""

import torch

from figlib import paths
from src.models import immuvis as _immuvis
from src.models.immuvis import NUM_CHANNELS, encode

__all__ = ["NUM_CHANNELS", "channel_ids", "encode", "load_immuvis"]


def load_immuvis(device: str = "cpu") -> tuple[torch.nn.Module, tuple[list[str], list[str]]]:
    """The reference ImmuVis encoder; returns ``(model, (missing_keys, unexpected_keys))``."""
    return _immuvis.load_immuvis(paths.IMMUVIS_CKPT, paths.IMMUVIS_CONFIG, device)


def channel_ids(dataset: str = "cords") -> tuple[torch.Tensor, list[str]]:
    """Encoder token ids ``[C]`` and marker names of a panel dataset (cords: 43 markers)."""
    return _immuvis.channel_ids(paths.PANEL_CONFIG, paths.MARKER_TOKENIZER, dataset)
