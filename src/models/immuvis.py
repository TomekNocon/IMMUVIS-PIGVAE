"""The ImmuVis encoder that produced the stored 16x16x768 crop embeddings.

The architecture comes from the ``external/multiplex-image-model`` submodule (branch ``abmil``,
the refactored code); with the ``ImmuVis-475-768-ma1-refactored`` weights it reproduces the
stored embeddings exactly. ``encode`` is a plain differentiable forward.
"""

import sys
from pathlib import Path

import torch
import yaml

DEFAULT_CODE_ROOT = Path(__file__).resolve().parents[2] / "external" / "multiplex-image-model"
NUM_CHANNELS = 265  # tokenizer vocabulary size the encoder was built with


def load_immuvis(
    ckpt_path: str | Path,
    config_path: str | Path,
    device: str = "cpu",
    code_root: str | Path = DEFAULT_CODE_ROOT,
) -> tuple[torch.nn.Module, tuple[list[str], list[str]]]:
    """Return ``(model, (missing_keys, unexpected_keys))``, in eval mode on ``device``.

    Empty key lists mean a clean load. ``code_root`` is the ImmuVis repository to import
    ``multiplex_model`` from.
    """
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from multiplex_model.modules.immuvis import MultiplexAutoencoder

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    encoder, decoder = dict(cfg["encoder"]), dict(cfg["decoder"])
    for part in (encoder, decoder):
        if "hyperkernel" in part:
            part["hyperkernel_config"] = part.pop("hyperkernel")
    model = MultiplexAutoencoder(
        num_channels=NUM_CHANNELS, encoder_config=encoder, decoder_config=decoder
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, (list(result.missing_keys), list(result.unexpected_keys))


def encode(model: torch.nn.Module, x: torch.Tensor, channel_ids: torch.Tensor) -> torch.Tensor:
    """``[B, C, 128, 128]`` preprocessed markers -> encoder latent ``[B, 768, 16, 16]``."""
    return model.encode(x, channel_ids)["output"]


def channel_ids(
    panel_config: str | Path, tokenizer: str | Path, dataset: str = "cords"
) -> tuple[torch.Tensor, list[str]]:
    """Encoder token ids ``[C]`` and marker names of a panel (cords: 43 markers, TIFF order)."""
    with open(panel_config) as f:
        names = list(yaml.safe_load(f)["markers"][dataset])
    with open(tokenizer) as f:
        tokens = yaml.safe_load(f)
    if isinstance(tokens, dict) and len(tokens) == 1:
        inner = next(iter(tokens.values()))
        if isinstance(inner, dict):
            tokens = inner
    return torch.tensor([int(tokens[str(m)]) for m in names], dtype=torch.long), names
