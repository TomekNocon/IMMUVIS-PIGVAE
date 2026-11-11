"""The ImmuVis encoder that produced the stored 16x16x768 embeddings, as a differentiable module.

The refactored ImmuVis-475 checkpoint reproduces the stored embeddings exactly (corr 1.0000,
max abs 6e-4 on 4 crops of slide 7064); it needs the refactored ImmuVis code, imported from the
``external/multiplex-image-model`` submodule (branch abmil). ``encode`` is a plain forward, so
Integrated Gradients can attribute a downstream target all the way back to the raw IMC markers.
"""

import sys

import torch
import yaml
from omegaconf import OmegaConf

from figlib import paths

NUM_CHANNELS = 265  # tokenizer vocabulary size the encoder was built with


def load_immuvis(device: str = "cpu") -> tuple[torch.nn.Module, tuple[list[str], list[str]]]:
    """Return ``(model, (missing_keys, unexpected_keys))``; empty key lists mean a clean load."""
    if str(paths.IMMUVIS_SUBMODULE) not in sys.path:
        sys.path.insert(0, str(paths.IMMUVIS_SUBMODULE))
    from multiplex_model.modules.immuvis import MultiplexAutoencoder

    with open(paths.IMMUVIS_CONFIG) as f:
        cfg = yaml.safe_load(f)
    encoder, decoder = dict(cfg["encoder"]), dict(cfg["decoder"])
    for part in (encoder, decoder):
        if "hyperkernel" in part:
            part["hyperkernel_config"] = part.pop("hyperkernel")
    model = MultiplexAutoencoder(
        num_channels=NUM_CHANNELS, encoder_config=encoder, decoder_config=decoder
    )
    ckpt = torch.load(paths.IMMUVIS_CKPT, map_location="cpu", weights_only=False)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, (list(result.missing_keys), list(result.unexpected_keys))


def encode(model: torch.nn.Module, x: torch.Tensor, channel_ids: torch.Tensor) -> torch.Tensor:
    """``[B, C, 128, 128]`` preprocessed markers -> encoder latent ``[B, 768, 16, 16]``."""
    return model.encode(x, channel_ids)["output"]


def channel_ids(dataset: str = "cords") -> tuple[torch.Tensor, list[str]]:
    """Encoder token ids ``[C]`` and marker names of a panel dataset (cords: 43 markers)."""
    names = list(
        OmegaConf.to_container(OmegaConf.load(paths.PANEL_CONFIG), resolve=True)["markers"][dataset]
    )
    tokens = OmegaConf.to_container(OmegaConf.load(paths.MARKER_TOKENIZER), resolve=True)
    if (
        isinstance(tokens, dict)
        and len(tokens) == 1
        and isinstance(next(iter(tokens.values())), dict)
    ):
        tokens = next(iter(tokens.values()))
    return torch.tensor([int(tokens[str(m)]) for m in names], dtype=torch.long), names
