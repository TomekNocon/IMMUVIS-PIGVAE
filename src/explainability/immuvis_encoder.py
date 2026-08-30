"""Load the ImmuVis-768-UN-475 encoder and expose a differentiable raw-markers -> latent block.

UN-475 is the upstream ImmuVis model (colleague's `multiplex-image-model`, linked as a git
submodule pinned to branch ``immuvis_v0`` at ``external/multiplex-image-model``) that produced the
embeddings feeding PIGVAE. This module reuses that code WITHOUT copying it -- the submodule's flat
module dir is put on ``sys.path`` (the way its own ``generate_patches.py`` imports ``modules_vit``),
and the weights + arch config are loaded by path from szary/raid (never copied).

The point is version-B explainability: ``encode`` is a plain differentiable ``nn.Module`` forward,
so Captum Integrated Gradients can attribute a downstream target (a z_global dim, or the ABMIL
decision logit) all the way back to the raw 40+ IMC markers.

Arch (from probing the checkpoint): ConvNeXt-V2 marker-agnostic encoder + hyperkernel + conv
decoder = ``MultiplexAutoencoder`` (encoder_type ``convnext``). ``encode(x, channel_ids)`` returns
``{'output': latent}``; ``x`` is ``[B, C, 128, 128]`` and ``channel_ids`` is ``[B, C]`` marker
token ids (marker-agnostic: the shared backbone is conditioned on which marker each channel is).
"""

from __future__ import annotations

import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SUBMODULE = os.path.join(_REPO, "external/multiplex-image-model")
_ENCODER_DIR = os.path.join(_SUBMODULE, "evaluation/transformer_backbones")

# szary/raid defaults for UN-475 (loaded by path, never copied -- raid is ~96% full).
# Verified clean strict load (missing=0/unexpected=0), latent [B,768,16,16], gradients flow.
# NB: the arch config is keyed by the model BASE name (ImmuVis-768_model), not the run suffix;
# the *475-refactored* yaml is an older layout (hyperkernel alias) and does NOT load this ckpt.
DEFAULT_CKPT = "/raid_encrypted/immucan/models/ImmuVis-768_model-UN-475.pth"
DEFAULT_CONFIG = "/raid_encrypted/immucan/models/config.ImmuVis-768_model.yaml"
DEFAULT_TOKENIZER = os.path.join(_SUBMODULE, "configs/all_markers_tokenizer.yaml")
DEFAULT_PANELS = os.path.join(_SUBMODULE, "configs/all_panels_config.yaml")
NUM_CHANNELS = 265  # tokenizer vocab size the encoder was built with (per generate_patches.py)


def _ensure_importable() -> None:
    """Put the submodule's flat encoder dir on ``sys.path`` (matches its own import style)."""
    if not os.path.isdir(_ENCODER_DIR):
        raise FileNotFoundError(
            f"encoder submodule not found at {_ENCODER_DIR}. Run "
            f"`git submodule update --init external/multiplex-image-model`."
        )
    if _ENCODER_DIR not in sys.path:
        sys.path.insert(0, _ENCODER_DIR)


def load_immuvis(config_path: str = DEFAULT_CONFIG, ckpt_path: str = DEFAULT_CKPT,
                 device: str = "cpu", num_channels: int = NUM_CHANNELS):
    """Build ``MultiplexAutoencoder`` from the arch config and load UN-475 weights.

    Returns ``(model, (missing, unexpected))`` -- empty missing/unexpected means a clean strict
    load. Mirrors ``generate_patches.load_model_from_config`` but reads the yaml with OmegaConf
    (no ruamel) and reports the key diffs instead of silently falling back to non-strict.
    """
    _ensure_importable()
    from modules_vit import MultiplexAutoencoder  # noqa: E402 (submodule import)
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    enc, dec = cfg["encoder"], cfg["decoder"]
    model_config = {
        "num_channels": num_channels,
        "encoder_config": enc,
        "decoder_config": dec,
        "encoder_type": enc.get("encoder_type", "convnext"),
        "input_image_size": tuple(cfg.get("input_image_size", [128, 128])),
    }
    for key in ("vit_config", "swin_config", "dinov2_config", "dinov3_config"):
        if key in enc:
            model_config[key] = enc[key]

    model = MultiplexAutoencoder(**model_config)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, (list(missing), list(unexpected))


def encode(model, x: torch.Tensor, channel_ids: torch.Tensor) -> torch.Tensor:
    """Differentiable forward: ``[B, C, 128, 128]`` raw markers -> encoder latent (``['output']``)."""
    return model.encode(x, channel_ids)["output"]


def marker_channel_ids(marker_names, tokenizer_path: str = DEFAULT_TOKENIZER) -> torch.Tensor:
    """Map marker names to their encoder token ids (``[C]``) using the model's tokenizer."""
    from omegaconf import OmegaConf

    tok = OmegaConf.to_container(OmegaConf.load(tokenizer_path), resolve=True)
    if isinstance(tok, dict) and len(tok) == 1 and isinstance(next(iter(tok.values())), dict):
        tok = next(iter(tok.values()))  # unwrap a single top-level key if present
    return torch.tensor([int(tok[str(m)]) for m in marker_names], dtype=torch.long)


def dataset_channel_ids(dataset: str = "cords", panels_path: str = DEFAULT_PANELS,
                        tokenizer_path: str = DEFAULT_TOKENIZER):
    """Return ``(channel_ids [C], marker_names)`` for a panel dataset (e.g. ``cords`` -> 43)."""
    from omegaconf import OmegaConf

    panels = OmegaConf.to_container(OmegaConf.load(panels_path), resolve=True)
    names = list(panels["markers"][dataset])
    return marker_channel_ids(names, tokenizer_path), names
