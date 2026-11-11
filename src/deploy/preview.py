"""RGB preview of a preprocessed IMC image, so a browser can show what was analysed.

Three markers become the red, green and blue channels. Each channel is contrast-stretched to its
99.5th percentile, and the image is downscaled so its long side is at most ``max_side`` pixels.
"""

import base64
import io
from collections.abc import Sequence
from typing import Any

import numpy as np

# Tumour (red), immune (green), nuclei (blue).
DEFAULT_PREVIEW_MARKERS = ("panCK", "CD45", "DNA1")
PREVIEW_MAX_SIDE = 1024


class UnknownMarkerError(ValueError):
    """A requested preview marker is not in the panel."""


def resolve_markers(names: Sequence[str], panel: Sequence[str]) -> list[int]:
    """Channel indices of three marker ``names`` in ``panel`` order."""
    if len(names) != 3:
        raise UnknownMarkerError(f"preview needs exactly 3 markers (R, G, B), got {len(names)}")
    unknown = [n for n in names if n not in panel]
    if unknown:
        raise UnknownMarkerError(f"unknown markers {unknown}; valid: {list(panel)}")
    return [list(panel).index(n) for n in names]


def _stretch(channel: np.ndarray) -> np.ndarray:
    hi = float(np.percentile(channel, 99.5))
    if hi <= 0:
        return np.zeros_like(channel)
    return np.clip(channel / hi, 0.0, 1.0)


def render_preview(
    pre: np.ndarray,
    names: Sequence[str],
    panel: Sequence[str],
    max_side: int = PREVIEW_MAX_SIDE,
) -> dict[str, Any]:
    """Preprocessed ``[C, H, W]`` image -> ``{png_base64, width, height, scale, markers}``.

    ``scale`` is preview pixels per image pixel, so a crop at image pixel ``(row, col)`` sits at
    ``(row * scale, col * scale)`` in the preview.
    """
    from PIL import Image

    idx = resolve_markers(names, panel)
    rgb = np.stack([_stretch(pre[i]) for i in idx], axis=-1)  # [H, W, 3] in [0, 1]
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    img = Image.fromarray((rgb * 255).round().astype(np.uint8), mode="RGB")
    if scale < 1.0:
        img = img.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return {
        "png_base64": base64.b64encode(buf.getvalue()).decode(),
        "width": img.width,
        "height": img.height,
        "scale": scale,
        "markers": list(names),
    }
