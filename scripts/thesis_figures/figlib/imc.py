"""Raw IMC image handling: the encoder's preprocessing, crop tiling and display stretching."""

from functools import cache

import matplotlib.colors
import numpy as np
import tifffile
import yaml

from figlib import paths
from src.data.imc_raw import CROP, crops, preprocess

__all__ = ["CROP", "crops", "preprocess"]  # re-exported for the figure scripts


@cache
def cords_markers() -> list[str]:
    """The 43 cords marker names in TIFF channel order."""
    with open(paths.PANEL_CONFIG) as f:
        return yaml.safe_load(f)["markers"]["cords"]


def load_slide(path) -> np.ndarray:
    """Read a cords TIFF and apply ``preprocess``: [43, H, W] in [0, 1]."""
    return preprocess(tifffile.imread(path))


def tiff_path(image: str):
    """Locate a cords TIFF (``<id>.tiff``) in the train/test split, else in the full export."""
    for split in ("train", "test"):
        candidate = paths.TIFF_SPLIT / split / "cords" / "imgs" / image
        if candidate.exists():
            return candidate
    return paths.TIFF_FULL / image


def stretch(
    ch: np.ndarray, lo: float = 1, hi: float = 99.5, ref: np.ndarray | None = None
) -> np.ndarray:
    """Percentile contrast stretch to [0, 1]; ``ref`` sets the percentiles (e.g. the whole slide)."""
    ref = ch if ref is None else ref
    a, b = np.percentile(ref, lo), np.percentile(ref, hi)
    return np.clip((ch - a) / (b - a), 0, 1) if b > a else np.zeros_like(ch)


def rgb(colour: str) -> np.ndarray:
    return np.array(matplotlib.colors.to_rgb(colour))


def tint(ch: np.ndarray, colour: str, ref: np.ndarray | None = None) -> np.ndarray:
    """One stretched channel rendered in ``colour``: [H, W, 3]."""
    return stretch(ch, ref=ref)[..., None] * rgb(colour)
