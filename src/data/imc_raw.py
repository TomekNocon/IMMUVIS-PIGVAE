"""Raw IMC images as the ImmuVis encoder saw them: preprocessing and 128 px crop tiling.

The stored PIGVAE inputs were made by tiling each preprocessed whole image into non-overlapping
``CROP x CROP`` crops in row-major order, dropping the remainder at the right and bottom edges.
"""

import numpy as np
from skimage.filters import butterworth

CROP = 128  # crop edge in pixels (1 um/px); the ImmuVis encoder input


def preprocess(img: np.ndarray) -> np.ndarray:
    """Encoder input transform on a whole image ``[C, H, W]`` of raw ion counts.

    ``arcsinh(x / 5)`` -> per-channel Butterworth low-pass (cutoff 0.2) -> ``clip(0, 4) / 4``,
    returned as float32 in [0, 1].
    """
    img = np.arcsinh(img.astype(np.float32) / 5.0)
    img = np.stack([butterworth(c, cutoff_frequency_ratio=0.2, high_pass=False) for c in img])
    return (np.clip(img, 0, 4) / 4).astype(np.float32)


def crops(img: np.ndarray, size: int = CROP) -> list[tuple[tuple[int, int], np.ndarray]]:
    """Non-overlapping ``size x size`` tiling of ``[C, H, W]``: ``[((row, col), crop)]``.

    ``(row, col)`` is the top-left pixel; order is row-major; partial tiles are dropped.
    """
    _, height, width = img.shape
    return [
        ((i, j), img[:, i : i + size, j : j + size])
        for i in range(0, height - size + 1, size)
        for j in range(0, width - size + 1, size)
    ]
