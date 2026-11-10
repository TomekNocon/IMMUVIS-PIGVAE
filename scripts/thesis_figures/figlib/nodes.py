"""Colouring PCA node features of the 16x16 grid, shared by the pipeline, D4 and architecture figures."""

import numpy as np

from figlib import imc

PC = 1  # principal component shown on its own (0-based): PC2 tracks panCK here (r = -0.98)
NODE_RGB = [(-1, 1), (1, 0), (1, 3)]  # (sign, component) -> R, G, B: -PC2, PC1, PC4
HIGHLIGHT = "#d62728"  # the one cell followed through the pipeline
TRACKED_CELL = (5, 10)  # its (row, col) in the 16x16 grid


def node_rgb(
    z: np.ndarray, lo: float = 2, hi: float = 98, ref: np.ndarray | None = None
) -> np.ndarray:
    """Node features ``[C, 16, 16]`` -> RGB from three principal components (robust 2nd-98th pct).

    -PC2 drives red (tracks panCK, so tumour reads warm), PC1 green, PC4 blue. ``ref`` (e.g. the
    input when colouring a reconstruction) fixes the scale so the two are comparable.
    """
    ref = z if ref is None else ref
    channels = []
    for sign, k in NODE_RGB:
        v = sign * np.asarray(z[k], dtype=np.float32)
        r = sign * np.asarray(ref[k], dtype=np.float32)
        a, b = np.percentile(r, lo), np.percentile(r, hi)
        channels.append(np.clip((v - a) / (b - a), 0, 1))
    return np.stack(channels, -1)


def tinted(ch: np.ndarray, colour: str) -> np.ndarray:
    """One marker channel stretched and rendered in ``colour``: [H, W, 3]."""
    return imc.tint(ch.astype(np.float32), colour)
