"""Incremental writer for a fixed-shape ``.npy`` embedding matrix."""

import numpy as np


class MemmapWriter:
    """Pre-allocates an ``[n_rows, dim]`` ``.npy`` memmap and fills it in row blocks."""

    def __init__(self, path: str, n_rows: int, dim: int, dtype: str = "float32"):
        self.path = path
        self._arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(n_rows, dim))

    def write(self, start: int, rows: np.ndarray) -> None:
        self._arr[start : start + rows.shape[0]] = rows

    def close(self) -> None:
        self._arr.flush()
        del self._arr
