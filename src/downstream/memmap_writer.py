import numpy as np

class MemmapWriter:
    def __init__(self, path: str, n_rows: int, dim: int, dtype="float32"):
        self.path = path
        self._arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(n_rows, dim))
    def write(self, start: int, rows: np.ndarray) -> None:
        self._arr[start:start + rows.shape[0]] = rows
    def close(self) -> None:
        self._arr.flush(); del self._arr
