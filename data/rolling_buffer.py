import os
import threading
import numpy as np


class SurfaceHistoryBuffer:
    """
    Persists a ring buffer of the most recent `lookback` real (processed,
    interpolated) volatility surface grids, so serving-time inference can be
    driven by actual submitted market data instead of synthetic filler.
    """

    def __init__(self, path: str, lookback: int, grid_shape: tuple = (7, 7)):
        self.path = path
        self.lookback = lookback
        self.grid_shape = grid_shape
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _load(self):
        if not os.path.exists(self.path):
            return np.empty((0,) + self.grid_shape, dtype=np.float64), np.empty((0,), dtype="U64")
        with np.load(self.path, allow_pickle=False) as data:
            return data["grids"], data["timestamps"]

    def _save(self, grids: np.ndarray, timestamps: np.ndarray) -> None:
        np.savez(self.path, grids=grids, timestamps=timestamps)

    def append(self, grid: np.ndarray, timestamp: str) -> None:
        grid = np.asarray(grid, dtype=np.float64).reshape(self.grid_shape)
        with self._lock:
            grids, timestamps = self._load()
            grids = np.concatenate([grids, grid[np.newaxis, ...]], axis=0)
            timestamps = np.concatenate([timestamps, np.array([timestamp], dtype="U64")])
            # Ring buffer: only the most recent `lookback` entries are ever needed for inference.
            if len(grids) > self.lookback:
                grids = grids[-self.lookback:]
                timestamps = timestamps[-self.lookback:]
            self._save(grids, timestamps)

    def get_window(self, degraded_padding: bool = False):
        """
        Returns (window, is_degraded).
        - window is None (cold start) if fewer than `lookback` real grids exist and
          degraded_padding is False.
        - Otherwise returns a (lookback, M, N) array. If padding was needed,
          is_degraded is True and the earliest real grid is repeated to fill the gap.
        """
        with self._lock:
            grids, _ = self._load()

        n = len(grids)
        if n >= self.lookback:
            return grids[-self.lookback:], False

        if not degraded_padding or n == 0:
            return None, False

        pad_count = self.lookback - n
        pad = np.repeat(grids[0][np.newaxis, ...], pad_count, axis=0)
        window = np.concatenate([pad, grids], axis=0)
        return window, True

    def count(self) -> int:
        with self._lock:
            grids, _ = self._load()
        return len(grids)
