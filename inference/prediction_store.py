import os
import threading
import numpy as np


class PredictionStore:
    """
    Persists forecasts keyed by their target date + horizon, so that once a
    real snapshot for that date is later observed, the drift monitor can
    compare the forecast made in advance against what actually happened
    (instead of comparing a prediction against itself).
    """

    def __init__(self, path: str, grid_shape: tuple = (7, 7)):
        self.path = path
        self.grid_shape = grid_shape
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _load(self):
        if not os.path.exists(self.path):
            return np.empty((0,), dtype="U64"), np.empty((0,) + self.grid_shape, dtype=np.float64)
        with np.load(self.path, allow_pickle=False) as data:
            return data["keys"], data["grids"]

    def _save(self, keys: np.ndarray, grids: np.ndarray) -> None:
        np.savez(self.path, keys=keys, grids=grids)

    @staticmethod
    def _make_key(target_date: str, horizon: int) -> str:
        return f"{target_date}__h{horizon}"

    def save_prediction(self, target_date: str, horizon: int, predicted_grid: np.ndarray) -> None:
        predicted_grid = np.asarray(predicted_grid, dtype=np.float64).reshape(self.grid_shape)
        key = self._make_key(target_date, horizon)
        with self._lock:
            keys, grids = self._load()
            keys = np.concatenate([keys, np.array([key], dtype="U64")])
            grids = np.concatenate([grids, predicted_grid[np.newaxis, ...]], axis=0)
            self._save(keys, grids)

    def pop_matching(self, observed_date: str) -> list:
        """
        Finds and removes all stored predictions whose target date equals
        `observed_date`, returning [{"horizon": h, "grid": grid}, ...].
        """
        prefix = f"{observed_date}__h"
        with self._lock:
            keys, grids = self._load()
            if len(keys) == 0:
                return []

            match_mask = np.array([k.startswith(prefix) for k in keys])
            if not match_mask.any():
                return []

            matched = []
            for key, grid in zip(keys[match_mask], grids[match_mask]):
                horizon = int(str(key).split("__h")[-1])
                matched.append({"horizon": horizon, "grid": grid})

            keep_mask = ~match_mask
            self._save(keys[keep_mask], grids[keep_mask])

        return matched
