import sys
import os
import shutil
import tempfile
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
import inference.app as app_module
from inference.app import app
from data.rolling_buffer import SurfaceHistoryBuffer
from inference.prediction_store import PredictionStore

SAMPLE_PAYLOAD = {
    "timestamp": "2026-05-24T16:00:00",
    "spot_price": 745.64,
    "risk_free_rate": 0.035,
    "dividend_yield": 0.010,
    "contracts": [
        {"strike": 700.0, "expiry": "2026-06-01", "option_type": "put", "bid": 2.50, "ask": 2.60, "volume": 10},
        {"strike": 720.0, "expiry": "2026-06-01", "option_type": "put", "bid": 5.00, "ask": 5.20, "volume": 20},
        {"strike": 740.0, "expiry": "2026-06-01", "option_type": "call", "bid": 12.00, "ask": 12.20, "volume": 15},
        {"strike": 760.0, "expiry": "2026-06-01", "option_type": "call", "bid": 4.00, "ask": 4.10, "volume": 30},
        {"strike": 780.0, "expiry": "2026-06-01", "option_type": "call", "bid": 1.00, "ask": 1.05, "volume": 5},
        {"strike": 700.0, "expiry": "2026-09-01", "option_type": "put", "bid": 15.00, "ask": 15.50, "volume": 12},
        {"strike": 750.0, "expiry": "2026-09-01", "option_type": "call", "bid": 25.00, "ask": 25.50, "volume": 25},
        {"strike": 800.0, "expiry": "2026-09-01", "option_type": "call", "bid": 8.00, "ask": 8.20, "volume": 8}
    ]
}


class TestPredictDeterminism(unittest.TestCase):
    """Direct regression test for the old bug where /predict filled 59/60 lookback
    steps with fresh random synthetic data, making repeated identical calls
    non-deterministic. With a real rolling buffer, the same payload replayed
    against a warm buffer must produce bit-identical output."""

    @classmethod
    def setUpClass(cls):
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.lookback = 3

        self._orig_lookback = app_module.config.get("data", {}).get("lookback")
        self._orig_history_buffer = app_module.history_buffer
        self._orig_prediction_store = app_module.prediction_store

        app_module.config.setdefault("data", {})["lookback"] = self.lookback
        app_module.history_buffer = SurfaceHistoryBuffer(
            os.path.join(self.tmp_dir, "buffer.npz"), lookback=self.lookback
        )
        app_module.prediction_store = PredictionStore(os.path.join(self.tmp_dir, "predictions.npz"))

    def tearDown(self):
        app_module.config["data"]["lookback"] = self._orig_lookback
        app_module.history_buffer = self._orig_history_buffer
        app_module.prediction_store = self._orig_prediction_store
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_cold_start_then_deterministic_once_warm(self):
        # Cold start: fewer than `lookback` real snapshots submitted so far -> 425, not a guess.
        for _ in range(self.lookback - 1):
            resp = self.client.post("/predict?include_plots=false", json=SAMPLE_PAYLOAD)
            self.assertEqual(resp.status_code, 425)

        # The call that brings the buffer to exactly `lookback` real snapshots succeeds.
        first_ok = self.client.post("/predict?include_plots=false", json=SAMPLE_PAYLOAD)
        self.assertEqual(first_ok.status_code, 200)

        # A repeat call with the identical payload, buffer now warm and unchanged, must be
        # deterministic — this is what the old synthetic-filler bug violated.
        second_ok = self.client.post("/predict?include_plots=false", json=SAMPLE_PAYLOAD)
        self.assertEqual(second_ok.status_code, 200)

        self.assertEqual(first_ok.json()["predictions"], second_ok.json()["predictions"])


if __name__ == "__main__":
    unittest.main()
