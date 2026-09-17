import sys
import os
import tempfile
import shutil
import numpy as np
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.rolling_buffer import SurfaceHistoryBuffer


class TestSurfaceHistoryBuffer(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp_dir, "buffer.npz")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _grid(self, value):
        return np.full((7, 7), value, dtype=np.float64)

    def test_cold_start_returns_none_without_degraded_padding(self):
        buf = SurfaceHistoryBuffer(self.path, lookback=5)
        buf.append(self._grid(0.2), "2026-01-01")

        window, degraded = buf.get_window(degraded_padding=False)
        self.assertIsNone(window)
        self.assertFalse(degraded)
        self.assertEqual(buf.count(), 1)

    def test_degraded_padding_repeats_earliest_real_grid(self):
        buf = SurfaceHistoryBuffer(self.path, lookback=5)
        buf.append(self._grid(0.1), "2026-01-01")
        buf.append(self._grid(0.2), "2026-01-02")

        window, degraded = buf.get_window(degraded_padding=True)
        self.assertTrue(degraded)
        self.assertEqual(window.shape, (5, 7, 7))
        # First 3 slots are padding (repeat of the earliest real grid, value 0.1)
        np.testing.assert_array_equal(window[0], self._grid(0.1))
        np.testing.assert_array_equal(window[1], self._grid(0.1))
        np.testing.assert_array_equal(window[2], self._grid(0.1))
        # Last two slots are the real grids in order
        np.testing.assert_array_equal(window[3], self._grid(0.1))
        np.testing.assert_array_equal(window[4], self._grid(0.2))

    def test_ring_buffer_keeps_only_most_recent_lookback_and_persists(self):
        lookback = 3
        buf = SurfaceHistoryBuffer(self.path, lookback=lookback)
        for i in range(lookback + 2):
            buf.append(self._grid(float(i)), f"2026-01-{i+1:02d}")

        window, degraded = buf.get_window()
        self.assertFalse(degraded)
        self.assertEqual(window.shape, (lookback, 7, 7))
        # Oldest two entries (0, 1) should have been dropped; last 3 (2, 3, 4) remain.
        expected_values = [2.0, 3.0, 4.0]
        for i, val in enumerate(expected_values):
            np.testing.assert_array_equal(window[i], self._grid(val))

        # Durability: a fresh buffer instance pointed at the same path sees the same state.
        buf_reloaded = SurfaceHistoryBuffer(self.path, lookback=lookback)
        self.assertEqual(buf_reloaded.count(), lookback)
        window_reloaded, _ = buf_reloaded.get_window()
        np.testing.assert_array_equal(window_reloaded, window)


if __name__ == "__main__":
    unittest.main()
