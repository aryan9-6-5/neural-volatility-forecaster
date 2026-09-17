import sys
import os
import pandas as pd
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import build_surface_dataset
from data.processor import process_raw_snapshot

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


@unittest.skipUnless(
    os.path.isdir(RAW_DIR) and any(f.endswith(".parquet") for f in os.listdir(RAW_DIR)),
    "No real snapshot parquet files present in data/raw/",
)
class TestRealDataSmoke(unittest.TestCase):
    """Exercises the real-data path against the actual collected parquet snapshots
    (no fetching) to make sure the vectorized processor doesn't crash on real data."""

    def test_build_surface_dataset_from_real_snapshots(self):
        dataset, timestamps = build_surface_dataset(RAW_DIR)
        self.assertGreaterEqual(dataset.shape[0], 1)
        self.assertEqual(dataset.shape[1:], (7, 7))
        self.assertEqual(len(timestamps), dataset.shape[0])

    def test_process_raw_snapshot_on_a_real_file(self):
        parquet_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".parquet")]
        sample_path = os.path.join(RAW_DIR, sorted(parquet_files)[0])
        raw_df = pd.read_parquet(sample_path)

        processed = process_raw_snapshot(raw_df, volume_filter=False)

        self.assertFalse(processed.empty)
        for col in ("kappa", "tau", "impliedVolatility"):
            self.assertIn(col, processed.columns)
        self.assertTrue((processed["impliedVolatility"] >= 0.01).all())
        self.assertTrue((processed["impliedVolatility"] <= 5.0).all())


if __name__ == "__main__":
    unittest.main()
