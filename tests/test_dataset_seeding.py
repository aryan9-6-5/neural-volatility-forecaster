import sys
import os
import numpy as np
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import generate_synthetic_dataset


class TestDatasetSeeding(unittest.TestCase):

    def test_same_seed_gives_identical_output(self):
        """Same seed must produce bit-identical synthetic surfaces across calls."""
        dataset_a, timestamps_a = generate_synthetic_dataset(num_days=15, seed=123)
        dataset_b, timestamps_b = generate_synthetic_dataset(num_days=15, seed=123)

        np.testing.assert_array_equal(dataset_a, dataset_b)
        self.assertEqual(timestamps_a, timestamps_b)

    def test_different_seeds_give_different_output(self):
        """Sanity check that seeding actually has an effect (guards against a no-op)."""
        dataset_a, _ = generate_synthetic_dataset(num_days=15, seed=1)
        dataset_b, _ = generate_synthetic_dataset(num_days=15, seed=2)

        self.assertFalse(np.array_equal(dataset_a, dataset_b))


if __name__ == "__main__":
    unittest.main()
