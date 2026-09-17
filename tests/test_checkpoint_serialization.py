import sys
import os
import tempfile
import unittest
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.architectures import StackedLSTM
from models.serialization import save_checkpoint, load_checkpoint


class TestCheckpointSerialization(unittest.TestCase):

    def test_round_trip_preserves_forward_pass_and_metadata(self):
        """save_checkpoint -> load_checkpoint must reconstruct a model that behaves identically
        and carries back the serving/data-contract metadata that doesn't live in the state_dict."""
        torch.manual_seed(0)
        model_kwargs = dict(grid_size=(7, 7), hidden_dim=8, num_layers=1, horizon=2)
        model = StackedLSTM(**model_kwargs)
        model.eval()

        x = torch.randn(1, 5, 1, 7, 7)
        with torch.no_grad():
            out_before = model(x)

        extra_meta = {
            "data_contract_version": "v1.0",
            "lookback_window": 5,
            "horizons": [1, 2],
            "tensor_orientation": "(B, L, C, E, M)",
            "train_mean": 0.2,
            "train_std": 0.05,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "checkpoint_test.pt")
            save_checkpoint(model, checkpoint_path, "StackedLSTM", model_kwargs, extra_meta=extra_meta)

            self.assertTrue(os.path.exists(checkpoint_path))
            self.assertTrue(os.path.exists(checkpoint_path + ".json"))

            reloaded = load_checkpoint(checkpoint_path, map_location="cpu")

        with torch.no_grad():
            out_after = reloaded(x)

        self.assertTrue(torch.allclose(out_before, out_after))
        self.assertEqual(reloaded.data_contract_version, "v1.0")
        self.assertEqual(reloaded.lookback_window, 5)
        self.assertEqual(reloaded.horizons, [1, 2])
        self.assertEqual(reloaded.tensor_orientation, "(B, L, C, E, M)")
        self.assertAlmostEqual(reloaded.train_mean, 0.2)
        self.assertAlmostEqual(reloaded.train_std, 0.05)

    def test_missing_sidecar_raises_clear_error(self):
        """Legacy torch.save(model, ...) checkpoints (no sidecar) must fail loudly, not silently."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "legacy.pt")
            torch.save({"some": "tensor_dict"}, checkpoint_path)  # no sidecar written

            with self.assertRaises(FileNotFoundError):
                load_checkpoint(checkpoint_path)


if __name__ == "__main__":
    unittest.main()
