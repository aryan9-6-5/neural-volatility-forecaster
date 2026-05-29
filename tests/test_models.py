import sys
import os
import unittest
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.architectures import StackedLSTM, ConvLSTM, TransformerEncoderModel
from models.loss import SmoothnessRegularizedLoss
from models.baselines import NaiveRandomWalk, HistoricalMean, ExponentialSmoothing, GARCHModel, HARRVModel

class TestForecastingModels(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 4
        self.lookback = 20
        self.horizon = 5
        self.M = 7
        self.N = 7
        
        # Create dummy inputs
        # X shape: (Batch, Lookback, Channels=1, M, N)
        self.X_np = np.random.uniform(0.15, 0.35, (self.batch_size, self.lookback, 1, self.M, self.N))
        self.X_torch = torch.from_numpy(self.X_np).float()
        
        # y shape: (Batch, Horizon, M, N)
        self.y_np = np.random.uniform(0.15, 0.35, (self.batch_size, self.horizon, self.M, self.N))
        self.y_torch = torch.from_numpy(self.y_np).float()

    def test_stacked_lstm_shape(self):
        """Verify StackedLSTM shape inputs, outputs, and parameters."""
        model = StackedLSTM(grid_size=(self.M, self.N), hidden_dim=64, num_layers=2, horizon=self.horizon)
        out = model(self.X_torch)
        self.assertEqual(out.shape, (self.batch_size, self.horizon, self.M, self.N))
        
    def test_conv_lstm_shape(self):
        """Verify ConvLSTM shape inputs, outputs, and parameters."""
        model = ConvLSTM(in_channels=1, hidden_dims=[16, 32], kernel_size=3, num_layers=2, horizon=self.horizon)
        out = model(self.X_torch)
        self.assertEqual(out.shape, (self.batch_size, self.horizon, self.M, self.N))
        
    def test_transformer_shape(self):
        """Verify TransformerEncoderModel shape inputs, outputs, and parameters."""
        model = TransformerEncoderModel(grid_size=(self.M, self.N), d_model=32, nhead=2, num_layers=2, horizon=self.horizon)
        out = model(self.X_torch)
        self.assertEqual(out.shape, (self.batch_size, self.horizon, self.M, self.N))
        
    def test_smoothness_loss_and_gradients(self):
        """Verify SmoothnessRegularizedLoss evaluates and flows gradients."""
        model = ConvLSTM(in_channels=1, hidden_dims=[8, 16], kernel_size=3, num_layers=2, horizon=self.horizon)
        criterion = SmoothnessRegularizedLoss(
            lambda_strike=0.05, 
            lambda_expiry=0.05, 
            lambda_calendar=0.05, 
            lambda_butterfly=0.05
        )
        
        y_pred = model(self.X_torch)
        # Ensure loss returns values and is differentiable
        loss, metrics = criterion(y_pred, self.y_torch)
        
        self.assertTrue(loss.item() > 0)
        self.assertIn("mse_loss", metrics)
        self.assertIn("strike_loss", metrics)
        self.assertIn("expiry_loss", metrics)
        self.assertIn("total_loss", metrics)
        
        # Test backward pass
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isnan(p.grad).sum() == 0)

    def test_statistical_baselines(self):
        """Verify baselines process and return appropriate forecast dimensions."""
        # 1. Random Walk
        rw = NaiveRandomWalk(horizon=self.horizon)
        rw_pred = rw.predict(self.X_np)
        self.assertEqual(rw_pred.shape, (self.batch_size, self.horizon, self.M, self.N))
        np.testing.assert_array_equal(rw_pred[:, 0, :, :], self.X_np[:, -1, 0, :, :])
        
        # 2. Historical Mean
        hm = HistoricalMean(horizon=self.horizon)
        hm_pred = hm.predict(self.X_np)
        self.assertEqual(hm_pred.shape, (self.batch_size, self.horizon, self.M, self.N))
        expected_mean = np.mean(self.X_np[:, :, 0, :, :], axis=1)
        np.testing.assert_allclose(hm_pred[:, 0, :, :], expected_mean, rtol=1e-5)
        
        # 3. Exponential Smoothing
        es = ExponentialSmoothing(horizon=self.horizon)
        es_pred = es.predict(self.X_np)
        self.assertEqual(es_pred.shape, (self.batch_size, self.horizon, self.M, self.N))
        
        # 4. HAR-RV
        har = HARRVModel(horizon=self.horizon)
        # fit on dummy training sequences
        har.fit(self.X_np, self.y_np)
        har_pred = har.predict(self.X_np)
        self.assertEqual(har_pred.shape, (self.batch_size, self.horizon, self.M, self.N))

if __name__ == "__main__":
    unittest.main()
