import sys
import os
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from inference.app import app

class TestServingAPI(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()
        
    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)
        
    def test_root_route(self):
        """Test GET / returns the dashboard HTML."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Neural Volatility Surface Forecaster", response.text)

    def test_health_endpoint(self):
        """Test GET /health returns standard healthy status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn(data["model_type"], ["Mock", "HAR-RV + LSTM Hybrid", "StackedLSTM", "ConvLSTM", "TransformerEncoderModel"])
        
    def test_drift_endpoint(self):
        """Test GET /monitor/drift returns drift metrics."""
        response = self.client.get("/monitor/drift")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("rolling_rmse", data)
        self.assertIn("rolling_kl_divergence", data)
        self.assertIn("retrain_recommended", data)
        
    def test_predict_endpoint(self):
        """Test POST /predict with a mock options snapshot payload."""
        payload = {
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
        
        # Call /predict with plots enabled
        response = self.client.post("/predict?include_plots=true", json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("forecast_horizons", data)
        self.assertIn("predictions", data)
        self.assertIn("drift_status", data)
        self.assertIn("plots", data)
        
        # Verify forecast keys
        self.assertIn("1", data["predictions"])
        self.assertIn("5", data["predictions"])
        self.assertIn("10", data["predictions"])
        
        # Verify grid shape (should be 7x7)
        grid_1 = data["predictions"]["1"]
        self.assertEqual(len(grid_1), 7)
        self.assertEqual(len(grid_1[0]), 7)
        
        # Verify plots JSON keys
        self.assertIn("surface_3d", data["plots"])
        self.assertIn("evolution_heatmap", data["plots"])
        self.assertIn("cross_sections", data["plots"])

if __name__ == "__main__":
    unittest.main()
