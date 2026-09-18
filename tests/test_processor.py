import sys
import os
import numpy as np
import pandas as pd
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.processor import (
    black_scholes_price,
    implied_volatility_newton_raphson,
    apply_arbitrage_filters,
    interpolate_to_grid,
    GRID_KAPPAS,
    GRID_TAUS
)

class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):
        # Sample parameters
        self.spot = 100.0
        self.strike = 100.0
        self.tau = 0.25 # 3 months
        self.r = 0.05   # 5% risk-free rate
        self.q = 0.02   # 2% dividend yield
        self.true_sigma = 0.30 # 30% volatility
        
    def test_bs_pricing_and_inversion(self):
        """Test BSM call/put pricing and subsequent numeric IV inversion."""
        # 1. Price Call
        call_price = black_scholes_price(
            spot=self.spot,
            strike=self.strike,
            tau=self.tau,
            r=self.r,
            q=self.q,
            sigma=self.true_sigma,
            option_type="call"
        )
        self.assertTrue(call_price > 0)
        
        # 2. Invert Call
        solved_call_iv = implied_volatility_newton_raphson(
            market_price=call_price,
            spot=self.spot,
            strike=self.strike,
            tau=self.tau,
            r=self.r,
            q=self.q,
            option_type="call"
        )
        self.assertAlmostEqual(solved_call_iv, self.true_sigma, places=5)
        
        # 3. Price Put
        put_price = black_scholes_price(
            spot=self.spot,
            strike=self.strike,
            tau=self.tau,
            r=self.r,
            q=self.q,
            sigma=self.true_sigma,
            option_type="put"
        )
        self.assertTrue(put_price > 0)
        
        # 4. Invert Put
        solved_put_iv = implied_volatility_newton_raphson(
            market_price=put_price,
            spot=self.spot,
            strike=self.strike,
            tau=self.tau,
            r=self.r,
            q=self.q,
            option_type="put"
        )
        self.assertAlmostEqual(solved_put_iv, self.true_sigma, places=5)

    def test_arbitrage_filtering(self):
        """Verify that options violating calendar spread requirements are filtered out."""
        # Create a dataframe representing options chain on the same kappa bin
        # Let's say:
        # T1 = 0.1, IV1 = 0.30 -> Total Variance = 0.30^2 * 0.1 = 0.009
        # T2 = 0.2, IV2 = 0.20 -> Total Variance = 0.20^2 * 0.2 = 0.008 (DECREASING! Calendar Spread Violation!)
        data = {
            "timestamp": ["2026-05-24T16:00:00", "2026-05-24T16:00:00"],
            "ticker": ["SPY", "SPY"],
            "spot_price": [100.0, 100.0],
            "strike": [100.0, 100.0],
            "expiry": ["2026-06-01", "2026-07-01"],
            "option_type": ["call", "call"],
            "impliedVolatility": [0.30, 0.20],
            "bid": [2.0, 2.2],
            "ask": [2.1, 2.3],
            "volume": [10, 20],
            "openInterest": [100, 200],
            "tau": [0.1, 0.2],
            "kappa": [0.0, 0.0]
        }
        df = pd.DataFrame(data)
        
        # Apply filters
        # Volume filtering enabled
        filtered_df = apply_arbitrage_filters(df, volume_filter=True)
        
        # The second option violates calendar spread (decreasing total variance: 0.008 < 0.009)
        # So only the first option (T=0.1) should remain!
        self.assertEqual(len(filtered_df), 1)
        self.assertEqual(filtered_df.iloc[0]["tau"], 0.1)

    def test_rbf_interpolation(self):
        """Test RBF grid interpolation output dimensions and boundary bounds."""
        # Create scattered coordinates
        np.random.seed(42)
        N_points = 50
        
        # Generate random scattered points in moneyness [-0.2, 0.2] and expiry [0.05, 0.9]
        kappas = np.random.uniform(-0.2, 0.2, N_points)
        taus = np.random.uniform(0.05, 0.9, N_points)
        ivs = 0.20 + 0.15 * kappas**2 + 0.05 * np.log(taus)
        ivs = np.clip(ivs, 0.05, 0.8) # Keep it positive
        
        # Run interpolation
        grid_iv = interpolate_to_grid(kappas, taus, ivs)
        
        # Check output properties
        self.assertEqual(grid_iv.shape, (7, 7))
        self.assertTrue(np.all(grid_iv >= 0.01))
        self.assertTrue(np.all(grid_iv <= 5.0))

    def test_rbf_interpolation_large_chain_matches_small_chain_default(self):
        """The neighbors-bounded default must be numerically identical to the
        unbounded global fit whenever point count <= neighbors (regression
        test for the interpolate_to_grid performance fix)."""
        np.random.seed(7)
        n = 40  # below the default neighbors=50 bound
        kappas = np.random.uniform(-0.2, 0.2, n)
        taus = np.random.uniform(0.05, 0.9, n)
        ivs = np.clip(0.20 + 0.15 * kappas**2 + 0.05 * np.log(taus), 0.05, 0.8)

        bounded = interpolate_to_grid(kappas, taus, ivs, neighbors=50)
        unbounded = interpolate_to_grid(kappas, taus, ivs, neighbors=None)
        np.testing.assert_array_equal(bounded, unbounded)

    def test_rbf_interpolation_falls_back_on_degenerate_local_neighborhood(self):
        """A dense single-expiry strike ladder (many points sharing the same
        tau) can make a local k-nearest-neighbors fit's monomial matrix
        rank-deficient (LinAlgError). interpolate_to_grid must recover via
        the global-fit fallback instead of raising, since the equivalent
        unbounded call would have succeeded fine."""
        np.random.seed(11)
        # 60 points all sharing one tau (a dense 0DTE-style strike ladder) plus
        # a few points at other expiries -- the nearest-50 neighborhood for a
        # grid query point near that ladder will be tau-constant.
        kappas_same_tau = np.random.uniform(-0.3, 0.3, 60)
        taus_same_tau = np.full(60, 0.05)
        ivs_same_tau = np.clip(0.20 + 0.15 * kappas_same_tau**2, 0.05, 0.8)

        kappas_other = np.random.uniform(-0.3, 0.3, 10)
        taus_other = np.random.uniform(0.2, 1.0, 10)
        ivs_other = np.clip(0.20 + 0.05 * np.log(taus_other), 0.05, 0.8)

        kappas = np.concatenate([kappas_same_tau, kappas_other])
        taus = np.concatenate([taus_same_tau, taus_other])
        ivs = np.concatenate([ivs_same_tau, ivs_other])

        grid_iv = interpolate_to_grid(kappas, taus, ivs, neighbors=50)
        self.assertEqual(grid_iv.shape, (7, 7))
        self.assertTrue(np.all(np.isfinite(grid_iv)))

if __name__ == "__main__":
    unittest.main()
