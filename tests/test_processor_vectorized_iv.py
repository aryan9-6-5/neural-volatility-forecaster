import sys
import os
import numpy as np
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.processor import (
    black_scholes_price,
    implied_volatility_newton_raphson,
    implied_volatility_newton_raphson_vec,
)


class TestVectorizedIVSolver(unittest.TestCase):

    def test_vectorized_matches_scalar_on_random_valid_inputs(self):
        """The batched Newton-Raphson solver must agree with the scalar, per-row
        version it replaces in data/processor.py::process_raw_snapshot."""
        rng = np.random.default_rng(7)
        n = 200

        spot = rng.uniform(50, 500, n)
        strike = spot * rng.uniform(0.7, 1.3, n)
        tau = rng.uniform(1 / 252, 2.0, n)
        r = rng.uniform(0.0, 0.06, n)
        q = rng.uniform(0.0, 0.03, n)
        true_sigma = rng.uniform(0.05, 1.5, n)
        is_call = rng.random(n) < 0.5

        market_price = np.array([
            black_scholes_price(spot[i], strike[i], tau[i], r[i], q[i], true_sigma[i],
                                 "call" if is_call[i] else "put")
            for i in range(n)
        ])

        initial_guess = np.full(n, 0.20)

        scalar_iv = np.array([
            implied_volatility_newton_raphson(
                market_price=market_price[i], spot=spot[i], strike=strike[i], tau=tau[i],
                r=r[i], q=q[i], option_type="call" if is_call[i] else "put",
                initial_guess=initial_guess[i]
            )
            for i in range(n)
        ])

        vec_iv = implied_volatility_newton_raphson_vec(
            market_price=market_price, spot=spot, strike=strike, tau=tau,
            r=r, q=q, is_call=is_call, initial_guess=initial_guess
        )

        both_valid = ~np.isnan(scalar_iv) & ~np.isnan(vec_iv)
        # The vast majority of random-but-valid cases should solve on both paths.
        self.assertGreater(both_valid.sum(), 0.9 * n)
        np.testing.assert_allclose(scalar_iv[both_valid], vec_iv[both_valid], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
