# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import warnings
from scipy.stats import norm
from scipy.interpolate import RBFInterpolator
from scipy.optimize import brentq
import logging

logger = logging.getLogger(__name__)

# Grid constants defined by the project contract
GRID_KAPPAS = np.array([-0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30])
GRID_TAUS = np.array([1/52, 2/52, 1/12, 2/12, 3/12, 6/12, 1.0])

def black_scholes_price(spot: float, strike: float, tau: float, r: float, q: float, sigma: float, option_type: str = "call") -> float:
    """
    Calculate Black-Scholes European option price.
    """
    if tau <= 0 or sigma <= 0:
        return 0.0
        
    d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    if option_type.lower() == "call":
        price = spot * np.exp(-q * tau) * norm.cdf(d1) - strike * np.exp(-r * tau) * norm.cdf(d2)
    else:
        price = strike * np.exp(-r * tau) * norm.cdf(-d2) - spot * np.exp(-q * tau) * norm.cdf(-d1)
        
    return float(price)

def black_scholes_vega(spot: float, strike: float, tau: float, r: float, q: float, sigma: float) -> float:
    """
    Calculate Black-Scholes option Vega (derivative of price w.r.t volatility).
    """
    if tau <= 0 or sigma <= 0:
        return 0.0
        
    d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    vega = spot * np.exp(-q * tau) * np.sqrt(tau) * norm.pdf(d1)
    return float(vega)

def implied_volatility_newton_raphson(
    market_price: float, 
    spot: float, 
    strike: float, 
    tau: float, 
    r: float, 
    q: float, 
    option_type: str = "call", 
    initial_guess: float = 0.20, 
    max_iter: int = 100, 
    tol: float = 1e-6
) -> float:
    """
    Solve for implied volatility numerically using Newton-Raphson iteration.
    Falls back to Brent's root-finding method if Newton-Raphson fails to converge.
    """
    if tau <= 0 or market_price <= 0:
        return np.nan
        
    # Check intrinsic value lower bound
    intrinsic_val = max(0.0, spot * np.exp(-q * tau) - strike * np.exp(-r * tau)) if option_type.lower() == "call" else max(0.0, strike * np.exp(-r * tau) - spot * np.exp(-q * tau))
    if market_price <= intrinsic_val:
        return np.nan
        
    # Newton-Raphson Iteration
    sigma = initial_guess if (initial_guess and initial_guess > 0) else 0.20
    for _ in range(max_iter):
        price = black_scholes_price(spot, strike, tau, r, q, sigma, option_type)
        vega = black_scholes_vega(spot, strike, tau, r, q, sigma)
        
        if abs(vega) < 1e-8:
            break
            
        diff = price - market_price
        if abs(diff) < tol:
            return float(sigma)
            
        # Update step
        sigma_next = sigma - diff / vega
        
        # Keep volatility in reasonable bounds during search
        if sigma_next <= 0.001:
            sigma = 0.001
        elif sigma_next > 5.0:
            sigma = 5.0
        else:
            sigma = sigma_next

    # Fallback to Brent's method if Newton-Raphson didn't converge
    from scipy.optimize import brentq
    try:
        def objective(s):
            return black_scholes_price(spot, strike, tau, r, q, s, option_type) - market_price
        # Solve in interval [0.001, 5.0]
        return float(brentq(objective, 0.001, 5.0, xtol=tol))
    except Exception:
        # If Brent fails, return NaN
        return np.nan

def black_scholes_price_vec(spot: np.ndarray, strike: np.ndarray, tau: np.ndarray, r: np.ndarray,
                             q: np.ndarray, sigma: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    """Vectorized Black-Scholes price. Rows with tau<=0 or sigma<=0 price to 0.0 (matches the scalar version)."""
    spot, strike, tau, r, q, sigma = (np.asarray(a, dtype=float) for a in (spot, strike, tau, r, q, sigma))
    is_call = np.asarray(is_call, dtype=bool)

    valid = (tau > 0) & (sigma > 0)
    safe_tau = np.where(valid, tau, 1.0)
    safe_sigma = np.where(valid, sigma, 1.0)
    sqrt_tau = np.sqrt(safe_tau)

    d1 = (np.log(spot / strike) + (r - q + 0.5 * safe_sigma ** 2) * safe_tau) / (safe_sigma * sqrt_tau)
    d2 = d1 - safe_sigma * sqrt_tau

    call_price = spot * np.exp(-q * safe_tau) * norm.cdf(d1) - strike * np.exp(-r * safe_tau) * norm.cdf(d2)
    put_price = strike * np.exp(-r * safe_tau) * norm.cdf(-d2) - spot * np.exp(-q * safe_tau) * norm.cdf(-d1)
    price = np.where(is_call, call_price, put_price)
    return np.where(valid, price, 0.0)


def black_scholes_vega_vec(spot: np.ndarray, strike: np.ndarray, tau: np.ndarray, r: np.ndarray,
                            q: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Vectorized Black-Scholes vega. Rows with tau<=0 or sigma<=0 vega to 0.0 (matches the scalar version)."""
    spot, strike, tau, r, q, sigma = (np.asarray(a, dtype=float) for a in (spot, strike, tau, r, q, sigma))

    valid = (tau > 0) & (sigma > 0)
    safe_tau = np.where(valid, tau, 1.0)
    safe_sigma = np.where(valid, sigma, 1.0)
    sqrt_tau = np.sqrt(safe_tau)

    d1 = (np.log(spot / strike) + (r - q + 0.5 * safe_sigma ** 2) * safe_tau) / (safe_sigma * sqrt_tau)
    vega = spot * np.exp(-q * safe_tau) * sqrt_tau * norm.pdf(d1)
    return np.where(valid, vega, 0.0)


def implied_volatility_newton_raphson_vec(
    market_price: np.ndarray,
    spot: np.ndarray,
    strike: np.ndarray,
    tau: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    is_call: np.ndarray,
    initial_guess: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Batched Newton-Raphson IV solver across all rows simultaneously (same
    convergence logic as implied_volatility_newton_raphson, vectorized).
    Rows that fail to converge within max_iter fall back to a per-row
    scipy.optimize.brentq call, mirroring the scalar function's fallback.
    """
    market_price, spot, strike, tau, r, q, initial_guess = (
        np.asarray(a, dtype=float) for a in (market_price, spot, strike, tau, r, q, initial_guess)
    )
    is_call = np.asarray(is_call, dtype=bool)
    n = market_price.shape[0]
    result = np.full(n, np.nan)

    call_intrinsic = np.maximum(0.0, spot * np.exp(-q * tau) - strike * np.exp(-r * tau))
    put_intrinsic = np.maximum(0.0, strike * np.exp(-r * tau) - spot * np.exp(-q * tau))
    intrinsic = np.where(is_call, call_intrinsic, put_intrinsic)

    valid = (tau > 0) & (market_price > 0) & (market_price > intrinsic)

    sigma = np.where((initial_guess > 0) & ~np.isnan(initial_guess), initial_guess, 0.20)
    active = valid.copy()

    for _ in range(max_iter):
        if not np.any(active):
            break

        price = black_scholes_price_vec(spot, strike, tau, r, q, sigma, is_call)
        vega = black_scholes_vega_vec(spot, strike, tau, r, q, sigma)
        diff = price - market_price

        newly_converged = active & (np.abs(diff) < tol)
        result[newly_converged] = sigma[newly_converged]
        active &= ~newly_converged

        # Rows with ~zero vega can't be updated further (matches scalar's `break`);
        # they fall through to the brentq fallback below.
        stuck = active & (np.abs(vega) < 1e-8)
        active &= ~stuck

        if np.any(active):
            safe_vega = np.where(vega == 0, 1e-8, vega)
            sigma_next = sigma - diff / safe_vega
            sigma_next = np.clip(sigma_next, 0.001, 5.0)
            sigma = np.where(active, sigma_next, sigma)

    unresolved = valid & np.isnan(result)
    for i in np.where(unresolved)[0]:
        try:
            option_type_i = "call" if is_call[i] else "put"

            def objective(s, i=i, option_type_i=option_type_i):
                return black_scholes_price(spot[i], strike[i], tau[i], r[i], q[i], s, option_type_i) - market_price[i]

            result[i] = brentq(objective, 0.001, 5.0, xtol=tol)
        except Exception:
            result[i] = np.nan

    return result


def apply_arbitrage_filters(df: pd.DataFrame, volume_filter: bool = True, stats: dict = None) -> pd.DataFrame:
    """
    Apply arbitrage, quality, and liquidity filters to options snapshot.

    Args:
        stats: optional dict to record the row-count dropped at each filter stage
            (keys: positivity, iv_bounds, volume, bid_ask_validity, spread_ratio,
            calendar_spread). Pure instrumentation of the predicates below — does
            not change what gets filtered, only reports why, so a quality report
            built from this can never diverge from the actual filtering behavior.
    """
    filtered = df.copy()
    initial_len = len(filtered)
    prev_len = initial_len

    # 1. Ensure basic numerical columns are positive and not null
    filtered = filtered[
        (filtered["strike"] > 0) &
        (filtered["spot_price"] > 0) &
        (filtered["tau"] > 0)
    ]
    if stats is not None:
        stats["positivity"] = prev_len - len(filtered)
        prev_len = len(filtered)

    # 2. Hard IV limits: remove IVs outside [1%, 500%]
    filtered = filtered[
        (filtered["impliedVolatility"] >= 0.01) &
        (filtered["impliedVolatility"] <= 5.00)
    ]
    if stats is not None:
        stats["iv_bounds"] = prev_len - len(filtered)
        prev_len = len(filtered)

    # 3. Liquidity & price sanity
    # For historical files we might allow zero volume, so volume_filter is configurable
    if volume_filter:
        filtered = filtered[filtered["volume"] > 0]
        if stats is not None:
            stats["volume"] = prev_len - len(filtered)
            prev_len = len(filtered)
    elif stats is not None:
        stats["volume"] = 0

    filtered = filtered[
        (filtered["bid"] > 0) &
        (filtered["ask"] > 0) &
        (filtered["ask"] >= filtered["bid"])
    ]
    if stats is not None:
        stats["bid_ask_validity"] = prev_len - len(filtered)
        prev_len = len(filtered)

    # 4. Bid-Ask spread ratio: remove options where (ask - bid) / mid > 50%
    filtered["mid"] = (filtered["bid"] + filtered["ask"]) / 2.0
    filtered["spread_ratio"] = (filtered["ask"] - filtered["bid"]) / filtered["mid"]
    filtered = filtered[filtered["spread_ratio"] < 0.50]
    if stats is not None:
        stats["spread_ratio"] = prev_len - len(filtered)
        prev_len = len(filtered)

    if len(filtered) == 0:
        logger.warning("No contracts left after initial quality filtering.")
        if stats is not None:
            stats["calendar_spread"] = 0
        return filtered

    # 5. Calendar spread check: total variance w = IV^2 * tau must be non-decreasing in tau
    # Assign each option to its closest moneyness bin
    filtered["kappa_bin"] = filtered["kappa"].apply(lambda k: GRID_KAPPAS[np.argmin(np.abs(GRID_KAPPAS - k))])
    filtered["total_variance"] = filtered["impliedVolatility"] ** 2 * filtered["tau"]

    calendar_valid = []
    # Check calendar spread separately for calls and puts to preserve structure, grouping by strike
    for (opt_type, strike_val), group in filtered.groupby(["option_type", "strike"]):
        group = group.sort_values("tau")
        tv_values = group["total_variance"].values
        tau_values = group["tau"].values

        valid_mask = [True]
        last_valid_tv = tv_values[0]

        for idx in range(1, len(tv_values)):
            # Total variance must be non-decreasing in expiry.
            # If tau is the same (e.g. multiple options at same expiry), they don't form a calendar spread.
            if np.isclose(tau_values[idx], tau_values[idx-1]) or tv_values[idx] >= last_valid_tv:
                valid_mask.append(True)
                if not np.isclose(tau_values[idx], tau_values[idx-1]):
                    last_valid_tv = tv_values[idx]
            else:
                valid_mask.append(False)
        calendar_valid.append(group[valid_mask])

    if calendar_valid:
        filtered = pd.concat(calendar_valid, ignore_index=True)
    else:
        filtered = pd.DataFrame(columns=filtered.columns)

    if stats is not None:
        stats["calendar_spread"] = prev_len - len(filtered)

    logger.info(f"Filtering completed: retained {len(filtered)} / {initial_len} contracts.")
    return filtered

def process_raw_snapshot(df: pd.DataFrame, volume_filter: bool = True, stats: dict = None) -> pd.DataFrame:
    """
    Perform moneyness calculations, BSM inversion verification, and assign tau coordinates.
    Vectorized across all rows (no per-row .apply()) so it scales to full option chains.

    Args:
        stats: optional dict, forwarded to apply_arbitrage_filters to record a
            stage-by-stage rejection breakdown (see its docstring). Also gets
            "raw_contracts_count" (rows in df before any filtering) set here.
    """
    processed = df.copy()
    if stats is not None:
        stats["raw_contracts_count"] = len(processed)

    # Calculate tau (days to expiry normalized by 252 trading days).
    # yfinance provides expiry as string 'YYYY-MM-DD'; timestamp is ISO8601 (optionally 'Z'-suffixed).
    expiry_date = pd.to_datetime(processed["expiry"], format="%Y-%m-%d", errors="coerce").dt.normalize()
    timestamp_date = (
        pd.to_datetime(processed["timestamp"], errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    days = (expiry_date - timestamp_date).dt.days
    # Fallback to at least 1 day to avoid tau=0 division
    days = days.clip(lower=1)
    tau = days / 252.0
    processed["tau"] = tau.where(expiry_date.notna() & timestamp_date.notna(), np.nan)
    pre_tau_len = len(processed)
    processed = processed.dropna(subset=["tau"])
    if stats is not None:
        stats["date_parse_failure"] = pre_tau_len - len(processed)

    # Calculate Forward Price F = S_0 * e^((r - q)*tau)
    processed["forward_price"] = processed["spot_price"] * np.exp(
        (processed["risk_free_rate"] - processed["dividend_yield"]) * processed["tau"]
    )

    # Calculate log-moneyness kappa = ln(K / F)
    processed["kappa"] = np.log(processed["strike"] / processed["forward_price"])

    # Re-calculate implied volatilities using the vectorized Newton-Raphson solver to check/overwrite raw IVs
    bid = processed["bid"]
    ask = processed["ask"]
    last_price = processed["lastPrice"] if "lastPrice" in processed.columns else pd.Series(np.nan, index=processed.index)
    orig_iv = processed["impliedVolatility"] if "impliedVolatility" in processed.columns else pd.Series(np.nan, index=processed.index)

    has_bid_ask = bid.notna() & ask.notna()
    mid_price = ((bid + ask) / 2.0).where(has_bid_ask, last_price)

    valid_price_mask = mid_price.notna() & (mid_price > 0)
    initial_iv = orig_iv.where(orig_iv.notna() & (orig_iv > 0), 0.20)
    is_call = processed["option_type"].str.lower() == "call"

    solved = np.full(len(processed), np.nan)
    idx = valid_price_mask.to_numpy()
    if idx.any():
        solved[idx] = implied_volatility_newton_raphson_vec(
            market_price=mid_price.to_numpy()[idx],
            spot=processed["spot_price"].to_numpy()[idx],
            strike=processed["strike"].to_numpy()[idx],
            tau=processed["tau"].to_numpy()[idx],
            r=processed["risk_free_rate"].to_numpy()[idx],
            q=processed["dividend_yield"].to_numpy()[idx],
            is_call=is_call.to_numpy()[idx],
            initial_guess=initial_iv.to_numpy()[idx],
        )

    solved_series = pd.Series(solved, index=processed.index)
    # If the solver failed to converge (or the price was invalid to begin with), fall back to the original IV.
    processed["impliedVolatility"] = solved_series.where(solved_series.notna(), orig_iv)

    # Run no-arbitrage and quality filters
    filtered_df = apply_arbitrage_filters(processed, volume_filter=volume_filter, stats=stats)
    if stats is not None:
        stats["valid_contracts_count"] = len(filtered_df)

    return filtered_df

def interpolate_to_grid(kappas: np.ndarray, taus: np.ndarray, ivs: np.ndarray, neighbors: int = 50) -> np.ndarray:
    """
    Interpolate scattered options coordinates (kappa, tau) onto the standard 7x7 grid.
    Utilizes Scipy's RBFInterpolator with a thin-plate spline kernel.

    Args:
        neighbors: bounds the fit to a local k-nearest-neighbors RBF per query
            point instead of one global fit using every input point. A global
            fit's cost scales roughly cubically with point count (empirically
            ~1s at 2,000 points, ~60s at 8,000) -- a real problem for a full
            EOD option chain (thousands of contracts) or an unusually liquid
            live snapshot, both of which this function must handle without
            hanging. Verified bit-for-bit identical to the unbounded global
            fit (neighbors=None) whenever the point count is <= neighbors
            (i.e. no behavior change for the typical small live-snapshot
            case), and empirically <0.001 max IV difference on realistic
            smile-shaped test data even at high point counts where they
            diverge. Pass None to force the exact global fit.
    """
    if len(ivs) < 3:
        raise ValueError(f"Insufficient options data points ({len(ivs)}) to perform RBF interpolation. Need at least 3.")

    points = np.column_stack([kappas, taus])

    # Create the regular evaluation grid mesh
    grid_k, grid_t = np.meshgrid(GRID_KAPPAS, GRID_TAUS)
    grid_points = np.column_stack([grid_k.ravel(), grid_t.ravel()])

    # Fit and evaluate. A local (neighbors-bounded) fit can raise LinAlgError
    # on real chains where a query point's nearest neighbors are dominated by
    # a single expiry (e.g. SPY lists a dense 0DTE strike ladder every single
    # trading day) -- the resulting local point set has a near-constant tau,
    # making the default degree-1 monomial matrix (1, kappa, tau) rank
    # deficient. This is the common case on real SPY data, not a rare edge
    # case, so the recovery path must stay fast (same kernel, same neighbor
    # bound, just a relaxed polynomial degree) rather than jumping straight to
    # the much slower exact global fit -- which is kept only as a last-resort
    # safety net in case even that somehow still fails.
    try:
        interpolator = RBFInterpolator(points, ivs, kernel="thin_plate_spline", smoothing=0.01, neighbors=neighbors)
        grid_iv = interpolator(grid_points).reshape(grid_k.shape)
    except np.linalg.LinAlgError:
        if neighbors is None:
            raise
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # expected: "degree should not be below 1..."
                interpolator = RBFInterpolator(
                    points, ivs, kernel="thin_plate_spline", smoothing=0.01, neighbors=neighbors, degree=0
                )
                grid_iv = interpolator(grid_points).reshape(grid_k.shape)
        except np.linalg.LinAlgError:
            logger.warning(
                f"Local RBF fit (neighbors={neighbors}, degree=0) still hit a degenerate point set; "
                f"falling back to the exact global fit."
            )
            interpolator = RBFInterpolator(points, ivs, kernel="thin_plate_spline", smoothing=0.01, neighbors=None)
            grid_iv = interpolator(grid_points).reshape(grid_k.shape)
    
    # Clean boundaries: clip outputs to reasonable range [1%, 500%]
    return np.clip(grid_iv, 0.01, 5.0)
