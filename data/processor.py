import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm
from scipy.interpolate import RBFInterpolator
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

def apply_arbitrage_filters(df: pd.DataFrame, volume_filter: bool = True) -> pd.DataFrame:
    """
    Apply arbitrage, quality, and liquidity filters to options snapshot.
    """
    filtered = df.copy()
    initial_len = len(filtered)
    
    # 1. Ensure basic numerical columns are positive and not null
    filtered = filtered[
        (filtered["strike"] > 0) & 
        (filtered["spot_price"] > 0) & 
        (filtered["tau"] > 0)
    ]
    
    # 2. Hard IV limits: remove IVs outside [1%, 500%]
    filtered = filtered[
        (filtered["impliedVolatility"] >= 0.01) &
        (filtered["impliedVolatility"] <= 5.00)
    ]
    
    # 3. Liquidity & price sanity
    # For historical files we might allow zero volume, so volume_filter is configurable
    if volume_filter:
        filtered = filtered[filtered["volume"] > 0]
        
    filtered = filtered[
        (filtered["bid"] > 0) & 
        (filtered["ask"] > 0) & 
        (filtered["ask"] >= filtered["bid"])
    ]
    
    # 4. Bid-Ask spread ratio: remove options where (ask - bid) / mid > 50%
    filtered["mid"] = (filtered["bid"] + filtered["ask"]) / 2.0
    filtered["spread_ratio"] = (filtered["ask"] - filtered["bid"]) / filtered["mid"]
    filtered = filtered[filtered["spread_ratio"] < 0.50]
    
    if len(filtered) == 0:
        logger.warning("No contracts left after initial quality filtering.")
        return filtered

    # 5. Calendar spread check: total variance w = IV^2 * tau must be non-decreasing in tau
    # Assign each option to its closest moneyness bin
    filtered["kappa_bin"] = filtered["kappa"].apply(lambda k: GRID_KAPPAS[np.argmin(np.abs(GRID_KAPPAS - k))])
    filtered["total_variance"] = filtered["impliedVolatility"] ** 2 * filtered["tau"]
    
    calendar_valid = []
    # Check calendar spread separately for calls and puts to preserve structure
    for (opt_type, kappa_val), group in filtered.groupby(["option_type", "kappa_bin"]):
        group = group.sort_values("tau")
        tv_values = group["total_variance"].values
        tau_values = group["tau"].values
        
        valid_mask = [True]
        last_valid_tv = tv_values[0]
        
        for idx in range(1, len(tv_values)):
            # Total variance must be non-decreasing in expiry
            if tv_values[idx] >= last_valid_tv:
                valid_mask.append(True)
                last_valid_tv = tv_values[idx]
            else:
                valid_mask.append(False)
        calendar_valid.append(group[valid_mask])
        
    if calendar_valid:
        filtered = pd.concat(calendar_valid, ignore_index=True)
    else:
        filtered = pd.DataFrame(columns=filtered.columns)
        
    logger.info(f"Filtering completed: retained {len(filtered)} / {initial_len} contracts.")
    return filtered

def process_raw_snapshot(df: pd.DataFrame, volume_filter: bool = True) -> pd.DataFrame:
    """
    Perform moneyness calculations, BSM inversion verification, and assign tau coordinates.
    """
    processed = df.copy()
    
    # Calculate tau (days to expiry normalized by 252 trading days)
    # yfinance provides expiry as string 'YYYY-MM-DD'
    def calculate_tau(row):
        try:
            exp_date = datetime.strptime(row["expiry"], "%Y-%m-%d").date()
            snap_date = datetime.fromisoformat(row["timestamp"]).date()
            days = (exp_date - snap_date).days
            # Fallback to at least 1 day to avoid tau=0 division
            days = max(1, days)
            return days / 252.0
        except Exception:
            return np.nan
            
    processed["tau"] = processed.apply(calculate_tau, axis=1)
    processed = processed.dropna(subset=["tau"])
    
    # Calculate Forward Price F = S_0 * e^((r - q)*tau)
    processed["forward_price"] = processed["spot_price"] * np.exp(
        (processed["risk_free_rate"] - processed["dividend_yield"]) * processed["tau"]
    )
    
    # Calculate log-moneyness kappa = ln(K / F)
    processed["kappa"] = np.log(processed["strike"] / processed["forward_price"])
    
    # Re-calculate implied volatilities using custom Newton-Raphson solver to check/overwrite raw IVs
    def calculate_iv(row):
        mid_price = (row["bid"] + row["ask"]) / 2.0 if (pd.notna(row["bid"]) and pd.notna(row["ask"])) else row.get("lastPrice", np.nan)
        if pd.isna(mid_price) or mid_price <= 0:
            return row.get("impliedVolatility", np.nan)
            
        initial_iv = row.get("impliedVolatility")
        if pd.isna(initial_iv) or initial_iv is None or initial_iv <= 0:
            initial_iv = 0.20  # Sensible default guess for numerical solver
            
        solved_iv = implied_volatility_newton_raphson(
            market_price=mid_price,
            spot=row["spot_price"],
            strike=row["strike"],
            tau=row["tau"],
            r=row["risk_free_rate"],
            q=row["dividend_yield"],
            option_type=row["option_type"],
            initial_guess=initial_iv
        )
        # If solver fails to converge, fall back to initial value or NaN
        if pd.notna(solved_iv):
            return solved_iv
        return row.get("impliedVolatility", np.nan)
        
    processed["impliedVolatility"] = processed.apply(calculate_iv, axis=1)
    
    # Run no-arbitrage and quality filters
    filtered_df = apply_arbitrage_filters(processed, volume_filter=volume_filter)
    
    return filtered_df

def interpolate_to_grid(kappas: np.ndarray, taus: np.ndarray, ivs: np.ndarray) -> np.ndarray:
    """
    Interpolate scattered options coordinates (kappa, tau) onto the standard 7x7 grid.
    Utilizes Scipy's RBFInterpolator with a thin-plate spline kernel.
    """
    if len(ivs) < 5:
        raise ValueError(f"Insufficient options data points ({len(ivs)}) to perform RBF interpolation. Need at least 5.")
        
    points = np.column_stack([kappas, taus])
    
    # Fit the interpolator
    interpolator = RBFInterpolator(points, ivs, kernel="thin_plate_spline", smoothing=0.01)
    
    # Create the regular evaluation grid mesh
    grid_k, grid_t = np.meshgrid(GRID_KAPPAS, GRID_TAUS)
    grid_points = np.column_stack([grid_k.ravel(), grid_t.ravel()])
    
    # Predict and reshape back to grid dimensions
    grid_iv = interpolator(grid_points).reshape(grid_k.shape)
    
    # Clean boundaries: clip outputs to reasonable range [1%, 500%]
    return np.clip(grid_iv, 0.01, 5.0)
