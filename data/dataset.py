import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from data.processor import process_raw_snapshot, interpolate_to_grid, GRID_KAPPAS, GRID_TAUS

logger = logging.getLogger(__name__)

def build_surface_dataset(snapshot_dir: str, volume_filter: bool = True) -> tuple:
    """
    Search snapshot_dir for Parquet files, process them sequentially,
    and build a time-ordered implied volatility surface tensor.
    Returns:
        dataset: np.ndarray of shape (T, 7, 7)
        timestamps: list of str (length T)
    """
    logger.info(f"Scanning {snapshot_dir} for snapshot files...")
    files = sorted(glob.glob(os.path.join(snapshot_dir, "*.parquet")))
    
    if not files:
        logger.warning(f"No snapshot files found in {snapshot_dir}.")
        return np.empty((0, 7, 7)), []
        
    surfaces = []
    timestamps = []
    
    for filepath in files:
        try:
            snap = pd.read_parquet(filepath)
            if snap.empty:
                continue
                
            processed = process_raw_snapshot(snap, volume_filter=volume_filter)
            if processed.empty:
                logger.warning(f"File {filepath} had no valid records after processing.")
                continue
                
            # Perform grid interpolation
            grid_iv = interpolate_to_grid(
                processed["kappa"].values,
                processed["tau"].values,
                processed["impliedVolatility"].values
            )
            
            surfaces.append(grid_iv)
            # Use the first timestamp in the snapshot
            timestamps.append(snap["timestamp"].iloc[0])
            logger.info(f"Successfully processed surface for {timestamps[-1]} from {os.path.basename(filepath)}")
            
        except Exception as e:
            logger.error(f"Error processing snapshot {filepath}: {e}")
            continue
            
    if not surfaces:
        return np.empty((0, 7, 7)), []
        
    dataset = np.stack(surfaces, axis=0) # Shape: (T, 7, 7)
    return dataset, timestamps

def import_historical_dataset(file_path_or_dir: str, volume_filter: bool = False) -> tuple:
    """
    Import and process historical option chain datasets (such as CSV or Parquet downloads).
    Attempts to map common column names dynamically to build standardized (T, 7, 7) tensors.
    
    Columns map dynamically:
        date/timestamp: 'date', 'quote_date', 'timestamp', 'datetime'
        strike: 'strike', 'strike_price'
        expiry: 'expiry', 'expiration', 'exdate'
        option_type: 'option_type', 'type', 'cp_flag' (C/P)
        bid: 'bid', 'bid_price'
        ask: 'ask', 'ask_price'
        lastPrice: 'lastPrice', 'last_price', 'price'
        impliedVolatility: 'impliedVolatility', 'implied_volatility', 'iv'
        spot_price: 'spot_price', 'underlying_price', 'spot'
    """
    logger.info(f"Importing historical dataset from {file_path_or_dir}...")
    
    # 1. Load data files
    if os.path.isdir(file_path_or_dir):
        files = glob.glob(os.path.join(file_path_or_dir, "*.parquet")) + glob.glob(os.path.join(file_path_or_dir, "*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV or Parquet files found in directory {file_path_or_dir}")
        dfs = []
        for f in sorted(files):
            if f.endswith(".parquet"):
                dfs.append(pd.read_parquet(f))
            else:
                dfs.append(pd.read_csv(f))
        df = pd.concat(dfs, ignore_index=True)
    else:
        if file_path_or_dir.endswith(".parquet"):
            df = pd.read_parquet(file_path_or_dir)
        elif file_path_or_dir.endswith(".csv"):
            df = pd.read_csv(file_path_or_dir)
        else:
            raise ValueError("Supported historical formats are CSV or Parquet.")
            
    if df.empty:
        raise ValueError("Loaded historical dataset is empty.")

    return import_historical_dataframe(df, volume_filter=volume_filter)


def import_historical_dataframe(df: pd.DataFrame, volume_filter: bool = False, collect_stats: bool = False) -> tuple:
    """
    Core of import_historical_dataset, split out so an in-memory DataFrame
    (e.g. already assembled by a dataset-specific adapter, such as
    data.adapters.spy_options_dataset) can be processed directly without a
    round-trip through disk. Column-mapping/dynamic-synonym behavior is
    identical to import_historical_dataset's docstring.

    Args:
        collect_stats: if True, also returns a list of per-day dicts (one per
            unique timestamp) with raw/valid contract counts, a stage-by-stage
            rejection breakdown (see data.processor.apply_arbitrage_filters),
            and a skip_reason ("insufficient_valid_contracts" or
            "interpolation_error") for days that produced no surface.

    Returns:
        (dataset, timestamps) normally, or (dataset, timestamps, per_day_stats)
        when collect_stats=True.
    """
    df = df.copy()

    # 2. Dynamic column schema mapping
    col_mapping = {}

    # helper mapping keys
    synonyms = {
        "timestamp": ["date", "quote_date", "timestamp", "datetime", "QuoteDate"],
        "strike": ["strike", "strike_price", "Strike", "StrikePrice"],
        "expiry": ["expiry", "expiration", "exdate", "ExpiryDate", "Expiration"],
        "option_type": ["option_type", "type", "cp_flag", "Type", "OptionType"],
        "bid": ["bid", "bid_price", "Bid"],
        "ask": ["ask", "ask_price", "Ask"],
        "lastPrice": ["lastPrice", "last_price", "price", "LastPrice"],
        "impliedVolatility": ["impliedVolatility", "implied_volatility", "iv", "ImpliedVolatility"],
        "spot_price": ["spot_price", "underlying_price", "spot", "UnderlyingPrice", "Spot"]
    }

    for standard_name, options in synonyms.items():
        matched = False
        for opt in options:
            if opt in df.columns:
                col_mapping[opt] = standard_name
                matched = True
                break
        if not matched and standard_name in ["timestamp", "strike", "expiry", "option_type"]:
            raise ValueError(f"Required column representing '{standard_name}' could not be matched. Columns found: {list(df.columns)}")

    # Rename columns to standard names
    df = df.rename(columns=col_mapping)

    # Standardize option type values (C/call -> call, P/put -> put)
    def clean_opt_type(val):
        val_str = str(val).lower()
        if 'c' in val_str:
            return "call"
        elif 'p' in val_str:
            return "put"
        return "call"

    df["option_type"] = df["option_type"].apply(clean_opt_type)

    # Fill standard auxiliary fields if missing
    if "risk_free_rate" not in df.columns:
        df["risk_free_rate"] = 0.045 # Default fallback
    if "dividend_yield" not in df.columns:
        df["dividend_yield"] = 0.015 # Default fallback
    if "spot_price" not in df.columns:
        # If spot price is completely missing, approximate from ATM option strikes, or raise error
        raise ValueError("Missing 'spot_price' column in historical data.")

    # Standardize timestamp to string ISO format
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Sort chronologically
    df = df.sort_values("timestamp")

    # 3. Process day-by-day snapshots
    surfaces = []
    timestamps = []
    per_day_stats = []

    grouped = df.groupby("timestamp")
    total_groups = len(grouped)
    logger.info(f"Processing {total_groups} unique historical daily snapshots...")

    idx = 0
    for ts, snap in grouped:
        day_stats = {} if collect_stats else None
        try:
            processed = process_raw_snapshot(snap, volume_filter=volume_filter, stats=day_stats)
            if len(processed) < 5:
                if day_stats is not None:
                    day_stats["skip_reason"] = "insufficient_valid_contracts"
                    per_day_stats.append({"date": ts, **day_stats})
                continue

            if day_stats is not None:
                from data.historical_quality_report import compute_coverage_pct
                day_stats["coverage_pct"] = compute_coverage_pct(processed["kappa"].values, processed["tau"].values)

            grid_iv = interpolate_to_grid(
                processed["kappa"].values,
                processed["tau"].values,
                processed["impliedVolatility"].values
            )
            surfaces.append(grid_iv)
            timestamps.append(ts)
            if day_stats is not None:
                day_stats["skip_reason"] = None
                per_day_stats.append({"date": ts, **day_stats})

            idx += 1
            if idx % 50 == 0 or idx == total_groups:
                logger.info(f"Processed {idx}/{total_groups} snapshots...")
        except Exception as e:
            logger.error(f"Error processing historical date {ts}: {e}")
            if day_stats is not None:
                day_stats["skip_reason"] = "interpolation_error"
                per_day_stats.append({"date": ts, **day_stats})
            continue

    if not surfaces:
        raise ValueError("No historical surfaces could be successfully constructed.")

    dataset = np.stack(surfaces, axis=0)
    logger.info(f"Historical import complete. Built surface tensor of shape {dataset.shape}")

    if collect_stats:
        return dataset, timestamps, per_day_stats
    return dataset, timestamps

def create_sequences(data: np.ndarray, lookback: int, horizon: int) -> tuple:
    """
    Format chronologically stacked surfaces into sequence tensors for training/evaluation.
    Args:
        data: Volatility surface tensor of shape (T, M, N)
        lookback: Input length L
        horizon: Forecasting horizon h
    Returns:
        X: Sequence input array of shape (N_samples, L, 1, M, N)
        y: Future target array of shape (N_samples, h, M, N)
    """
    T, M, N = data.shape
    if T < (lookback + horizon):
        raise ValueError(f"Dataset length T={T} is too short for lookback={lookback} and horizon={horizon}")
        
    num_samples = T - lookback - horizon + 1
    
    # Initialize tensors
    X = np.zeros((num_samples, lookback, 1, M, N), dtype=np.float32)
    y = np.zeros((num_samples, horizon, M, N), dtype=np.float32)
    
    for i in range(num_samples):
        # Extract past lookback windows (add channel dimension for ConvLSTM input)
        X[i, :, 0, :, :] = data[i : i + lookback]
        # Extract forecast target windows
        y[i, :, :, :] = data[i + lookback : i + lookback + horizon]
        
    return X, y

def generate_synthetic_dataset(num_days: int = 120, seed: int = None) -> tuple:
    """
    Generate a highly realistic synthetic volatility surface time-series.
    Incorporates Markov regime shifts (Calm vs. Crisis), Poisson jumps,
    leverage effects (skew-level coupling), and GARCH-like volatility clustering.

    Formula for surface at time t:
        IV(kappa, tau) = Level_t - Skew_t * kappa + Curvature_t * kappa^2 + Term_t * ln(tau / 0.25)

    Args:
        num_days: number of daily surfaces to generate.
        seed: RNG seed. Pass an explicit seed for reproducible output (e.g. for
            training/evaluation runs); leave as None for non-deterministic
            output (e.g. ad-hoc exploration).
    """
    logger.info(f"Generating realistic regime-switching synthetic volatility surface dataset for {num_days} days...")
    rng = np.random.default_rng(seed)
    
    # Latent state variables
    level = 0.18
    skew = 0.05
    term = 0.03
    regime = 0  # 0: Calm, 1: Stressed/Crisis
    
    # Jump decay state
    jump_effect = 0.0
    
    surfaces = []
    timestamps = []
    base_date = datetime.now() - timedelta(days=num_days)
    
    # GARCH-like conditional variance
    cond_vol = 0.015
    
    for day in range(num_days):
        # 1. Markov Regime Transition
        if regime == 0:
            if rng.random() < 0.04:  # Calm -> Stress transition
                regime = 1
        else:
            if rng.random() < 0.12:  # Stress -> Calm transition
                regime = 0
                
        # 2. Setup regime-specific parameters
        if regime == 0:
            mu_l, mu_s, mu_t = 0.16, 0.05, 0.03
            phi_l, phi_s, phi_t = 0.95, 0.90, 0.93
            sigma_l_base, sigma_s, sigma_t = 0.010, 0.004, 0.003
        else:
            mu_l, mu_s, mu_t = 0.35, 0.14, -0.02  # Inverted term structure in crisis
            phi_l, phi_s, phi_t = 0.97, 0.92, 0.95
            sigma_l_base, sigma_s, sigma_t = 0.025, 0.009, 0.006

        # 3. GARCH Volatility Clustering for level shocks
        cond_vol = 0.6 * cond_vol + 0.3 * (level - mu_l)**2 + 0.1 * sigma_l_base**2
        cond_vol = np.clip(np.sqrt(cond_vol), 0.005, 0.04)

        # 4. Level shock & Skew/Term structure coupling (leverage/spillover effects)
        shock_l = rng.normal(0, cond_vol)
        shock_s = 0.4 * shock_l + rng.normal(0, sigma_s)
        shock_t = -0.2 * shock_l + rng.normal(0, sigma_t)

        # 5. Poisson Jumps
        jump = 0.0
        if rng.random() < 0.03:  # 3% chance of macro shock jump
            jump = rng.exponential(0.15)
            skew += 0.08  # Volatility jump steepens skew immediately (puts bid up)
            
        # Decay previous jump effects (half-life of ~4 days)
        jump_effect = 0.8 * jump_effect + jump
        
        # 6. Evolve latent parameters
        level = mu_l + phi_l * (level - mu_l) + shock_l + jump
        skew = mu_s + phi_s * (skew - mu_s) + shock_s
        term = mu_t + phi_t * (term - mu_t) + shock_t
        
        # Effective level includes persistent jump impact
        effective_level = level + jump_effect
        
        # Keep physical parameter bounds
        effective_level = np.clip(effective_level, 0.08, 0.70)
        skew = np.clip(skew, 0.01, 0.25)
        term = np.clip(term, -0.06, 0.14)
        
        # 7. Build 7x7 grid surface
        grid_iv = np.zeros((len(GRID_TAUS), len(GRID_KAPPAS)))
        for i, tau in enumerate(GRID_TAUS):
            for j, kappa in enumerate(GRID_KAPPAS):
                # Volatility smile (curvature depends on regime)
                smile_curvature = 0.16 if regime == 1 else 0.10
                iv = effective_level - skew * kappa + smile_curvature * kappa**2 + term * np.log(tau / 0.25)
                iv += rng.normal(0, 0.001)
                grid_iv[i, j] = iv
                
        # Clip surface values
        grid_iv = np.clip(grid_iv, 0.02, 5.0)
        surfaces.append(grid_iv)
        
        # Generate daily timestamp string
        ts = (base_date + timedelta(days=day)).strftime("%Y-%m-%d %H:%M:%S")
        timestamps.append(ts)
        
    dataset = np.stack(surfaces, axis=0) # Shape: (T, 7, 7)
    logger.info(f"Synthetic generation complete. Built tensor shape: {dataset.shape}")
    return dataset, timestamps
