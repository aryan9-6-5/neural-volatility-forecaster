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
    
    grouped = df.groupby("timestamp")
    total_groups = len(grouped)
    logger.info(f"Processing {total_groups} unique historical daily snapshots...")
    
    idx = 0
    for ts, snap in grouped:
        try:
            processed = process_raw_snapshot(snap, volume_filter=volume_filter)
            if len(processed) < 5:
                continue
                
            grid_iv = interpolate_to_grid(
                processed["kappa"].values,
                processed["tau"].values,
                processed["impliedVolatility"].values
            )
            surfaces.append(grid_iv)
            timestamps.append(ts)
            
            idx += 1
            if idx % 50 == 0 or idx == total_groups:
                logger.info(f"Processed {idx}/{total_groups} snapshots...")
        except Exception as e:
            logger.error(f"Error processing historical date {ts}: {e}")
            continue
            
    if not surfaces:
        raise ValueError("No historical surfaces could be successfully constructed.")
        
    dataset = np.stack(surfaces, axis=0)
    logger.info(f"Historical import complete. Built surface tensor of shape {dataset.shape}")
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

def generate_synthetic_dataset(num_days: int = 120) -> tuple:
    """
    Generate a realistic synthetic volatility surface time-series.
    Simulates temporal changes by evolving level, skew, and term structure parameters
    using daily auto-regressive (AR(1)) processes.
    
    Formula for surface at time t:
        IV(kappa, tau) = Level_t - Skew_t * kappa + 0.1 * kappa^2 + Term_t * ln(tau / 0.25)
    """
    logger.info(f"Generating synthetic volatility surface dataset for {num_days} days...")
    
    # Setup AR(1) state variables for level, skew, and term structure
    level = 0.22      # baseline standard deviation level (22% IV)
    skew = 0.08       # volatility smile skew (puts higher than calls)
    term = 0.04       # term structure slope (long expiries higher vol)
    
    # AR(1) parameters
    phi_l, phi_s, phi_t = 0.95, 0.90, 0.93
    mu_l, mu_s, mu_t = 0.20, 0.06, 0.03
    sigma_l, sigma_s, sigma_t = 0.015, 0.006, 0.004
    
    surfaces = []
    timestamps = []
    
    base_date = datetime.now() - timedelta(days=num_days)
    
    for day in range(num_days):
        # Evolve latent parameters with AR(1) dynamics + noise
        level = mu_l + phi_l * (level - mu_l) + np.random.normal(0, sigma_l)
        skew = mu_s + phi_s * (skew - mu_s) + np.random.normal(0, sigma_s)
        term = mu_t + phi_t * (term - mu_t) + np.random.normal(0, sigma_t)
        
        # Keep physical parameter bounds
        level = np.clip(level, 0.08, 0.60)
        skew = np.clip(skew, 0.01, 0.20)
        term = np.clip(term, -0.05, 0.12)
        
        # Build 7x7 grid surface
        grid_iv = np.zeros((len(GRID_TAUS), len(GRID_KAPPAS)))
        for i, tau in enumerate(GRID_TAUS):
            for j, kappa in enumerate(GRID_KAPPAS):
                # Parametric volatility smile shape (parabolic in kappa, log-scale in tau)
                iv = level - skew * kappa + 0.12 * kappa**2 + term * np.log(tau / 0.25)
                # Add tiny local observational noise
                iv += np.random.normal(0, 0.001)
                grid_iv[i, j] = iv
                
        # Clip surface values to ensure no negative volatility
        grid_iv = np.clip(grid_iv, 0.01, 5.0)
        surfaces.append(grid_iv)
        
        # Generate daily timestamp string
        ts = (base_date + timedelta(days=day)).strftime("%Y-%m-%d %H:%M:%S")
        timestamps.append(ts)
        
    dataset = np.stack(surfaces, axis=0) # Shape: (T, 7, 7)
    logger.info(f"Synthetic generation complete. Built tensor shape: {dataset.shape}")
    return dataset, timestamps
