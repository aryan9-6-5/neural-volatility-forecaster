import os
import yaml
import yfinance as yf
import pandas as pd
from datetime import datetime
import logging
from apscheduler.schedulers.background import BackgroundScheduler

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path="configs/base_config.yaml"):
    """Load configuration file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config path {config_path} not found. Using default values.")
        return {
            "ticker": "SPY",
            "data": {"raw_dir": "data/raw"}
        }
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def fetch_risk_free_rate(fallback=0.045) -> float:
    """
    Fetch the 13-week US Treasury Bill rate (^IRX) from yfinance as proxy for risk-free rate.
    Converts percent to decimal (e.g., 4.5% -> 0.045).
    """
    try:
        irx = yf.Ticker("^IRX")
        history = irx.history(period="1d")
        if not history.empty:
            rate_pct = history["Close"].iloc[-1]
            rate_decimal = rate_pct / 100.0
            logger.info(f"Successfully fetched risk-free rate (^IRX) from yfinance: {rate_decimal:.5f} ({rate_pct:.3f}%)")
            return rate_decimal
        else:
            logger.warning("No history returned for ^IRX. Using fallback rate.")
    except Exception as e:
        logger.error(f"Error fetching risk-free rate: {e}. Using fallback rate.")
    
    logger.info(f"Using fallback risk-free rate: {fallback}")
    return fallback

def fetch_dividend_yield(ticker_symbol: str, fallback=0.015) -> float:
    """
    Fetch trailing/forward dividend yield for the ticker from yfinance.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Attempt to get dividend yield from info dictionary
        info = ticker.info
        yield_val = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
        if yield_val is not None:
            val = float(yield_val)
            # If yield is > 0.20 (20%), it is likely expressed in percentage points (e.g., 1.3% -> 1.3)
            if val > 0.20:
                val = val / 100.0
            logger.info(f"Successfully fetched dividend yield for {ticker_symbol}: {val:.5f}")
            return val
    except Exception as e:
        logger.warning(f"Error fetching dividend yield for {ticker_symbol}: {e}. Using fallback.")
    
    logger.info(f"Using fallback dividend yield for {ticker_symbol}: {fallback}")
    return fallback

def collect_snapshot(ticker_symbol: str = None, output_dir: str = None) -> pd.DataFrame:
    """
    Fetch current options chain snapshot for ticker_symbol and save as timestamped Parquet.
    """
    config = load_config()
    ticker_symbol = ticker_symbol or config.get("ticker", "SPY")
    output_dir = output_dir or config.get("data", {}).get("raw_dir", "data/raw")
    
    logger.info(f"Starting option chain collection snapshot for {ticker_symbol}...")
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        expiries = ticker.options
        if not expiries:
            logger.error(f"No option expiries found for ticker {ticker_symbol}.")
            return pd.DataFrame()
            
        # Get spot price
        history = ticker.history(period="1d")
        if history.empty:
            logger.error(f"Failed to fetch spot price for ticker {ticker_symbol}.")
            return pd.DataFrame()
        spot_price = history["Close"].iloc[-1]
        
        # Fetch auxiliary parameters
        r = fetch_risk_free_rate()
        q = fetch_dividend_yield(ticker_symbol)
        
        timestamp_str = datetime.now().isoformat()
        records = []
        
        for exp in expiries:
            try:
                chain = ticker.option_chain(exp)
                for df, opt_type in [(chain.calls, "call"), (chain.puts, "put")]:
                    if df is None or df.empty:
                        continue
                    for _, row in df.iterrows():
                        records.append({
                            "timestamp": timestamp_str,
                            "ticker": ticker_symbol,
                            "spot_price": spot_price,
                            "risk_free_rate": r,
                            "dividend_yield": q,
                            "strike": float(row["strike"]),
                            "expiry": exp,
                            "option_type": opt_type,
                            "impliedVolatility": float(row["impliedVolatility"]) if pd.notna(row.get("impliedVolatility")) else None,
                            "lastPrice": float(row["lastPrice"]) if pd.notna(row.get("lastPrice")) else None,
                            "bid": float(row["bid"]) if pd.notna(row.get("bid")) else None,
                            "ask": float(row["ask"]) if pd.notna(row.get("ask")) else None,
                            "volume": int(row["volume"]) if pd.notna(row.get("volume")) else 0,
                            "openInterest": int(row["openInterest"]) if pd.notna(row.get("openInterest")) else 0,
                        })
            except Exception as e:
                logger.error(f"Failed to fetch expiry {exp} for {ticker_symbol}: {e}")
                continue
                
        if not records:
            logger.warning(f"No records collected for {ticker_symbol}.")
            return pd.DataFrame()
            
        snapshot_df = pd.DataFrame(records)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output filename with date and hour-minute
        file_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = os.path.join(output_dir, f"{file_timestamp}.parquet")
        snapshot_df.to_parquet(filename, index=False)
        
        logger.info(f"Saved options snapshot: {filename} ({len(snapshot_df)} records)")
        return snapshot_df
        
    except Exception as e:
        logger.error(f"Critical error during snapshot collection: {e}")
        return pd.DataFrame()

def start_scheduler():
    """Start APScheduler to run collection daily at 4:05 PM on weekdays."""
    config = load_config()
    ticker = config.get("ticker", "SPY")
    raw_dir = config.get("data", {}).get("raw_dir", "data/raw")
    
    scheduler = BackgroundScheduler()
    # Runs Monday-Friday at 16:05 (4:05 PM)
    scheduler.add_job(
        collect_snapshot,
        trigger="cron",
        day_of_week="mon-fri",
        hour=16,
        minute=5,
        args=[ticker, raw_dir],
        id="options_collector_daily"
    )
    scheduler.start()
    logger.info(f"Background collector scheduler started. Job configured for {ticker} at 16:05 Mon-Fri.")
    return scheduler
