"""
Adapter for the public "options-dataset-hist" SPY historical options mirror
(https://github.com/anahatsingh-ui/options-dataset-hist, MIT license,
anahatsingh-ui, a preservation mirror of a dataset originally published by
Philipp D. Dubach).

This module ONLY prepares a flat DataFrame in the schema
data.dataset.import_historical_dataframe already expects (timestamp, strike,
expiry, option_type, bid, ask, lastPrice, volume, impliedVolatility,
spot_price, risk_free_rate, dividend_yield). It deliberately does NOT
duplicate any filtering, IV solving, or surface-construction logic --
that all happens in the existing pipeline (data/processor.py's
process_raw_snapshot -> apply_arbitrage_filters -> interpolate_to_grid, called
via data.dataset.import_historical_dataframe), per project decision: the
source's own precomputed `implied_volatility` is NOT treated as authoritative
and is re-solved from bid/ask independently -- it is only used as an initial
guess / fallback (see process_raw_snapshot).

Authoritative-field decisions (used as-is, not recomputed):
  - `close` (NOT `adjusted_close`) as the historical spot price. adjusted_close
    is retroactively recomputed using dividend/split events that postdate the
    row's own date, so using it would leak future corporate-action
    information into a historical price -- exactly the look-ahead bias this
    project must avoid.
  - `dividend_amount` as the factual basis for a computed trailing realized
    dividend yield (see compute_trailing_dividend_yield) -- it's a plain
    historical fact, not something we could or should recompute.
  - Risk-free rate has NO source column at all and is fetched externally via
    yfinance's ^IRX (13-week T-Bill), the same proxy data/collector.py already
    uses for live snapshots, aligned backward-only (never a future rate).

Dropped, not used anywhere in the pipeline: delta, gamma, theta, vega, rho,
mark, contract_id, open_interest, in_the_money, bid_size, ask_size.
"""
import os
import logging
import requests
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SOURCE_REPO = "anahatsingh-ui/options-dataset-hist"
SOURCE_REF = "main"
BASE_URL = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_REF}"


def download_source_files(year: int, cache_dir: str = "data/historical/raw_cache", ticker: str = "spy") -> tuple:
    """
    Download {ticker}/options_{year}.parquet and {ticker}/underlying_prices.parquet
    from the public mirror, caching under cache_dir. Idempotent: skips
    download if the file already exists locally.
    """
    os.makedirs(cache_dir, exist_ok=True)
    options_path = os.path.join(cache_dir, f"{ticker}_options_{year}.parquet")
    underlying_path = os.path.join(cache_dir, f"{ticker}_underlying_prices.parquet")

    for local_path, remote_name in [
        (options_path, f"{ticker}/options_{year}.parquet"),
        (underlying_path, f"{ticker}/underlying_prices.parquet"),
    ]:
        if os.path.exists(local_path):
            logger.info(f"Using cached {local_path}")
            continue
        url = f"{BASE_URL}/{remote_name}"
        logger.info(f"Downloading {url} -> {local_path}")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        logger.info(f"Saved {local_path} ({len(resp.content)} bytes)")

    return options_path, underlying_path


def fetch_historical_risk_free_rate(dates, fallback: float = 0.045) -> pd.Series:
    """
    Fetch historical 13-week T-Bill (^IRX) daily closes from yfinance for the
    given trading dates, aligned via an as-of BACKWARD merge so each date gets
    the most recent prior-or-same-day rate -- never a future one. Falls back
    to a flat constant (with a warning) if the fetch fails or returns nothing,
    mirroring data/collector.py::fetch_risk_free_rate's fallback philosophy.
    Returns a Series indexed by date.
    """
    unique_dates = pd.to_datetime(pd.Series(dates).unique())
    start = unique_dates.min() - pd.Timedelta(days=10)
    end = unique_dates.max() + pd.Timedelta(days=1)

    try:
        import yfinance as yf
        irx = yf.Ticker("^IRX").history(start=start, end=end)
        if irx.empty:
            raise ValueError("Empty ^IRX history returned")
        irx = irx.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "rate_pct"})
        irx["date"] = pd.to_datetime(irx["date"])
        if irx["date"].dt.tz is not None:
            irx["date"] = irx["date"].dt.tz_localize(None)
        irx = irx.sort_values("date")
        irx["risk_free_rate"] = irx["rate_pct"] / 100.0

        dates_df = pd.DataFrame({"date": sorted(unique_dates)})
        merged = pd.merge_asof(dates_df, irx[["date", "risk_free_rate"]], on="date", direction="backward")
        merged["risk_free_rate"] = merged["risk_free_rate"].fillna(fallback)
        logger.info(
            f"Fetched historical ^IRX risk-free rate for {len(unique_dates)} dates "
            f"(range {merged['risk_free_rate'].min():.4f}-{merged['risk_free_rate'].max():.4f})"
        )
        return merged.set_index("date")["risk_free_rate"]
    except Exception as e:
        logger.warning(f"Failed to fetch historical ^IRX rate ({e}). Using flat fallback {fallback}.")
        return pd.Series(fallback, index=sorted(unique_dates))


def compute_trailing_dividend_yield(underlying_df: pd.DataFrame, window_days: int = 365, fallback: float = 0.015) -> pd.Series:
    """
    For each date, sum dividend_amount over the trailing window
    (date - window_days, date] -- strictly backward-looking: only dividends
    already paid as of that date -- and divide by that date's own close price
    to get a trailing realized dividend yield. Avoids the look-ahead bias of
    a yield computed with knowledge of dividends that hadn't happened yet.
    Returns a Series indexed by date, covering every date in underlying_df.
    """
    df = underlying_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].to_numpy()
    div = df["dividend_amount"].fillna(0.0).to_numpy()
    close = df["close"].to_numpy()

    window = np.timedelta64(window_days, "D")
    trailing_sum = np.zeros(len(df))
    start_idx = 0
    running = 0.0
    for i in range(len(df)):
        running += div[i]
        while dates[i] - dates[start_idx] > window:
            running -= div[start_idx]
            start_idx += 1
        trailing_sum[i] = running

    with np.errstate(divide="ignore", invalid="ignore"):
        yield_est = np.where(close > 0, trailing_sum / close, np.nan)
    yield_series = pd.Series(yield_est, index=pd.DatetimeIndex(df["date"]))
    return yield_series.fillna(fallback)


def build_spy_dataframe(year: int, cache_dir: str = "data/historical/raw_cache",
                         fallback_rate: float = 0.045, fallback_dividend_yield: float = 0.015) -> tuple:
    """
    Build a flat DataFrame for SPY options in `year`, in the schema
    data.dataset.import_historical_dataframe expects. Performs NO filtering
    or IV solving -- that is the existing pipeline's job.

    Returns (options_df, raw_options_df): raw_options_df is the untouched
    source frame, kept for quality-report checks (duplicates, suspicious
    values) against the *original* data before any renaming/joining.
    """
    options_path, underlying_path = download_source_files(year, cache_dir=cache_dir)

    raw_options = pd.read_parquet(options_path)
    underlying = pd.read_parquet(underlying_path)
    underlying = underlying[underlying["symbol"] == "SPY"].copy()
    underlying["date"] = pd.to_datetime(underlying["date"])

    options = raw_options.copy()
    options["date"] = pd.to_datetime(options["date"])

    # Spot price: raw `close`, not `adjusted_close` -- see module docstring.
    spot = underlying.set_index("date")["close"].rename("spot_price")
    options = options.join(spot, on="date")

    # Dividend yield: trailing realized yield, strictly backward-looking.
    div_yield = compute_trailing_dividend_yield(underlying, fallback=fallback_dividend_yield)
    options = options.join(div_yield.rename("dividend_yield"), on="date")
    options["dividend_yield"] = options["dividend_yield"].fillna(fallback_dividend_yield)

    # Risk-free rate: fetched externally, backward-aligned only.
    rf_series = fetch_historical_risk_free_rate(options["date"], fallback=fallback_rate)
    options = options.join(rf_series.rename("risk_free_rate"), on="date")
    options["risk_free_rate"] = options["risk_free_rate"].fillna(fallback_rate)

    # Column mapping onto the standard schema (see module docstring for why
    # impliedVolatility is mapped straight across but NOT treated as
    # authoritative, and which columns are dropped).
    options = options.rename(columns={
        "date": "timestamp",
        "expiration": "expiry",
        "type": "option_type",
        "implied_volatility": "impliedVolatility",
        "last": "lastPrice",
    })
    options["timestamp"] = options["timestamp"].dt.strftime("%Y-%m-%d")
    options["expiry"] = pd.to_datetime(options["expiry"]).dt.strftime("%Y-%m-%d")

    keep_cols = [
        "timestamp", "strike", "expiry", "option_type", "bid", "ask", "lastPrice",
        "volume", "impliedVolatility", "spot_price", "risk_free_rate", "dividend_yield",
    ]
    options = options[keep_cols]

    logger.info(
        f"Built SPY {year} source dataframe: {len(options)} rows, "
        f"{options['timestamp'].nunique()} unique dates, "
        f"spot range [{options['spot_price'].min():.2f}, {options['spot_price'].max():.2f}]"
    )

    return options, raw_options
