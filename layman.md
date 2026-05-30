# Layman's Guide — Neural Volatility Surface Forecaster

This guide explains how this system works, what engineering changes we made, and how the components fit together in plain, simple terms.

---

## 1. What Does This Project Do?

Imagine you want to buy financial insurance (called an **Option**) on a stock like SPY (the S&P 500 ETF). The price of that insurance depends on how much the market expects the stock price to wiggle (called **Volatility**) before the insurance expires.

Since people buy insurance for different price levels (e.g., protection against a crash vs. betting on a boom) and different durations (e.g., 1 week from now vs. 1 year from now), we have thousands of option prices at any moment.
- If we plot **Volatility** against the **Strike Price** (horizontal axis) and **Expiry Date** (depth axis) on a 3D chart, we get a wavy, bowl-shaped surface called the **Implied Volatility Surface**.
- **The Goal**: We want to predict how this entire 3D surface will evolve and look in the future (1 day, 5 days, and 10 days from now). This helps traders price insurance and manage risk.

---

## 2. Platform Core Architecture

We have built a production-grade **Data Foundation**, **Cleaning Pipeline**, **Deep Learning Model Engine**, **Serving API**, and **Automated Quality Checks**.

### Summary of what we built:
1. **The Ingestor (Layer 1)**: A scheduled scheduler module that downloads real-time option prices from Yahoo Finance every market day at 4:05 PM and persists them securely in compressed Parquet databases.
2. **The Cleaner & Sculptor (Layer 2)**:
   - Raw market quotes are messy and contain illiquid options or spread errors. The processing filters discard contracts violating fundamental no-arbitrage bounds (e.g. calendar spread spread inequalities).
   - An RBF thin-plate spline interpolator maps scattered strike/date quotes onto a neat, standardized **$7 \times 7$ grid** representing expiries from 1 week to 1 year and log-moneyness levels from tail-OTM puts to tail-OTM calls.
3. **The Deep Learning Engine (Layer 4 & 5)**: High-performance stacked LSTM, ConvLSTM, and Transformer Encoder networks trained under custom composite loss functions that enforce spatial surface smoothness alongside structural reconstruction.
4. **The Web Server (Layer 7 serving)**: A low-latency FastAPI application that receives raw chain inputs, standardizes the surfaces, runs GPU-accelerated forward predictions, and renders interactively styled 3D Plotly visualizations. It is optimized with Gzip payload compression for instant load times.
5. **The Quality Guard (Layer 7 monitoring)**: Real-time drift detection that measures Kullback-Leibler (KL) Divergence and RMSE. If options incoming distribution drifts beyond predefined limits, the monitor triggers an automated training pipeline.
6. **Continuous Integration & Testing**: Integrated unit testing structures (`pytest`), portable containerization (`docker-compose`), and automated GitHub Actions CI.

---

## 3. How Did We Do It? (The Technical Implementation)

We created a highly modular and decoupled architecture:

- **[`configs/base_config.yaml`](file:///d:/projects/neural-volatility-forecaster/configs/base_config.yaml)**: The system's "control panel". It defines the standard $7 \times 7$ coordinate boundaries, file directories, learning rates, and drift alerts. All components share this unified configuration.
- **[`data/collector.py`](file:///d:/projects/neural-volatility-forecaster/data/collector.py)**: Fetches options, handles dynamic risk-free interest rates (using the US 13-week T-Bill index `^IRX`), and handles ETF dividend yields (correcting percentage-to-decimal errors dynamically).
- **[`data/processor.py`](file:///d:/projects/neural-volatility-forecaster/data/processor.py)**: Performs mathematical Black-Scholes inversion using a Newton-Raphson numerical solver to extract volatilities, checks calendar spread inequalities, and maps scattered points to the grid.
- **[`data/dataset.py`](file:///d:/projects/neural-volatility-forecaster/data/dataset.py)**: Contains the sequence generator for neural network training and a CSV/Parquet importer for historical data.
- **[`utils/plotting.py`](file:///d:/projects/neural-volatility-forecaster/utils/plotting.py)**: Leverages Plotly to output interactive 3D graphs, cross-sections, and residual error heatmaps.
- **[`inference/app.py`](file:///d:/projects/neural-volatility-forecaster/inference/app.py)**: FastAPI server managing the `/predict`, `/monitor/drift`, and `/health` REST endpoints.
- **[`tests/`](file:///d:/projects/neural-volatility-forecaster/tests/)**: Contains unit tests for math equations (`test_processor.py`) and FastAPI client endpoints (`test_api.py`).
