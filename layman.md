# Layman's Guide — Neural Volatility Surface Forecaster

This guide explains how this system works, what engineering changes we made, and how the components fit together in plain, simple terms.

---

## 1. What Does This Project Do?

Imagine you want to buy financial insurance (called an **Option**) on a stock like SPY (the S&P 500 ETF). The price of that insurance depends on how much the market expects the stock price to wiggle (called **Volatility**) before the insurance expires.

Since people buy insurance for different price levels (e.g., protection against a crash vs. betting on a boom) and different durations (e.g., 1 week from now vs. 1 year from now), we have thousands of option prices at any moment.
- If we plot **Volatility** against the **Strike Price** (horizontal axis) and **Expiry Date** (depth axis) on a 3D chart, we get a wavy, bowl-shaped surface called the **Implied Volatility Surface**.
- **The Goal**: We want to predict how this entire 3D surface will evolve and look in the future (1 day, 5 days, and 10 days from now). This helps traders price insurance and manage risk.

---

## 2. What Work is Completed? (Progress: ~45%)

We have built the entire **Data Foundation**, **Cleaning Pipeline**, **Web Serving API**, and **Safety Checks**. 

Think of this like building a water treatment and delivery system: we built the reservoir, the pipes, the filtration plant, the delivery trucks, and the water testing lab. The only thing left to build is the specialized filtration chemistry (the final AI models), which Developer B will design.

### Summary of what we built:
1. **The Ingestor (Layer 1)**: A script that automatically downloads real-time option prices from Yahoo Finance every market day at 4:05 PM and saves them securely in Parquet files (a highly compressed data format).
2. **The Cleaner & Sculptor (Layer 2)**:
   - Raw market quotes are messy. They contain typos, illiquid options, and pricing anomalies that violate financial rules (e.g., calendar arbitrage, where long-term insurance is priced cheaper than short-term insurance). We built filters to throw away these bad records.
   - Options trade at scattered strikes and dates. To feed them to a neural network, we must organize them. We built an **RBF Interpolator** (a mathematical grid-fitting tool) that takes scattered data points and projects them onto a neat, standard **$7 \times 7$ grid** (like a digital image with 49 pixels), representing maturities from 1 week to 1 year and strikes from deep out-of-the-money puts to deep out-of-the-money calls.
3. **The Web Server (Layer 7 serving)**: A FastAPI web application that accepts raw option data, cleans and standardizes it, runs it through the AI model, and returns predicted 3D surfaces alongside interactive 3D visualizations.
   - *Resilience*: Since the real AI model isn't trained yet, we built a **Smart Mock Model** that wiggles synthetic surfaces realistically. This allowed us to test the entire server and ensure it functions perfectly before the modeling begins.
4. **The Guard (Layer 7 monitoring)**: Markets change. If the stock market crashes or enters a new regime, the AI's old training becomes obsolete. We built a **Drift Monitor** using a mathematical comparison called **KL Divergence** to calculate if incoming options data is drifting too far from what the model originally learned, triggering a retraining alert.
5. **The Safety Net (CI/CD and Testing)**: 
   - We wrote automated testing scripts that check if our pricing math is correct.
   - We configured **Docker** and **Docker Compose** to run the server and an **MLflow dashboard** (an experiment tracker) inside portable containers.
   - We set up **GitHub Actions CI** so that every time a developer commits code, the test suite runs automatically in the cloud to guarantee nothing is broken.

---

## 3. How Did We Do It? (The Technical Implementation)

We created a modular and isolated architecture to avoid merge conflicts:

- **[`configs/base_config.yaml`](file:///d:/projects/neural-volatility-forecaster/configs/base_config.yaml)**: The system's "control panel". It defines the standard $7 \times 7$ coordinate boundaries, file directories, learning rates, and drift alerts. Both developers share this file.
- **[`data/collector.py`](file:///d:/projects/neural-volatility-forecaster/data/collector.py)**: Fetches options, handles dynamic risk-free interest rates (using the US 13-week T-Bill index `^IRX`), and handles ETF dividend yields (correcting percentage-to-decimal errors dynamically).
- **[`data/processor.py`](file:///d:/projects/neural-volatility-forecaster/data/processor.py)**: Performs mathematical Black-Scholes inversion using a Newton-Raphson numerical solver to extract volatilities, checks calendar spread inequalities, and maps scattered points to the grid.
- **[`data/dataset.py`](file:///d:/projects/neural-volatility-forecaster/data/dataset.py)**: Contains the sequence generator for neural network training and a CSV/Parquet importer for historical data.
- **[`utils/plotting.py`](file:///d:/projects/neural-volatility-forecaster/utils/plotting.py)**: Leverages Plotly to output interactive 3D graphs, cross-sections, and residual error heatmaps.
- **[`inference/app.py`](file:///d:/projects/neural-volatility-forecaster/inference/app.py)**: FastAPI server managing the `/predict`, `/monitor/drift`, and `/health` REST endpoints.
- **[`tests/`](file:///d:/projects/neural-volatility-forecaster/tests/)**: Contains unit tests for math equations (`test_processor.py`) and FastAPI client endpoints (`test_api.py`).
