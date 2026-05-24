# Volatility Surface Data Contract Specification

This document freezes the data layout, normalization rules, coordinate indexing, and metadata specifications for the Neural Volatility Surface Forecaster. Decoupling the data engineering pipeline (Developer A) from the modeling framework (Developer B) requires strict adherence to this contract.

---

## 1. Contract Identification
- **DATA_CONTRACT_VERSION**: `"v1.0"`
- **Scope**: Implied Volatility Surface (IVS) forecasting on standard $7 \times 7$ spatial grids.

---

## 2. Tensor Conventions

### 2.1 Model Input Shape (Lookback Sequence)
Inputs to PyTorch forecasting models must be strictly 5D spatiotemporal tensors representing sequences of historical surfaces:
$$\mathbf{X} \in \mathbb{R}^{B \times L \times C \times E \times M}$$

Where:
- **$B$ (Batch Size)**: Configurable (default `32`), representing the training mini-batch size.
- **$L$ (Lookback Window)**: Defined as `20` business days (representing ~1 month of trading days).
- **$C$ (Channels)**: Fixed to `1`, representing the single channel of implied volatility ($\sigma^{IV}$).
- **$E$ (Expiry Grid Rows)**: Fixed to `7` standardized maturity buckets.
- **$M$ (Moneyness Grid Columns)**: Fixed to `7` standardized log-moneyness levels.

### 2.2 Model Target Shape (Forecast Horizons)
Targets represent a multi-step forecast sequence of future surface states:
$$\mathbf{y} \in \mathbb{R}^{B \times h \times E \times M}$$

Where:
- **$h$ (Horizon)**: Represents the forecasting steps. Models are configured for $h \in \{1, 5, 10\}$ (corresponding to 1-day, 5-day, and 10-day ahead volatility surfaces).
- **$E$ and $M$**: Standard grid dimensions ($7 \times 7$).

### 2.3 Precision & Dtype
All tensor arrays must use standard **single-precision floating-point** format:
- **Python/NumPy**: `numpy.float32`
- **PyTorch**: `torch.float32`

---

## 3. Coordinate Indexing and Orientation

The volatility surface is represented as a 2D grid matrix of shape `(7, 7)`. The axes are mapped as follows:

| Dimension | Axis Index | Dimension Size | Coordinate Grid Variable | Interpretation |
|---|---|---|---|---|
| **Rows** | Axis `0` | 7 | Expiry ($\tau$) | Standardized time-to-expiry maturities |
| **Columns** | Axis `1` | 7 | Log-Moneyness ($\kappa$) | Standardized strike offsets |

### 3.1 Expiry Grid Rows (Axis 0)
Standardized annual trading day fractions sorted from shortest to longest:
$$\tau \in \{1/52, \; 2/52, \; 1/12, \; 2/12, \; 3/12, \; 6/12, \; 1.0\}$$
$$\tau \approx \{0.01923, \; 0.03846, \; 0.08333, \; 0.16667, \; 0.25000, \; 0.50000, \; 1.00000\}$$
- **Index `0`**: 1-week expiry (shortest duration, highest gamma risk)
- **Index `6`**: 1-year expiry (longest duration, long-term LEAPs)

### 3.2 Log-Moneyness Grid Columns (Axis 1)
Standardized strikes mapped to forward price offsets, sorted from left-wing to right-wing:
$$\kappa \in \{-0.30, \; -0.20, \; -0.10, \; 0.00, \; +0.10, \; +0.20, \; +0.30\}$$
- **Index `0`**: Deep OTM Put ($\kappa = -0.30$, tail risk protection)
- **Index `3`**: At-the-Money ($\kappa = 0.00$, highly liquid, primary trading asset)
- **Index `6`**: Deep OTM Call ($\kappa = +0.30$, speculative upside)

---

## 4. Normalization Rules

1. **Strike Pricing**: Normalized to log-moneyness coordinate:
   $$\kappa = \ln\left(\frac{K}{F}\right)$$
   Where the Forward Price is:
   $$F = S_0 \cdot e^{(r - q)\tau}$$
   - $S_0$: Underlying asset closing price
   - $r$: 13-week T-Bill annual yield (`^IRX` / 100.0)
   - $q$: Trailing asset dividend yield (e.g. SPY yield / 100.0)
2. **Time to Expiry**: Normalized to annual trading day fractions:
   $$\tau = \frac{\text{Calendar Days to Expiry}}{252}$$
   - Restricted to $\tau \ge 1/252$ (minimum 1 day to prevent divide-by-zero).
3. **Implied Volatility Bounds**: Output grids are clipped to ensure physical pricing limits:
   $$\sigma^{IV} \in [0.01, \; 5.00] \quad (\text{i.e., } 1\% \text{ to } 500\% \text{ volatility})$$

---

## 5. Missing-Value and Quality Strategy

1. **Missing IV Fallback**: Numerical BSM inversion defaults to Brent's root-finding method. If both Newton-Raphson and Brent fail, it falls back to the yfinance `impliedVolatility` field. If that is also absent/null, it returns `NaN`.
2. **Observation Removal**: Options with bid $\le 0$, ask $\le 0$, ask $<$ bid, volume $= 0$ (for live data), or spread ratios $(C_{ask} - C_{bid}) / C_{mid} \ge 0.50$ are discarded.
3. **Calendar Spread Arbitrage Check**: For each binned moneyness category, contracts must satisfy $w(\tau_1) \le w(\tau_2)$ for all $\tau_1 < \tau_2$. Violators are dropped.
4. **Interpolation Robustness**: If a daily snapshot yields fewer than `5` valid contracts after filtering, the entire day is skipped. Otherwise, the Scattered RBF thin-plate spline interpolation computes the full $7 \times 7$ grid, naturally filling in missing coordinates.

---

## 6. Reproducibility & Versioning Protocols

To ensure that research experiments remain reproducible once training commences, the following guidelines are frozen for this contract version:
1. **Grid Resolution Freeze**: The standardized $7 \times 7$ grid layout is locked for all baseline experiments. If higher-resolution surfaces are evaluated in subsequent iterations, they must be developed in separate, isolated git feature branches.
2. **Configuration Hashing**: When starting an MLflow run, the pipeline must compute and log a hash of the preprocessing configuration settings:
   - Standard grid coordinate array values
   - Interpolation kernel configuration (`thin_plate_spline` smoothing factor)
   - Discard thresholds (liquidity and arbitrage parameters)
3. **Checkpoint Metadata**: When saving model state checkpoints (e.g., `.pt` files), the model saver must package a dictionary containing metadata:
   - `data_contract_version`: `v1.0`
   - `lookback_window`: `20`
   - `horizons`: `[1, 5, 10]`
   - `tensor_orientation`: `(B, L, C, E, M)`
   This avoids deployment and serving runtime mismatches.

---

## 7. MLOps Preprocessing Metric Logging

During run orchestration or training pipeline execution, the following preprocessing statistics must be captured and logged into the MLflow tracking run:

- `raw_contracts_count`: Number of contracts parsed from the raw snapshot.
- `filtered_contracts_count`: Number of contracts remaining after filters are applied.
- `arbitrage_violations_count`: Number of contracts discarded due to calendar spread violations.
- `surface_coverage_pct`: Percentage of the 49 standard grid coordinate buckets covered by actual observations prior to RBF interpolation.

---

## 8. Developer B Onboarding Reading List

Before editing the codebase or instantiating training scripts, Developer B must read the following documentation files in order:

1. **[README.md](file:///d:/projects/neural-volatility-forecaster/README.md)**: High-level overview of the project scope, the Black-Scholes-Merton (BSM) analytical formulas, and numerical inversion solvers.
2. **[System_Design.md](file:///d:/projects/neural-volatility-forecaster/documentation/System_Design.md)**: Details the modular 7-layer architecture, highlighting Layer 4 (Model Training) and Layer 5 (Evaluation) directories and logic.
3. **[ML_Pipeline.md](file:///d:/projects/neural-volatility-forecaster/documentation/ML_Pipeline.md)**: Deep dive into baseline architectures (LSTM), spatiotemporal models (ConvLSTM), and our custom composite loss function (`SmoothnessRegularizedLoss`).
4. **[Data_Contract.md](file:///d:/projects/neural-volatility-forecaster/documentation/Data_Contract.md) (This Document)**: Reviews the inputs shape conventions, Axis orientation mappings (Axis 0 = Expiry, Axis 1 = Moneyness), dtypes, and configuration hash requirements.
5. **[Operations.md](file:///d:/projects/neural-volatility-forecaster/documentation/Operations.md)**: Reviews MLOps dashboard pipelines, drift metrics, and target milestones.
