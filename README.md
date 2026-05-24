# Smoothness-Regularized Deep Learning for Implied Volatility Surface Forecasting

## 1. Project Identity

### One-Line Definition
> An end-to-end deep learning system that forecasts future implied volatility surfaces from options market data using smoothness-regularized ConvLSTM architectures, evaluated through region-specific error decomposition and served via a production-grade automated pipeline.

### Executive Summary
This project develops a complete machine learning platform for multi-step-ahead forecasting of implied volatility surfaces derived from equity options chain data. Raw options snapshots are continuously collected via automated scripts, cleaned through arbitrage filtering, and interpolated onto standardized log-moneyness and time-to-expiry grids using radial basis function methods to produce discrete surface tensors. These tensors are fed as spatiotemporal sequences into ConvLSTM neural networks trained with a novel composite loss function that combines mean squared error reconstruction with spatial smoothness regularization penalties along both strike and expiry dimensions, ensuring that predicted surfaces maintain the financial consistency properties expected by practitioners. The system is evaluated against five baseline models including naive random walk, historical mean, and GARCH benchmarks using region-specific error metrics decomposed across ATM, OTM, and wing regions of the surface. The complete platform integrates automated data collection, MLflow experiment tracking, FastAPI inference serving, interactive Plotly three-dimensional visualizations, and drift-based automated retraining, constituting a production-ready research system for computational finance applications.

---

## 2. Problem Definition

### The Central Research Question
Options traders and risk managers need to know how market volatility will behave in the future. The implied volatility surface captures the market's collective expectation of future price movement across all available strike prices and expiration dates simultaneously. Forecasting how this entire surface evolves over time is critical for options pricing, portfolio hedging, and risk management.

### Mathematical Problem Formulation
The implied volatility surface at time $t$ is a bivariate function:

$$S_t : (\kappa, \tau) \rightarrow \sigma^{IV} \in \mathbb{R}^+$$

Where:

| Symbol | Definition | Description |
|---|---|---|
| $\kappa = \ln(K/F)$ | Log-moneyness | Normalized strike dimension |
| $K$ | Strike price | Contract exercise price |
| $F = S_0 e^{(r-q)\tau}$ | Forward price | Risk-adjusted current price |
| $\tau = (T-t)/252$ | Time to expiry | Annualized trading days |
| $\sigma^{IV}$ | Implied volatility | Market-embedded volatility expectation |

When discretized onto a uniform grid of $M$ moneyness levels and $N$ expiry buckets:

$$S_t \in \mathbb{R}^{M \times N}$$

The full historical dataset becomes a three-dimensional tensor:

$$\mathcal{D} = \{S_1, S_2, ..., S_T\} \in \mathbb{R}^{T \times M \times N}$$

The forecasting objective is to learn a parameterized mapping:

$$f_\theta : \mathbb{R}^{L \times M \times N} \rightarrow \mathbb{R}^{h \times M \times N}$$

$$\hat{S}_{t+1}, ..., \hat{S}_{t+h} = f_\theta(S_{t-L+1}, ..., S_t)$$

Optimized by minimizing:

$$\theta^* = \arg\min_\theta \sum_{t=L}^{T-h} \mathcal{L}(\hat{S}_{t+1:t+h}, S_{t+1:t+h})$$

### What Implied Volatility Actually Is
Implied volatility is not directly observable. It is extracted by numerically inverting the Black-Scholes-Merton call pricing formula:

$$C_{BS}(S_0, K, r, \tau, \sigma) = S_0 N(d_1) - K e^{-r\tau} N(d_2)$$

Where:

$$d_1 = \frac{\ln(S_0/K) + (r + \frac{\sigma^2}{2})\tau}{\sigma\sqrt{\tau}}, \quad d_2 = d_1 - \sigma\sqrt{\tau}$$

Given the observed market price $C_{mkt}$, implied volatility solves:

$$\sigma^{IV} = \arg\min_\sigma |C_{BS}(\sigma) - C_{mkt}|$$

Solved numerically via Newton-Raphson iteration:

$$\sigma_{n+1} = \sigma_n - \frac{C_{BS}(\sigma_n) - C_{mkt}}{\mathcal{V}(\sigma_n)}$$

Where Vega provides the derivative:

$$\mathcal{V} = S_0 \sqrt{\tau} \cdot \frac{e^{-d_1^2/2}}{\sqrt{2\pi}}$$

---

## 3. Documentation Navigation
- [System Design](file:///d:/projects/neural-volatility-forecaster/documentation/System_Design.md): Architecture, software principles, and layer-by-layer details.
- [ML Pipeline](file:///d:/projects/neural-volatility-forecaster/documentation/ML_Pipeline.md): Data engineering, custom losses, model architectures, and evaluation setups.
- [Operations](file:///d:/projects/neural-volatility-forecaster/documentation/Operations.md): FastAPI serving, MLOps, drift detection, and the week-by-week roadmap.

---

## 4. Technology Stack
- **Data Gathering**: `yfinance` API (scheduled snapshots via `APScheduler`)
- **Data Engineering**: `scipy` (RBF Interpolation), `pandas`, `numpy`, `pyarrow` (Parquet)
- **Deep Learning**: `PyTorch` (ConvLSTM, LSTM, Transformer Encoder models)
- **Inference Serving**: `FastAPI` (REST endpoints for surface forecasts)
- **Experiment Tracking**: `MLflow`
- **Visualization**: `Plotly` (Interactive 3D surfaces and heatmaps)

---

## 5. Project Progress & Developer B Handoff Status

To facilitate onboarding, the engineering progress of the project is estimated at **~45% complete** overall, with the entire data foundation, serving infrastructure, and MLOps monitoring stack fully verified.

### 5.1 Component Status Dashboard

| Component / Layer | Progress | Status | Completed Files |
|---|---|---|---|
| **Layer 1: Data Acquisition** | 100% | Complete | [collector.py](file:///d:/projects/neural-volatility-forecaster/data/collector.py), [run_collector.py](file:///d:/projects/neural-volatility-forecaster/data/run_collector.py) |
| **Layer 2: Data Engineering** | 100% | Complete | [processor.py](file:///d:/projects/neural-volatility-forecaster/data/processor.py), [dataset.py](file:///d:/projects/neural-volatility-forecaster/data/dataset.py) |
| **Layer 3: Feature Engineering** | 10% | Pending | Standard sequence builders are complete; PCA feature projection is pending |
| **Layer 4: Model Training** | 10% | Pending | Interfaces and input sequences defined; PyTorch architectures are pending |
| **Layer 5: Evaluation** | 10% | Pending | Evaluation metrics mathematically documented; model comparison loops are pending |
| **Layer 6: Visualization** | 90% | Complete | [plotting.py](file:///d:/projects/neural-volatility-forecaster/utils/plotting.py) (includes Plotly 3D surfaces, residuals heatmaps, cross-sections) |
| **Layer 7: Serving & Monitoring**| 90% | Complete | [app.py](file:///d:/projects/neural-volatility-forecaster/inference/app.py), [monitor.py](file:///d:/projects/neural-volatility-forecaster/inference/monitor.py) |
| **Infrastructure & CI/CD** | 100% | Complete | [Dockerfile](file:///d:/projects/neural-volatility-forecaster/Dockerfile), [docker-compose.yaml](file:///d:/projects/neural-volatility-forecaster/docker-compose.yaml), [.github/workflows/ci.yml](file:///d:/projects/neural-volatility-forecaster/.github/workflows/ci.yml) |

### 5.2 Developer B Checklist & Handover Tasks

Developer B should focus on the following tasks:
1. **Econometric Baselines (Layer 5)**:
   - Implement Random Walk, Historical Mean, Exponential Smoothing, GARCH(1,1), and HAR-RV benchmarks.
2. **Model Architectures (Layer 4)**:
   - Implement PyTorch models (LSTM, ConvLSTM, Transformer) in the `models/` directory.
   - Accept input sequence tensors of shape `(Batch, Lookback=20, Channels=1, Expiry=7, Moneyness=7)` and return predicted forecasts of shape `(Batch, Horizon=1/5/10, Expiry=7, Moneyness=7)`.
3. **Composite Loss Functions (Layer 4)**:
   - Build custom losses incorporating reconstruction error (MSE) and spatial strike/expiry smoothness penalties (using reflection padding gradients) as detailed in `documentation/ML_Pipeline.md`.
4. **Research Training & Benchmarking**:
   - Write train/validation/test loops in `training/` and evaluate all configurations.
   - Log runs (hyperparameters, loss curves, model checkpoints, preprocessing hashes) to the local MLflow registry.
5. **Production Deployment**:
   - Replace the `MockVolatilityModel` fallback in `inference/app.py` with the trained PyTorch checkpoint (`models/checkpoint.pt`).

