# Operations and Roadmap

This document outlines the system's production deployment, monitoring pipelines, literature-based novelty assessment, and week-by-week development roadmap.

---

## 1. Deployment Architecture

The system is designed for low-latency serving of multi-step implied volatility forecasts.

```text
       Incoming Client Request
                 │
                 ▼
       [ FastAPI Serving API ]
                 │
                 ▼
       [ MLflow Model Registry ] (Fetch Active production model)
                 │
                 ▼
       [ Options Chain Preprocessor ] (Filter & RBF Interpolate)
                 │
                 ▼
       [ PyTorch Inference Engine ] (Compute ConvLSTM predictions)
                 │
                 ▼
       [ Plotly 3D Output Generator ] (Return JSON forecasts & visual charts)
```

### 1.1 serving Endpoint (FastAPI)
- **Model Load**: On startup, FastAPI loads the active production model version from the local or remote **MLflow Model Registry**.
- **Payload Validation**: Requests submit raw options snapshots (json format), which are parsed and validated by Pydantic models.
- **Processing**: The API runs the incoming data through standard log-moneyness normalization, calendar/butterfly spreads filters, and projects it onto the $7 \times 7$ grid via thin-plate RBF interpolation.
- **Inference**: Reshapes the input tensor into sequence dimensions (e.g., $1 \times L \times 1 \times 7 \times 7$) and runs a forward pass on a GPU device.

---

## 2. Monitoring & MLOps

Markets are highly non-stationary. To counter performance degradation during regime shifts, the operations pipeline monitors drift and automates updates:

### 2.1 Drift Detection Metrics
- **Data Drift**: Tracks structural changes in market options distributions. The system calculates the **Kullback-Leibler (KL) Divergence** between the spatial densities of incoming implied volatility surfaces and the baseline training dataset:
  $$D_{KL}(P \parallel Q) = \sum_{i,j} P(\kappa_i, \tau_j) \ln\left(\frac{P(\kappa_i, \tau_j)}{Q(\kappa_i, \tau_j)}\right)$$
- **Prediction Drift**: Monitors forecasting decay by computing a rolling 10-day root-mean-squared error (RMSE) on historical observations.

### 2.2 Continuous Retraining Cycle
1. **Drift Alert**: Triggered if the 10-day rolling RMSE exceeds a pre-defined threshold (e.g., 1.5x the baseline validation error) or if KL divergence exceeds critical values.
2. **Data Assembly**: The system extracts the latest parquet snapshots accumulated in Layer 1.
3. **Automated Pipeline**: Instantiates a retraining runner to update weights (starting from the current production baseline to speed up training).
4. **Champion Verification**: Evaluates the retrained model. If it outperforms the existing model on the validation dataset and passes no-arbitrage checks, MLflow promotes it to the `Production` tag.

---

## 3. Novelty Assessment

### 3.1 Literature Positioning Table

| Paper | Year | Task | Smoothness in Loss | Arbitrage in Loss | Region Analysis | Pipeline |
|---|---|---|---|---|---|---|
| Medvedev & Wang | 2022 | Forecasting | ✗ Post-hoc S-G filter | ✗ | ✗ | ✗ |
| Stanford CS231n | 2022 | Forecasting | ✗ | ✗ | ✗ | ✗ |
| Deep Smoothing (NeurIPS) | 2020 | **Nowcasting** | ✓ (for fitting only) | ✗ | ✗ | ✗ |
| Operator Deep Smoothing | 2025 | **Nowcasting** | ✓ (for fitting only) | ✓ (for fitting only) | ✗ | ✗ |
| Hack EUR Thesis | 2021 | Forecasting | ✗ | ✗ | ✗ | ✗ |
| **This Project** | **2026** | **Forecasting** | **✓ In training loss** | **✓ Optional** | **✓** | **✓** |

### 3.2 The Critical Distinction
- **Nowcasting Papers** (e.g. NeurIPS 2020, Operator Deep Smoothing 2025): Attempt to *fit* or *calibrate* the current surface to noisy market data. They apply smoothness and arbitrage constraints for fitting, not for predicting future dates.
- **Forecasting Papers** (e.g. Medvedev & Wang 2022): Attempt to predict future surfaces using architectures like ConvLSTM. However, they train with standard MSE loss and apply smoothing (Savitzky-Golay filters) only as post-processing. Because this post-processing operates outside the PyTorch computational graph, it propagates no gradient signals and cannot regularize model weights.
- **This Project**: Combines **forecasting future surfaces** with a **smoothness-regularized training objective**. The model learns to produce structurally consistent and mathematically smooth volatility surfaces directly through backpropagation.

### 3.3 Confirmed Contributions

| Contribution | Novelty Level | Evidence |
|---|---|---|
| **Smoothness loss in forecasting** | Strong | First application in forecasting training objectives |
| **Arbitrage penalty in forecasting** | Strong | Differentiable calendar and butterfly penalties for forecasting |
| **Region-specific decomposition** | Moderate | Evaluates model performance based on trading significance |
| **Complete reproducible pipeline** | Strong | Integrates yfinance data collection, serving, and tracking |

### 3.4 Venue Assessment

| Venue | Novelty Fit | Likelihood | Strategy |
|---|---|---|---|
| **arXiv (q-fin.CP)** | Excellent | ~80% | Upload draft first to establish technical priority |
| **Thesis Deliverable** | Excellent | ~85-90% | Use as the primary research submission |
| **ACM ICAIF** | Good | ~65-75% | Emphasize smoothness loss and spatial ablatings |
| **IEEE ICMLA** | Good | ~65-75% | Emphasize system architecture and MLOps integrations |
| **Journal of Financial Data Science**| Good | ~70% | Focus on region-specific volatility smile behaviors |

---

## 4. Execution Roadmap

### Week-by-Week Checklist

#### Week 1-2: Data Foundation
- `[x]` Choose primary ticker (SPY for Medvedev comparison)
- `[x]` Run quick yfinance test (2-3 hours validation)
- `[x]` Build options chain collector with yfinance
- `[x]` Implement APScheduler for daily 4:05 PM collection
- `[x]` Build moneyness normalization: $\kappa = \ln(K/F)$
- `[x]` Build expiry standardization to 7 standard buckets
- `[x]` Implement all arbitrage filters (calendar, butterfly, hard)
- `[x]` Implement RBF interpolation to $7 \times 7$ grid
- `[x]` Build one surface, visualize with Plotly 3D (validation)
- `[x]` Collect minimum 6 months of daily snapshots
- `[x]` Store as Parquet files with timestamps
- `[x]` Document data quality statistics (missing rate, coverage)

#### Week 3-4: Baselines First
- `[x]` Implement naive random walk baseline
- `[x]` Implement historical mean baseline
- `[x]` Implement exponential smoothing baseline
- `[x]` Implement GARCH(1,1) per grid cell
- `[x]` Implement HAR-RV baseline
- `[x]` Evaluate all baselines on temporal test set
- `[x]` Store all baseline metrics in MLflow
- `[x]` Verify baselines produce reasonable numbers
- `[x]` Create initial baseline results table
- `[x]` Fix any data pipeline issues revealed by baselines

#### Week 5-6: Model Training
- `[x]` Implement LSTM baseline in PyTorch
- `[x]` Implement ConvLSTM in PyTorch
- `[x]` Implement `SmoothnessRegularizedLoss` class
- `[x]` Set up MLflow experiment tracking
- `[x]` Define temporal train/val/test split (60/20/20)
- `[x]` Train LSTM with MSE loss (5 seeds)
- `[x]` Train ConvLSTM with MSE loss (5 seeds)
- `[x]` Train ConvLSTM with MSE + Smoothness loss (5 seeds)
- `[x]` Log all experiments to MLflow
- `[x]` Save model checkpoints per epoch
- `[x]` Monitor for overfitting (early stopping patience=15)
- `[x]` Implement gradient clipping (max norm 1.0)

#### Week 7: Comprehensive Evaluation
- `[x]` Compute all metrics: RMSE, MAE, MAPE, DA, TV, AV
- `[x]` Decompose errors by 6 surface regions
- `[x]` Generate full results table with mean $\pm$ std
- `[x]` Generate 3D actual vs predicted surface plots
- `[x]` Generate error residual heatmaps
- `[x]` Generate volatility smile cross-section comparisons
- `[x]` Generate term structure cross-section comparisons
- `[x]` Run all 4 ablation studies
- `[x]` Run Wilcoxon signed-rank significance tests
- `[x]` Document all findings clearly

#### Week 8-9: Paper Writing
- `[x]` Write Abstract (250 words)
- `[x]` Write Introduction with clear research gap
- `[x]` Write Literature Review citing Medvedev & Wang correctly
- `[x]` Write Methodology
- `[x]` Write Data Description with statistics table
- `[x]` Write Experimental Setup
- `[x]` Write Results with all tables and figures
- `[x]` Write Discussion and limitations
- `[x]` Compile LaTeX source and submit
