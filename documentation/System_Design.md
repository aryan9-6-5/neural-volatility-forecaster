# System Design

This document details the high-level system architecture, software engineering principles, modular organization, and security framework of the Neural Volatility Surface Forecaster.

---

## 1. Architectural Principles

To deliver a production-ready research platform, the system is designed around four core principles:
- **Separation of Concerns**: The platform is strictly decoupled into independent operational layers (Acquisition, Engineering, Modeling, Serving) to make debugging, testing, and extension straightforward.
- **Modularity**: Components communicate via clearly defined interfaces, enabling seamless replacement of individual components (e.g., swapping yfinance for another provider or changing the interpolation kernel).
- **Scalability**: Designed to ingest high-frequency options chain data and process structured multi-dimensional tensors using GPU-accelerated operations.
- **Reproducibility**: Every model run, dataset snapshot, and experimental evaluation is fully versioned, and configurations are defined declaratively in configurations to ensure consistent outputs.

---

## 2. Complete Seven-Layer Architecture

The system is organized into a cohesive, seven-layer structure spanning raw data ingestion to production serving:

```text
┌───────────────────────────────────────────────────────────────┐
│                    LAYER 1: DATA ACQUISITION                  │
│                                                               │
│   yfinance API ──→ Options Chain Snapshot Collector           │
│   APScheduler ──→ Daily/Hourly Automated Execution           │
│   Underlying ──→ Stock/Index Price Collector                  │
│   Storage ──→ Parquet files with timestamps                   │
│                                                               │
│   Output: Raw timestamped options chain snapshots             │
├───────────────────────────────────────────────────────────────┤
│                    LAYER 2: DATA ENGINEERING                  │
│                                                               │
│   Raw Snapshot ──→ Timestamp Alignment                        │
│                 ──→ Moneyness Normalization: κ = ln(K/F)      │
│                 ──→ Expiry Standardization: τ = (T-t)/252     │
│                 ──→ IV Extraction via BSM Inversion           │
│                 ──→ Arbitrage-Free Filtering                  │
│                 ──→ RBF Grid Interpolation                    │
│                 ──→ Surface Tensor Construction               │
│                                                               │
│   Output: S_t ∈ ℝ^(M×N) per timestamp                        │
├───────────────────────────────────────────────────────────────┤
│                    LAYER 3: FEATURE ENGINEERING               │
│                                                               │
│   PCA Decomposition ──→ Level/Slope/Curvature components      │
│   ATM Extraction ──→ At-the-money volatility time series      │
│   Skew Metrics ──→ Put-call skew features                     │
│   Term Structure ──→ Short vs long dated slope                │
│   Realized Vol ──→ Rolling 22-day historical volatility       │
│                                                               │
│   Output: Enriched feature tensor for model input            │
├───────────────────────────────────────────────────────────────┤
│                    LAYER 4: MODEL TRAINING                    │
│                                                               │
│   Baseline: Stacked LSTM (flattened surface input)            │
│   Primary: ConvLSTM (2D spatial structure preserved)          │
│   Advanced: Transformer Encoder (long-range dependencies)     │
│                                                               │
│   Loss: MSE + λ₁·L_strike + λ₂·L_expiry                      │
│   Optional: + λ₃·L_calendar + λ₄·L_butterfly                 │
│   Optimizer: AdamW + CosineAnnealingLR                        │
│   Tracking: MLflow (every run logged)                         │
│                                                               │
│   Output: Trained model checkpoints per configuration         │
├───────────────────────────────────────────────────────────────┐
│                    LAYER 5: EVALUATION                        │
│                                                               │
│   Metrics: RMSE, MAE, MAPE, Directional Accuracy             │
│   Decomposition: ATM / OTM-P / OTM-C / Wings / Short / Long  │
│   Benchmarks: Random Walk, Mean, GARCH, HAR, ExpSmoothing    │
│   Ablation: Architecture, Loss, Lookback, Horizon             │
│   Statistics: Wilcoxon signed-rank test (mean ± std)         │
│                                                               │
│   Output: Complete evaluation report with all metrics         │
├───────────────────────────────────────────────────────────────┤
│                    LAYER 6: VISUALIZATION                     │
│                                                               │
│   Plotly 3D ──→ Interactive actual surface rendering          │
│   Plotly 3D ──→ Interactive predicted surface rendering       │
│   Heatmaps ──→ Residual error spatial visualization          │
│   Cross-sections ──→ Volatility smile and term structure      │
│   Training ──→ Loss convergence and gradient norm curves      │
│   Animation ──→ Temporal surface evolution over time          │
│                                                               │
│   Output: Interactive HTML visualization dashboards           │
├───────────────────────────────────────────────────────────────┤
│                    LAYER 7: SERVING AND MONITORING            │
│                                                               │
│   FastAPI ──→ REST inference endpoints                        │
│   MLflow ──→ Model registry and version control              │
│   KL Divergence ──→ Data drift detection                      │
│   Rolling RMSE ──→ Prediction drift detection                 │
│   Retraining ──→ Automated trigger when drift detected        │
│                                                               │
│   Output: Production-ready continuously learning system       │
└───────────────────────────────────────────────────────────────┘
```

---

## 3. Deep Dive into the Seven Layers

### Layer 1: Data Acquisition
- **Sources**: Yahoo Finance APIs for options chains, spot prices, and risk-free rates.
- **Scheduling**: Automated collector scripts run on a schedule (e.g., using `APScheduler` to trigger daily at 4:05 PM on market days).
- **Storage**: Raw snapshots are persisted as Parquet files with ISO-8601 timestamps to maintain an auditable timeline.

### Layer 2: Data Engineering
- **Alignment**: Standardizes timestamps across varying expiries.
- **Normalization**: Translates strikes into log-moneyness $\kappa = \ln(K/F)$ and expiries into annualized trading days $\tau = (T-t)/252$.
- **BSM Inversion**: Numerically solves for implied volatility (IV) via Newton-Raphson iteration.
- **No-Arbitrage Filters**: Removes contracts violating calendar spread inequalities or butterfly/convexity requirements.
- **Interpolation**: Projects scattered points onto a standardized $7 \times 7$ grid using Radial Basis Function (RBF) thin-plate spline interpolation.

### Layer 3: Feature Engineering
- **PCA Decomposition**: Projects the surface onto level, slope, and curvature components.
- **ATM Extraction**: Extracts at-the-money implied volatility time series.
- **Volatility Skew**: Captures the put-call skew ratio.
- **Term Structure**: Computes the slope across short vs. long-dated contracts.
- **Realized Volatility**: Calculates rolling historical asset volatility (e.g., rolling 22-day window) as an auxiliary input feature.

### Layer 4: Model Training
- **Architectures**: PyTorch implementations of stacked LSTM (baseline), spatiotemporal ConvLSTM (primary model), and a Transformer Encoder (advanced variant).
- **Loss Functions**: Differentiable training objectives incorporating Reconstruction Loss (MSE) and spatial curvature penalties (strike/expiry smoothness).
- **Optimization**: Cosine Annealing learning rate schedulers paired with the AdamW optimizer.
- **Experiment Tracking**: Automatic logging of hyperparameters, weights, loss curves, and artifact checkpoints using MLflow.

### Layer 5: Evaluation
- **Quantitative Metrics**: Implements RMSE, MAE, MAPE, Directional Accuracy (DA), Total Variation (TV), and Arbitrage Violation counts (AV).
- **Region-Specific Error Decomposition**: Segregates performance tracking across ATM, OTM puts, OTM calls, wings, and short/long expiries.
- **Benchmarks**: Evaluates forecast gains against Naive Random Walk, Historical Mean, Exponential Smoothing, GARCH(1,1), and HAR-RV.
- **Ablation Studies**: Systematic trials isolating the effects of lookback windows, forecasting horizons, model architecture depth, and regularization weights.

### Layer 6: Visualization
- **Plotly 3D Surfaces**: Renders interactive 3D visualizations of the actual versus predicted volatility surfaces.
- **Heatmaps**: Creates spatial visual representations of local residual error.
- **Cross-Sectional Curves**: Inspects the volatility smile (IV vs. strike) and term structure (IV vs. expiry).
- **Animations**: Generates visual sequences displaying the temporal evolution of the surface.

### Layer 7: Serving and Monitoring
- **REST Endpoints**: Serves low-latency predictions via FastAPI endpoints.
- **Model Registry**: MLflow manages model versions and deployment statuses (Staging vs. Production).
- **Drift Monitors**: Computes Kullback-Leibler (KL) Divergence on incoming option chain distributions and tracks rolling RMSE to trigger automated retraining tasks.

---

## 4. Software Engineering Practices

The codebase applies standard software patterns to structure machine learning workflows:
- **Abstraction**: Uses generic builder interfaces (`build_model`, `build_dataset`, `build_loss`) to instantiate classes dynamically from parameters without hardcoding.
- **Encapsulation**: Encapsulates model states, epoch handling, and metric logging in a robust `Trainer` class.
- **Config-Driven Development**: Experiments are run using declarative configuration files (YAML/JSON) to control model layers, learning rates, lookbacks, and training regimes.

### Planned Directory Layout
```text
project/
├── data/           # Data collection (yfinance, APScheduler), cleaning, and RBF interpolation
├── models/         # PyTorch architectures (LSTM, ConvLSTM, Transformer)
├── training/       # Custom loss functions (smoothness/arbitrage regularizations) and training loops
├── evaluation/     # Metrics, region-specific error analysis, and benchmark calculations
├── inference/      # FastAPI serving, request validation, and MLflow model load logic
├── configs/        # Declarative configuration files for training and pipelines
└── utils/          # Plotly 3D surface plot generation and helper utilities
```

---

## 5. Security & Ethical Design

The system implements safety, explainability, and auditable logging controls:
- **Input Validation**: Hard bounds verify that input pricing, volumes, and expiries are non-negative, clipping extreme outliers to prevent model instability.
- **Explainability**: Applies Grad-CAM (for the Conv2D layers in ConvLSTM) and SHAP analysis to reveal which regions of the past surface drive changes in the forecasted surface.
- **Auditability**: Retains complete history of incoming request data, outgoing predictions, model version IDs, and serving latency.
- **Robustness**: Error-handling wrappers isolate bad raw snapshots and skip them to ensure the online pipeline remains stable.
