# ML Pipeline

This document describes the end-to-end machine learning lifecycle, from raw financial data to a validated forecasting model.

## 1. Data Engineering

The system is data-centric; model performance depends on the quality and reproducibility of the input pipeline.

### Ingestion & Cleaning
- **Data Sources**: NSE APIs, Yahoo Finance, Polygon.io, and Binance Options API.
- **Cleaning**: Removes illiquid contracts, handles missing values via interpolation, and aligns timestamps to ensure synchronized market snapshots.
- **Fields**: Strike Price, Expiry Date, Option Premium, Volume, Open Interest, and Underlying Price.

### Transformation & Augmentation
- **Volatility Surface Construction**: Computes implied volatility (IV) and maps it onto structured grids (Strike vs. Expiry).
- **Normalization**: Scales prices, volume, and IV to stabilize neural network training.
- **Augmentation**: Employs noise injection, temporal window shifting, and surface smoothing variations to reduce overfitting.
- **Versioning**: Uses **DVC** to track dataset iterations and ensure reproducibility.

## 2. Model Design

The project follows a **Simplicity First** strategy:
1. **Baseline**: Start with lightweight models (e.g., Simple CNN, Shallow LSTM) to establish benchmarks.
2. **Incremental Improvement**: Introduce complexity (Attention, Transformers, Hybrid architectures) only when justified by performance gains.
3. **Config-Driven**: Architectures and hyperparameters are managed via YAML/JSON configs, never hardcoded.

## 3. Training System

Designed for reliability and performance during long-running sessions:
- **Reliability**: Implements periodic **Checkpoint Recovery** to save weights, optimizer state, and scheduler state, allowing resumes after failures.
- **Optimization**: Uses **Mixed Precision Training** (PyTorch AMP) to reduce memory usage and accelerate GPU computation.
- **Data Loading**: Parallel dataloaders with prefetching and memory pinning minimize GPU idle time.
- **Parallelism**: Supports **Distributed Data Parallel (DDP)** for scaling across multiple GPUs.

## 4. Evaluation & Validation

A rigorous framework ensures the model generalizes to unseen market conditions:
- **Core Metrics**: RMSE, MAE, and MAPE for surface reconstruction; ROC-AUC for classification-based tasks.
- **Calibration**: Analyzes reliability diagrams to ensure prediction confidence matches actual correctness probability.
- **Robustness Testing**: Evaluates stability under noisy inputs, sparse market data, and extreme volatility spikes.
- **Interpretability**: Uses **Grad-CAM** and feature importance analysis to visualize which market factors drive specific forecasts.
- **Uncertainty Estimation**: Employs Monte Carlo Dropout to assign risk/confidence scores to each prediction.
