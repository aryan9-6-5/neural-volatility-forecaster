# Machine Learning Pipeline

This document provides a detailed specification of the machine learning pipeline, including data engineering, model architectures, custom loss functions, and the evaluation framework.

---

## 1. Data Engineering

### 1.1 Data Source and Fields Collected
The primary data source is Yahoo Finance via the `yfinance` Python library. The following fields are collected per option contract per snapshot:

| Field | Symbol | Role in System |
|---|---|---|
| Strike Price | $K$ | Surface X-axis coordinate |
| Expiry Date | $T$ | Surface Y-axis coordinate |
| Implied Volatility | $\sigma^{IV}$ | Primary prediction target |
| Last Price | $C_{mkt}$ | IV verification |
| Bid Price | $C_{bid}$ | Spread quality filter |
| Ask Price | $C_{ask}$ | Spread quality filter |
| Volume | $V$ | Liquidity signal |
| Open Interest | $OI$ | Market activity signal |
| Option Type | $\phi$ | Call/Put separation |
| Underlying Price | $S_0$ | Moneyness normalization input |
| Timestamp | $t$ | Time-series ordering key |

### 1.2 Automated Collection Pipeline
Options chain snapshots are collected on a daily schedule using `APScheduler`. Below is the automated pipeline script:

```python
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import yfinance as yf
import pandas as pd
import os

def collect_snapshot(ticker_symbol: str, output_dir: str) -> pd.DataFrame:
    """
    Collect full options chain snapshot and store with timestamp.
    """
    ticker = yf.Ticker(ticker_symbol)
    expiries = ticker.options
    spot_price = ticker.history(period="1d")["Close"].iloc[-1]

    records = []
    for exp in expiries:
        try:
            chain = ticker.option_chain(exp)
            for df, opt_type in [(chain.calls, "call"), (chain.puts, "put")]:
                for _, row in df.iterrows():
                    records.append({
                        "timestamp": datetime.now().isoformat(),
                        "ticker": ticker_symbol,
                        "spot_price": spot_price,
                        "strike": row["strike"],
                        "expiry": exp,
                        "option_type": opt_type,
                        "impliedVolatility": row.get("impliedVolatility", None),
                        "lastPrice": row.get("lastPrice", None),
                        "bid": row.get("bid", None),
                        "ask": row.get("ask", None),
                        "volume": row.get("volume", 0),
                        "openInterest": row.get("openInterest", 0),
                    })
        except Exception as e:
            print(f"Failed to fetch expiry {exp}: {e}")
            continue

    snapshot = pd.DataFrame(records)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M')}.parquet"
    snapshot.to_parquet(filename, index=False)
    print(f"Saved snapshot: {filename} ({len(snapshot)} records)")
    return snapshot

# Schedule daily collection at 4:05 PM on market days
scheduler = BackgroundScheduler()
scheduler.add_job(
    collect_snapshot,
    trigger="cron",
    day_of_week="mon-fri",
    hour=16,
    minute=5,
    args=["SPY", "data/raw"]
)
scheduler.start()
```

### 1.3 Moneyness Normalization
To create time-invariant coordinates across varying spot prices, strikes are normalized into log-moneyness:

$$\kappa = \ln\left(\frac{K}{F}\right)$$

Where the forward price is:
$$F = S_0 \cdot e^{(r - q)\tau}$$
with $r$ representing the risk-free interest rate, and $q$ the dividend yield.

**Standardized Moneyness Grid:**
$$\kappa \in \{-0.30, -0.20, -0.10, 0.00, +0.10, +0.20, +0.30\}$$

| $\kappa$ Value | Interpretation | Trading Meaning |
|---|---|---|
| $-0.30$ | Deep OTM put | Extreme downside protection |
| $-0.20$ | OTM put | Standard downside hedge |
| $-0.10$ | Slight OTM put | Near-money put |
| $0.00$ | At-the-money | Most liquid contracts |
| $+0.10$ | Slight OTM call | Near-money call |
| $+0.20$ | OTM call | Standard upside exposure |
| $+0.30$ | Deep OTM call | Extreme upside speculation |

### 1.4 Expiry Standardization
Expiries are annualized into standard maturity buckets:
$$\tau_i \in \{1/52, 2/52, 1/12, 2/12, 3/12, 6/12, 1.0\}$$

| Bucket | Annualized Value | Interpretation |
|---|---|---|
| $1/52$ | $\approx 0.019$ | 1 week |
| $2/52$ | $\approx 0.038$ | 2 weeks |
| $1/12$ | $\approx 0.083$ | 1 month |
| $2/12$ | $\approx 0.167$ | 2 months |
| $3/12$ | $0.250$ | 1 quarter |
| $6/12$ | $0.500$ | 6 months |
| $1.0$ | $1.000$ | 1 year |

### 1.5 Arbitrage-Free Filtering
To ensure physical consistency, the raw option chains are passed through no-arbitrage checks:
1. **Calendar Spread Condition**: Total variance $w(\kappa, \tau) = (\sigma^{IV})^2 \cdot \tau$ must be non-decreasing in expiry time:
   $$w(\kappa, \tau_1) \leq w(\kappa, \tau_2) \quad \forall \tau_1 < \tau_2$$
2. **Butterfly Condition**: Implied density must be non-negative, corresponding to local smile convexity.
3. **Data Quality Constraints**: Drops illiquid contracts, stale quotes, or excessive spreads.

```python
def apply_arbitrage_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply arbitrage and quality filters to options snapshot.
    """
    filtered = df.copy()

    # Hard IV limits: remove σᴵᵛ outside [1%, 500%]
    filtered = filtered[
        (filtered["impliedVolatility"] > 0.01) &
        (filtered["impliedVolatility"] < 5.00)
    ]

    # Liquidity: remove zero volume and stale quotes
    filtered = filtered[
        (filtered["volume"] > 0) &
        (filtered["bid"] > 0)
    ]

    # Spread quality: remove (ask - bid) / mid > 50%
    filtered["mid"] = (filtered["bid"] + filtered["ask"]) / 2.0
    filtered["spread_ratio"] = (filtered["ask"] - filtered["bid"]) / filtered["mid"]
    filtered = filtered[filtered["spread_ratio"] < 0.50]

    # Calendar spread: total variance must be non-decreasing in τ
    filtered["total_variance"] = filtered["impliedVolatility"] ** 2 * filtered["tau"]
    filtered = filtered.sort_values(["kappa_bin", "tau"])

    calendar_valid = []
    for kappa_val, group in filtered.groupby("kappa_bin"):
        group = group.sort_values("tau")
        tv_values = group["total_variance"].values
        valid_mask = [True]
        for idx in range(1, len(tv_values)):
            if tv_values[idx] >= tv_values[idx - 1]:
                valid_mask.append(True)
            else:
                valid_mask.append(False)
        calendar_valid.append(group[valid_mask])

    filtered = pd.concat(calendar_valid, ignore_index=True)
    print(f"Retained {len(filtered)} contracts after filtering")
    return filtered
```

### 1.6 Grid Interpolation and Dataset Assembly
After filtering, scattered points are projected onto the regular $7 \times 7$ grid using Radial Basis Function (RBF) thin-plate spline interpolation:

$$\sigma^{IV}(\kappa, \tau) = \sum_{i=1}^{N_{pts}} w_i \cdot \phi(\|\mathbf{x} - \mathbf{x}_i\|)$$

Where the kernel function $\phi(r) = r^2 \ln(r)$.

```python
from scipy.interpolate import RBFInterpolator
import numpy as np

GRID_KAPPAS = np.array([-0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30])
GRID_TAUS = np.array([1/52, 2/52, 1/12, 2/12, 3/12, 6/12, 1.0])

def interpolate_to_grid(kappas: np.ndarray, taus: np.ndarray, ivs: np.ndarray) -> np.ndarray:
    """
    Interpolate scattered IV points onto standardized 7x7 grid.
    """
    points = np.column_stack([kappas, taus])
    interpolator = RBFInterpolator(points, ivs, kernel="thin_plate_spline", smoothing=0.01)
    grid_k, grid_t = np.meshgrid(GRID_KAPPAS, GRID_TAUS)
    grid_points = np.column_stack([grid_k.ravel(), grid_t.ravel()])
    grid_iv = interpolator(grid_points).reshape(grid_k.shape)
    return np.clip(grid_iv, 0.01, 5.0)

def build_surface_dataset(snapshot_dir: str) -> tuple:
    """
    Load all snapshots and assemble time-ordered surface tensor.
    """
    import glob
    files = sorted(glob.glob(f"{snapshot_dir}/*.parquet"))
    surfaces, timestamps = [], []

    for filepath in files:
        snap = pd.read_parquet(filepath)
        try:
            filtered = apply_arbitrage_filters(snap)
            grid_iv = interpolate_to_grid(
                filtered["kappa"].values,
                filtered["tau"].values,
                filtered["impliedVolatility"].values,
            )
            surfaces.append(grid_iv)
            timestamps.append(snap["timestamp"].iloc[0])
        except Exception as e:
            print(f"Skipping {filepath}: {e}")
            continue

    dataset = np.stack(surfaces, axis=0) # Shape: (T, 7, 7)
    return dataset, timestamps
```

**Dataset Split**: The full chronological dataset $\mathcal{D} = \{S_1, ..., S_T\}$ is split into Train ($[0, 0.6T]$), Validation ($[0.6T, 0.8T]$), and Test ($[0.8T, T]$). *Data is never shuffled to preserve the temporal ordering.*

---

## 2. Model Design

### 2.1 Stacked LSTM Baseline
Processes flattened surface vectors through stacked LSTM layers.

**Input**: $X \in \mathbb{R}^{B \times L \times (M \cdot N)}$ where $M \cdot N = 49$

**LSTM Equations**:
$$\mathbf{f}_t = \sigma(\mathbf{W}_f[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$
$$\mathbf{i}_t = \sigma(\mathbf{W}_i[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$
$$\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_C[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C)$$
$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$$

```text
Input: (B, L, 49) -> LSTM 1 (256 units) -> LSTM 2 (256 units) -> LSTM 3 (128 units) -> Linear(128->512) -> Linear(512->h*49) -> Reshape -> (B, h, 7, 7)
```

### 2.2 ConvLSTM Primary Proposed Model
Preserves spatial 2D surface correlations during temporal transitions by using convolutions inside the recurrent cells instead of matrix multiplications.

**Input**: $X \in \mathbb{R}^{B \times L \times C \times M \times N}$

**ConvLSTM Cell Equations**:
$$\mathbf{I}_t = \sigma(\mathbf{W}_{xi} * \mathcal{X}_t + \mathbf{W}_{hi} * \mathcal{H}_{t-1} + \mathbf{W}_{ci} \odot \mathcal{C}_{t-1} + \mathbf{b}_i)$$
$$\mathbf{F}_t = \sigma(\mathbf{W}_{xf} * \mathcal{X}_t + \mathbf{W}_{hf} * \mathcal{H}_{t-1} + \mathbf{W}_{cf} \odot \mathcal{C}_{t-1} + \mathbf{b}_f)$$
$$\mathcal{C}_t = \mathbf{F}_t \odot \mathcal{C}_{t-1} + \mathbf{I}_t \odot \tanh(\mathbf{W}_{xc} * \mathcal{X}_t + \mathbf{W}_{hc} * \mathcal{H}_{t-1} + \mathbf{b}_c)$$
$$\mathbf{O}_t = \sigma(\mathbf{W}_{xo} * \mathcal{X}_t + \mathbf{W}_{ho} * \mathcal{H}_{t-1} + \mathbf{W}_{co} \odot \mathcal{C}_t + \mathbf{b}_o)$$
$$\mathcal{H}_t = \mathbf{O}_t \odot \tanh(\mathcal{C}_t)$$

Where $*$ is spatial 2D convolution and all state tensors $\mathcal{X}_t, \mathcal{H}_t, \mathcal{C}_t \in \mathbb{R}^{B \times C \times M \times N}$.

```text
Input: (B, L, 1, 7, 7)
    │
ConvLSTM 1 (hidden=32, kernel=3x3, pad=1x1) -> BatchNorm3D
    │
ConvLSTM 2 (hidden=64, kernel=3x3, pad=1x1) -> BatchNorm3D
    │
ConvLSTM 3 (hidden=64, kernel=3x3, pad=1x1)
    │
Last Hidden State: (B, 64, 7, 7) -> Conv2D(64->32, kernel=3) -> Conv2D(32->h, kernel=1) -> Output: (B, h, 7, 7)
```
*Why ConvLSTM is superior*: It prevents loss of spatial dependencies. Flipping volatility smile shapes (strike dimension) and steepening term structures (expiry dimension) are preserved using spatial convolutions.

### 2.3 Transformer Encoder Advanced Variant
Applies multi-head self-attention mechanisms to capture long-range temporal surface correlations:
- **Surface Tokenization**: $\mathbf{e}_t = \mathbf{W}_{emb} \cdot \text{vec}(S_t) + \mathbf{b}_{emb} \in \mathbb{R}^{d_{model}}$
- **Positional Encoding**: Uses sinusoidal encodings $PE_{(t, 2i)} = \sin\left(\frac{t}{10000^{2i/d_{model}}}\right)$.
- **Multi-Head Attention**: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.

---

## 3. Training System

### 3.1 Smoothness-Regularized Loss Function
Existing papers (e.g. Medvedev and Wang, 2022) apply post-prediction smoothing (Savitzky-Golay filters), which does not affect model weights during optimization. In contrast, this system integrates smoothness constraints directly into the loss function:

| Aspect | Savitzky-Golay Post-Processing | Smoothness Loss (This Project) |
|---|---|---|
| **Application** | Post-inference | Active training parameter update |
| **Gradient Signal** | None (non-differentiable filter) | Backpropagated gradients |
| **Learned Logic** | Surface structure unaffected | Regularizes representations |

**Mathematical Penalties**:
- **MSE Loss**:
  $$\mathcal{L}_{MSE} = \frac{1}{B \cdot h \cdot M \cdot N} \sum_{b,t,i,j} (\hat{\sigma}_{b,t,i,j} - \sigma_{b,t,i,j})^2$$
- **Strike Smoothness Penalty**:
  $$\mathcal{L}_{strike} = \frac{1}{B \cdot h} \sum_{b,t} \sum_{i=2}^{M-1} \sum_{j=1}^{N} \left(\hat{\sigma}_{i+1,j} - 2\hat{\sigma}_{i,j} + \hat{\sigma}_{i-1,j}\right)^2$$
- **Expiry Smoothness Penalty**:
  $$\mathcal{L}_{expiry} = \frac{1}{B \cdot h} \sum_{b,t} \sum_{i=1}^{M} \sum_{j=2}^{N-1} \left(\hat{\sigma}_{i,j+1} - 2\hat{\sigma}_{i,j} + \hat{\sigma}_{i,j-1}\right)^2$$
- **Composite Training Loss**:
  $$\mathcal{L}_{total} = \mathcal{L}_{MSE} + \lambda_1 \mathcal{L}_{strike} + \lambda_2 \mathcal{L}_{expiry}$$

#### PyTorch Implementation:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmoothnessRegularizedLoss(nn.Module):
    """
    Composite loss for volatility surface forecasting.
    Combines MSE reconstruction with strike & expiry smoothness regularizations.
    """
    def __init__(self, lambda_strike: float = 0.01, lambda_expiry: float = 0.01):
        super().__init__()
        self.lambda_strike = lambda_strike
        self.lambda_expiry = lambda_expiry

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple:
        # y_pred, y_true shape: (B, h, M, N)
        mse_loss = F.mse_loss(y_pred, y_true)

        # Strike smoothness (dim=-2) with reflection padding
        padded_strike = F.pad(y_pred, pad=(0, 0, 1, 1), mode="reflect")
        strike_d2 = padded_strike[:, :, 2:, :] - 2 * padded_strike[:, :, 1:-1, :] + padded_strike[:, :, :-2, :]
        strike_loss = torch.mean(strike_d2 ** 2)

        # Expiry smoothness (dim=-1) with reflection padding
        padded_expiry = F.pad(y_pred, pad=(1, 1, 0, 0), mode="reflect")
        expiry_d2 = padded_expiry[:, :, :, 2:] - 2 * padded_expiry[:, :, :, 1:-1] + padded_expiry[:, :, :, :-2]
        expiry_loss = torch.mean(expiry_d2 ** 2)

        total_loss = mse_loss + self.lambda_strike * strike_loss + self.lambda_expiry * expiry_loss
        return total_loss, {
            "mse_loss": mse_loss.item(),
            "strike_loss": strike_loss.item(),
            "expiry_loss": expiry_loss.item(),
            "total_loss": total_loss.item()
        }
```

### 3.2 Optional Enhancement: Arbitrage-Aware Loss
To enforce financial constraints directly, we add calendar spread and butterfly penalty terms to $\mathcal{L}_{total}$:
- **Calendar Spread Penalty**:
  $$\mathcal{L}_{calendar} = \frac{1}{M} \sum_{i=1}^{M} \sum_{j=1}^{N-1} \max\left(0, \; \hat{\sigma}_{i,j}^2 \tau_j - \hat{\sigma}_{i,j+1}^2 \tau_{j+1}\right)^2$$
- **Butterfly Spread Penalty**:
  $$\mathcal{L}_{butterfly} = \frac{1}{N} \sum_{j=1}^{N} \sum_{i=2}^{M-1} \max\left(0, \; -\left(\hat{\sigma}_{i+1,j} - 2\hat{\sigma}_{i,j} + \hat{\sigma}_{i-1,j}\right)\right)^2$$

### 3.3 Training Configuration
The primary training parameters are defined as:

| Hyperparameter | Value | Reason |
|---|---|---|
| Optimizer | AdamW | Decouples weight decay regularization |
| Learning Rate | $1 \times 10^{-3}$ | Standard starting point |
| LR Scheduler | CosineAnnealingLR | Decays optimizer steps smoothly to zero |
| Batch Size | 32 | Memory efficiency |
| Lookback Window ($L$) | 20 | Incorporates one month of trading days |
| Forecast Horizon ($h$) | 1, 5, 10 | Captures daily, weekly, and bi-weekly targets |
| Early Stopping | 15 epochs patience | Avoids model overfitting |
| Gradient Clipping | 1.0 | Guards against gradient explosion |

---

## 4. Evaluation Framework

### 4.1 Quantitative Metrics
- **Root Mean Squared Error (RMSE)**:
  $$RMSE = \sqrt{\frac{1}{M \cdot N} \sum_{i=1}^{M} \sum_{j=1}^{N} (\hat{\sigma}_{ij} - \sigma_{ij})^2}$$
- **Mean Absolute Error (MAE)**:
  $$MAE = \frac{1}{M \cdot N} \sum_{i=1}^{M} \sum_{j=1}^{N} |\hat{\sigma}_{ij} - \sigma_{ij}|$$
- **Mean Absolute Percentage Error (MAPE)**:
  $$MAPE = \frac{100\%}{M \cdot N} \sum_{i=1}^{M} \sum_{j=1}^{N} \frac{|\hat{\sigma}_{ij} - \sigma_{ij}|}{\sigma_{ij}}$$
- **Directional Accuracy (DA)**:
  $$DA = \frac{1}{T_{test}} \sum_{t} \mathbf{1}\left[\text{sign}(\hat{S}_t - S_{t-1}) = \text{sign}(S_t - S_{t-1})\right]$$
- **Total Variation (TV)** (smoothness tracker):
  $$TV(\hat{S}) = \sum_{i=1}^{M-1} \sum_{j=1}^{N} |\hat{\sigma}_{i+1,j} - \hat{\sigma}_{i,j}| + \sum_{i=1}^{M} \sum_{j=1}^{N-1} |\hat{\sigma}_{i,j+1} - \hat{\sigma}_{i,j}|$$
- **Arbitrage Violation Count (AV)**:
  $$AV(\hat{S}) = \sum_{i=1}^{M} \sum_{j=1}^{N-1} \mathbf{1}\left[\hat{\sigma}_{i,j}^2 \tau_j > \hat{\sigma}_{i,j+1}^2 \tau_{j+1}\right]$$

### 4.2 Region-Specific Error Decomposition
The surface error is evaluated across sub-regions to assess trading performance:

| Region | Filter Condition | Financial Significance |
|---|---|---|
| **ATM** | $|\kappa| < 0.05$ | Most liquid contracts; major pricing driver |
| **OTM Puts** | $\kappa < -0.10$ | Downside hedging tracking |
| **OTM Calls** | $\kappa > +0.10$ | Upside return capture |
| **Deep Wings** | $|\kappa| > 0.20$ | Extreme tail risk & crash protection |
| **Short-dated** | $\tau < 1/12$ | Near-term event risk |
| **Long-dated** | $\tau > 6/12$ | LEAPS contracts & term structure trends |

### 4.3 Baseline Models

| Model | Type | Prediction Rule |
|---|---|---|
| **Naive Random Walk** | Statistical | $\hat{S}_{t+1} = S_t$ |
| **Historical Mean** | Statistical | $\hat{S}_{t+1} = \frac{1}{L}\sum_{i=1}^{L} S_{t-i}$ |
| **Exponential Smoothing** | Statistical | $\hat{S}_{t+1} = \alpha S_t + (1-\alpha)\hat{S}_t$ |
| **GARCH(1,1)** | Econometric | Grid cell ARMA-GARCH volatility prediction |
| **HAR-RV** | Econometric | Heterogeneous autoregressive (daily, weekly, monthly) |

### 4.4 Ablation Studies
1. **Architecture Comparison**: Stacked LSTM (1D) vs. ConvLSTM (2D, 1 layer) vs. ConvLSTM (2D, 3 layers).
2. **Loss Regularization**: MSE-only loss vs. MSE + Smoothness ($\lambda \in \{0.001, 0.01, 0.1\}$) vs. MSE + Smoothness + Arbitrage.
3. **Lookback Windows**: Testing $L \in \{5, 10, 20, 40\}$.
4. **Forecast Horizons**: Testing $h \in \{1, 5, 10\}$.

### 4.5 Evaluation Results Template
Results are averaged across 5 random seeds (reported as `mean ± std`):

| Model | RMSE | ATM RMSE | OTM-P RMSE | OTM-C RMSE | TV | AV | DA % |
|---|---|---|---|---|---|---|---|
| Random Walk | · ± · | · | · | · | · | · | · |
| Historical Mean | · ± · | · | · | · | · | · | · |
| Exp. Smoothing | · ± · | · | · | · | · | · | · |
| GARCH(1,1) | · ± · | · | · | · | · | · | · |
| HAR-RV | · ± · | · | · | · | · | · | · |
| LSTM | · ± · | · | · | · | · | · | · |
| ConvLSTM (MSE) | · ± · | · | · | · | · | · | · |
| **ConvLSTM (MSE+Smooth)** | **· ± ·** | **·** | **·** | **·** | **·** | **·** | **·** |
