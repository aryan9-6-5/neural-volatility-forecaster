import os
import yaml
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import logging

# Import project utilities
from data.processor import process_raw_snapshot, interpolate_to_grid, GRID_KAPPAS, GRID_TAUS
from data.dataset import generate_synthetic_dataset, build_surface_dataset
from data.rolling_buffer import SurfaceHistoryBuffer
from utils.plotting import plot_3d_surface, plot_residuals_heatmap, plot_cross_sections, fig_to_json
from inference.monitor import DriftMonitor
from inference.prediction_store import PredictionStore
from models.architectures import HARRVLSTMHybrid
from models.serialization import load_checkpoint

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Neural Volatility Surface Serving API",
    description="Low-latency REST endpoints for serving multi-step implied volatility surface forecasts.",
    version="1.0.0"
)

# Mount Gzip middleware for speed/compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount static files
app.mount("/static", StaticFiles(directory="inference/static"), name="static")

@app.get("/")
def read_root():
    """Serve the interactive dashboard frontend."""
    return FileResponse("inference/static/index.html")

# Global variables
config = {}
model = None
drift_monitor = None
baseline_avg_surface = None
history_buffer = None
prediction_store = None

# Pydantic Schemas for validation
class OptionContract(BaseModel):
    strike: float
    expiry: str = Field(..., description="String date in YYYY-MM-DD format")
    option_type: str = Field(..., description="Either 'call' or 'put'")
    bid: float
    ask: float
    lastPrice: Optional[float] = None
    volume: int = 0
    openInterest: int = 0

class SnapshotPayload(BaseModel):
    timestamp: str = Field(..., description="ISO timestamp of the snapshot collection")
    spot_price: float
    risk_free_rate: float
    dividend_yield: float
    contracts: List[OptionContract]

class PredictionResponse(BaseModel):
    forecast_horizons: List[int]
    predictions: dict  # maps horizon step to 7x7 grid list
    drift_status: dict
    plots: dict        # maps plot name to Plotly JSON strings

# Setup a Mock Model class for testing when no PyTorch checkpoint is available
class MockVolatilityModel:
    def __init__(self, forecast_horizons=[1, 5, 10]):
        self.forecast_horizons = forecast_horizons
    def __call__(self, x: torch.Tensor) -> dict:
        # x is (B, L, 1, 7, 7)
        # Squeeze the batch dimension (since B=1 for serving) to return grids of shape (7, 7)
        last_step = x[0, -1, 0, :, :].numpy() # shape (7, 7)
        forecasts = {}
        for h in self.forecast_horizons:
            # Add some slight decay and wave noise for mock predictions
            decay = 0.98 ** h
            noise = np.sin(np.arange(49).reshape(7, 7) * h) * 0.005
            forecasts[h] = last_step * decay + noise
        return forecasts

@app.on_event("startup")
def startup_event():
    global config, model, drift_monitor, baseline_avg_surface, history_buffer, prediction_store

    # 1. Load config
    config_path = "configs/base_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "ticker": "SPY",
            "data": {"lookback": 20, "raw_dir": "data/raw"},
            "serving": {"model_name": "volatility_surface_forecaster"},
            "monitoring": {"rolling_window_days": 10, "kl_threshold": 0.5, "rmse_threshold_factor": 1.5}
        }

    seed = config.get("training", {}).get("seed", 42)

    # 2. Instantiate drift monitor
    m_cfg = config.get("monitoring", {})
    drift_monitor = DriftMonitor(
        rolling_window_days=m_cfg.get("rolling_window_days", 10),
        kl_threshold=m_cfg.get("kl_threshold", 0.5),
        rmse_threshold_factor=m_cfg.get("rmse_threshold_factor", 1.5)
    )

    # 3. Real rolling history buffer (drives /predict) and matured-forecast store (drives drift RMSE)
    lookback = config.get("data", {}).get("lookback", 20)
    s_cfg = config.get("serving", {})
    history_buffer = SurfaceHistoryBuffer(
        path=s_cfg.get("history_buffer_path", "data/state/rolling_buffer.npz"),
        lookback=lookback
    )
    prediction_store = PredictionStore(path=s_cfg.get("prediction_store_path", "data/state/pending_predictions.npz"))

    # 4. Baseline surface for data-drift KL comparisons: use the real historical mean
    # surface when we have real collected data, otherwise a seeded (reproducible) synthetic one.
    raw_dir = config.get("data", {}).get("raw_dir", "data/raw")
    real_dataset = np.empty((0, 7, 7))
    if os.path.isdir(raw_dir) and len(os.listdir(raw_dir)) > 0:
        real_dataset, _ = build_surface_dataset(raw_dir)
    if len(real_dataset) > 0:
        baseline_avg_surface = np.mean(real_dataset, axis=0)
        logger.info(f"Using real historical mean surface ({len(real_dataset)} snapshots) as drift baseline.")
    else:
        synthetic_data, _ = generate_synthetic_dataset(num_days=30, seed=seed)
        baseline_avg_surface = np.mean(synthetic_data, axis=0)
        logger.info("No real historical data available; using seeded synthetic mean surface as drift baseline.")

    # 5. Load Active Model from the local checkpoint (state_dict + JSON sidecar).
    # MLflow (see training/train.py) is used for experiment tracking only — production
    # serving always loads from the local checkpoint, never a registry round-trip.
    model_loaded = False
    checkpoint_path = "models/checkpoint.pt"
    if os.path.exists(checkpoint_path):
        try:
            model = load_checkpoint(checkpoint_path, map_location=torch.device("cpu"))
            logger.info(f"Loaded active production model from local checkpoint: {checkpoint_path}")
            model_loaded = True
        except Exception as e:
            logger.error(f"Error loading model from {checkpoint_path}: {e}")

    if not model_loaded:
        logger.warning("No production model checkpoint found. Initializing Mock Forecasting Model.")
        model = MockVolatilityModel(forecast_horizons=config.get("data", {}).get("forecast_horizons", [1, 5, 10]))

@app.get("/health")
def health_check():
    """Report API status and load condition."""
    if model is None:
        model_type = "None"
    elif isinstance(model, MockVolatilityModel):
        model_type = "Mock"
    elif isinstance(model, HARRVLSTMHybrid):
        model_type = "HAR-RV + LSTM Hybrid"
    else:
        model_type = type(model).__name__
    return {
        "status": "healthy",
        "ticker": config.get("ticker", "SPY"),
        "model_loaded": model is not None,
        "model_type": model_type
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_surface(payload: SnapshotPayload, include_plots: bool = Query(True, description="Whether to generate and return Plotly figures")):
    """
    Accept raw option snapshot, execute engineering filters, interpolate,
    gather historical lookback sequence, and serve future forecasting grids.
    """
    global model, drift_monitor, baseline_avg_surface, history_buffer, prediction_store

    if not model:
        raise HTTPException(status_code=503, detail="Forecasting model is not initialized yet.")
        
    try:
        # 1. Parse incoming payload into DataFrame
        records = []
        for contract in payload.contracts:
            records.append({
                "timestamp": payload.timestamp,
                "ticker": config.get("ticker", "SPY"),
                "spot_price": payload.spot_price,
                "risk_free_rate": payload.risk_free_rate,
                "dividend_yield": payload.dividend_yield,
                "strike": contract.strike,
                "expiry": contract.expiry,
                "option_type": contract.option_type,
                "bid": contract.bid,
                "ask": contract.ask,
                "lastPrice": contract.lastPrice,
                "volume": contract.volume,
                "openInterest": contract.openInterest
            })
            
        df = pd.DataFrame(records)
        
        # 2. Preprocess snapshot & grid-interpolate
        # We disable volume filtering for inference requests to support illiquid queries
        processed_df = process_raw_snapshot(df, volume_filter=False)
        if processed_df.empty:
            raise HTTPException(status_code=400, detail="Payload has no valid options records after BSM filtering.")
            
        current_grid = interpolate_to_grid(
            processed_df["kappa"].values,
            processed_df["tau"].values,
            processed_df["impliedVolatility"].values
        ) # Shape (7, 7)
        
        # 3. Append the real processed grid to the rolling history buffer and fetch the
        # actual lookback window — no synthetic filler (see data/rolling_buffer.py).
        lookback = config.get("data", {}).get("lookback", 20)
        degraded_padding = config.get("serving", {}).get("degraded_mode_padding", False)

        history_buffer.append(current_grid, payload.timestamp)
        window, degraded = history_buffer.get_window(degraded_padding=degraded_padding)
        if window is None:
            have = history_buffer.count()
            raise HTTPException(
                status_code=425,
                detail=(
                    f"Insufficient real snapshot history: have {have}, need {lookback}. "
                    f"Submit {lookback - have} more /predict snapshot(s) before forecasting is available."
                )
            )

        # Normalize if model has train_mean and train_std attributes
        model_has_norm = not isinstance(model, MockVolatilityModel) and hasattr(model, "train_mean") and hasattr(model, "train_std")
        if model_has_norm:
            window_norm = (window - model.train_mean) / model.train_std
            input_tensor = torch.from_numpy(window_norm).float().unsqueeze(0).unsqueeze(2)
        else:
            input_tensor = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(2)
        
        # 4. Perform Inference
        # In real PyTorch model, we run forward pass
        if isinstance(model, MockVolatilityModel):
            forecasts = model(input_tensor)
        else:
            with torch.no_grad():
                # PyTorch model outputs prediction tensor
                # Let's assume output shape is (1, max_horizon, 7, 7)
                out_tensor = model(input_tensor)
                
                # Denormalize predicted outputs back to original scale if model was trained with normalization
                if hasattr(model, "train_mean") and hasattr(model, "train_std"):
                    out_tensor_denorm = out_tensor * model.train_std + model.train_mean
                else:
                    out_tensor_denorm = out_tensor
                
                # Convert back to numpy
                out_numpy = out_tensor_denorm.cpu().numpy()
                forecasts = {
                    1: out_numpy[0, 0],
                    5: out_numpy[0, 4] if out_numpy.shape[1] >= 5 else out_numpy[0, -1],
                    10: out_numpy[0, 9] if out_numpy.shape[1] >= 10 else out_numpy[0, -1]
                }
                
        # 5. Match this snapshot against any previously-made forecasts that targeted this
        # date to compute real performance drift, log data drift for the current snapshot,
        # then persist today's forecasts (keyed by target date) for future matching.
        observed_date = payload.timestamp[:10]
        matured = prediction_store.pop_matching(observed_date)
        for entry in matured:
            drift_monitor.log_performance(y_true=current_grid, y_pred=entry["grid"])

        drift_monitor.log_data_drift(current_surface=current_grid, baseline_surface=baseline_avg_surface)
        drift_status = drift_monitor.get_status()
        drift_status["degraded_mode"] = degraded

        snapshot_date = pd.Timestamp(observed_date)
        for h, f_grid in forecasts.items():
            # Approximate "h forecast steps" as "h business days ahead", matching the
            # collector's Mon-Fri daily cadence (see data/collector.py::start_scheduler).
            # A gap in daily collection will desynchronize step-count vs. calendar-day
            # matching — acceptable for this lightweight, no-extra-infra store.
            target_date = (snapshot_date + pd.tseries.offsets.BDay(int(h))).strftime("%Y-%m-%d")
            prediction_store.save_prediction(target_date, int(h), np.asarray(f_grid))
        
        # 6. Generate Plotly Visualizations (Layer 6 integration)
        plots = {}
        if include_plots:
            # 3D surface plot for horizon 1 prediction
            fig_3d = plot_3d_surface(forecasts[1], title=f"Forecasted Volatility Surface (1-Day Horizon)")
            plots["surface_3d"] = fig_to_json(fig_3d)
            
            # Heatmap comparison: current grid vs predicted horizon 1
            fig_heatmap = plot_residuals_heatmap(current_grid, forecasts[1], title="Expected Surface Evolution (Current - 1-Day Forecast)")
            plots["evolution_heatmap"] = fig_to_json(fig_heatmap)
            
            # Cross sections
            fig_cross = plot_cross_sections(current_grid, forecasts[1], title="Cross-Sectional Smile & Term Evolution")
            plots["cross_sections"] = fig_to_json(fig_cross)
            
        # Format return payload
        predictions_serialized = {str(h): f_grid.tolist() for h, f_grid in forecasts.items()}
        
        return PredictionResponse(
            forecast_horizons=config.get("data", {}).get("forecast_horizons", [1, 5, 10]),
            predictions=predictions_serialized,
            drift_status=drift_status,
            plots=plots
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Prediction processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/monitor/drift")
def get_drift_metrics():
    """Retrieve rolling drift parameters and alerts."""
    global drift_monitor
    if not drift_monitor:
        raise HTTPException(status_code=503, detail="Drift monitor is not initialized.")

    status = drift_monitor.get_status()
    status["retrain_recommended"] = status["trigger_retrain"]
    return status

from fastapi import BackgroundTasks

@app.post("/retrain")
def trigger_retraining(background_tasks: BackgroundTasks):
    """
    Trigger automated model retraining using accumulated options snapshot data.
    Runs in the background and reloads the model checkpoint upon completion.
    """
    global config, model
    
    def run_retrain_task():
        try:
            logger.info("Starting background retraining runner...")
            # Retrain using the active model name from config.
            # In a production context, this searches data/raw; here it defaults to synthetic if empty.
            from training.train import train_model
            model_name = config.get("model", {}).get("name", "hybrid")
            
            logger.info(f"Retraining model type: {model_name}")
            train_model(config, model_name=model_name, use_synthetic=True)
            
            # Reload the model weights
            checkpoint_path = "models/checkpoint.pt"
            if os.path.exists(checkpoint_path):
                global model
                model = load_checkpoint(checkpoint_path, map_location=torch.device("cpu"))
                logger.info("Successfully reloaded new production model checkpoint.")
        except Exception as e:
            logger.error(f"Error during background model retraining: {e}")
            
    background_tasks.add_task(run_retrain_task)
    return {"status": "retraining triggered", "detail": "Model retraining job running in the background."}

if __name__ == "__main__":
    uvicorn.run("inference.app:app", host="127.0.0.1", port=8000, reload=True)

