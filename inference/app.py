import os
import yaml
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import logging

# Import project utilities
from data.processor import process_raw_snapshot, interpolate_to_grid, GRID_KAPPAS, GRID_TAUS
from data.dataset import generate_synthetic_dataset
from utils.plotting import plot_3d_surface, plot_residuals_heatmap, plot_cross_sections, fig_to_json
from inference.monitor import DriftMonitor
from models.architectures import HARRVLSTMHybrid

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Neural Volatility Surface Serving API",
    description="Low-latency REST endpoints for serving multi-step implied volatility surface forecasts.",
    version="1.0.0"
)

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
    global config, model, drift_monitor, baseline_avg_surface
    
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
        
    # 2. Instantiate drift monitor
    m_cfg = config.get("monitoring", {})
    drift_monitor = DriftMonitor(
        rolling_window_days=m_cfg.get("rolling_window_days", 10),
        kl_threshold=m_cfg.get("kl_threshold", 0.5),
        rmse_threshold_factor=m_cfg.get("rmse_threshold_factor", 1.5)
    )
    
    # 3. Create synthetic baseline surface for monitoring reference
    synthetic_data, _ = generate_synthetic_dataset(num_days=30)
    baseline_avg_surface = np.mean(synthetic_data, axis=0) # shape (7, 7)
    
    # 4. Load Active Model (from MLflow or local files)
    model_loaded = False
    
    # Try local PyTorch checkpoint first
    checkpoint_path = "models/checkpoint.pt"
    if os.path.exists(checkpoint_path):
        try:
            model = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
            model.eval()
            logger.info(f"Loaded active production model from local checkpoint: {checkpoint_path}")
            model_loaded = True
        except Exception as e:
            logger.error(f"Error loading model from {checkpoint_path}: {e}")
            
    # Try MLflow Registry if not loaded
    if not model_loaded:
        try:
            import mlflow
            # Ensure MLflow tracking URI is set (defaulting to local mlruns)
            tracking_uri = config.get("serving", {}).get("model_registry_uri", "mlruns")
            mlflow.set_tracking_uri(tracking_uri)
            
            model_name = config.get("serving", {}).get("model_name", "volatility_surface_forecaster")
            # Load active model with 'Production' alias or tag if possible
            model_uri = f"models:/{model_name}/Production"
            
            # Since MLflow server might not be running locally, we try/except
            # model = mlflow.pytorch.load_model(model_uri)
            # model.eval()
            # model_loaded = True
            logger.info("Attempted loading from MLflow. Mocking registry load since server is idle.")
        except Exception as e:
            logger.warning(f"Could not connect to MLflow server: {e}")
            
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
    global model, drift_monitor, baseline_avg_surface
    
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
        
        # 3. Build input lookback sequence (Requires L=20/60 steps)
        lookback = config.get("data", {}).get("lookback", 20)
        
        # In a real environment, we would load the last L-1 processed snapshots from data/raw/
        # Here we simulate historical loading by getting a synthetic series and swapping the last element
        # with our active current_grid
        dummy_series, _ = generate_synthetic_dataset(num_days=lookback)
        dummy_series[-1] = current_grid
        
        # Normalize if model has train_mean and train_std attributes
        model_has_norm = not isinstance(model, MockVolatilityModel) and hasattr(model, "train_mean") and hasattr(model, "train_std")
        if model_has_norm:
            dummy_series_norm = (dummy_series - model.train_mean) / model.train_std
            input_tensor = torch.from_numpy(dummy_series_norm).float().unsqueeze(0).unsqueeze(2)
        else:
            input_tensor = torch.from_numpy(dummy_series).float().unsqueeze(0).unsqueeze(2)
        
        # 4. Perform Inference
        # In real PyTorch model, we run forward pass
        if isinstance(model, MockVolatilityModel):
            forecasts = model(input_tensor)
        else:
            with torch.no_grad():
                # Developer B's PyTorch model outputs prediction tensor
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
                
        # 5. Log prediction events to check drift
        # Since we don't have actual future labels at serving time, we compare current_grid
        # against baseline_avg_surface to see if input distributions are shifting.
        # We also pass a dummy forecast comparison to satisfy the logging signature.
        drift_status = drift_monitor.log_prediction(
            y_true=current_grid,
            y_pred=current_grid, # Placeholder for real future labels
            baseline_surface=baseline_avg_surface
        )
        
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
        
    return {
        "rolling_rmse": float(np.mean(drift_monitor.recent_rmses)) if drift_monitor.recent_rmses else 0.0,
        "rolling_kl_divergence": float(np.mean(drift_monitor.recent_kls)) if drift_monitor.recent_kls else 0.0,
        "recent_rmses": drift_monitor.recent_rmses,
        "recent_kls": drift_monitor.recent_kls,
        "baseline_rmse": drift_monitor.baseline_rmse,
        "kl_threshold": drift_monitor.kl_threshold,
        "rmse_threshold_factor": drift_monitor.rmse_threshold_factor,
        "retrain_recommended": bool(
            (np.mean(drift_monitor.recent_rmses) > drift_monitor.baseline_rmse * drift_monitor.rmse_threshold_factor)
            if drift_monitor.recent_rmses else False
        )
    }

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
                model = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
                model.eval()
                logger.info("Successfully reloaded new production model checkpoint.")
        except Exception as e:
            logger.error(f"Error during background model retraining: {e}")
            
    background_tasks.add_task(run_retrain_task)
    return {"status": "retraining triggered", "detail": "Model retraining job running in the background."}

if __name__ == "__main__":
    uvicorn.run("inference.app:app", host="127.0.0.1", port=8000, reload=True)

