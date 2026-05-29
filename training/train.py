import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import mlflow
import mlflow.pytorch
import logging
from datetime import datetime


# Import project classes
from data.dataset import build_surface_dataset, create_sequences, generate_synthetic_dataset
from data.processor import GRID_KAPPAS, GRID_TAUS
from models.architectures import StackedLSTM, ConvLSTM, TransformerEncoderModel
from models.loss import SmoothnessRegularizedLoss
from models.baselines import NaiveRandomWalk, HistoricalMean, ExponentialSmoothing, GARCHModel, HARRVModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class SurfaceDataset(Dataset):
    """
    Custom PyTorch Dataset for volatility surfaces sequences.
    """
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_region_masks(kappas, taus):
    """
    Create boolean masks for different regions on the 7x7 volatility surface.
    """
    grid_k, grid_t = np.meshgrid(kappas, taus)
    
    masks = {
        "Overall": np.ones_like(grid_k, dtype=bool),
        "ATM": np.abs(grid_k) < 0.05,
        "OTM Puts": grid_k <= -0.10,
        "OTM Calls": grid_k >= 0.10,
        "Deep Wings": np.abs(grid_k) >= 0.20,
        "Short-dated": grid_t < 1/12,
        "Long-dated": grid_t > 6/12
    }
    return masks


def calculate_metrics(y_pred, y_true):
    """
    Calculate RMSE, MAE, MAPE, and Directional Accuracy.
    Args:
        y_pred: numpy array of shape (N, h, M, N)
        y_true: numpy array of shape (N, h, M, N)
    """
    # 1. RMSE
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    # 2. MAE
    mae = np.mean(np.abs(y_pred - y_true))
    # 3. MAPE
    mape = np.mean(np.abs(y_pred - y_true) / np.clip(y_true, 1e-5, None)) * 100.0
    
    # 4. Total Variation (TV) - measure of surface roughness
    # TV along strike (dim -2) + TV along expiry (dim -1)
    tv_strike = np.mean(np.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]))
    tv_expiry = np.mean(np.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]))
    tv = tv_strike + tv_expiry

    # 5. Arbitrage Violation count (AV)
    # Check calendar spread violations: W_j > W_{j+1} where W = IV^2 * tau
    # grid_taus shape: (7,)
    w_pred = (y_pred ** 2) * GRID_TAUS.reshape(1, 1, 1, -1)
    av_violations = np.sum(w_pred[:, :, :, :-1] > w_pred[:, :, :, 1:])
    av_ratio = av_violations / (y_pred.shape[0] * y_pred.shape[1] * y_pred.shape[2] * (y_pred.shape[3] - 1))
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "TV": tv,
        "AV": av_ratio
    }


def evaluate_regions(y_pred, y_true, masks):
    """
    Decompose RMSE across surface regions.
    """
    results = {}
    for region_name, mask in masks.items():
        # y_pred, y_true are shape (N, h, M, N).
        # We index along the last two dimensions (M, N) using the region mask.
        y_pred_reg = y_pred[:, :, mask]
        y_true_reg = y_true[:, :, mask]
        if y_pred_reg.size > 0:
            reg_rmse = np.sqrt(np.mean((y_pred_reg - y_true_reg) ** 2))
            results[region_name] = reg_rmse
        else:
            results[region_name] = np.nan
    return results


def train_model(config, model_name="convlstm", use_synthetic=False):
    """
    Core function to train PyTorch models and log to MLflow.
    """
    # 1. Load configuration parameters
    lookback = config["data"]["lookback"]
    forecast_horizons = config["data"]["forecast_horizons"]
    horizon = max(forecast_horizons)
    
    # 2. Prepare Data
    raw_dir = config["data"]["raw_dir"]
    if not os.path.exists(raw_dir) or len(os.listdir(raw_dir)) == 0 or use_synthetic:
        logger.info("Raw data directory empty or synthetic flag enabled. Generating synthetic dataset.")
        dataset, timestamps = generate_synthetic_dataset(num_days=300)
    else:
        logger.info(f"Loading data from raw directory: {raw_dir}")
        dataset, timestamps = build_surface_dataset(raw_dir, volume_filter=True)
        if len(dataset) < (lookback + horizon):
            logger.warning("Insufficient actual collected data. Generating synthetic dataset instead.")
            dataset, timestamps = generate_synthetic_dataset(num_days=300)
            
    # Chronological Split
    T = len(dataset)
    train_end = int(T * config["data"]["train_ratio"])
    val_end = int(T * (config["data"]["train_ratio"] + config["data"]["val_ratio"]))
    
    train_data = dataset[:train_end]
    val_data = dataset[train_end:val_end]
    test_data = dataset[val_end:]
    
    logger.info(f"Dataset Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Create Sequences
    X_train, y_train = create_sequences(train_data, lookback, horizon)
    X_val, y_val = create_sequences(val_data, lookback, horizon)
    X_test, y_test = create_sequences(test_data, lookback, horizon)
    
    # PyTorch DataLoaders
    train_dataset = SurfaceDataset(X_train, y_train)
    val_dataset = SurfaceDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    
    # 3. Initialize Model, Loss, Optimizer, and Scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using training device: {device}")
    
    if model_name == "convlstm":
        model = ConvLSTM(
            in_channels=1,
            hidden_dims=config["model"]["convlstm"]["hidden_dim"],
            kernel_size=config["model"]["convlstm"]["kernel_size"][0],
            num_layers=config["model"]["convlstm"]["num_layers"],
            horizon=horizon
        )
    elif model_name == "lstm":
        model = StackedLSTM(
            grid_size=(7, 7),
            hidden_dim=config["model"]["lstm"]["hidden_dim"],
            num_layers=config["model"]["lstm"]["num_layers"],
            horizon=horizon
        )
    elif model_name == "transformer":
        model = TransformerEncoderModel(
            grid_size=(7, 7),
            horizon=horizon
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    model = model.to(device)
    
    criterion = SmoothnessRegularizedLoss(
        lambda_strike=config["training"]["loss"]["lambda_strike"],
        lambda_expiry=config["training"]["loss"]["lambda_expiry"],
        lambda_calendar=config["training"]["loss"].get("lambda_calendar", 0.0),
        lambda_butterfly=config["training"]["loss"].get("lambda_butterfly", 0.0),
        grid_taus=GRID_TAUS
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    
    # MLflow Setup
    mlflow.set_tracking_uri(config["serving"]["model_registry_uri"])
    mlflow.set_experiment(f"implied_volatility_forecasting_{model_name}")
    
    best_val_loss = float("inf")
    patience = config["training"]["early_stopping_patience"]
    patience_counter = 0
    
    os.makedirs("models", exist_ok=True)
    checkpoint_path = "models/checkpoint.pt"
    
    with mlflow.start_run() as run:
        mlflow.log_params({
            "model_type": model_name,
            "lookback": lookback,
            "horizon": horizon,
            "epochs": config["training"]["epochs"],
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "lambda_strike": config["training"]["loss"]["lambda_strike"],
            "lambda_expiry": config["training"]["loss"]["lambda_expiry"],
        })
        
        for epoch in range(1, config["training"]["epochs"] + 1):
            # Training phase
            model.train()
            train_loss_accum = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                y_pred = model(X_batch)
                
                loss, metrics = criterion(y_pred, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
                optimizer.step()
                
                train_loss_accum += loss.item() * X_batch.size(0)
                
            train_loss = train_loss_accum / len(train_loader.dataset)
            scheduler.step()
            
            # Validation phase
            model.eval()
            val_loss_accum = 0.0
            val_metrics_accum = {"mse_loss": 0.0, "strike_loss": 0.0, "expiry_loss": 0.0, "calendar_loss": 0.0, "butterfly_loss": 0.0}
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = model(X_batch)
                    
                    loss, metrics = criterion(y_pred, y_batch)
                    val_loss_accum += loss.item() * X_batch.size(0)
                    for k in val_metrics_accum:
                        val_metrics_accum[k] += metrics[k] * X_batch.size(0)
                        
            val_loss = val_loss_accum / len(val_loader.dataset)
            for k in val_metrics_accum:
                val_metrics_accum[k] /= len(val_loader.dataset)
                
            # Log metrics per epoch
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mse": val_metrics_accum["mse_loss"],
                "val_strike_smoothness": val_metrics_accum["strike_loss"],
                "val_expiry_smoothness": val_metrics_accum["expiry_loss"]
            }, step=epoch)
            
            logger.info(f"Epoch {epoch:02d} | Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f}")
            
            # Early stopping & Checkpoint saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best checkpoint
                torch.save(model, checkpoint_path)
                logger.info(f"--> Saved champion model checkpoint with Val Loss={best_val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break
                    
        # Load best model for final test set evaluation
        logger.info("Evaluating champion model on test set...")
        best_model = torch.load(checkpoint_path, weights_only=False)
        best_model.eval()
        
        # Run test set predictions
        with torch.no_grad():
            test_X_tensor = torch.from_numpy(X_test).float().to(device)
            test_y_pred = best_model(test_X_tensor).cpu().numpy()
            
        # Log Pytorch model checkpoint to MLflow
        mlflow.pytorch.log_model(best_model, "volatility_surface_forecaster")
        
    return test_y_pred, y_test


def evaluate_all(config, use_synthetic=False):
    """
    Fits and evaluates all baselines and PyTorch models, reporting a region-decomposed comparison table.
    """
    lookback = config["data"]["lookback"]
    forecast_horizons = config["data"]["forecast_horizons"]
    horizon = max(forecast_horizons)
    
    # Load Data
    raw_dir = config["data"]["raw_dir"]
    if not os.path.exists(raw_dir) or len(os.listdir(raw_dir)) == 0 or use_synthetic:
        dataset, _ = generate_synthetic_dataset(num_days=300)
    else:
        dataset, _ = build_surface_dataset(raw_dir, volume_filter=True)
        if len(dataset) < (lookback + horizon):
            dataset, _ = generate_synthetic_dataset(num_days=300)
            
    # Splits
    T = len(dataset)
    train_end = int(T * config["data"]["train_ratio"])
    val_end = int(T * (config["data"]["train_ratio"] + config["data"]["val_ratio"]))
    
    train_data = dataset[:train_end]
    val_data = dataset[train_end:val_end]
    test_data = dataset[val_end:]
    
    # Sequences
    X_train, y_train = create_sequences(train_data, lookback, horizon)
    X_test, y_test = create_sequences(test_data, lookback, horizon)
    
    # Create evaluation region masks
    masks = get_region_masks(GRID_KAPPAS, GRID_TAUS)
    
    # Dictionary to collect results
    all_results = {}
    
    # 1. Naive Random Walk
    rw = NaiveRandomWalk(horizon=horizon)
    rw_pred = rw.predict(X_test)
    all_results["Random Walk"] = (rw_pred, calculate_metrics(rw_pred, y_test), evaluate_regions(rw_pred, y_test, masks))
    
    # 2. Historical Mean
    hm = HistoricalMean(horizon=horizon)
    hm_pred = hm.predict(X_test)
    all_results["Historical Mean"] = (hm_pred, calculate_metrics(hm_pred, y_test), evaluate_regions(hm_pred, y_test, masks))
    
    # 3. Exponential Smoothing
    es = ExponentialSmoothing(horizon=horizon)
    es_pred = es.predict(X_test)
    all_results["Exp. Smoothing"] = (es_pred, calculate_metrics(es_pred, y_test), evaluate_regions(es_pred, y_test, masks))
    
    # 4. GARCH(1,1)
    garch = GARCHModel(horizon=horizon)
    garch_pred = garch.predict(X_test)
    all_results["GARCH(1,1)"] = (garch_pred, calculate_metrics(garch_pred, y_test), evaluate_regions(garch_pred, y_test, masks))
    
    # 5. HAR-RV
    har = HARRVModel(horizon=horizon)
    har.fit(X_train, y_train)
    har_pred = har.predict(X_test)
    all_results["HAR-RV"] = (har_pred, calculate_metrics(har_pred, y_test), evaluate_regions(har_pred, y_test, masks))
    
    # 6. LSTM Model (train and predict)
    logger.info("---------- Training Stacked LSTM Baseline ----------")
    lstm_pred, _ = train_model(config, model_name="lstm", use_synthetic=use_synthetic)
    all_results["LSTM"] = (lstm_pred, calculate_metrics(lstm_pred, y_test), evaluate_regions(lstm_pred, y_test, masks))
    
    # 7. ConvLSTM Model (train and predict)
    logger.info("---------- Training ConvLSTM Model ----------")
    convlstm_pred, _ = train_model(config, model_name="convlstm", use_synthetic=use_synthetic)
    all_results["ConvLSTM (Smooth)"] = (convlstm_pred, calculate_metrics(convlstm_pred, y_test), evaluate_regions(convlstm_pred, y_test, masks))

    # Print results summary table
    logger.info("========================================= FINAL RESULTS COMPARISON =========================================")
    print(f"{'Model':<20} | {'RMSE':<8} | {'MAE':<8} | {'TV (Rough)':<10} | {'AV (Arbitrage)':<14} | {'ATM RMSE':<9} | {'OTM-P RMSE':<10} | {'Long-dated RMSE':<15}")
    print("-" * 115)
    for model_name, (preds, metrics, regions) in all_results.items():
        av_val = metrics.get('AV', 0.0)
        av_str = f"{av_val:.5%}" if av_val is not None else "0.00000%"
        print(f"{model_name:<20} | {metrics['RMSE']:.5f} | {metrics['MAE']:.5f} | {metrics['TV']:.5f}    | {av_str:<14} | {regions['ATM']:.5f}  | {regions['OTM Puts']:.5f}   | {regions['Long-dated']:.5f}")
    logger.info("==========================================================================================================")
    
    return all_results


if __name__ == "__main__":
    # Load configs
    with open("configs/base_config.yaml", "r") as f:
        config_data = yaml.safe_load(f)
        
    evaluate_all(config_data, use_synthetic=True)
