import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
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
from models.architectures import StackedLSTM, ConvLSTM, TransformerEncoderModel, HARRVLSTMHybrid
from models.loss import SmoothnessRegularizedLoss
from models.baselines import NaiveRandomWalk, HistoricalMean, ExponentialSmoothing, GARCHModel, HARRVModel
from models.serialization import save_checkpoint, load_checkpoint


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch RNGs so training and synthetic data generation are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    # TV along strike (dim -1, kappas) + TV along expiry (dim -2, taus)
    tv_strike = np.mean(np.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]))
    tv_expiry = np.mean(np.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]))
    tv = tv_strike + tv_expiry

    # 5. Arbitrage Violation count (AV)
    # Check calendar spread violations: W_j > W_{j+1} where W = IV^2 * tau along dim=-2 (taus)
    # grid_taus shape: (7,) matches dim=-2 (taus)
    w_pred = (y_pred ** 2) * GRID_TAUS.reshape(1, 1, -1, 1)
    av_violations = np.sum(w_pred[:, :, :-1, :] > w_pred[:, :, 1:, :])
    av_ratio = av_violations / (y_pred.shape[0] * y_pred.shape[1] * (y_pred.shape[2] - 1) * y_pred.shape[3])
    
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


def train_model(config, model_name="convlstm", use_synthetic=False, dataset=None):
    """
    Core function to train PyTorch models and log to MLflow.
    """
    # 1. Load configuration parameters
    lookback = config["data"]["lookback"]
    forecast_horizons = config["data"]["forecast_horizons"]
    horizon = max(forecast_horizons)

    # Seed all RNGs up front so this run (data generation, model init, training) is reproducible.
    seed = config.get("training", {}).get("seed", 42)
    seed_everything(seed)

    # 2. Prepare Data
    # Preprocessing metric counters — populated when loading real market data
    preproc_metrics = {
        "raw_contracts_count": 0,
        "filtered_contracts_count": 0,
        "arbitrage_violations_count": 0,
        "surface_coverage_pct": 100.0,  # synthetic always gives 100%
    }

    if dataset is None:
        raw_dir = config["data"]["raw_dir"]
        if not os.path.exists(raw_dir) or len(os.listdir(raw_dir)) == 0 or use_synthetic:
            logger.info("Raw data directory empty or synthetic flag enabled. Generating synthetic dataset.")
            dataset, timestamps = generate_synthetic_dataset(num_days=3000, seed=seed)
        else:
            logger.info(f"Loading data from raw directory: {raw_dir}")

            # Collect preprocessing stats across all Parquet snapshots
            import glob
            import pandas as pd
            from data.processor import process_raw_snapshot
            snapshot_files = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
            total_raw, total_filtered, total_av = 0, 0, 0
            for fp in snapshot_files:
                try:
                    raw_df = pd.read_parquet(fp)
                    total_raw += len(raw_df)
                    proc_df = process_raw_snapshot(raw_df, volume_filter=True)
                    total_filtered += len(proc_df)
                    # Count calendar arbitrage violations before interpolation
                    if not proc_df.empty and "total_variance" in proc_df.columns:
                        proc_df = proc_df.sort_values(["kappa", "tau"])
                        av = int((proc_df.groupby("kappa")["total_variance"]
                                  .apply(lambda s: (s.diff().dropna() < 0).sum())
                                  .sum()))
                        total_av += av
                except Exception:
                    pass
            grid_cells = len(GRID_KAPPAS) * len(GRID_TAUS)  # 49
            preproc_metrics = {
                "raw_contracts_count": total_raw,
                "filtered_contracts_count": total_filtered,
                "arbitrage_violations_count": total_av,
                "surface_coverage_pct": min(100.0, (total_filtered / max(total_raw, 1)) * 100.0),
            }
            logger.info(
                f"Preprocessing stats — raw: {total_raw}, filtered: {total_filtered}, "
                f"AV violations: {total_av}, coverage: {preproc_metrics['surface_coverage_pct']:.1f}%"
            )

            dataset, timestamps = build_surface_dataset(raw_dir, volume_filter=True)
            if len(dataset) < (lookback + horizon):
                logger.warning("Insufficient actual collected data. Generating synthetic dataset instead.")
                dataset, timestamps = generate_synthetic_dataset(num_days=3000, seed=seed)
            
    # Chronological Split
    T = len(dataset)
    train_end = int(T * config["data"]["train_ratio"])
    val_end = int(T * (config["data"]["train_ratio"] + config["data"]["val_ratio"]))
    
    train_data = dataset[:train_end]
    val_data = dataset[train_end:val_end]
    test_data = dataset[val_end:]
    
    # Calculate normalization parameters on training set
    train_mean = float(np.mean(train_data))
    train_std = float(np.std(train_data)) + 1e-8
    logger.info(f"Dataset normalization parameters: Mean={train_mean:.6f}, Std={train_std:.6f}")
    
    # Normalize datasets
    train_data_norm = (train_data - train_mean) / train_std
    val_data_norm = (val_data - train_mean) / train_std
    test_data_norm = (test_data - train_mean) / train_std
    
    # Create Sequences
    X_train, y_train = create_sequences(train_data_norm, lookback, horizon)
    X_val, y_val = create_sequences(val_data_norm, lookback, horizon)
    X_test, y_test_norm = create_sequences(test_data_norm, lookback, horizon)
    
    # Also generate unnormalized true targets for test evaluation
    _, y_test = create_sequences(test_data, lookback, horizon)
    
    # PyTorch DataLoaders
    train_dataset = SurfaceDataset(X_train, y_train)
    val_dataset = SurfaceDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    
    # 3. Initialize Model, Loss, Optimizer, and Scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using training device: {device}")
    
    if model_name == "convlstm":
        model_kwargs = dict(
            in_channels=1,
            hidden_dims=config["model"]["convlstm"]["hidden_dim"],
            kernel_size=config["model"]["convlstm"]["kernel_size"][0],
            num_layers=config["model"]["convlstm"]["num_layers"],
            horizon=horizon
        )
        model = ConvLSTM(**model_kwargs)
    elif model_name == "lstm":
        model_kwargs = dict(
            grid_size=(7, 7),
            hidden_dim=config["model"]["lstm"]["hidden_dim"],
            num_layers=config["model"]["lstm"]["num_layers"],
            horizon=horizon
        )
        model = StackedLSTM(**model_kwargs)
    elif model_name == "transformer":
        model_kwargs = dict(
            grid_size=(7, 7),
            horizon=horizon
        )
        model = TransformerEncoderModel(**model_kwargs)
    elif model_name == "hybrid":
        model_kwargs = dict(
            grid_size=(7, 7),
            hidden_dim=config["model"]["lstm"]["hidden_dim"],
            num_layers=config["model"]["lstm"]["num_layers"],
            horizon=horizon
        )
        model = HARRVLSTMHybrid(**model_kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Data Contract §6 — Checkpoint metadata for reproducibility and serving safety.
    # Set now (before training starts) so every checkpoint saved during training —
    # not just a final reloaded/re-saved copy — already carries full sidecar metadata.
    model.data_contract_version = "v1.0"
    model.lookback_window = lookback
    model.horizons = forecast_horizons
    model.tensor_orientation = "(B, L, C, E, M)"
    model.train_mean = train_mean
    model.train_std = train_std

    def _checkpoint_meta():
        return {
            "data_contract_version": model.data_contract_version,
            "lookback_window": model.lookback_window,
            "horizons": model.horizons,
            "tensor_orientation": model.tensor_orientation,
            "train_mean": model.train_mean,
            "train_std": model.train_std,
        }

    # ---- Two-step HAR-RV fit for hybrid model (must happen before moving to GPU) ----
    if model_name == "hybrid":
        logger.info("Fitting HAR-RV coefficients on unnormalized training sequences...")
        # We need unnormalized training sequences for HAR-RV fitting
        X_train_raw, y_train_raw = create_sequences(train_data, lookback, horizon)
        model.fit_har(X_train_raw, y_train_raw)
        logger.info("HAR-RV fit complete. Training LSTM residual sub-network on normalized data.")

    model = model.to(device)
    
    criterion = SmoothnessRegularizedLoss(
        lambda_strike=config["training"]["loss"]["lambda_strike"],
        lambda_expiry=config["training"]["loss"]["lambda_expiry"],
        lambda_calendar=config["training"]["loss"].get("lambda_calendar", 0.0),
        lambda_butterfly=config["training"]["loss"].get("lambda_butterfly", 0.0),
        lambda_residual_mean=config["training"]["loss"].get("lambda_residual_mean", 0.0),
        grid_taus=GRID_TAUS
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    
    # MLflow Setup
    mlflow.set_tracking_uri(config["serving"]["model_registry_uri"])
    mlflow.set_experiment(f"implied_volatility_forecasting_{model_name}")
    
    best_val_loss = float("inf")
    patience = config["training"]["early_stopping_patience"]
    patience_counter = 0
    
    os.makedirs("models", exist_ok=True)
    checkpoint_path = f"models/checkpoint_{model_name}.pt"
    
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
            "data_contract_version": "v1.0",
            "tensor_orientation": "(B, L, C, E, M)",
        })
        # Log Data Contract §7 preprocessing metrics (real data) or mark as synthetic
        mlflow.log_metrics({
            "preproc_raw_contracts": float(preproc_metrics["raw_contracts_count"]),
            "preproc_filtered_contracts": float(preproc_metrics["filtered_contracts_count"]),
            "preproc_arbitrage_violations": float(preproc_metrics["arbitrage_violations_count"]),
            "preproc_surface_coverage_pct": float(preproc_metrics["surface_coverage_pct"]),
            # Exact metric names requested for Data Contract compliance
            "raw_contracts_count": float(preproc_metrics["raw_contracts_count"]),
            "filtered_contracts_count": float(preproc_metrics["filtered_contracts_count"]),
            "arbitrage_violations_count": float(preproc_metrics["arbitrage_violations_count"]),
            "surface_coverage_pct": float(preproc_metrics["surface_coverage_pct"]),
        })
        
        for epoch in range(1, config["training"]["epochs"] + 1):
            # Training phase
            model.train()
            train_loss_accum = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                y_pred = model(X_batch)
                
                # Extract residual from hybrid model for zero-mean penalty
                residual = getattr(model, "last_residual", None)
                loss, metrics = criterion(y_pred, y_batch, residual=residual)
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
            val_metrics_accum = {"mse_loss": 0.0, "strike_loss": 0.0, "expiry_loss": 0.0, "calendar_loss": 0.0, "butterfly_loss": 0.0, "residual_mean_loss": 0.0}
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = model(X_batch)
                    
                    residual = getattr(model, "last_residual", None)
                    loss, metrics = criterion(y_pred, y_batch, residual=residual)
                    val_loss_accum += loss.item() * X_batch.size(0)
                    for k in val_metrics_accum:
                        val_metrics_accum[k] += metrics[k] * X_batch.size(0)
                        
            val_loss = val_loss_accum / len(val_loader.dataset)
            for k in val_metrics_accum:
                val_metrics_accum[k] /= len(val_loader.dataset)
                
            # Log metrics per epoch
            epoch_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mse": val_metrics_accum["mse_loss"],
                "val_strike_smoothness": val_metrics_accum["strike_loss"],
                "val_expiry_smoothness": val_metrics_accum["expiry_loss"]
            }
            # Log gate value for hybrid model
            if hasattr(model, "gate_value"):
                epoch_metrics["gate_value"] = model.gate_value
            mlflow.log_metrics(epoch_metrics, step=epoch)
            
            gate_str = f" | Gate={model.gate_value:.4f}" if hasattr(model, "gate_value") else ""
            logger.info(f"Epoch {epoch:02d} | Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f}{gate_str}")
            
            # Early stopping & Checkpoint saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best checkpoint (state_dict + JSON sidecar — see models/serialization.py)
                save_checkpoint(model, checkpoint_path, type(model).__name__, model_kwargs, extra_meta=_checkpoint_meta())
                logger.info(f"--> Saved champion model checkpoint with Val Loss={best_val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break
                    
        # Load best model for final test set evaluation
        logger.info("Evaluating champion model on test set...")
        best_model = load_checkpoint(checkpoint_path, map_location=device)
        logger.info(
            f"Checkpoint loaded with contract metadata: "
            f"version={best_model.data_contract_version}, "
            f"lookback={best_model.lookback_window}, "
            f"horizons={best_model.horizons}, "
            f"orientation={best_model.tensor_orientation}"
        )

        # Save as active served model if it matches config
        active_model_name = config.get("model", {}).get("name", "hybrid")
        if model_name == active_model_name:
            import shutil
            shutil.copy(checkpoint_path, "models/checkpoint.pt")
            shutil.copy(checkpoint_path + ".json", "models/checkpoint.pt.json")
            logger.info(f"Copied {checkpoint_path} (+ sidecar) to models/checkpoint.pt as active served model.")
            
        best_model.eval()
        
        # Run test set predictions (X_test is normalized)
        with torch.no_grad():
            test_X_tensor = torch.from_numpy(X_test).float().to(device)
            test_y_pred_norm = best_model(test_X_tensor).cpu().numpy()
            
        # Denormalize predictions back to original scale
        test_y_pred = test_y_pred_norm * train_std + train_mean
        
        # Persist raw predictions to disk for significance testing (no retraining needed)
        os.makedirs("models/predictions", exist_ok=True)
        np.save(f"models/predictions/{model_name}_pred.npy", test_y_pred)
        np.save(f"models/predictions/{model_name}_true.npy", y_test)
        logger.info(f"Saved test predictions to models/predictions/{model_name}_pred.npy")
        
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

    seed = config.get("training", {}).get("seed", 42)
    seed_everything(seed)

    # Load Data
    raw_dir = config["data"]["raw_dir"]
    if not os.path.exists(raw_dir) or len(os.listdir(raw_dir)) == 0 or use_synthetic:
        dataset, _ = generate_synthetic_dataset(num_days=3000, seed=seed)
    else:
        dataset, _ = build_surface_dataset(raw_dir, volume_filter=True)
        if len(dataset) < (lookback + horizon):
            dataset, _ = generate_synthetic_dataset(num_days=3000, seed=seed)
            
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
    logger.info("---------- Training Stacked LSTM ----------")
    lstm_pred, _ = train_model(config, model_name="lstm", use_synthetic=use_synthetic, dataset=dataset)
    all_results["LSTM"] = (lstm_pred, calculate_metrics(lstm_pred, y_test), evaluate_regions(lstm_pred, y_test, masks))
    
    # 7. Transformer Model (train and predict)
    logger.info("---------- Training Transformer Encoder ----------")
    transformer_pred, _ = train_model(config, model_name="transformer", use_synthetic=use_synthetic, dataset=dataset)
    all_results["Transformer"] = (transformer_pred, calculate_metrics(transformer_pred, y_test), evaluate_regions(transformer_pred, y_test, masks))

    # 8. HAR-RV + LSTM Hybrid (train and predict)
    logger.info("---------- Training HAR-RV + LSTM Hybrid ----------")
    hybrid_pred, _ = train_model(config, model_name="hybrid", use_synthetic=use_synthetic, dataset=dataset)
    all_results["Hybrid (HAR+LSTM)"] = (hybrid_pred, calculate_metrics(hybrid_pred, y_test), evaluate_regions(hybrid_pred, y_test, masks))

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
