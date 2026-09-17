import sys
import os
import random
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import generate_synthetic_dataset, create_sequences, build_surface_dataset
from data.processor import GRID_KAPPAS, GRID_TAUS
from models.serialization import load_checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def export_plots(use_synthetic=True):
    # 1. Load config
    config_path = "configs/base_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "data": {
                "lookback": 20,
                "forecast_horizons": [1, 5, 10],
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
                "raw_dir": "data/raw"
            }
        }

    lookback = config["data"]["lookback"]
    forecast_horizons = config["data"]["forecast_horizons"]
    horizon = max(forecast_horizons)

    seed = config.get("training", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    # 2. Get data
    raw_dir = config["data"]["raw_dir"]
    if not os.path.exists(raw_dir) or len(os.listdir(raw_dir)) == 0 or use_synthetic:
        logger.info("Using synthetic dataset for plotting.")
        dataset, _ = generate_synthetic_dataset(num_days=3000, seed=seed)
    else:
        logger.info(f"Loading raw options data from {raw_dir}...")
        dataset, _ = build_surface_dataset(raw_dir, volume_filter=True)
        if len(dataset) < (lookback + horizon):
            dataset, _ = generate_synthetic_dataset(num_days=3000, seed=seed)

    # Split and sequences
    T = len(dataset)
    val_end = int(T * (config["data"]["train_ratio"] + config["data"]["val_ratio"]))
    test_data = dataset[val_end:]
    X_test, y_test = create_sequences(test_data, lookback, horizon)

    # 3. Load PyTorch model checkpoint
    checkpoint_path = "models/checkpoint.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dl_pred = None
    if os.path.exists(checkpoint_path):
        try:
            logger.info(f"Loading trained PyTorch model from {checkpoint_path}...")
            model = load_checkpoint(checkpoint_path, map_location=device)
            with torch.no_grad():
                if hasattr(model, "train_mean") and hasattr(model, "train_std"):
                    logger.info(f"Applying normalization using checkpoint parameters: mean={model.train_mean:.6f}, std={model.train_std:.6f}")
                    X_test_norm = (X_test[:10] - model.train_mean) / model.train_std
                    test_X_tensor = torch.from_numpy(X_test_norm).float().to(device)
                    pred_norm = model(test_X_tensor).cpu().numpy()
                    dl_pred = pred_norm * model.train_std + model.train_mean
                else:
                    test_X_tensor = torch.from_numpy(X_test[:10]).float().to(device)
                    dl_pred = model(test_X_tensor).cpu().numpy()
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            
    if dl_pred is None:
        logger.warning("Using mock predictions for plotting.")
        dl_pred = y_test[:10] + np.random.normal(0, 0.015, y_test[:10].shape)

    # We take the first sample in test set for plotting
    actual_surf = y_test[0, 0]          # Actual surface at horizon 1, shape (7, 7)
    predicted_surf = dl_pred[0, 0]      # Predicted surface at horizon 1, shape (7, 7)
    current_surf = X_test[0, -1, 0]     # Input surface at last step (t), shape (7, 7)

    # Ensure output directory exists
    plots_dir = "documentation/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Enable dark background styling matching dashboard style
    plt.style.use("dark_background")

    # ------------------ Plot 1: 3D Forecasted Surface ------------------
    logger.info("Plotting 3D Forecasted Surface...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    K, T_grid = np.meshgrid(GRID_KAPPAS, GRID_TAUS)
    surf = ax.plot_surface(K, T_grid, predicted_surf, cmap='viridis', edgecolor='none', alpha=0.9)
    
    fig.colorbar(surf, ax=ax, label="Implied Volatility", shrink=0.5, aspect=10)
    ax.set_title("Forecasted Implied Volatility Surface (1-Day Horizon)", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Moneyness κ = ln(K/F)", labelpad=10)
    ax.set_ylabel("Expiry τ (years)", labelpad=10)
    ax.set_zlabel("Implied Volatility", labelpad=10)
    ax.view_init(elev=30, azim=135)
    
    plot1_path = os.path.join(plots_dir, "forecast_surface_3d.png")
    plt.tight_layout()
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {plot1_path}")

    # ------------------ Plot 2: Residuals Heatmap ------------------
    logger.info("Plotting Residuals Heatmap...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Residuals: Actual - Predicted
    residuals = actual_surf - predicted_surf
    max_err = max(float(np.max(np.abs(residuals))), 0.02)
    
    im = ax.imshow(
        residuals,
        extent=[GRID_KAPPAS[0], GRID_KAPPAS[-1], GRID_TAUS[-1], GRID_TAUS[0]],
        aspect='auto',
        cmap='RdBu_r', # Red is underpricing, Blue is overpricing (or vice-versa)
        vmin=-max_err,
        vmax=max_err
    )
    
    fig.colorbar(im, ax=ax, label="Forecast Residual (Actual - Predicted)")
    ax.set_title("Expected Surface Evolution & Residuals (1-Day Horizon)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Moneyness κ = ln(K/F)")
    ax.set_ylabel("Expiry τ (years)")
    
    # Custom grid and ticks
    ax.set_xticks(GRID_KAPPAS)
    ax.set_yticks(GRID_TAUS)
    ax.invert_yaxis()
    ax.grid(True, color='gray', alpha=0.2, linestyle='--')
    
    plot2_path = os.path.join(plots_dir, "forecast_residuals_heatmap.png")
    plt.tight_layout()
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {plot2_path}")

    # ------------------ Plot 3: Cross-Sectional Smiles ------------------
    logger.info("Plotting Cross-Sectional Smiles & Term Structure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Smiles: Short (index 0), Medium (index 3), Long (index 6)
    expiry_indices = [0, 3, 6]
    colors = ["#ff4f4f", "#4fff4f", "#4f9fff"]
    labels = ["1 Week Expiry", "2 Months Expiry", "1 Year Expiry"]
    
    for idx, col, label in zip(expiry_indices, colors, labels):
        ax1.plot(GRID_KAPPAS, actual_surf[idx, :], marker='o', linestyle='-', color=col, label=f"Actual ({label})")
        ax1.plot(GRID_KAPPAS, predicted_surf[idx, :], marker='x', linestyle='--', color=col, label=f"Predicted ({label})")
        
    ax1.set_title("Volatility Smile comparison (IV vs Moneyness)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Moneyness κ = ln(K/F)")
    ax1.set_ylabel("Implied Volatility")
    ax1.grid(True, color='gray', alpha=0.2, linestyle=':')
    ax1.legend(fontsize=9)

    # Term Structure: OTM Put (index 1), ATM (index 3), OTM Call (index 5)
    moneyness_indices = [1, 3, 5]
    colors_ts = ["#df4fff", "#ff9f4f", "#4fdfdf"]
    labels_ts = ["OTM Put (κ=-0.20)", "ATM (κ=0.00)", "OTM Call (κ=0.20)"]
    
    for idx, col, label in zip(moneyness_indices, colors_ts, labels_ts):
        ax2.plot(GRID_TAUS, actual_surf[:, idx], marker='o', linestyle='-', color=col, label=f"Actual ({label})")
        ax2.plot(GRID_TAUS, predicted_surf[:, idx], marker='x', linestyle='--', color=col, label=f"Predicted ({label})")
        
    ax2.set_title("Term Structure comparison (IV vs Expiry)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Expiry τ (years)")
    ax2.set_ylabel("Implied Volatility")
    ax2.grid(True, color='gray', alpha=0.2, linestyle=':')
    ax2.legend(fontsize=9)

    plot3_path = os.path.join(plots_dir, "cross_sections.png")
    plt.tight_layout()
    plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {plot3_path}")
    logger.info("Plot exporting complete!")

if __name__ == "__main__":
    import glob
    # Check if we have collected raw live data in data/raw
    has_real_data = os.path.exists("data/raw") and len(glob.glob("data/raw/*.parquet")) >= 3
    export_plots(use_synthetic=not has_real_data)
