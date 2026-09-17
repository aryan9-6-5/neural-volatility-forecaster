"""Train ONLY the hybrid model for the final stabilization experiment."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import numpy as np
from training.train import train_model, calculate_metrics, evaluate_regions, get_region_masks, seed_everything
from data.dataset import generate_synthetic_dataset, create_sequences
from data.processor import GRID_KAPPAS, GRID_TAUS

with open("configs/base_config.yaml", "r") as f:
    config = yaml.safe_load(f)

lookback = config["data"]["lookback"]
horizon = max(config["data"]["forecast_horizons"])

seed = config.get("training", {}).get("seed", 42)
seed_everything(seed)

# Generate shared dataset
dataset, _ = generate_synthetic_dataset(num_days=3000, seed=seed)

# Splits
T = len(dataset)
train_end = int(T * config["data"]["train_ratio"])
val_end = int(T * (config["data"]["train_ratio"] + config["data"]["val_ratio"]))
test_data = dataset[val_end:]
X_test, y_test = create_sequences(test_data, lookback, horizon)

masks = get_region_masks(GRID_KAPPAS, GRID_TAUS)

# Train hybrid only
print("=" * 80)
print("HYBRID-ONLY TRAINING (lambda_calendar=0.01)")
print("=" * 80)
hybrid_pred, _ = train_model(config, model_name="hybrid", use_synthetic=True, dataset=dataset)

metrics = calculate_metrics(hybrid_pred, y_test)
regions = evaluate_regions(hybrid_pred, y_test, masks)

print("\n" + "=" * 80)
print("HYBRID RESULTS")
print("=" * 80)
av_str = f"{metrics['AV']:.5%}"
print(f"RMSE:           {metrics['RMSE']:.5f}")
print(f"MAE:            {metrics['MAE']:.5f}")
print(f"TV (Roughness): {metrics['TV']:.5f}")
print(f"AV (Arbitrage): {av_str}")
print(f"ATM RMSE:       {regions['ATM']:.5f}")
print(f"OTM-P RMSE:     {regions['OTM Puts']:.5f}")
print(f"Long-dated RMSE:{regions['Long-dated']:.5f}")
print("=" * 80)
