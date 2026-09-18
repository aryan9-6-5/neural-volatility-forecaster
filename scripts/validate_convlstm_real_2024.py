"""
Supplementary validation: training/train.py::evaluate_all doesn't include
ConvLSTM in its comparison loop (a pre-existing gap, not introduced by the
real-data work) even though it's an implemented, spatial/convolutional model.
This reuses the already-built real 2024 surface artifact (no need to redo the
~24-minute historical import) and trains+evaluates ConvLSTM the same way
evaluate_all does for the other three DL models, for an apples-to-apples
comparison row.
"""
import os
import sys
import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.train import train_model, calculate_metrics, evaluate_regions, get_region_masks
from data.processor import GRID_KAPPAS, GRID_TAUS

ARTIFACT_PATH = "data/historical/spy_2024_surfaces.npz"
CONFIG_PATH = "configs/real_data_config.yaml"
OUTPUT_DIR = "models/real_data"


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    with np.load(ARTIFACT_PATH, allow_pickle=False) as data:
        surfaces = data["surfaces"]

    masks = get_region_masks(GRID_KAPPAS, GRID_TAUS)

    test_y_pred, y_test = train_model(config, model_name="convlstm", dataset=surfaces, output_dir=OUTPUT_DIR)
    metrics = calculate_metrics(test_y_pred, y_test)
    regions = evaluate_regions(test_y_pred, y_test, masks)

    print("\n" + "=" * 115)
    print(f"{'Model':<20} | {'RMSE':<8} | {'MAE':<8} | {'TV (Rough)':<10} | {'AV (Arbitrage)':<14} | {'ATM RMSE':<9} | {'OTM-P RMSE':<10} | {'Long-dated RMSE':<15}")
    print("-" * 115)
    av_str = f"{metrics['AV']:.5%}"
    print(f"{'ConvLSTM':<20} | {metrics['RMSE']:.5f} | {metrics['MAE']:.5f} | {metrics['TV']:.5f}    | {av_str:<14} | {regions['ATM']:.5f}  | {regions['OTM Puts']:.5f}   | {regions['Long-dated']:.5f}")
    print("=" * 115)


if __name__ == "__main__":
    main()
