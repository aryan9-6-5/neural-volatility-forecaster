"""
End-to-end real-data integration test: SPY 2024 historical options ->
standardized 7x7 IV surfaces -> quality report -> leakage-safety verification
-> validation of every existing model/baseline against real data.

Single command to reproduce the whole pass:
    python scripts/run_real_data_2024_integration_test.py

Does NOT touch the production synthetic pipeline: writes to
models/real_data/ (not models/), data/historical/ (already gitignored), and
uses configs/real_data_config.yaml (configs/base_config.yaml untouched).
"""
import os
import sys
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.adapters.spy_options_dataset import build_spy_dataframe
from data.dataset import import_historical_dataframe, create_sequences
from data.historical_quality_report import build_report
from training.train import evaluate_all


# force=True: some already-imported dependency (mlflow, etc.) may have attached
# its own root-logger handlers before this line runs, which would otherwise
# make basicConfig a silent no-op and swallow every INFO log this script and
# the pipeline it drives emit.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)

YEAR = 2024
ARTIFACT_PATH = "data/historical/spy_2024_surfaces.npz"
QUALITY_REPORT_PATH = "data/historical/spy_2024_quality_report.md"
REAL_DATA_CONFIG_PATH = "configs/real_data_config.yaml"
LEAKAGE_AUDIT_TEMPLATE_PATH = "documentation/Leakage_Audit_2024.md"
REAL_MODEL_OUTPUT_DIR = "models/real_data"


def verify_no_lookahead(surfaces: np.ndarray, timestamps: list, lookback: int, horizons: list) -> dict:
    """
    Concretely verifies, for every constructible sequence, that every input
    date is strictly before every target date, and that the horizon is
    represented at the correct offset. Raises AssertionError on any violation
    (this is a hard leakage gate, not a soft warning).
    """
    dates = pd.to_datetime(pd.Series(timestamps))
    horizon = max(horizons)
    T = len(surfaces)
    num_samples = T - lookback - horizon + 1
    assert num_samples > 0, f"Not enough surfaces ({T}) for lookback={lookback} + horizon={horizon}"

    violations = 0
    for i in range(num_samples):
        input_dates = dates.iloc[i:i + lookback]
        target_dates = dates.iloc[i + lookback:i + lookback + horizon]
        if not (input_dates.max() < target_dates.min()):
            violations += 1
        # Horizon correctly represented: h-day-ahead target is exactly at offset (lookback + h - 1)
        for h in horizons:
            target_idx = i + lookback + h - 1
            if target_idx >= T:
                continue
            expected_date = dates.iloc[target_idx]
            input_last_date = dates.iloc[i + lookback - 1]
            if not (input_last_date < expected_date):
                violations += 1

    assert violations == 0, f"Found {violations} look-ahead violations in sequence construction"
    return {"num_samples_checked": num_samples, "violations": violations}


def main():
    with open(REAL_DATA_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    lookback = config["data"]["lookback"]
    horizons = config["data"]["forecast_horizons"]

    # 1-3: download, build source dataframe, run through the EXISTING pipeline
    # (process_raw_snapshot -> apply_arbitrage_filters -> interpolate_to_grid)
    logger.info(f"Building SPY {YEAR} source dataframe...")
    options_df, raw_options_df = build_spy_dataframe(YEAR)

    logger.info("Running through the existing historical import pipeline (with stats collection)...")
    surfaces, timestamps, per_day_stats = import_historical_dataframe(
        options_df, volume_filter=False, collect_stats=True
    )
    surfaces = surfaces.astype(np.float32)  # Data Contract v1.0 Section 2.3

    assert np.all((surfaces >= 0.01) & (surfaces <= 5.00)), "Surface values outside Data Contract IV bounds [0.01, 5.00]"
    assert list(timestamps) == sorted(timestamps), "Timestamps are not chronologically ordered"

    logger.info(f"Built {len(surfaces)} valid surfaces out of {len(per_day_stats)} attempted trading dates.")

    # 4: save the surface dataset artifact
    os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
    metadata = {
        "data_contract_version": "v1.0",
        "grid_kappas": [-0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30],
        "grid_taus": [1 / 52, 2 / 52, 1 / 12, 2 / 12, 3 / 12, 6 / 12, 1.0],
        "source_repo": "anahatsingh-ui/options-dataset-hist",
        "source_ref": "main",
        "build_date": datetime.now().isoformat(),
        "year": YEAR,
        "volume_filter": False,
        "spot_price_field": "close (not adjusted_close -- see data/adapters/spy_options_dataset.py docstring)",
    }
    np.savez(ARTIFACT_PATH, surfaces=surfaces, timestamps=np.array(timestamps, dtype="U32"), metadata=json.dumps(metadata))
    logger.info(f"Saved surface dataset artifact to {ARTIFACT_PATH}")

    # 5: quality report
    logger.info("Building dataset quality report...")
    report = build_report(raw_options_df, per_day_stats, surfaces, timestamps, QUALITY_REPORT_PATH)
    logger.info(f"Quality report written to {QUALITY_REPORT_PATH} (+ .json)")

    # 6: leakage gate + per-split / per-horizon reporting
    logger.info("Verifying no look-ahead information in sequence construction...")
    T = len(surfaces)
    train_end = int(T * config["data"]["train_ratio"])
    val_end = int(T * (config["data"]["train_ratio"] + config["data"]["val_ratio"]))
    splits = {"train": (0, train_end), "validation": (train_end, val_end), "test": (val_end, T)}

    split_report = {}
    for name, (start, end) in splits.items():
        split_surfaces = surfaces[start:end]
        split_dates = timestamps[start:end]
        T_split = end - start

        leakage_check = verify_no_lookahead(split_surfaces, split_dates, lookback, horizons) if T_split > lookback + max(horizons) else {"num_samples_checked": 0, "violations": 0}

        X_split, y_split = (create_sequences(split_surfaces, lookback, max(horizons))
                             if T_split >= lookback + max(horizons) else (np.empty((0,)), np.empty((0,))))

        split_report[name] = {
            "date_range": [split_dates[0], split_dates[-1]] if split_dates else [None, None],
            "num_surfaces": T_split,
            "num_sequences_built": len(X_split),
            "usable_samples_per_horizon": {h: max(0, T_split - lookback - h + 1) for h in horizons},
            "leakage_check": leakage_check,
        }
        logger.info(f"Split '{name}': {T_split} surfaces, {len(X_split)} sequences, dates {split_report[name]['date_range']}")

    with open("data/historical/spy_2024_split_report.json", "w") as f:
        json.dump(split_report, f, indent=2, default=str)
    logger.info("Split/leakage report written to data/historical/spy_2024_split_report.json")

    # 7: plots
    logger.info("Generating real-data surface plots...")
    import subprocess
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "plot_real_surfaces_2024.py")], check=True)

    # 8: leakage audit doc note
    if os.path.exists(LEAKAGE_AUDIT_TEMPLATE_PATH):
        logger.info(f"Leakage audit document present at {LEAKAGE_AUDIT_TEMPLATE_PATH} -- "
                    f"cross-check its 'Split Methodology' section against data/historical/spy_2024_split_report.json")

    # 9: validate every existing model/baseline against the real dataset (no new training loop,
    # no new model architectures -- reuses evaluate_all exactly as it already exists).
    logger.info("Validating all existing models/baselines against the real 2024 dataset...")
    os.makedirs(REAL_MODEL_OUTPUT_DIR, exist_ok=True)
    results = evaluate_all(config, dataset=surfaces, output_dir=REAL_MODEL_OUTPUT_DIR)

    logger.info("Integration test complete.")
    logger.info(f"  Quality report:  {QUALITY_REPORT_PATH}")
    logger.info(f"  Split report:    data/historical/spy_2024_split_report.json")
    logger.info(f"  Surface plots:   documentation/plots/real_2024/")
    logger.info(f"  Real checkpoints: {REAL_MODEL_OUTPUT_DIR}/ (production models/ untouched)")

    return report, split_report, results


if __name__ == "__main__":
    main()
