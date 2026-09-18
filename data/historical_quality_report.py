"""
Builds a dataset-quality report for a historical surface-import run (see
data.dataset.import_historical_dataframe's collect_stats=True output). Pure
reporting: reuses the exact rejection-stage counts recorded by
data.processor.apply_arbitrage_filters/process_raw_snapshot at filtering time,
so the numbers here can never diverge from what was actually filtered.
"""
import json
import numpy as np
import pandas as pd

from data.processor import GRID_KAPPAS, GRID_TAUS

REJECTION_STAGES = [
    "date_parse_failure", "positivity", "iv_bounds", "volume",
    "bid_ask_validity", "spread_ratio", "calendar_spread",
]


def compute_coverage_pct(kappas: np.ndarray, taus: np.ndarray) -> float:
    """
    Fraction of the 49 standard (kappa, tau) grid cells touched by at least
    one observation, computed BEFORE RBF interpolation -- matches Data
    Contract v1.0 Section 7's surface_coverage_pct definition exactly.
    """
    if len(kappas) == 0:
        return 0.0
    kappa_bins = GRID_KAPPAS[np.argmin(np.abs(GRID_KAPPAS[:, None] - np.asarray(kappas)[None, :]), axis=0)]
    tau_bins = GRID_TAUS[np.argmin(np.abs(GRID_TAUS[:, None] - np.asarray(taus)[None, :]), axis=0)]
    occupied = set(zip(kappa_bins.tolist(), tau_bins.tolist()))
    return 100.0 * len(occupied) / (len(GRID_KAPPAS) * len(GRID_TAUS))


def build_report(raw_df: pd.DataFrame, per_day_stats: list, surfaces: np.ndarray, timestamps: list,
                  output_path: str) -> dict:
    """
    Writes a human-readable Markdown quality report to output_path and returns
    the same data as a dict (also written alongside as <output_path>.json).
    """
    stats_df = pd.DataFrame(per_day_stats)
    total_dates_attempted = len(stats_df)
    valid_dates = stats_df[stats_df["skip_reason"].isna()] if not stats_df.empty else stats_df
    skipped_dates = stats_df[stats_df["skip_reason"].notna()] if not stats_df.empty else stats_df

    skip_reason_counts = skipped_dates["skip_reason"].value_counts().to_dict() if not skipped_dates.empty else {}

    raw_contracts_total = int(stats_df["raw_contracts_count"].sum()) if not stats_df.empty else 0
    valid_contracts_total = int(stats_df["valid_contracts_count"].sum()) if "valid_contracts_count" in stats_df else 0
    rejection_totals = {
        stage: int(stats_df[stage].sum()) if stage in stats_df else 0
        for stage in REJECTION_STAGES
    }
    rejection_sum = sum(rejection_totals.values())

    contracts_per_date = stats_df["raw_contracts_count"] if not stats_df.empty else pd.Series(dtype=float)

    nan_count = int(np.isnan(surfaces).sum())
    inf_count = int(np.isinf(surfaces).sum())

    ts_series = pd.Series(timestamps)
    duplicate_dates = int(ts_series.duplicated().sum())
    duplicate_contracts = int(raw_df.duplicated(subset=["contract_id", "date"]).sum()) if "contract_id" in raw_df.columns else None

    suspicious = {}
    if "implied_volatility" in raw_df.columns:
        suspicious["iv_gt_3"] = int((raw_df["implied_volatility"] > 3.0).sum())
        suspicious["iv_gt_5"] = int((raw_df["implied_volatility"] > 5.0).sum())
    if "bid" in raw_df.columns and "ask" in raw_df.columns:
        suspicious["bid_gt_ask"] = int((raw_df["bid"] > raw_df["ask"]).sum())
        suspicious["bid_le_0"] = int((raw_df["bid"] <= 0).sum())
        suspicious["ask_le_0"] = int((raw_df["ask"] <= 0).sum())

    coverage_stats = {}
    if "coverage_pct" in stats_df.columns:
        cov = stats_df.loc[stats_df["skip_reason"].isna(), "coverage_pct"].dropna()
        if len(cov):
            coverage_stats = {"mean_pct": float(cov.mean()), "min_pct": float(cov.min()), "max_pct": float(cov.max())}

    report = {
        "total_trading_dates": total_dates_attempted,
        "valid_surfaces": len(valid_dates),
        "rejected_surfaces": len(skipped_dates),
        "skip_reason_counts": skip_reason_counts,
        "contracts_per_date": {
            "min": float(contracts_per_date.min()) if len(contracts_per_date) else None,
            "median": float(contracts_per_date.median()) if len(contracts_per_date) else None,
            "max": float(contracts_per_date.max()) if len(contracts_per_date) else None,
        },
        "raw_contracts_count": raw_contracts_total,
        "valid_contracts_count": valid_contracts_total,
        "rejected_contracts_count": raw_contracts_total - valid_contracts_total,
        "rejection_reasons": rejection_totals,
        "rejection_reasons_sum_check": {
            "sum_of_stages": rejection_sum,
            "raw_minus_valid": raw_contracts_total - valid_contracts_total,
            "consistent": rejection_sum == (raw_contracts_total - valid_contracts_total),
        },
        "coverage_pct": coverage_stats,
        "nan_cells_in_final_surfaces": nan_count,
        "inf_cells_in_final_surfaces": inf_count,
        "date_range": {"start": str(min(timestamps)) if timestamps else None, "end": str(max(timestamps)) if timestamps else None},
        "duplicate_dates": duplicate_dates,
        "duplicate_contracts": duplicate_contracts,
        "suspicious_values": suspicious,
    }

    with open(output_path, "w") as f:
        f.write("# SPY Historical Options Dataset Quality Report\n\n")
        f.write(f"## Trading dates\n- Total dates attempted: {total_dates_attempted}\n")
        f.write(f"- Valid surfaces built: {len(valid_dates)}\n- Rejected (skipped) dates: {len(skipped_dates)}\n")
        for reason, count in skip_reason_counts.items():
            f.write(f"  - {reason}: {count}\n")
        f.write(f"- Date range: {report['date_range']['start']} to {report['date_range']['end']}\n")
        f.write(f"- Duplicate dates: {duplicate_dates}\n")
        if duplicate_contracts is not None:
            f.write(f"- Duplicate contracts (contract_id+date): {duplicate_contracts}\n")

        f.write("\n## Contracts per date\n")
        f.write(f"- min: {report['contracts_per_date']['min']}\n- median: {report['contracts_per_date']['median']}\n- max: {report['contracts_per_date']['max']}\n")

        f.write("\n## Contract-level filtering\n")
        f.write(f"- Raw contracts (total across all dates): {raw_contracts_total}\n")
        f.write(f"- Valid contracts (survived all filters): {valid_contracts_total}\n")
        f.write(f"- Rejected contracts: {raw_contracts_total - valid_contracts_total}\n\n")
        f.write("| Rejection stage | Contracts dropped |\n|---|---|\n")
        for stage, count in rejection_totals.items():
            f.write(f"| {stage} | {count} |\n")
        f.write(f"\nConsistency check (sum of stages == raw - valid): **{report['rejection_reasons_sum_check']['consistent']}** "
                f"({rejection_sum} vs {raw_contracts_total - valid_contracts_total})\n")

        f.write("\n## Surface coverage\n")
        if coverage_stats:
            f.write(f"- Mean pre-interpolation grid coverage: {coverage_stats['mean_pct']:.1f}%\n")
            f.write(f"- Min: {coverage_stats['min_pct']:.1f}%, Max: {coverage_stats['max_pct']:.1f}%\n")
        else:
            f.write("- Not computed for this run.\n")

        f.write("\n## Data integrity\n")
        f.write(f"- NaN cells in final float32 surfaces: {nan_count}\n")
        f.write(f"- Infinite cells in final float32 surfaces: {inf_count}\n")

        f.write("\n## Suspicious values (raw source data)\n")
        for k, v in suspicious.items():
            f.write(f"- {k}: {v}\n")

    with open(output_path + ".json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report
