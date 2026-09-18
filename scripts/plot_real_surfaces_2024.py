"""
Generates a handful of real-data SPY implied-volatility surface plots for
manual visual inspection, from the artifact built by
scripts/run_real_data_2024_integration_test.py.

Picks: first date, last date, lowest-mean-IV date (calmest), highest-mean-IV
date (most stressed) in the 2024 dataset.
"""
import os
import sys
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.processor import GRID_KAPPAS, GRID_TAUS
from utils.plotting import plot_3d_surface

ARTIFACT_PATH = "data/historical/spy_2024_surfaces.npz"
OUTPUT_DIR = "documentation/plots/real_2024"


def _plot_one(grid_iv: np.ndarray, date: str, label: str) -> None:
    title = f"SPY Real IV Surface — {date} ({label})"

    # Plotly interactive HTML
    fig = plot_3d_surface(grid_iv, title=title)
    fig.write_html(os.path.join(OUTPUT_DIR, f"{date}_{label}.html"))

    # Matplotlib static PNG, matching training/export_plots.py's style
    plt.style.use("dark_background")
    fig_mpl = plt.figure(figsize=(10, 7))
    ax = fig_mpl.add_subplot(111, projection="3d")
    k_grid, t_grid = np.meshgrid(GRID_KAPPAS, GRID_TAUS)
    surf = ax.plot_surface(k_grid, t_grid, grid_iv, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    ax.set_xlabel("Moneyness κ = ln(K/F)", labelpad=10)
    ax.set_ylabel("Expiry τ (years)", labelpad=10)
    ax.set_zlabel("Implied Volatility", labelpad=10)
    ax.view_init(elev=25, azim=135)
    fig_mpl.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Implied Volatility")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{date}_{label}.png"), dpi=150)
    plt.close(fig_mpl)

    print(f"Saved plots for {date} ({label})")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with np.load(ARTIFACT_PATH, allow_pickle=False) as data:
        surfaces = data["surfaces"]
        timestamps = data["timestamps"]

    means = surfaces.mean(axis=(1, 2))
    picks = {
        "first_date": 0,
        "last_date": len(timestamps) - 1,
        "calmest": int(np.argmin(means)),
        "most_stressed": int(np.argmax(means)),
    }

    for label, idx in picks.items():
        # Filename-safe date: our timestamps are "YYYY-MM-DD HH:MM:SS" (always
        # midnight for this daily-EOD dataset) -- take just the date part,
        # since ':' is invalid in Windows filenames.
        date_str = str(timestamps[idx]).split(" ")[0]
        _plot_one(surfaces[idx], date_str, label)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
