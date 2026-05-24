import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg") # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import generate_synthetic_dataset
from data.processor import GRID_KAPPAS, GRID_TAUS
from utils.plotting import plot_3d_surface, plot_residuals_heatmap, plot_cross_sections

def main():
    print("Generating synthetic dataset...")
    # Generate 1 day of synthetic surface (index 0)
    dataset, _ = generate_synthetic_dataset(num_days=2)
    grid_iv = dataset[0] # Shape (7, 7)
    
    # Create forecast grid (with a slight decay to simulate residuals)
    forecast_iv = grid_iv * 0.95
    
    print("Generating Matplotlib 3D Volatility Surface PNG...")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    
    # Create meshgrid for plotting
    k_grid, t_grid = np.meshgrid(GRID_KAPPAS, GRID_TAUS)
    
    # Plot surface
    surf = ax.plot_surface(
        k_grid, t_grid, grid_iv, 
        cmap="viridis", 
        edgecolor="none",
        alpha=0.9
    )
    
    # Format labels
    ax.set_title("Implied Volatility Surface (Matplotlib)", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Moneyness κ = ln(K/F)", fontsize=10, labelpad=10)
    ax.set_ylabel("Expiry τ (years)", fontsize=10, labelpad=10)
    ax.set_zlabel("Implied Volatility", fontsize=10, labelpad=10)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Implied Volatility")
    
    # Adjust view angle
    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    
    # Save PNG
    output_png = "tests/sample_surface_matplotlib.png"
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Saved Matplotlib PNG: {output_png}")
    
    # Generate Plotly HTML files
    print("Generating Plotly interactive HTML files...")
    
    # 1. 3D Surface
    fig_3d = plot_3d_surface(grid_iv, title="Implied Volatility Surface (Plotly 3D)")
    fig_3d.write_html("tests/sample_surface_3d.html")
    print("Saved: tests/sample_surface_3d.html")
    
    # 2. Residual Heatmap
    fig_heatmap = plot_residuals_heatmap(grid_iv, forecast_iv, title="Forecast Residuals (Actual - Predicted)")
    fig_heatmap.write_html("tests/sample_residuals.html")
    print("Saved: tests/sample_residuals.html")
    
    # 3. Cross Sections
    fig_cross = plot_cross_sections(grid_iv, forecast_iv, title="Smile & Term Structure Comparisons")
    fig_cross.write_html("tests/sample_cross_sections.html")
    print("Saved: tests/sample_cross_sections.html")
    
    print("\nVisual verification assets generated successfully!")

if __name__ == "__main__":
    main()
