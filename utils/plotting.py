import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from data.processor import GRID_KAPPAS, GRID_TAUS

def plot_3d_surface(grid_iv: np.ndarray, title: str = "Implied Volatility Surface") -> go.Figure:
    """
    Generate an interactive Plotly 3D surface plot.
    Args:
        grid_iv: Volatility surface array of shape (7, 7) or (N_expiry, N_moneyness)
        title: Title of the chart
    """
    fig = go.Figure(data=[
        go.Surface(
            x=GRID_KAPPAS, 
            y=GRID_TAUS, 
            z=grid_iv,
            colorscale="Viridis",
            colorbar=dict(title="IV", thickness=15, len=0.6)
        )
    ])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="Moneyness κ = ln(K/F)", gridcolor="rgb(220, 220, 220)"),
            yaxis=dict(title="Expiry τ (years)", gridcolor="rgb(220, 220, 220)"),
            zaxis=dict(title="Implied Volatility", gridcolor="rgb(220, 220, 220)"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        width=800,
        height=600,
        template="plotly_dark"
    )
    return fig

def plot_residuals_heatmap(actual: np.ndarray, predicted: np.ndarray, title: str = "Forecast Residuals (Actual - Predicted)") -> go.Figure:
    """
    Generate a 2D Heatmap of forecast residual errors.
    """
    residuals = actual - predicted
    max_val = float(np.max(np.abs(residuals)))
    
    fig = go.Figure(data=go.Heatmap(
        x=GRID_KAPPAS,
        y=GRID_TAUS,
        z=residuals,
        colorscale="RdBu",
        zmin=-max_val if max_val > 1e-5 else -0.05,
        zmax=max_val if max_val > 1e-5 else 0.05,
        zmid=0,
        colorbar=dict(title="Error", thickness=15),
        hovertemplate="Moneyness: %{x}<br>Expiry: %{y:.4f}y<br>Residual: %{z:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(title="Moneyness κ = ln(K/F)", tickmode="array", tickvals=GRID_KAPPAS),
        yaxis=dict(title="Expiry τ (years)", tickmode="array", tickvals=GRID_TAUS, tickformat=".4f"),
        width=700,
        height=500,
        template="plotly_dark"
    )
    return fig

def plot_cross_sections(actual: np.ndarray, predicted: np.ndarray, title: str = "Smile & Term Structure Comparisons") -> go.Figure:
    """
    Create side-by-side subplot panels:
    - Left: Volatility Smile curves for short, medium, and long expiries.
    - Right: Term Structure curves for OTM put, ATM, and OTM call.
    """
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Volatility Smile (IV vs Strike)", "Term Structure (IV vs Expiry)"),
        horizontal_spacing=0.15
    )
    
    # 1. Left Panel: Smiles at chosen expiries (Index 0 = 1w, Index 3 = 2m, Index 6 = 1y)
    expiry_indices = [0, 3, 6]
    colors = ["#FF4136", "#2ECC40", "#0074D9"] # Red, Green, Blue
    
    for idx, col in zip(expiry_indices, colors):
        exp_year = GRID_TAUS[idx]
        exp_label = f"1 Week" if idx==0 else f"2 Months" if idx==3 else f"1 Year"
        
        # Actual Smile
        fig.add_trace(
            go.Scatter(
                x=GRID_KAPPAS, y=actual[idx, :],
                mode="lines+markers",
                name=f"Actual ({exp_label})",
                line=dict(color=col, width=2),
                legendgroup="smile"
            ),
            row=1, col=1
        )
        # Predicted Smile
        fig.add_trace(
            go.Scatter(
                x=GRID_KAPPAS, y=predicted[idx, :],
                mode="lines",
                name=f"Predicted ({exp_label})",
                line=dict(color=col, width=2, dash="dash"),
                legendgroup="smile"
            ),
            row=1, col=1
        )
        
    # 2. Right Panel: Term structure at chosen moneyness (Index 1 = -0.20, Index 3 = 0.00, Index 5 = 0.20)
    moneyness_indices = [1, 3, 5]
    colors_ts = ["#B10DC9", "#FF851B", "#39CCCC"] # Purple, Orange, Teal
    
    for idx, col in zip(moneyness_indices, colors_ts):
        kappa_val = GRID_KAPPAS[idx]
        kappa_label = f"OTM Put (κ={kappa_val:.1f})" if idx==1 else f"ATM (κ={kappa_val:.1f})" if idx==3 else f"OTM Call (κ={kappa_val:.1f})"
        
        # Actual Term Structure
        fig.add_trace(
            go.Scatter(
                x=GRID_TAUS, y=actual[:, idx],
                mode="lines+markers",
                name=f"Actual ({kappa_label})",
                line=dict(color=col, width=2),
                legendgroup="term"
            ),
            row=1, col=2
        )
        # Predicted Term Structure
        fig.add_trace(
            go.Scatter(
                x=GRID_TAUS, y=predicted[:, idx],
                mode="lines",
                name=f"Predicted ({kappa_label})",
                line=dict(color=col, width=2, dash="dash"),
                legendgroup="term"
            ),
            row=1, col=2
        )

    fig.update_layout(
        title_text=title,
        width=1000,
        height=500,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )
    
    fig.update_xaxes(title_text="Moneyness κ = ln(K/F)", row=1, col=1)
    fig.update_yaxes(title_text="Implied Volatility", row=1, col=1)
    
    fig.update_xaxes(title_text="Expiry τ (years)", row=1, col=2)
    fig.update_yaxes(title_text="Implied Volatility", row=1, col=2)
    
    return fig

def fig_to_json(fig: go.Figure) -> str:
    """Convert Plotly figure to JSON representation for web serving."""
    return fig.to_json()
