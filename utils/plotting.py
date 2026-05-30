import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import base64
import struct
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

def _decode_bdata(obj):
    """Recursively walk a parsed Plotly JSON dict and decode any binary-encoded
    arrays (the {dtype, bdata, shape?} objects that Plotly Python 6.x emits)
    back into plain Python lists so any version of Plotly.js can render them."""
    if isinstance(obj, dict):
        if 'bdata' in obj and 'dtype' in obj:
            # Decode base64 binary data
            raw = base64.b64decode(obj['bdata'])
            dtype_map = {
                'f8': ('d', 8),  # float64
                'f4': ('f', 4),  # float32
                'i4': ('i', 4),  # int32
                'i2': ('h', 2),  # int16
                'i1': ('b', 1),  # int8
                'u4': ('I', 4),  # uint32
                'u2': ('H', 2),  # uint16
                'u1': ('B', 1),  # uint8
            }
            fmt_char, byte_size = dtype_map.get(obj['dtype'], ('d', 8))
            count = len(raw) // byte_size
            values = list(struct.unpack(f'<{count}{fmt_char}', raw))
            # Reshape if shape is provided (e.g., "7, 7" for 2D arrays)
            if 'shape' in obj:
                shape = [int(s.strip()) for s in obj['shape'].split(',')]
                if len(shape) == 2:
                    rows, cols = shape
                    values = [values[i * cols:(i + 1) * cols] for i in range(rows)]
                elif len(shape) == 3:
                    d0, d1, d2 = shape
                    values = [
                        [values[(i * d1 + j) * d2:(i * d1 + j + 1) * d2] for j in range(d1)]
                        for i in range(d0)
                    ]
            return values
        else:
            return {k: _decode_bdata(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_decode_bdata(item) for item in obj]
    return obj


def fig_to_json(fig: go.Figure) -> str:
    """Convert Plotly figure to JSON representation for web serving.
    
    Plotly Python 6.x uses binary bdata encoding for numpy arrays by default
    in to_json(). This is incompatible with older Plotly.js CDN versions.
    We decode all bdata fields back to plain JSON number arrays.
    """
    raw_json = fig.to_json()
    parsed = json.loads(raw_json)
    decoded = _decode_bdata(parsed)
    return json.dumps(decoded)

