"""
Combined visualization showing geodesic trajectory in pattern space
alongside the time series candlestick chart with curvature.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc
from datetime import datetime, timedelta


def plot_geodesic_with_timeseries(
    ohlc_data: pd.DataFrame,
    geometry: CurvedCandleGeometry,
    start_index: int,
    initial_velocity: np.ndarray,
    n_steps: int = None,
    title: str = "Geodesic Trajectory: Pattern Space vs Time Series"
) -> go.Figure:
    """
    Create a comprehensive visualization showing:
    1. Geodesic trajectory in 2D pattern space
    2. Time series candlestick chart
    3. Curvature overlay
    4. Trajectory position markers on time series
    """
    
    # Default to extending trajectory to end of data
    if n_steps is None:
        n_steps = len(ohlc_data) - start_index - 1
    
    # Compute geodesic trajectory
    if initial_velocity is None:
        # Historical mode - show actual path
        trajectory = geometry.predict_geodesic_path(start_index, use_historical=True, n_steps=n_steps)
    else:
        # Prediction mode - use geodesic equation
        trajectory = geometry.predict_geodesic_path(start_index, initial_velocity, n_steps, use_historical=False)
    traj_s = [p[0] for p in trajectory]
    traj_u = [p[1] for p in trajectory]
    
    # Compute curvatures
    curvatures = geometry.compute_curvature_series()
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.5, 0.5],
        column_widths=[0.5, 0.5],
        subplot_titles=(
            "Geodesic Path in Pattern Space",
            "Candlestick Time Series",
            "Pattern Evolution Along Trajectory", 
            "Gaussian Curvature"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "candlestick", "rowspan": 1}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.15
    )
    
    # --- 1. Pattern Space Trajectory (Top Left) ---
    
    # Background candles
    sentiments = [c.sentiment for c in geometry.candles]
    uwrs = [c.upper_wick_ratio for c in geometry.candles]
    
    fig.add_trace(
        go.Scatter(
            x=sentiments,
            y=uwrs,
            mode='markers',
            marker=dict(size=4, color='#30363d'),
            name='All Candles',
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    # Geodesic path
    fig.add_trace(
        go.Scatter(
            x=traj_s,
            y=traj_u,
            mode='lines+markers',
            line=dict(color='#00ffff', width=3),
            marker=dict(size=8, color='#00ffff'),
            name='Geodesic Path',
            text=[f"Step {i}" for i in range(len(trajectory))],
            hovertemplate='Step: %{text}<br>Sentiment: %{x:.3f}<br>UWR: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Start and end markers
    fig.add_trace(
        go.Scatter(
            x=[traj_s[0]], y=[traj_u[0]],
            mode='markers',
            marker=dict(size=15, color='#00ff00', symbol='circle', line=dict(color='#ffffff', width=2)),
            name='Start',
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[traj_s[-1]], y=[traj_u[-1]],
            mode='markers',
            marker=dict(size=15, color='#ff0000', symbol='square', line=dict(color='#ffffff', width=2)),
            name='End',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Valid region boundary
    boundary_s = np.linspace(-1, 1, 100)
    boundary_u = 1 - np.abs(boundary_s)
    fig.add_trace(
        go.Scatter(
            x=boundary_s, y=boundary_u,
            mode='lines',
            line=dict(color='#ffffff', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    # --- 2. Candlestick Chart (Top Right) ---
    
    fig.add_trace(
        go.Candlestick(
            x=ohlc_data.index,
            open=ohlc_data['open'],
            high=ohlc_data['high'],
            low=ohlc_data['low'],
            close=ohlc_data['close'],
            name='OHLC',
            increasing_line_color='#00ff00',
            decreasing_line_color='#ff0000',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Add trajectory position markers on time series
    trajectory_indices = list(range(start_index, min(start_index + len(trajectory), len(ohlc_data))))
    trajectory_times = ohlc_data.index[trajectory_indices] if len(trajectory_indices) <= len(ohlc_data) else []
    
    # Create trajectory overlay showing the predicted evolution
    if len(trajectory_times) > 0:
        # Use high prices for visibility
        trajectory_high_prices = [ohlc_data.iloc[i]['high'] if i < len(ohlc_data) else np.nan 
                                for i in trajectory_indices]
        trajectory_low_prices = [ohlc_data.iloc[i]['low'] if i < len(ohlc_data) else np.nan 
                               for i in trajectory_indices]
        
        # Add trajectory path as a line connecting high points
        fig.add_trace(
            go.Scatter(
                x=trajectory_times,
                y=trajectory_high_prices,
                mode='lines+markers',
                line=dict(color='#00ffff', width=4, dash='dash'),
                marker=dict(size=8, color='#00ffff', symbol='circle',
                           line=dict(color='#ffffff', width=2)),
                name='Geodesic Timeline',
                hovertemplate='Time: %{x}<br>Price: %{y}<br>Trajectory Step: %{customdata}<extra></extra>',
                customdata=list(range(len(trajectory_times))),
                showlegend=True
            ),
            row=1, col=2
        )
        
        # Add start and end markers on time series
        fig.add_trace(
            go.Scatter(
                x=[trajectory_times[0]],
                y=[trajectory_high_prices[0]],
                mode='markers',
                marker=dict(size=15, color='#00ff00', symbol='star',
                           line=dict(color='#ffffff', width=2)),
                name='Start',
                showlegend=False
            ),
            row=1, col=2
        )
        
        if len(trajectory_times) > 1:
            fig.add_trace(
                go.Scatter(
                    x=[trajectory_times[-1]],
                    y=[trajectory_high_prices[-1]],
                    mode='markers',
                    marker=dict(size=15, color='#ff0000', symbol='square',
                               line=dict(color='#ffffff', width=2)),
                    name='End',
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # Highlight start position
    if start_index < len(ohlc_data):
        fig.add_vline(
            x=ohlc_data.index[start_index], 
            line_width=2, 
            line_dash="dash", 
            line_color="#00ff00",
            row=1, col=2
        )
    
    # --- 3. Pattern Evolution (Bottom Left) ---
    
    # Show how sentiment and UWR evolve along the trajectory
    step_numbers = list(range(len(trajectory)))
    
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=traj_s,
            mode='lines+markers',
            line=dict(color='#00ff00', width=2),
            marker=dict(size=6),
            name='Sentiment',
            yaxis='y3',
            showlegend=True
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=traj_u,
            mode='lines+markers',
            line=dict(color='#ff6b6b', width=2),
            marker=dict(size=6),
            name='Upper Wick Ratio',
            yaxis='y3',
            showlegend=True
        ),
        row=2, col=1
    )
    
    # --- 4. Curvature (Bottom Right) ---
    
    colors = ['#00ff00' if k > 0 else '#ff0000' if k < 0 else '#666666' for k in curvatures]
    
    fig.add_trace(
        go.Scatter(
            x=ohlc_data.index,
            y=curvatures,
            mode='lines+markers',
            line=dict(width=2, color='#00ffff'),
            marker=dict(size=4, color=colors),
            name='Curvature',
            hovertemplate='Curvature: %{y:.6f}<br>%{text}<extra></extra>',
            text=['Trending' if k > 0 else 'Volatile' if k < 0 else 'Flat' for k in curvatures],
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Add zero line for curvature
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", row=2, col=2)
    
    # Highlight trajectory region on curvature plot
    if start_index < len(ohlc_data) and len(trajectory_times) > 0:
        fig.add_vrect(
            x0=ohlc_data.index[start_index],
            x1=trajectory_times[-1],
            fillcolor="#00ffff",
            opacity=0.1,
            line_width=0,
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#ffffff')),
        height=900,
        showlegend=True,
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        legend=dict(
            bgcolor='#161b22',
            bordercolor='#30363d',
            borderwidth=1
        )
    )
    
    # Update axes
    # Pattern space
    fig.update_xaxes(title_text="Sentiment", row=1, col=1, 
                     gridcolor='#30363d', zerolinecolor='#30363d',
                     range=[-1.1, 1.1])
    fig.update_yaxes(title_text="Upper Wick Ratio", row=1, col=1,
                     gridcolor='#30363d', zerolinecolor='#30363d',
                     range=[-0.1, 1.1])
    
    # Time series
    fig.update_xaxes(title_text="Time", row=1, col=2,
                     gridcolor='#30363d')
    fig.update_yaxes(title_text="Price", row=1, col=2,
                     gridcolor='#30363d', zerolinecolor='#30363d')
    
    # Pattern evolution
    fig.update_xaxes(title_text="Trajectory Step", row=2, col=1,
                     gridcolor='#30363d')
    fig.update_yaxes(title_text="Pattern Values", row=2, col=1,
                     gridcolor='#30363d', zerolinecolor='#30363d')
    
    # Curvature
    fig.update_xaxes(title_text="Time", row=2, col=2,
                     gridcolor='#30363d')
    fig.update_yaxes(title_text="Gaussian Curvature", row=2, col=2,
                     gridcolor='#30363d', zerolinecolor='#30363d')
    
    return fig


def create_trajectory_comparison(
    ohlc_data: pd.DataFrame,
    geometry: CurvedCandleGeometry,
    trajectories: list,
    title: str = "Multiple Geodesic Trajectories Comparison"
) -> go.Figure:
    """
    Compare multiple geodesic trajectories with different initial conditions.
    
    trajectories: list of tuples (start_index, initial_velocity, label, color)
    """
    
    # Create subplot
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=("Pattern Space Trajectories", "Time Series with Trajectory Markers"),
        horizontal_spacing=0.12
    )
    
    # Background for pattern space
    sentiments = [c.sentiment for c in geometry.candles]
    uwrs = [c.upper_wick_ratio for c in geometry.candles]
    
    fig.add_trace(
        go.Scatter(
            x=sentiments, y=uwrs,
            mode='markers',
            marker=dict(size=3, color='#30363d'),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=ohlc_data.index,
            open=ohlc_data['open'],
            high=ohlc_data['high'],
            low=ohlc_data['low'],
            close=ohlc_data['close'],
            name='OHLC',
            increasing_line_color='#00ff00',
            decreasing_line_color='#ff0000',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Add each trajectory
    for start_idx, velocity, label, color in trajectories:
        # Compute full trajectory
        max_steps = len(ohlc_data) - start_idx - 1
        trajectory = geometry.predict_geodesic_path(start_idx, velocity, n_steps=max_steps)
        traj_s = [p[0] for p in trajectory]
        traj_u = [p[1] for p in trajectory]
        
        # Pattern space path
        fig.add_trace(
            go.Scatter(
                x=traj_s, y=traj_u,
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=6),
                name=label,
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Time series markers
        trajectory_indices = list(range(start_idx, min(start_idx + len(trajectory), len(ohlc_data))))
        if trajectory_indices:
            trajectory_times = ohlc_data.index[trajectory_indices]
            trajectory_prices = [ohlc_data.iloc[i]['close'] for i in trajectory_indices]
            
            fig.add_trace(
                go.Scatter(
                    x=trajectory_times,
                    y=trajectory_prices,
                    mode='markers',
                    marker=dict(size=8, color=color, symbol='circle'),
                    name=f"{label} (TS)",
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # Add boundary
    boundary_s = np.linspace(-1, 1, 100)
    boundary_u = 1 - np.abs(boundary_s)
    fig.add_trace(
        go.Scatter(
            x=boundary_s, y=boundary_u,
            mode='lines',
            line=dict(color='#ffffff', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#ffffff')),
        height=500,
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        legend=dict(
            bgcolor='#161b22',
            bordercolor='#30363d',
            borderwidth=1,
            x=0.02,
            y=0.98
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text="Sentiment", row=1, col=1,
                     gridcolor='#30363d', zerolinecolor='#30363d',
                     range=[-1.1, 1.1])
    fig.update_yaxes(title_text="Upper Wick Ratio", row=1, col=1,
                     gridcolor='#30363d', zerolinecolor='#30363d',
                     range=[-0.1, 1.1])
    fig.update_xaxes(title_text="Time", row=1, col=2,
                     gridcolor='#30363d')
    fig.update_yaxes(title_text="Price", row=1, col=2,
                     gridcolor='#30363d', zerolinecolor='#30363d')
    
    return fig