"""
Visualization tools for curved candle geometry.

This module provides interactive visualizations for:
- Market curvature over time
- Geodesic trajectories
- Pattern space with metric distortion
- Parallel transport of patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Optional
from curved_candle_geometry import CurvedCandleGeometry, CandleMetric, create_candle_metrics_from_ohlc


def plot_curvature_analysis(ohlc_data: pd.DataFrame, 
                           curvatures: np.ndarray,
                           title: str = "Market Curvature Analysis") -> go.Figure:
    """
    Create an interactive plot showing candlesticks with curvature overlay.
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=("Candlestick Chart with Curved Geometry", "Gaussian Curvature")
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=ohlc_data.index,
            open=ohlc_data['open'],
            high=ohlc_data['high'],
            low=ohlc_data['low'],
            close=ohlc_data['close'],
            name='OHLC',
            increasing_line_color='#00ff00',
            decreasing_line_color='#ff0000'
        ),
        row=1, col=1
    )
    
    # Add curvature trace
    colors = ['#00ff00' if k > 0 else '#ff0000' if k < 0 else '#666666' for k in curvatures]
    
    fig.add_trace(
        go.Scatter(
            x=ohlc_data.index,
            y=curvatures,
            mode='lines+markers',
            name='Curvature',
            line=dict(width=2, color='#00ffff'),
            marker=dict(size=6, color=colors),
            hovertemplate='Curvature: %{y:.6f}<br>%{text}',
            text=['Trending' if k > 0 else 'Volatile' if k < 0 else 'Flat' for k in curvatures]
        ),
        row=2, col=1
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", row=2, col=1)
    
    # Update layout with dark theme
    fig.update_layout(
        title=dict(text=title, font=dict(color='#ffffff')),
        height=800,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9')
    )
    
    # Update axes styling
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor='#30363d', zerolinecolor='#30363d')
    fig.update_yaxes(title_text="Gaussian Curvature", row=2, col=1, gridcolor='#30363d', zerolinecolor='#30363d')
    fig.update_xaxes(title_text="Time", row=2, col=1, gridcolor='#30363d')
    fig.update_xaxes(gridcolor='#30363d', row=1, col=1)
    
    return fig


def plot_pattern_space_with_metric(candle_metrics: List[CandleMetric],
                                 highlight_indices: Optional[List[int]] = None) -> go.Figure:
    """
    Visualize the 2D pattern space with metric distortion effects.
    """
    fig = go.Figure()
    
    # Extract pattern coordinates
    sentiments = [c.sentiment for c in candle_metrics]
    uwrs = [c.upper_wick_ratio for c in candle_metrics]
    ranges = [c.range_value for c in candle_metrics]
    
    # Create main scatter plot
    fig.add_trace(go.Scatter(
        x=sentiments,
        y=uwrs,
        mode='markers',
        marker=dict(
            size=[r/np.mean(ranges) * 10 for r in ranges],  # Size proportional to range
            color=ranges,
            colorscale='Viridis',
            line=dict(color='#30363d', width=0.5),
            colorbar=dict(title="Range"),
            showscale=True
        ),
        text=[f"Candle {i}<br>Range: {r:.2f}<br>Sentiment: {s:.3f}<br>UWR: {u:.3f}" 
              for i, (r, s, u) in enumerate(zip(ranges, sentiments, uwrs))],
        hoverinfo='text',
        name='Candles'
    ))
    
    # Add valid region boundary
    boundary_s = np.linspace(-1, 1, 100)
    boundary_u = 1 - np.abs(boundary_s)
    
    fig.add_trace(go.Scatter(
        x=boundary_s,
        y=boundary_u,
        mode='lines',
        line=dict(color='#ffffff', width=2, dash='dash'),
        name='Valid Boundary',
        hoverinfo='skip'
    ))
    
    # Highlight specific candles if requested
    if highlight_indices is not None and len(highlight_indices) > 0:
        highlight_s = [sentiments[i] for i in highlight_indices if i < len(sentiments)]
        highlight_u = [uwrs[i] for i in highlight_indices if i < len(uwrs)]
        
        fig.add_trace(go.Scatter(
            x=highlight_s,
            y=highlight_u,
            mode='markers',
            marker=dict(size=15, color='#ff6b6b', symbol='star', line=dict(color='#ffffff', width=1)),
            name='Highlighted',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(text="Pattern Space with Metric Distortion", font=dict(color='#ffffff')),
        xaxis_title="Sentiment",
        yaxis_title="Upper Wick Ratio",
        xaxis=dict(range=[-1.1, 1.1], gridcolor='#30363d', zerolinecolor='#30363d'),
        yaxis=dict(range=[-0.1, 1.1], gridcolor='#30363d', zerolinecolor='#30363d'),
        height=600,
        hovermode='closest',
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9')
    )
    
    return fig


def plot_geodesic_trajectory(geometry: CurvedCandleGeometry,
                           start_index: int,
                           initial_velocity: np.ndarray,
                           n_steps: int = 20) -> go.Figure:
    """
    Visualize a geodesic trajectory in pattern space.
    """
    # Compute geodesic path
    trajectory = geometry.predict_geodesic_path(start_index, initial_velocity, n_steps)
    
    # Extract coordinates
    traj_s = [p[0] for p in trajectory]
    traj_u = [p[1] for p in trajectory]
    
    # Create figure
    fig = go.Figure()
    
    # Add all candles as background
    sentiments = [c.sentiment for c in geometry.candles]
    uwrs = [c.upper_wick_ratio for c in geometry.candles]
    
    fig.add_trace(go.Scatter(
        x=sentiments,
        y=uwrs,
        mode='markers',
        marker=dict(size=5, color='#30363d', line=dict(color='#30363d', width=0)),
        name='All Candles',
        hoverinfo='skip'
    ))
    
    # Add geodesic trajectory
    fig.add_trace(go.Scatter(
        x=traj_s,
        y=traj_u,
        mode='lines+markers',
        line=dict(color='#00ffff', width=3),
        marker=dict(size=8),
        name='Geodesic Path',
        text=[f"Step {i}" for i in range(len(trajectory))],
        hoverinfo='text+x+y'
    ))
    
    # Mark start and end
    fig.add_trace(go.Scatter(
        x=[traj_s[0]],
        y=[traj_u[0]],
        mode='markers',
        marker=dict(size=15, color='#00ff00', symbol='circle', line=dict(color='#ffffff', width=2)),
        name='Start',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=[traj_s[-1]],
        y=[traj_u[-1]],
        mode='markers',
        marker=dict(size=15, color='#ff0000', symbol='square', line=dict(color='#ffffff', width=2)),
        name='End',
        showlegend=False
    ))
    
    # Add velocity vector at start
    fig.add_annotation(
        x=traj_s[0],
        y=traj_u[0],
        ax=traj_s[0] + initial_velocity[0] * 0.1,
        ay=traj_u[0] + initial_velocity[1] * 0.1,
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=2,
        arrowwidth=2,
        arrowcolor='#00ff00'
    )
    
    # Add boundary
    boundary_s = np.linspace(-1, 1, 100)
    boundary_u = 1 - np.abs(boundary_s)
    
    fig.add_trace(go.Scatter(
        x=boundary_s,
        y=boundary_u,
        mode='lines',
        line=dict(color='#ffffff', width=2, dash='dash'),
        name='Boundary',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(text=f"Geodesic Trajectory from Candle {start_index}", font=dict(color='#ffffff')),
        xaxis_title="Sentiment",
        yaxis_title="Upper Wick Ratio",
        xaxis=dict(range=[-1.1, 1.1], gridcolor='#30363d', zerolinecolor='#30363d'),
        yaxis=dict(range=[-0.1, 1.1], gridcolor='#30363d', zerolinecolor='#30363d'),
        height=600,
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9')
    )
    
    return fig


def plot_metric_field(candle_metrics: List[CandleMetric], 
                     grid_size: int = 20) -> plt.Figure:
    """
    Visualize the metric tensor field as ellipses showing local distortion.
    """
    # Set dark theme
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 10), facecolor='#0d1117')
    ax = fig.add_subplot(111, facecolor='#0d1117')
    
    # Create grid
    s_grid = np.linspace(-0.9, 0.9, grid_size)
    u_grid = np.linspace(0.05, 0.9, grid_size)
    
    # Plot metric ellipses
    for i, s in enumerate(s_grid):
        for j, u in enumerate(u_grid):
            if u <= 1 - abs(s):  # Valid region only
                # Find nearest candle
                distances = [(c.sentiment - s)**2 + (c.upper_wick_ratio - u)**2 
                           for c in candle_metrics]
                nearest_idx = np.argmin(distances)
                
                # Get metric at this point
                g = candle_metrics[nearest_idx].metric_tensor
                
                # Compute eigenvalues and eigenvectors for ellipse
                eigenvalues, eigenvectors = np.linalg.eigh(g)
                
                # Create ellipse
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                width = 2 * np.sqrt(eigenvalues[0]) * 0.05  # Scale for visibility
                height = 2 * np.sqrt(eigenvalues[1]) * 0.05
                
                ellipse = Ellipse((s, u), width, height, angle=angle,
                                facecolor='#00ffff', alpha=0.3, edgecolor='#00cccc')
                ax.add_patch(ellipse)
    
    # Add candle positions
    sentiments = [c.sentiment for c in candle_metrics]
    uwrs = [c.upper_wick_ratio for c in candle_metrics]
    ax.scatter(sentiments, uwrs, c='#ff6b6b', s=20, zorder=5, alpha=0.8)
    
    # Add boundary
    boundary_s = np.linspace(-1, 1, 100)
    boundary_u = 1 - np.abs(boundary_s)
    ax.plot(boundary_s, boundary_u, 'w--', linewidth=2, alpha=0.8)
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('Sentiment', fontsize=12, color='#c9d1d9')
    ax.set_ylabel('Upper Wick Ratio', fontsize=12, color='#c9d1d9')
    ax.set_title('Metric Tensor Field Visualization', fontsize=14, color='#ffffff')
    ax.grid(True, alpha=0.2, color='#30363d')
    
    # Style the axes
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['top'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['right'].set_color('#30363d')
    ax.tick_params(colors='#c9d1d9')
    
    plt.tight_layout()
    return fig


def create_curvature_heatmap(geometry: CurvedCandleGeometry) -> go.Figure:
    """
    Create a heatmap showing how curvature varies across pattern space.
    """
    # Create grid for interpolation
    n_grid = 50
    sentiment_range = np.linspace(-0.9, 0.9, n_grid)
    uwr_range = np.linspace(0.05, 0.9, n_grid)
    
    # Initialize curvature grid
    curvature_grid = np.zeros((n_grid, n_grid))
    
    # Compute curvatures
    curvatures = geometry.compute_curvature_series()
    
    # Map curvatures to grid positions
    for i, candle in enumerate(geometry.candles[2:-2]):
        # Find nearest grid point
        s_idx = np.argmin(np.abs(sentiment_range - candle.sentiment))
        u_idx = np.argmin(np.abs(uwr_range - candle.upper_wick_ratio))
        
        if 0 <= s_idx < n_grid and 0 <= u_idx < n_grid:
            curvature_grid[u_idx, s_idx] = curvatures[i + 2]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        x=sentiment_range,
        y=uwr_range,
        z=curvature_grid,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(
            title=dict(text="Gaussian Curvature", font=dict(color='#c9d1d9')),
            tickfont=dict(color='#c9d1d9'),
            bordercolor='#30363d',
            borderwidth=1
        ),
        hovertemplate='Sentiment: %{x:.3f}<br>UWR: %{y:.3f}<br>Curvature: %{z:.6f}<extra></extra>'
    ))
    
    # Add candle positions
    sentiments = [c.sentiment for c in geometry.candles]
    uwrs = [c.upper_wick_ratio for c in geometry.candles]
    
    fig.add_trace(go.Scatter(
        x=sentiments,
        y=uwrs,
        mode='markers',
        marker=dict(size=3, color='#ffff00', line=dict(color='#000000', width=1)),
        name='Candles',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(text="Market Curvature Heatmap", font=dict(color='#ffffff')),
        xaxis_title="Sentiment",
        yaxis_title="Upper Wick Ratio",
        height=600,
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        xaxis=dict(gridcolor='#30363d', zerolinecolor='#30363d'),
        yaxis=dict(gridcolor='#30363d', zerolinecolor='#30363d')
    )
    
    return fig


def animate_parallel_transport(geometry: CurvedCandleGeometry,
                              initial_vector: np.ndarray,
                              start_index: int = 0,
                              end_index: Optional[int] = None) -> FuncAnimation:
    """
    Animate parallel transport of a vector along the candle series.
    """
    if end_index is None:
        end_index = len(geometry.candles) - 1
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Initialize plots
    sentiments = [c.sentiment for c in geometry.candles]
    uwrs = [c.upper_wick_ratio for c in geometry.candles]
    
    # Pattern space plot
    ax1.scatter(sentiments, uwrs, c='lightgray', s=20, alpha=0.5)
    path_line, = ax1.plot([], [], 'b-', linewidth=2)
    current_point, = ax1.plot([], [], 'ro', markersize=10)
    vector_arrow = None
    
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel('Sentiment')
    ax1.set_ylabel('Upper Wick Ratio')
    ax1.set_title('Parallel Transport in Pattern Space')
    
    # Vector components plot
    component_bars = ax2.bar(['Sentiment', 'UWR'], [0, 0])
    ax2.set_ylim(-2, 2)
    ax2.set_ylabel('Component Value')
    ax2.set_title('Transported Vector Components')
    
    def animate(frame):
        nonlocal vector_arrow
        
        current_idx = start_index + frame
        if current_idx > end_index:
            current_idx = end_index
        
        # Compute transported vector
        transported = geometry.compute_parallel_transport(
            initial_vector, start_index, current_idx
        )
        
        # Update pattern space
        path_x = sentiments[start_index:current_idx+1]
        path_y = uwrs[start_index:current_idx+1]
        path_line.set_data(path_x, path_y)
        
        current_x = sentiments[current_idx]
        current_y = uwrs[current_idx]
        current_point.set_data([current_x], [current_y])
        
        # Remove old arrow
        if vector_arrow:
            vector_arrow.remove()
        
        # Add new arrow
        vector_arrow = ax1.arrow(
            current_x, current_y,
            transported[0] * 0.1, transported[1] * 0.1,
            head_width=0.02, head_length=0.02,
            fc='red', ec='red'
        )
        
        # Update bar chart
        component_bars[0].set_height(transported[0])
        component_bars[1].set_height(transported[1])
        
        return path_line, current_point, component_bars
    
    n_frames = end_index - start_index + 1
    anim = FuncAnimation(fig, animate, frames=n_frames, 
                        interval=100, blit=False, repeat=True)
    
    plt.tight_layout()
    return anim