"""
3D visualization of geodesic trajectories in (Sentiment, UWR, Time) space.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Optional
from curved_candle_geometry import CurvedCandleGeometry, CandleMetric


def create_3d_geodesic_visualization(
    ohlc_data: pd.DataFrame,
    geometry: CurvedCandleGeometry,
    start_index: int = 5,
    title: str = "3D Geodesic Trajectory: Pattern Space Evolution Through Time"
) -> go.Figure:
    """
    Create a 3D visualization showing the geodesic path through 
    (Sentiment, Upper Wick Ratio, Time) space.
    """
    
    # Get the historical trajectory
    trajectory = geometry.get_historical_geodesic(start_index, None)
    
    # Extract coordinates
    sentiments = [p[0] for p in trajectory]
    uwrs = [p[1] for p in trajectory]
    time_indices = list(range(start_index, start_index + len(trajectory)))
    
    # Normalize time for better visualization (0 to 1)
    time_normalized = [(t - start_index) / (len(trajectory) - 1) for t in time_indices]
    
    # Get curvature values for coloring
    curvatures = geometry.compute_curvature_series()
    traj_curvatures = curvatures[start_index:start_index + len(trajectory)]
    
    # Create the main 3D plot
    fig = go.Figure()
    
    # Add the main geodesic trajectory (Time, UWR, Sentiment)
    fig.add_trace(go.Scatter3d(
        x=time_normalized,
        y=uwrs, 
        z=sentiments,
        mode='lines+markers',
        line=dict(
            color=traj_curvatures,
            colorscale='RdBu',
            width=6,
            cmid=0,
            colorbar=dict(
                title=dict(text="Gaussian Curvature", font=dict(color='#c9d1d9')),
                tickfont=dict(color='#c9d1d9'),
                x=1.02
            )
        ),
        marker=dict(
            size=4,
            color=traj_curvatures,
            colorscale='RdBu',
            cmid=0,
            showscale=False
        ),
        name='Geodesic Path',
        hovertemplate='<b>Candle %{customdata}</b><br>' +
                     'Time Step: %{x:.3f}<br>' +
                     'Upper Wick Ratio: %{y:.3f}<br>' +
                     'Sentiment: %{z:.3f}<br>' +
                     'Curvature: %{marker.color:.4f}<extra></extra>',
        customdata=time_indices
    ))
    
    # Add start and end markers
    fig.add_trace(go.Scatter3d(
        x=[time_normalized[0]],
        y=[uwrs[0]],
        z=[sentiments[0]],
        mode='markers',
        marker=dict(size=12, color='#00ff00', symbol='diamond'),
        name='Start',
        hovertemplate='<b>START</b><br>Candle: %{customdata}<br>' +
                     'Time: %{x:.3f}<br>UWR: %{y:.3f}<br>Sentiment: %{z:.3f}<extra></extra>',
        customdata=[time_indices[0]]
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[time_normalized[-1]],
        y=[uwrs[-1]],
        z=[sentiments[-1]],
        mode='markers',
        marker=dict(size=12, color='#ff0000', symbol='square'),
        name='End',
        hovertemplate='<b>END</b><br>Candle: %{customdata}<br>' +
                     'Time: %{x:.3f}<br>UWR: %{y:.3f}<br>Sentiment: %{z:.3f}<extra></extra>',
        customdata=[time_indices[-1]]
    ))
    
    # Add the triangular boundary constraint as a surface
    # Create boundary at different time levels
    n_time_levels = 20
    time_levels = np.linspace(0, 1, n_time_levels)
    
    # Create boundary coordinates
    boundary_sentiment = np.linspace(-1, 1, 100)
    boundary_uwr = 1 - np.abs(boundary_sentiment)
    
    # Create meshgrid for the boundary surface (Time, UWR, Sentiment)
    S_boundary, T_boundary = np.meshgrid(boundary_sentiment, time_levels)
    U_boundary = np.tile(boundary_uwr, (n_time_levels, 1))
    
    # Add boundary surface (swap x and z)
    fig.add_trace(go.Surface(
        x=T_boundary,
        y=U_boundary,
        z=S_boundary,
        opacity=0.1,
        colorscale=[[0, '#30363d'], [1, '#30363d']],
        showscale=False,
        name='Boundary Constraint',
        hoverinfo='skip'
    ))
    
    # Add floor grid (time = 0 plane)
    sentiment_grid = np.linspace(-1.1, 1.1, 20)
    uwr_grid = np.linspace(-0.1, 1.1, 20)
    S_grid, U_grid = np.meshgrid(sentiment_grid, uwr_grid)
    T_grid = np.zeros_like(S_grid)  # Time = 0
    
    fig.add_trace(go.Surface(
        x=T_grid,
        y=U_grid,
        z=S_grid,
        opacity=0.05,
        colorscale=[[0, '#161b22'], [1, '#161b22']],
        showscale=False,
        name='Base Plane',
        hoverinfo='skip'
    ))
    
    # Update layout for dark theme
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#ffffff')),
        scene=dict(
            xaxis=dict(
                title='Time (Normalized)',
                gridcolor='#30363d',
                zerolinecolor='#30363d',
                backgroundcolor='#0d1117',
                color='#c9d1d9',
                range=[0, 1.1]
            ),
            yaxis=dict(
                title='Upper Wick Ratio',
                gridcolor='#30363d',
                zerolinecolor='#30363d',
                backgroundcolor='#0d1117',
                color='#c9d1d9',
                range=[-0.1, 1.1]
            ),
            zaxis=dict(
                title='Sentiment',
                gridcolor='#30363d',
                zerolinecolor='#30363d',
                backgroundcolor='#0d1117',
                color='#c9d1d9',
                range=[-1.1, 1.1]
            ),
            bgcolor='#0d1117',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0)
            )
        ),
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        legend=dict(
            bgcolor='#161b22',
            bordercolor='#30363d',
            borderwidth=1,
            font=dict(color='#c9d1d9')
        ),
        height=700
    )
    
    return fig


def create_3d_multiple_trajectories(
    ohlc_data: pd.DataFrame,
    geometry: CurvedCandleGeometry,
    trajectories: List[tuple],
    title: str = "3D Multiple Geodesic Trajectories"
) -> go.Figure:
    """
    Create 3D visualization with multiple trajectory paths.
    
    trajectories: List of (start_index, label, color) tuples
    """
    fig = go.Figure()
    
    for start_idx, label, color in trajectories:
        # Get trajectory
        trajectory = geometry.get_historical_geodesic(start_idx, None)
        
        if len(trajectory) < 2:
            continue
            
        # Extract coordinates
        sentiments = [p[0] for p in trajectory]
        uwrs = [p[1] for p in trajectory]
        time_indices = list(range(start_idx, start_idx + len(trajectory)))
        time_normalized = [(t - start_idx) / max(1, len(trajectory) - 1) for t in time_indices]
        
        # Add trajectory
        fig.add_trace(go.Scatter3d(
            x=sentiments,
            y=uwrs,
            z=time_normalized,
            mode='lines+markers',
            line=dict(color=color, width=5),
            marker=dict(size=3, color=color),
            name=label,
            hovertemplate=f'<b>{label}</b><br>' +
                         'Sentiment: %{x:.3f}<br>' +
                         'UWR: %{y:.3f}<br>' +
                         'Time: %{z:.3f}<extra></extra>'
        ))
        
        # Add start marker
        fig.add_trace(go.Scatter3d(
            x=[sentiments[0]],
            y=[uwrs[0]],
            z=[time_normalized[0]],
            mode='markers',
            marker=dict(size=10, color=color, symbol='diamond'),
            showlegend=False,
            hovertemplate=f'<b>{label} START</b><br>' +
                         'Sentiment: %{x:.3f}<br>UWR: %{y:.3f}<extra></extra>'
        ))
    
    # Add boundary constraint surface
    n_time_levels = 15
    time_levels = np.linspace(0, 1, n_time_levels)
    boundary_sentiment = np.linspace(-1, 1, 50)
    boundary_uwr = 1 - np.abs(boundary_sentiment)
    
    S_boundary, T_boundary = np.meshgrid(boundary_sentiment, time_levels)
    U_boundary = np.tile(boundary_uwr, (n_time_levels, 1))
    
    fig.add_trace(go.Surface(
        x=S_boundary,
        y=U_boundary,
        z=T_boundary,
        opacity=0.15,
        colorscale=[[0, '#ffffff'], [1, '#ffffff']],
        showscale=False,
        name='Valid Region',
        hoverinfo='skip'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#ffffff')),
        scene=dict(
            xaxis=dict(
                title='Sentiment',
                gridcolor='#30363d',
                backgroundcolor='#0d1117',
                color='#c9d1d9',
                range=[-1.1, 1.1]
            ),
            yaxis=dict(
                title='Upper Wick Ratio',
                gridcolor='#30363d',
                backgroundcolor='#0d1117',
                color='#c9d1d9',
                range=[-0.1, 1.1]
            ),
            zaxis=dict(
                title='Time (Normalized)',
                gridcolor='#30363d',
                backgroundcolor='#0d1117',
                color='#c9d1d9',
                range=[0, 1.1]
            ),
            bgcolor='#0d1117',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.0),
                center=dict(x=0, y=0, z=0.5)
            )
        ),
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        legend=dict(
            bgcolor='#161b22',
            bordercolor='#30363d',
            borderwidth=1
        ),
        height=700
    )
    
    return fig


def create_3d_curvature_field(
    geometry: CurvedCandleGeometry,
    title: str = "3D Market Curvature Field"
) -> go.Figure:
    """
    Create a 3D visualization showing how curvature varies through space and time.
    """
    # Get curvatures
    curvatures = geometry.compute_curvature_series()
    
    # Sample points from the trajectory
    sample_indices = range(5, len(geometry.candles) - 5, 5)  # Every 5th candle
    
    sentiments = []
    uwrs = []
    times = []
    curvs = []
    
    for i in sample_indices:
        sentiments.append(geometry.candles[i].sentiment)
        uwrs.append(geometry.candles[i].upper_wick_ratio)
        times.append((i - 5) / (len(geometry.candles) - 10))  # Normalized time
        curvs.append(curvatures[i])
    
    fig = go.Figure()
    
    # Add curvature points as 3D scatter
    fig.add_trace(go.Scatter3d(
        x=sentiments,
        y=uwrs,
        z=times,
        mode='markers',
        marker=dict(
            size=8,
            color=curvs,
            colorscale='RdBu',
            cmid=0,
            colorbar=dict(
                title="Curvature",
                titlefont=dict(color='#c9d1d9'),
                tickfont=dict(color='#c9d1d9')
            ),
            line=dict(color='#ffffff', width=1)
        ),
        name='Curvature Field',
        hovertemplate='Sentiment: %{x:.3f}<br>' +
                     'UWR: %{y:.3f}<br>' +
                     'Time: %{z:.3f}<br>' +
                     'Curvature: %{marker.color:.4f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#ffffff')),
        scene=dict(
            xaxis=dict(title='Sentiment', gridcolor='#30363d', color='#c9d1d9'),
            yaxis=dict(title='Upper Wick Ratio', gridcolor='#30363d', color='#c9d1d9'),
            zaxis=dict(title='Time', gridcolor='#30363d', color='#c9d1d9'),
            bgcolor='#0d1117'
        ),
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        height=600
    )
    
    return fig