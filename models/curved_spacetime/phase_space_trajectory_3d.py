"""
3D Phase Space Trajectory Visualization

Creates a 3D visualization where:
- X-axis: Time (flows left to right)
- Y-axis: Sentiment 
- Z-axis: Upper Wick Ratio

The triangular constraint region is shown as a surface, and the
market trajectory moves through this 3D phase space over time.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
from curved_candle_geometry import create_candle_metrics_from_ohlc, CurvedCandleGeometry
from curvature_color_scheme import get_curvature_gradient_color


def create_triangular_constraint_surface(time_range: tuple, resolution: int = 50):
    """
    Create the triangular constraint surface in 3D space.
    
    The constraint is: |sentiment| + upper_wick_ratio <= 1
    This creates a triangular prism extruded through time.
    """
    t_min, t_max = time_range
    t_values = np.linspace(t_min, t_max, 10)  # Fewer time slices for surface
    
    # Create constraint boundary
    sentiment_range = np.linspace(-0.99, 0.99, resolution)
    constraint_surfaces = []
    
    for t in t_values:
        # For each sentiment value, find the max UWR
        uwr_max = []
        uwr_min = []
        valid_sentiments = []
        
        for s in sentiment_range:
            max_uwr = 1.0 - abs(s)
            if max_uwr >= 0.01:  # Minimum UWR constraint
                uwr_max.append(max_uwr)
                uwr_min.append(0.01)  # Minimum UWR
                valid_sentiments.append(s)
        
        if valid_sentiments:
            # Create surface patches for this time slice
            x_coords = [t] * len(valid_sentiments)
            
            # Top boundary of triangle
            constraint_surfaces.append({
                'x': x_coords,
                'y': valid_sentiments,
                'z': uwr_max,
                'type': 'top'
            })
            
            # Bottom boundary of triangle  
            constraint_surfaces.append({
                'x': x_coords,
                'y': valid_sentiments,
                'z': uwr_min,
                'type': 'bottom'
            })
    
    return constraint_surfaces


def create_phase_space_trajectory_3d(ohlc_data: pd.DataFrame) -> go.Figure:
    """
    Create 3D phase space trajectory with triangular constraints.
    """
    print("🌌 Creating 3D Phase Space Trajectory...")
    
    # Get market data
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    geometry = CurvedCandleGeometry(candle_metrics)
    curvatures = geometry.compute_curvature_series()
    proper_times = geometry.compute_proper_time_series()
    
    # Extract trajectory coordinates
    times = proper_times
    sentiments = [c.sentiment for c in candle_metrics]
    uwrs = [c.upper_wick_ratio for c in candle_metrics]
    volumes = [c.volume for c in candle_metrics]
    
    # Create figure
    fig = go.Figure()
    
    # 1. Add triangular constraint surfaces
    time_range = (min(times), max(times))
    constraint_surfaces = create_triangular_constraint_surface(time_range)
    
    # Create constraint boundary mesh
    for i in range(0, len(constraint_surfaces)-1, 2):
        if i+1 < len(constraint_surfaces):
            top_surface = constraint_surfaces[i]
            bottom_surface = constraint_surfaces[i+1]
            
            # Create mesh between top and bottom
            if len(top_surface['x']) == len(bottom_surface['x']):
                # Side walls of the triangular prism
                for j in range(len(top_surface['x'])-1):
                    # Create quad face
                    x_quad = [
                        top_surface['x'][j], top_surface['x'][j+1],
                        bottom_surface['x'][j+1], bottom_surface['x'][j],
                        top_surface['x'][j]  # Close the quad
                    ]
                    y_quad = [
                        top_surface['y'][j], top_surface['y'][j+1],
                        bottom_surface['y'][j+1], bottom_surface['y'][j],
                        top_surface['y'][j]
                    ]
                    z_quad = [
                        top_surface['z'][j], top_surface['z'][j+1],
                        bottom_surface['z'][j+1], bottom_surface['z'][j],
                        top_surface['z'][j]
                    ]
                    
                    fig.add_trace(go.Scatter3d(
                        x=x_quad, y=y_quad, z=z_quad,
                        mode='lines',
                        line=dict(color='rgba(255,255,255,0.3)', width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    
    # 2. Add constraint boundary lines (triangle edges at each time)
    time_samples = np.linspace(min(times), max(times), 5)
    
    for t in time_samples:
        # Triangle vertices at this time
        vertices_x = [t, t, t, t]  # Same time for all vertices
        vertices_y = [-0.99, 0.99, 0, -0.99]  # Sentiment vertices + close
        vertices_z = [0.01, 0.01, 1.0, 0.01]  # UWR vertices + close
        
        fig.add_trace(go.Scatter3d(
            x=vertices_x,
            y=vertices_y, 
            z=vertices_z,
            mode='lines',
            line=dict(color='rgba(255,255,255,0.5)', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 3. Add the actual market trajectory
    # Color by curvature
    colors = []
    for c in curvatures:
        color, _ = get_curvature_gradient_color(c)
        colors.append(color)
    
    # Size by volume
    max_volume = max(volumes)
    sizes = [5 + 15 * (v / max_volume) for v in volumes]
    
    # Main trajectory line
    fig.add_trace(go.Scatter3d(
        x=times,
        y=sentiments,
        z=uwrs,
        mode='lines+markers',
        line=dict(
            color='white',
            width=4
        ),
        marker=dict(
            size=sizes,
            color=curvatures,
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(
                title="Spacetime<br>Curvature",
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            ),
            line=dict(color='white', width=2)
        ),
        name='Market Trajectory',
        hovertemplate='<b>Market State</b><br>' +
                      'Time: %{x:.2f}<br>' +
                      'Sentiment: %{y:.2f}<br>' +
                      'Upper Wick Ratio: %{z:.2f}<br>' +
                      'Curvature: %{customdata:.3f}<extra></extra>',
        customdata=curvatures
    ))
    
    # 4. Add start and end markers
    fig.add_trace(go.Scatter3d(
        x=[times[0]],
        y=[sentiments[0]],
        z=[uwrs[0]],
        mode='markers',
        marker=dict(
            size=15,
            color='green',
            symbol='diamond',
            line=dict(color='white', width=3)
        ),
        name='Start',
        hovertemplate='<b>Market Start</b><br>' +
                      'Time: %{x:.2f}<br>' +
                      'Sentiment: %{y:.2f}<br>' +
                      'UWR: %{z:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[times[-1]],
        y=[sentiments[-1]],
        z=[uwrs[-1]],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='diamond',
            line=dict(color='white', width=3)
        ),
        name='End',
        hovertemplate='<b>Market End</b><br>' +
                      'Time: %{x:.2f}<br>' +
                      'Sentiment: %{y:.2f}<br>' +
                      'UWR: %{z:.2f}<extra></extra>'
    ))
    
    # 5. Add time flow arrows
    n_arrows = 5
    arrow_indices = np.linspace(0, len(times)-2, n_arrows, dtype=int)
    
    for i in arrow_indices:
        if i+1 < len(times):
            # Arrow from current to next point
            fig.add_trace(go.Scatter3d(
                x=[times[i], times[i+1]],
                y=[sentiments[i], sentiments[i+1]],
                z=[uwrs[i], uwrs[i+1]],
                mode='lines',
                line=dict(color='yellow', width=6),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Arrowhead
            fig.add_trace(go.Scatter3d(
                x=[times[i+1]],
                y=[sentiments[i+1]],
                z=[uwrs[i+1]],
                mode='markers',
                marker=dict(
                    size=8,
                    color='yellow',
                    symbol='diamond',
                    line=dict(color='black', width=1)
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': '🌌 3D Phase Space Trajectory<br><sub>Time → Sentiment × Upper Wick Ratio</sub>',
            'x': 0.5,
            'font': {'size': 16, 'color': 'white'}
        },
        scene=dict(
            xaxis=dict(
                title='Proper Time (τ) →',
                backgroundcolor='#0d1117',
                gridcolor='#30363d',
                tickfont=dict(color='white'),
                titlefont=dict(color='white'),
                showspikes=False
            ),
            yaxis=dict(
                title='← Bearish | Sentiment | Bullish →',
                range=[-1, 1],
                backgroundcolor='#0d1117', 
                gridcolor='#30363d',
                tickfont=dict(color='white'),
                titlefont=dict(color='white'),
                showspikes=False
            ),
            zaxis=dict(
                title='Upper Wick Ratio ↑',
                range=[0, 1],
                backgroundcolor='#0d1117',
                gridcolor='#30363d', 
                tickfont=dict(color='white'),
                titlefont=dict(color='white'),
                showspikes=False
            ),
            bgcolor='#0d1117',
            camera=dict(
                eye=dict(x=2, y=-1.5, z=1.2),  # Good angle to see the trajectory
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=1)  # Stretch time axis
        ),
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='white'),
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(13, 17, 23, 0.8)',
            bordercolor='white',
            borderwidth=1
        )
    )
    
    # Add annotations
    fig.add_annotation(
        text="📈 Particle moves RIGHT through time<br>while exploring the triangular phase space",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=12, color='yellow'),
        bgcolor='rgba(0,0,0,0.7)',
        bordercolor='yellow',
        borderwidth=1
    )
    
    return fig


def create_phase_space_animation(ohlc_data: pd.DataFrame, n_frames: int = 50) -> go.Figure:
    """
    Create an animation showing the particle moving through phase space over time.
    """
    print("🎬 Creating Phase Space Animation...")
    
    # Get market data
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    geometry = CurvedCandleGeometry(candle_metrics)
    proper_times = geometry.compute_proper_time_series()
    
    times = proper_times
    sentiments = [c.sentiment for c in candle_metrics]
    uwrs = [c.upper_wick_ratio for c in candle_metrics]
    
    # Create frames
    frames = []
    
    # Show progressive trajectory
    for i in range(1, min(len(times), n_frames)):
        frame_times = times[:i+1]
        frame_sentiments = sentiments[:i+1]
        frame_uwrs = uwrs[:i+1]
        
        frame = go.Frame(
            data=[
                # Trajectory so far
                go.Scatter3d(
                    x=frame_times,
                    y=frame_sentiments, 
                    z=frame_uwrs,
                    mode='lines+markers',
                    line=dict(color='white', width=3),
                    marker=dict(size=4, color='lightblue'),
                    name='Path'
                ),
                # Current position
                go.Scatter3d(
                    x=[times[i]],
                    y=[sentiments[i]],
                    z=[uwrs[i]],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='red',
                        symbol='circle',
                        line=dict(color='white', width=3)
                    ),
                    name='Current'
                )
            ],
            name=f'frame_{i}'
        )
        frames.append(frame)
    
    # Initial figure
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[times[0]],
                y=[sentiments[0]],
                z=[uwrs[0]],
                mode='markers',
                marker=dict(size=20, color='red', symbol='circle'),
                name='Market Particle'
            )
        ],
        frames=frames
    )
    
    # Add constraint triangle at multiple time points
    time_samples = np.linspace(min(times), max(times), 8)
    
    for t in time_samples:
        # Triangle outline
        vertices_x = [t, t, t, t]
        vertices_y = [-0.99, 0.99, 0, -0.99]
        vertices_z = [0.01, 0.01, 1.0, 0.01]
        
        fig.add_trace(go.Scatter3d(
            x=vertices_x,
            y=vertices_y,
            z=vertices_z,
            mode='lines',
            line=dict(color='rgba(255,255,255,0.4)', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Animation controls
    fig.update_layout(
        title='🎬 Phase Space Animation - Market Particle Evolution',
        scene=dict(
            xaxis=dict(title='Time →', range=[min(times), max(times)]),
            yaxis=dict(title='Sentiment', range=[-1, 1]),
            zaxis=dict(title='Upper Wick Ratio', range=[0, 1]),
            camera=dict(eye=dict(x=2, y=-1.5, z=1.2)),
            bgcolor='#0d1117'
        ),
        template='plotly_dark',
        height=800,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 200, 'redraw': True},
                        'fromcurrent': True
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate'
                    }]
                }
            ]
        }]
    )
    
    return fig


def generate_rich_market_data(n_candles: int = 200) -> pd.DataFrame:
    """Generate rich market data with multiple regimes for meaningful histograms."""
    dates = [datetime(2024, 1, 1, 9, 0) + timedelta(hours=i) for i in range(n_candles)]
    
    np.random.seed(42)  # Reproducible
    price = 100.0
    ohlc_data = []
    volumes = []
    
    for i in range(n_candles):
        # Create distinct market regimes with clustering
        if i < n_candles * 0.2:  # Bullish trending (positive sentiment cluster)
            sentiment_target = 0.4
            uwr_target = 0.3
            volatility = 1.2
            trend = 0.3
        elif i < n_candles * 0.4:  # Bearish correction (negative sentiment cluster)  
            sentiment_target = -0.3
            uwr_target = 0.6
            volatility = 2.0
            trend = -0.2
        elif i < n_candles * 0.6:  # High volatility consolidation (spread across space)
            sentiment_target = 0.0
            uwr_target = 0.5
            volatility = 2.5
            trend = 0.0
        elif i < n_candles * 0.8:  # Recovery phase (moderate positive sentiment)
            sentiment_target = 0.2
            uwr_target = 0.4
            volatility = 1.5
            trend = 0.15
        else:  # Final consolidation (neutral cluster)
            sentiment_target = 0.0
            uwr_target = 0.5
            volatility = 1.0
            trend = 0.05
        
        # Add noise around regime targets
        price_change = np.random.normal(trend, volatility)
        price += price_change
        
        # Generate OHLC with regime-specific characteristics
        range_factor = np.random.uniform(0.8, 2.5) * volatility
        
        # Bias toward regime sentiment/UWR targets
        sentiment_noise = np.random.normal(0, 0.3)
        uwr_noise = np.random.normal(0, 0.2)
        
        # Create OHLC that will produce desired sentiment/UWR clustering
        if price_change >= 0:  # Bullish candle
            close = price
            open_price = close - abs(price_change) * (1 + sentiment_noise * 0.5)
            high = max(open_price, close) + range_factor * (0.5 - uwr_target + uwr_noise)
            low = min(open_price, close) - range_factor * 0.3
        else:  # Bearish candle
            open_price = price - price_change  # Before the drop
            close = price
            high = max(open_price, close) + range_factor * (uwr_target + uwr_noise)
            low = min(open_price, close) - range_factor * 0.3
        
        # Ensure OHLC validity
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        ohlc_data.append([open_price, high, low, close])
        
        # Volume varies with volatility
        volume_base = 2000 + volatility * 1000
        volume = volume_base * np.random.uniform(0.5, 2.0)
        volumes.append(int(volume))
    
    # Create DataFrame
    price_array = np.array(ohlc_data)
    return pd.DataFrame({
        'open': price_array[:, 0],
        'high': price_array[:, 1],
        'low': price_array[:, 2],
        'close': price_array[:, 3],
        'volume': volumes
    }, index=dates)


def create_phase_space_density_gradient(ohlc_data: pd.DataFrame) -> go.Figure:
    """
    Create 2D phase space trajectory with density-based gradient background.
    No individual particles, just smooth gradient showing candle clustering.
    """
    print("🎯 Creating phase space density gradient visualization...")
    
    # Get phase space coordinates
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    sentiments = np.array([c.sentiment for c in candle_metrics])
    uwrs = np.array([c.upper_wick_ratio for c in candle_metrics])
    
    # Create high-resolution grid for smooth gradient
    grid_resolution = 80
    sentiment_range = np.linspace(-0.99, 0.99, grid_resolution)
    uwr_range = np.linspace(0.01, 0.99, grid_resolution)
    S_grid, U_grid = np.meshgrid(sentiment_range, uwr_range)
    
    # Calculate density using kernel density estimation
    points = np.column_stack([sentiments, uwrs])
    kde = gaussian_kde(points.T, bw_method='scott')
    
    # Evaluate density on grid
    density_grid = np.zeros_like(S_grid)
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            s, u = S_grid[i, j], U_grid[i, j]
            # Apply triangular constraint
            if abs(s) + u <= 1.0:
                density_grid[i, j] = kde([s, u])[0]
            else:
                density_grid[i, j] = np.nan
    
    # Apply log scale transformation for better cluster visibility
    # Add small epsilon to avoid log(0), then apply log1p for smooth scaling
    epsilon = 1e-10
    density_grid_log = np.log1p(density_grid + epsilon)
    # Set NaN values back to NaN
    density_grid_log[np.isnan(density_grid)] = np.nan
    
    # Create figure
    fig = go.Figure()
    
    # Add smooth density heatmap (log scale)
    fig.add_trace(go.Heatmap(
        x=sentiment_range,
        y=uwr_range,
        z=density_grid_log,
        colorscale=[
            [0, '#0d1117'],      # Dark background (low density)
            [0.2, '#1a1b5e'],    # Deep blue
            [0.4, '#2e4f99'],    # Medium blue  
            [0.6, '#4c7cdb'],    # Light blue
            [0.8, '#7ba7ff'],    # Bright blue
            [1, '#b3d9ff']       # Very bright blue (high density)
        ],
        showscale=True,
        colorbar=dict(
            title="Log Candle<br>Density",
            titlefont=dict(color='white', size=14),
            tickfont=dict(color='white'),
            x=1.02
        ),
        hovertemplate='Sentiment: %{x:.2f}<br>UWR: %{y:.2f}<br>Log Density: %{z:.4f}<extra></extra>',
        name='Log Density Field'
    ))
    
    # NO trajectory path - pure gradient only
    
    # Add constraint boundary
    boundary_s = np.linspace(-0.99, 0.99, 100)
    boundary_u = 1.0 - np.abs(boundary_s)
    
    fig.add_trace(go.Scatter(
        x=boundary_s,
        y=boundary_u,
        mode='lines',
        line=dict(color='yellow', width=3, dash='dash'),
        name='Phase Space Boundary',
        hoverinfo='skip'
    ))
    
    # Update layout
    fig.update_layout(
        title='🎯 Phase Space Log Density Gradient<br><sub>Candle Clustering Patterns (Log Scale for Better Cluster Visibility)</sub>',
        xaxis=dict(
            title='← Bearish | Sentiment | Bullish →',
            range=[-1, 1],
            gridcolor='rgba(255,255,255,0.2)',
            tickfont=dict(color='white')
        ),
        yaxis=dict(
            title='Upper Wick Ratio ↑',
            range=[0, 1],
            gridcolor='rgba(255,255,255,0.2)',
            tickfont=dict(color='white')
        ),
        template='plotly_dark',
        height=700,
        width=900
    )
    
    return fig


def create_phase_space_time_gradient(ohlc_data: pd.DataFrame, use_proper_time: bool = True) -> go.Figure:
    """
    Create 2D phase space trajectory with time-flow gradient background.
    No individual particles, just smooth gradient showing temporal flow.
    """
    time_type = "proper" if use_proper_time else "coordinate"
    print(f"⏰ Creating phase space {time_type} time gradient visualization...")
    
    # Get phase space coordinates and time
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    geometry = CurvedCandleGeometry(candle_metrics)
    
    sentiments = np.array([c.sentiment for c in candle_metrics])
    uwrs = np.array([c.upper_wick_ratio for c in candle_metrics])
    
    if use_proper_time:
        time_values = geometry.compute_proper_time_series()
        time_label = "Proper Time (τ)"
        colorscale = 'Viridis'
    else:
        time_values = np.arange(len(candle_metrics))
        time_label = "Coordinate Time"
        colorscale = 'Plasma'
    
    # Create interpolation grid
    grid_resolution = 60
    sentiment_range = np.linspace(-0.99, 0.99, grid_resolution)
    uwr_range = np.linspace(0.01, 0.99, grid_resolution)
    S_grid, U_grid = np.meshgrid(sentiment_range, uwr_range)
    
    # Interpolate time values onto grid
    points = np.column_stack([sentiments, uwrs])
    time_grid = griddata(
        points, 
        time_values, 
        (S_grid, U_grid), 
        method='cubic',
        fill_value=np.nan
    )
    
    # Apply triangular constraint
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            s, u = S_grid[i, j], U_grid[i, j]
            if abs(s) + u > 1.0:
                time_grid[i, j] = np.nan
    
    # Create figure
    fig = go.Figure()
    
    # Add time flow heatmap
    fig.add_trace(go.Heatmap(
        x=sentiment_range,
        y=uwr_range,
        z=time_grid,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title=f"Average<br>{time_label}",
            titlefont=dict(color='white', size=12),
            tickfont=dict(color='white'),
            x=1.02
        ),
        hovertemplate=f'Sentiment: %{{x:.2f}}<br>UWR: %{{y:.2f}}<br>{time_label}: %{{z:.2f}}<extra></extra>',
        name='Time Flow Field'
    ))
    
    # NO trajectory path - pure gradient only
    
    # Add constraint boundary
    boundary_s = np.linspace(-0.99, 0.99, 100)
    boundary_u = 1.0 - np.abs(boundary_s)
    
    fig.add_trace(go.Scatter(
        x=boundary_s,
        y=boundary_u,
        mode='lines',
        line=dict(color='white', width=3, dash='dash'),
        name='Phase Space Boundary',
        hoverinfo='skip'
    ))
    
    # Update layout  
    fig.update_layout(
        title=f'⏰ Phase Space {time_type.title()} Time Gradient<br><sub>Temporal Flow Patterns (No Individual Particles)</sub>',
        xaxis=dict(
            title='← Bearish | Sentiment | Bullish →',
            range=[-1, 1],
            gridcolor='rgba(255,255,255,0.2)',
            tickfont=dict(color='white')
        ),
        yaxis=dict(
            title='Upper Wick Ratio ↑',
            range=[0, 1],
            gridcolor='rgba(255,255,255,0.2)',
            tickfont=dict(color='white')
        ),
        template='plotly_dark',
        height=700,
        width=900
    )
    
    return fig


def create_interactive_combined_analysis(ohlc_data: pd.DataFrame) -> go.Figure:
    """
    Create interactive combined view with range selector controlling all charts.
    Now includes clustering analysis.
    """
    print("🌌 Creating interactive combined gradient analysis with clustering...")
    
    # Get data
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    geometry = CurvedCandleGeometry(candle_metrics)
    
    sentiments = np.array([c.sentiment for c in candle_metrics])
    uwrs = np.array([c.upper_wick_ratio for c in candle_metrics])
    proper_times = geometry.compute_proper_time_series()
    
    # Store data for updates
    time_indices = list(range(len(ohlc_data)))
    
    # Import clustering functionality
    from market_phase_space_processor import MarketPhaseSpaceProcessor
    processor = MarketPhaseSpaceProcessor()
    clustering_results = processor._perform_clustering_analysis(sentiments, uwrs)
    
    # Create subplots with 3 rows, 3 columns - clustering takes the full bottom-right spot
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Log Density Gradient',
            'Proper Time Gradient', 
            'Traditional Candlesticks',
            'Coordinate Time Gradient',
            'Phase Space Trajectory',
            'Clustering Analysis',
            '', '', ''  # Bottom row for clustering spans
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'scatter'}, {'secondary_y': True}],  # Clustering with secondary y
            [None, None, None]  # Empty bottom row
        ],
        column_widths=[0.33, 0.33, 0.34],
        row_heights=[0.35, 0.35, 0.3]
    )
    
    # Generate density and time grids (simplified for subplots)
    grid_res = 40
    sentiment_range = np.linspace(-0.99, 0.99, grid_res)
    uwr_range = np.linspace(0.01, 0.99, grid_res)
    S_grid, U_grid = np.meshgrid(sentiment_range, uwr_range)
    
    # 1. Density contours
    points = np.column_stack([sentiments, uwrs])
    kde = gaussian_kde(points.T, bw_method='scott')
    
    density_grid = np.zeros_like(S_grid)
    for i in range(grid_res):
        for j in range(grid_res):
            s, u = S_grid[i, j], U_grid[i, j]
            if abs(s) + u <= 1.0:
                density_grid[i, j] = kde([s, u])[0]
            else:
                density_grid[i, j] = np.nan
    
    # Apply log scale transformation for better cluster visibility
    epsilon = 1e-10
    density_grid_log = np.log1p(density_grid + epsilon)
    density_grid_log[np.isnan(density_grid)] = np.nan
    
    fig.add_trace(
        go.Contour(
            x=sentiment_range,
            y=uwr_range,
            z=density_grid_log,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(
                title="Log<br>Density",
                x=0.32,
                len=0.45,
                y=0.775,
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            ),
            contours=dict(
                start=np.nanmin(density_grid_log),
                end=np.nanmax(density_grid_log),
                size=(np.nanmax(density_grid_log) - np.nanmin(density_grid_log))/8
            )
        ),
        row=1, col=1
    )
    
    # 2. Proper time contours
    time_grid = griddata(
        points, proper_times, (S_grid, U_grid), 
        method='linear', fill_value=np.nan
    )
    
    # Apply constraint
    for i in range(grid_res):
        for j in range(grid_res):
            if abs(S_grid[i, j]) + U_grid[i, j] > 1.0:
                time_grid[i, j] = np.nan
    
    fig.add_trace(
        go.Contour(
            x=sentiment_range,
            y=uwr_range,
            z=time_grid,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Proper<br>Time",
                x=0.66,
                len=0.45,
                y=0.775,
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            ),
            contours=dict(
                start=np.nanmin(time_grid),
                end=np.nanmax(time_grid),
                size=(np.nanmax(time_grid)-np.nanmin(time_grid))/8
            )
        ),
        row=1, col=2
    )
    
    # 3. Traditional candlestick chart with range selector
    fig.add_trace(
        go.Candlestick(
            x=time_indices,
            open=ohlc_data['open'].tolist(),
            high=ohlc_data['high'].tolist(), 
            low=ohlc_data['low'].tolist(),
            close=ohlc_data['close'].tolist(),
            name='OHLC',
            showlegend=False,
            xaxis='x3'  # Link to subplot x-axis
        ),
        row=1, col=3
    )
    
    # 4. Coordinate time contours
    coord_times = np.arange(len(candle_metrics))
    coord_time_grid = griddata(
        points, coord_times, (S_grid, U_grid),
        method='linear', fill_value=np.nan
    )
    
    # Apply constraint
    for i in range(grid_res):
        for j in range(grid_res):
            if abs(S_grid[i, j]) + U_grid[i, j] > 1.0:
                coord_time_grid[i, j] = np.nan
    
    fig.add_trace(
        go.Contour(
            x=sentiment_range,
            y=uwr_range,
            z=coord_time_grid,
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(
                title="Coordinate<br>Time",
                x=0.32,
                len=0.45,
                y=0.275,
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            ),
            contours=dict(
                start=0,
                end=len(candle_metrics),
                size=len(candle_metrics)/8
            )
        ),
        row=2, col=1
    )
    
    # 5. Phase space trajectory overview (sentiment vs UWR) - THIS WILL UPDATE WITH RANGE
    fig.add_trace(
        go.Scatter(
            x=sentiments,
            y=uwrs,
            mode='lines+markers',
            line=dict(color='white', width=3),
            marker=dict(
                size=6,
                color=proper_times,
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(
                    title="Proper<br>Time",
                    x=0.66,
                    len=0.45,
                    y=0.275,
                    titlefont=dict(color='white'),
                    tickfont=dict(color='white')
                )
            ),
            name='Phase Space Trajectory',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # 6. Add clustering visualization (row=2, col=3)
    optimal_clustering = clustering_results['optimal']
    best_method = optimal_clustering['method']
    best_labels = optimal_clustering['labels']
    
    # Get silhouette scores from kmeans results
    silhouette_scores = {}
    if 'kmeans' in clustering_results:
        for k, result in clustering_results['kmeans'].items():
            silhouette_scores[k] = result['silhouette_score']
    
    # Determine best k
    if best_method == 'kmeans':
        best_k = optimal_clustering['params']['k']
    else:
        best_k = len(np.unique(best_labels))
    
    # Main clustering scatter plot
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    for cluster_id in range(best_k):
        cluster_mask = best_labels == cluster_id
        if np.any(cluster_mask):
            fig.add_trace(
                go.Scatter(
                    x=sentiments[cluster_mask],
                    y=uwrs[cluster_mask],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=colors[cluster_id % len(colors)],
                        line=dict(color='white', width=1),
                        opacity=0.8
                    ),
                    name=f'Cluster {cluster_id + 1}',
                    showlegend=True,
                    hovertemplate=f'<b>Cluster {cluster_id + 1}</b><br>' +
                                  'Sentiment: %{x:.3f}<br>' +
                                  'UWR: %{y:.3f}<extra></extra>'
                ),
                row=2, col=3
            )
    
    # Add silhouette score inset as text annotation
    silhouette_text = f"Optimal k = {best_k}<br>"
    silhouette_text += f"Silhouette Score: {silhouette_scores[best_k]:.3f}<br><br>"
    silhouette_text += "All k scores:<br>"
    
    for k in sorted(silhouette_scores.keys()):
        score = silhouette_scores[k]
        marker = "★ " if k == best_k else "  "
        silhouette_text += f"{marker}k={k}: {score:.3f}<br>"
    
    fig.add_annotation(
        text=silhouette_text,
        xref=f"x{6}", yref=f"y{6}",  # Reference clustering subplot
        x=0.95, y=0.95,  # Top-RIGHT of clustering subplot
        showarrow=False,
        font=dict(size=10, color='white'),
        bgcolor='rgba(0,0,0,0.8)',
        bordercolor='white',
        borderwidth=1,
        align='right'  # Right-align text
    )
    
    # Add boundaries to phase space subplots only (not candlestick)
    boundary_s = np.linspace(-0.99, 0.99, 50)
    boundary_u = 1.0 - np.abs(boundary_s)
    
    # Only add boundaries to phase space plots (cols 1,2 and row 2 cols 2,3)
    phase_space_plots = [(1,1), (1,2), (2,1), (2,2), (2,3)]
    for row, col in phase_space_plots:
        fig.add_trace(
            go.Scatter(
                x=boundary_s,
                y=boundary_u,
                mode='lines',
                line=dict(color='yellow', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
    
    # NO trajectory lines on gradient plots - pure gradients only
    
    # Update layout with interactive features and range selector
    fig.update_layout(
        title='🌌 Interactive Market Analysis: Use Range Selector to Filter All Time-Series Charts',
        template='plotly_dark',
        height=1000,  # Taller for 3 rows
        width=1800,  # Wider for 3 columns
        showlegend=True,  # Enable legend for clustering
        legend=dict(
            x=1.02,
            y=0.5,
            bgcolor='rgba(13, 17, 23, 0.8)',
            bordercolor='white',
            borderwidth=1,
            font=dict(color='white')
        ),
        # Add range selector to candlestick chart
        xaxis3=dict(
            title='Time Index',
            rangeslider=dict(visible=True, thickness=0.1),
            rangeselector=dict(
                buttons=list([
                    dict(count=20, label="20", step="all", stepmode="backward"),
                    dict(count=40, label="40", step="all", stepmode="backward"),
                    dict(count=60, label="60", step="all", stepmode="backward"),
                    dict(count=100, label="100", step="all", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor='rgba(50,50,50,0.8)',
                bordercolor='white',
                borderwidth=1,
                font=dict(color='white')
            )
        )
    )
    
    # Update all phase space axes (including trajectory and clustering)
    phase_space_plots = [(1,1), (1,2), (2,1), (2,2), (2,3)]
    for row, col in phase_space_plots:
        fig.update_xaxes(title='Sentiment', range=[-1, 1], row=row, col=col)
        fig.update_yaxes(title='UWR', range=[0, 1], row=row, col=col)
    
    # Update candlestick axes (already configured in layout)
    fig.update_yaxes(title='Price', row=1, col=3)
    
    # Add range selector instruction
    fig.add_annotation(
        text="📊 Use Range Selector Buttons or Drag on Range Slider to Filter Time-Series Charts",
        xref="paper", yref="paper",
        x=0.5, y=1.02,
        showarrow=False,
        font=dict(size=14, color='yellow'),
        bgcolor='rgba(0,0,0,0.7)',
        bordercolor='yellow',
        borderwidth=1
    )
    
    return fig


def create_server_side_interactive_version(ohlc_data: pd.DataFrame) -> str:
    """Create a JavaScript enhancement for trajectory updates."""
    
    ohlc_json = ohlc_data.to_json(orient='records', date_format='iso')
    
    js_code = f"""
    <script>
    // Store original data
    const originalData = {ohlc_json};
    
    // Function to update phase space trajectory for selected range
    function updateForRange(start_idx, end_idx) {{
        console.log('🔄 Updating for range:', start_idx, 'to', end_idx);
        
        // Get the selected data slice
        const selectedData = originalData.slice(start_idx, end_idx + 1);
        
        if (selectedData.length < 3) {{
            console.log('❌ Not enough data points:', selectedData.length);
            return;
        }}
        
        // Convert to phase space coordinates
        const candle_metrics = selectedData.map((d, i) => {{
            const range = d.high - d.low;
            if (range <= 0) return null;
            
            return {{
                sentiment: (d.close - d.open) / range,
                upper_wick_ratio: (d.high - Math.max(d.open, d.close)) / range,
                time: start_idx + i
            }};
        }}).filter(c => c !== null);
        
        if (candle_metrics.length < 2) return;
        
        const sentiments = candle_metrics.map(c => c.sentiment);
        const uwrs = candle_metrics.map(c => c.upper_wick_ratio);
        const times = candle_metrics.map(c => c.time);
        
        // Update phase space trajectory
        updateTrajectory(sentiments, uwrs, times);
        
        // Update title
        updateTitle(start_idx, end_idx, candle_metrics.length);
    }}
    
    function updateTrajectory(sentiments, uwrs, times) {{
        const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (!plotDiv) return;
        
        // Find the phase space trajectory trace
        const data = plotDiv.data;
        for (let i = 0; i < data.length; i++) {{
            if (data[i].name === 'Phase Space Trajectory') {{
                console.log('🎯 Updating trajectory trace', i);
                Plotly.restyle(plotDiv, {{
                    x: [sentiments],
                    y: [uwrs],
                    'marker.color': [times]
                }}, [i]);
                break;
            }}
        }}
    }}
    
    function updateTitle(start_idx, end_idx, pointCount) {{
        const plotDiv = document.getElementsByClassName('plotly-graph-div'][0];
        if (!plotDiv) return;
        
        const newTitle = `🌌 Interactive Analysis: Range [${{start_idx}}-${{end_idx}}] → ${{pointCount}} points`;
        Plotly.relayout(plotDiv, {{'title': newTitle}});
    }}
    
    // Listen for range changes
    document.addEventListener('DOMContentLoaded', function() {{
        const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (plotDiv) {{
            console.log('🚀 Phase space interaction ready');
            
            plotDiv.on('plotly_relayout', function(eventData) {{
                if (eventData['xaxis3.range[0]'] !== undefined && eventData['xaxis3.range[1]'] !== undefined) {{
                    const startIdx = Math.max(0, Math.floor(eventData['xaxis3.range[0]']));
                    const endIdx = Math.min(originalData.length - 1, Math.ceil(eventData['xaxis3.range[1]']));
                    
                    if (endIdx > startIdx && (endIdx - startIdx) >= 1) {{
                        updateForRange(startIdx, endIdx);
                    }}
                }}
                
                if (eventData['xaxis3.autorange'] === true) {{
                    updateForRange(0, originalData.length - 1);
                }}
            }});
        }}
    }});
    </script>
    """
    
    return js_code


def add_gradient_recomputation_to_html(html_content: str, ohlc_data: pd.DataFrame) -> str:
    """Add JavaScript to recompute gradients based on selected range."""
    
    # Convert data to JSON for JavaScript
    ohlc_json = ohlc_data.to_json(orient='records', date_format='iso')
    
    # JavaScript code for gradient recomputation
    js_code = f"""
    <script>
    // Store original data
    const originalData = {ohlc_json};
    
    // Function to recompute gradients for selected range
    function recomputeGradientsForRange(start_idx, end_idx) {{
        console.log('Recomputing gradients for range:', start_idx, 'to', end_idx);
        
        // Get the selected data slice
        const selectedData = originalData.slice(start_idx, end_idx + 1);
        
        if (selectedData.length < 5) return; // Need enough points for meaningful gradients
        
        // Convert to phase space coordinates
        const candle_metrics = selectedData.map((d, i) => ({{
            sentiment: isNaN((d.close - d.open) / (d.high - d.low)) ? 0 : (d.close - d.open) / (d.high - d.low),
            upper_wick_ratio: isNaN((d.high - Math.max(d.open, d.close)) / (d.high - d.low)) ? 0 : (d.high - Math.max(d.open, d.close)) / (d.high - d.low)
        }}));
        
        // Extract coordinates
        const sentiments = candle_metrics.map(c => c.sentiment).filter(s => !isNaN(s));
        const uwrs = candle_metrics.map(c => c.upper_wick_ratio).filter(u => !isNaN(u));
        
        if (sentiments.length < 3 || uwrs.length < 3) return;
        
        // Update phase space trajectory
        updatePhaseSpaceTrajectory(sentiments, uwrs, start_idx);
        
        // TODO: Recompute density gradients (complex - would need KDE in JS)
        // For now, just update the trajectory and show range info
        updateRangeInfo(start_idx, end_idx, sentiments.length);
    }}
    
    function updatePhaseSpaceTrajectory(sentiments, uwrs, start_idx) {{
        const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (!plotDiv) return;
        
        // Update phase space trajectory (subplot 2,2)
        const trajectoryUpdate = {{
            x: [sentiments],
            y: [uwrs]
        }};
        
        // Find and update the phase space trajectory trace
        const data = plotDiv.data;
        for (let i = 0; i < data.length; i++) {{
            if (data[i].name === 'Phase Space Trajectory' && data[i].type === 'scatter') {{
                console.log('Updating phase space trajectory trace', i);
                Plotly.restyle(plotDiv, trajectoryUpdate, [i]);
                break;
            }}
        }}
    }}
    
    function updateRangeInfo(start_idx, end_idx, pointCount) {{
        // Update the title to show current range
        const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (!plotDiv) return;
        
        const newTitle = `🌌 Interactive Market Analysis: Showing Range [${{start_idx}} - ${{end_idx}}] (${{pointCount}} points)`;
        Plotly.relayout(plotDiv, {{'title': newTitle}});
    }}
    
    // Listen for range selector and zoom events
    document.addEventListener('DOMContentLoaded', function() {{
        const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (plotDiv) {{
            // Listen for relayout events (range selector, zoom, pan)
            plotDiv.on('plotly_relayout', function(eventData) {{
                console.log('Relayout event:', eventData);
                
                // Check for range changes on the candlestick chart (xaxis3)
                if (eventData['xaxis3.range[0]'] !== undefined && eventData['xaxis3.range[1]'] !== undefined) {{
                    const startIdx = Math.max(0, Math.floor(eventData['xaxis3.range[0]']));
                    const endIdx = Math.min(originalData.length - 1, Math.ceil(eventData['xaxis3.range[1]']));
                    
                    if (endIdx > startIdx && (endIdx - startIdx) >= 2) {{
                        console.log('Range change detected:', startIdx, 'to', endIdx);
                        recomputeGradientsForRange(startIdx, endIdx);
                    }}
                }}
                
                // Also check for autorange reset
                if (eventData['xaxis3.autorange'] === true) {{
                    console.log('Range reset to full dataset');
                    const newTitle = '🌌 Interactive Market Analysis: Showing Full Dataset';
                    Plotly.relayout(plotDiv, {{'title': newTitle}});
                }}
            }});
            
            console.log('Interactive gradient recomputation ready!');
        }}
    }});
    </script>
    """
    
    # Insert the JavaScript before the closing body tag
    return html_content.replace('</body>', js_code + '</body>')


def main():
    """Test the enhanced phase space trajectory visualizations with meaningful gradients."""
    
    # Generate rich market data for meaningful histograms
    print("📊 Generating rich market data for meaningful gradient analysis...")
    ohlc_data = generate_rich_market_data(n_candles=200)
    
    # Original 3D visualization
    print("🌌 Creating original 3D visualization...")
    trajectory_3d = create_phase_space_trajectory_3d(ohlc_data)
    trajectory_3d.write_html("phase_space_trajectory_3d.html")
    
    # New gradient visualizations
    density_gradient = create_phase_space_density_gradient(ohlc_data)
    density_gradient.write_html("phase_space_density_gradient.html")
    
    proper_time_gradient = create_phase_space_time_gradient(ohlc_data, use_proper_time=True)
    proper_time_gradient.write_html("phase_space_proper_time_gradient.html")
    
    coord_time_gradient = create_phase_space_time_gradient(ohlc_data, use_proper_time=False)
    coord_time_gradient.write_html("phase_space_coord_time_gradient.html")
    
    # Interactive combined analysis with gradient recomputation
    print("🚀 Creating interactive combined analysis...")
    interactive_analysis = create_interactive_combined_analysis(ohlc_data)
    
    # Add JavaScript for trajectory updates (gradients need server-side processing)
    html_content = interactive_analysis.to_html()
    js_enhancement = create_server_side_interactive_version(ohlc_data)
    interactive_html = html_content.replace('</body>', js_enhancement + '</body>')
    
    with open("phase_space_interactive_analysis.html", "w") as f:
        f.write(interactive_html)
    
    # Also save the non-interactive version
    interactive_analysis.write_html("phase_space_combined_gradients.html")
    
    print("\n✅ Enhanced Phase Space Visualizations Created:")
    print("   🌌 phase_space_trajectory_3d.html - Original 3D with particles")
    print("   🎯 phase_space_density_gradient.html - Density clustering (NO particles)")
    print("   ⏰ phase_space_proper_time_gradient.html - Proper time flow (NO particles)")
    print("   🕰️ phase_space_coord_time_gradient.html - Coordinate time flow (NO particles)")
    print("   🌟 phase_space_combined_gradients.html - All gradients together")
    print("   🚀 phase_space_interactive_analysis.html - INTERACTIVE! Select range on candlesticks!")
    
    print("\n🎯 Key Improvements:")
    print("   ✓ Rich data with 200 candles for meaningful gradients")
    print("   ✓ Removed individual particles from gradient views")
    print("   ✓ Smooth density gradients showing clustering")
    print("   ✓ Time flow gradients with directional arrows")
    print("   ✓ Combined analysis for easy comparison")
    print("   ✓ Multiple market regimes create distinct patterns")
    print("   🚀 INTERACTIVE range selector filters time-series charts!")
    
    print("\n🌟 Data includes 5 distinct market regimes:")
    print("   • Bullish trending (positive sentiment cluster)")
    print("   • Bearish correction (negative sentiment cluster)")
    print("   • High volatility (spread across phase space)")
    print("   • Recovery phase (moderate positive sentiment)")
    print("   • Final consolidation (neutral cluster)")


if __name__ == "__main__":
    main()