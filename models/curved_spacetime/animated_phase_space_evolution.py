#!/usr/bin/env python3
"""
Animated Phase Space Evolution - Watch All Charts Build Up Candle by Candle
Creates an animation showing how density gradients, time flows, and trajectories evolve as each candle is added.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from curved_candle_geometry import CurvedCandleGeometry, CandleMetric, create_candle_metrics_from_ohlc

def generate_rich_market_data(n_candles: int = 100) -> pd.DataFrame:
    """Generate rich market data with multiple regimes for meaningful gradients."""
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


def compute_evolutionary_frame(ohlc_data: pd.DataFrame, frame_idx: int, min_candles: int = 5):
    """
    Compute all visualizations for a given frame (number of candles).
    Returns all data needed for that frame of the animation.
    """
    if frame_idx < min_candles:
        return None  # Need minimum candles for meaningful visualization
    
    # Get data up to this frame
    data_slice = ohlc_data.iloc[:frame_idx+1]
    
    # Create candle metrics
    candle_metrics = create_candle_metrics_from_ohlc(data_slice)
    geometry = CurvedCandleGeometry(candle_metrics)
    
    sentiments = np.array([c.sentiment for c in candle_metrics])
    uwrs = np.array([c.upper_wick_ratio for c in candle_metrics])
    proper_times = geometry.compute_proper_time_series()
    
    # Generate grid for contours
    grid_res = 30  # Smaller for animation performance
    sentiment_range = np.linspace(-0.99, 0.99, grid_res)
    uwr_range = np.linspace(0.01, 0.99, grid_res)
    S_grid, U_grid = np.meshgrid(sentiment_range, uwr_range)
    
    # 1. Density contours using KDE
    points = np.column_stack([sentiments, uwrs])
    if len(points) >= 2:  # Need at least 2 points for KDE
        kde = gaussian_kde(points.T, bw_method='scott')
        
        density_grid = np.zeros_like(S_grid)
        for i in range(grid_res):
            for j in range(grid_res):
                s, u = S_grid[i, j], U_grid[i, j]
                if abs(s) + u <= 1.0:  # Phase space constraint
                    density_grid[i, j] = kde([s, u])[0]
                else:
                    density_grid[i, j] = np.nan
    else:
        density_grid = np.full_like(S_grid, np.nan)
    
    # 2. Proper time gradients
    if len(points) >= 2:
        time_grid = griddata(
            points, proper_times, (S_grid, U_grid), 
            method='linear', fill_value=np.nan
        )
        
        # Apply constraint
        for i in range(grid_res):
            for j in range(grid_res):
                if abs(S_grid[i, j]) + U_grid[i, j] > 1.0:
                    time_grid[i, j] = np.nan
    else:
        time_grid = np.full_like(S_grid, np.nan)
    
    # 3. Coordinate time gradients
    coord_times = np.arange(len(candle_metrics))
    if len(points) >= 2:
        coord_time_grid = griddata(
            points, coord_times, (S_grid, U_grid),
            method='linear', fill_value=np.nan
        )
        
        # Apply constraint
        for i in range(grid_res):
            for j in range(grid_res):
                if abs(S_grid[i, j]) + U_grid[i, j] > 1.0:
                    coord_time_grid[i, j] = np.nan
    else:
        coord_time_grid = np.full_like(S_grid, np.nan)
    
    return {
        'frame_idx': frame_idx,
        'n_candles': len(candle_metrics),
        'sentiment_range': sentiment_range,
        'uwr_range': uwr_range,
        'density_grid': density_grid,
        'time_grid': time_grid,
        'coord_time_grid': coord_time_grid,
        'sentiments': sentiments,
        'uwrs': uwrs,
        'proper_times': proper_times,
        'ohlc_slice': data_slice
    }


def create_animated_evolution(ohlc_data: pd.DataFrame, start_frame: int = 10, end_frame: int = None):
    """
    Create animated evolution of all phase space visualizations.
    """
    if end_frame is None:
        end_frame = len(ohlc_data) - 1
    
    print(f"🎬 Creating animated evolution from candle {start_frame} to {end_frame}...")
    
    # Compute all frames
    frames_data = []
    for frame_idx in range(start_frame, end_frame + 1):
        frame_data = compute_evolutionary_frame(ohlc_data, frame_idx)
        if frame_data:
            frames_data.append(frame_data)
            if frame_idx % 10 == 0:
                print(f"   ✅ Computed frame {frame_idx}/{end_frame}")
    
    print(f"✅ Computed {len(frames_data)} frames")
    
    # Create figure with all subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Density Gradient Evolution',
            'Proper Time Gradient Evolution', 
            'Candlestick Chart Evolution',
            'Coordinate Time Gradient Evolution',
            'Phase Space Trajectory Evolution',
            ''  # Empty bottom-right
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'scatter'}, None]  # No bottom-right subplot
        ]
    )
    
    # Get the final frame for consistent scaling
    final_frame = frames_data[-1]
    
    # Create animation frames
    animation_frames = []
    
    for i, frame_data in enumerate(frames_data):
        frame_traces = []
        
        # 1. Density gradient contour
        density_max = np.nanmax(frame_data['density_grid']) if not np.isnan(frame_data['density_grid']).all() else 1
        density_min = np.nanmin(frame_data['density_grid']) if not np.isnan(frame_data['density_grid']).all() else 0
        
        frame_traces.append(
            go.Contour(
                x=frame_data['sentiment_range'],
                y=frame_data['uwr_range'],
                z=frame_data['density_grid'],
                colorscale='Blues',
                showscale=(i == 0),  # Only show colorbar on first trace
                colorbar=dict(
                    title="Density",
                    x=0.32,
                    len=0.45,
                    y=0.775,
                    titlefont=dict(color='white'),
                    tickfont=dict(color='white')
                ) if i == 0 else None,
                contours=dict(
                    start=density_min,
                    end=density_max,
                    size=max((density_max - density_min)/8, 0.001),
                    showlines=True,
                    coloring='fill'
                ),
                hovertemplate=f'Candles: {frame_data["n_candles"]}<br>Sentiment: %{{x:.3f}}<br>UWR: %{{y:.3f}}<br>Density: %{{z:.6f}}<extra></extra>',
                name='Density'
            )
        )
        
        # 2. Proper time gradient contour
        time_max = np.nanmax(frame_data['time_grid']) if not np.isnan(frame_data['time_grid']).all() else 1
        time_min = np.nanmin(frame_data['time_grid']) if not np.isnan(frame_data['time_grid']).all() else 0
        
        frame_traces.append(
            go.Contour(
                x=frame_data['sentiment_range'],
                y=frame_data['uwr_range'],
                z=frame_data['time_grid'],
                colorscale='Viridis',
                showscale=(i == 0),
                colorbar=dict(
                    title="Proper<br>Time",
                    x=0.66,
                    len=0.45,
                    y=0.775,
                    titlefont=dict(color='white'),
                    tickfont=dict(color='white')
                ) if i == 0 else None,
                contours=dict(
                    start=time_min,
                    end=time_max,
                    size=max((time_max - time_min)/8, 0.1),
                    showlines=True,
                    coloring='fill'
                ),
                hovertemplate=f'Candles: {frame_data["n_candles"]}<br>Sentiment: %{{x:.3f}}<br>UWR: %{{y:.3f}}<br>Proper Time: %{{z:.3f}}<extra></extra>',
                name='Proper Time'
            )
        )
        
        # 3. Candlestick chart (up to current frame)
        frame_traces.append(
            go.Candlestick(
                x=list(range(len(frame_data['ohlc_slice']))),
                open=frame_data['ohlc_slice']['open'].tolist(),
                high=frame_data['ohlc_slice']['high'].tolist(),
                low=frame_data['ohlc_slice']['low'].tolist(),
                close=frame_data['ohlc_slice']['close'].tolist(),
                name=f'OHLC (Candles 1-{frame_data["n_candles"]})',
                showlegend=False
            )
        )
        
        # 4. Coordinate time gradient contour
        coord_time_max = np.nanmax(frame_data['coord_time_grid']) if not np.isnan(frame_data['coord_time_grid']).all() else frame_data['n_candles']
        coord_time_min = np.nanmin(frame_data['coord_time_grid']) if not np.isnan(frame_data['coord_time_grid']).all() else 0
        
        frame_traces.append(
            go.Contour(
                x=frame_data['sentiment_range'],
                y=frame_data['uwr_range'],
                z=frame_data['coord_time_grid'],
                colorscale='Plasma',
                showscale=(i == 0),
                colorbar=dict(
                    title="Coordinate<br>Time",
                    x=0.32,
                    len=0.45,
                    y=0.275,
                    titlefont=dict(color='white'),
                    tickfont=dict(color='white')
                ) if i == 0 else None,
                contours=dict(
                    start=coord_time_min,
                    end=coord_time_max,
                    size=max((coord_time_max - coord_time_min)/8, 1),
                    showlines=True,
                    coloring='fill'
                ),
                hovertemplate=f'Candles: {frame_data["n_candles"]}<br>Sentiment: %{{x:.3f}}<br>UWR: %{{y:.3f}}<br>Coord Time: %{{z:.0f}}<extra></extra>',
                name='Coord Time'
            )
        )
        
        # 5. Phase space trajectory (growing path)
        frame_traces.append(
            go.Scatter(
                x=frame_data['sentiments'],
                y=frame_data['uwrs'],
                mode='lines+markers',
                line=dict(color='white', width=3),
                marker=dict(
                    size=6,
                    color=frame_data['proper_times'],
                    colorscale='RdYlBu_r',
                    showscale=(i == 0),
                    colorbar=dict(
                        title="Proper<br>Time",
                        x=0.66,
                        len=0.45,
                        y=0.275,
                        titlefont=dict(color='white'),
                        tickfont=dict(color='white')
                    ) if i == 0 else None
                ),
                name=f'Trajectory ({frame_data["n_candles"]} candles)',
                showlegend=False,
                hovertemplate=f'Candle: %{{pointNumber}}<br>Sentiment: %{{x:.3f}}<br>UWR: %{{y:.3f}}<br>Proper Time: %{{marker.color:.3f}}<extra></extra>'
            )
        )
        
        # Add phase space boundaries to all phase space plots
        boundary_s = np.linspace(-0.99, 0.99, 30)
        boundary_u = 1.0 - np.abs(boundary_s)
        
        phase_space_positions = [(1,1), (1,2), (2,1), (2,2)]  # All except candlestick
        for pos_idx, (row, col) in enumerate(phase_space_positions):
            frame_traces.append(
                go.Scatter(
                    x=boundary_s,
                    y=boundary_u,
                    mode='lines',
                    line=dict(color='yellow', width=2, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip',
                    name=f'Boundary_{pos_idx}'
                )
            )
        
        # Create frame
        animation_frames.append(
            go.Frame(
                data=frame_traces,
                name=f'frame_{i}',
                layout=dict(
                    title=f'🎬 Phase Space Evolution: Candle {frame_data["n_candles"]}/{end_frame} - Watch All Gradients Build Up!'
                )
            )
        )
    
    # Add initial traces from first frame with proper subplot assignment
    first_frame_traces = animation_frames[0].data
    
    # Map traces to correct subplots
    trace_mapping = [
        (0, 1, 1),  # Density contour -> row 1, col 1
        (1, 1, 2),  # Proper time contour -> row 1, col 2
        (2, 1, 3),  # Candlestick -> row 1, col 3
        (3, 2, 1),  # Coord time contour -> row 2, col 1
        (4, 2, 2),  # Phase space trajectory -> row 2, col 2
        (5, 1, 1),  # Boundary 1 -> row 1, col 1
        (6, 1, 2),  # Boundary 2 -> row 1, col 2
        (7, 2, 1),  # Boundary 3 -> row 2, col 1
        (8, 2, 2),  # Boundary 4 -> row 2, col 2
    ]
    
    for trace_idx, row, col in trace_mapping:
        if trace_idx < len(first_frame_traces):
            fig.add_trace(first_frame_traces[trace_idx], row=row, col=col)
    
    # Add frames to figure
    fig.frames = animation_frames
    
    # Update layout with animation controls
    fig.update_layout(
        title=f'🎬 Animated Phase Space Evolution: Watch All Charts Build Up Candle by Candle',
        template='plotly_dark',
        height=800,
        width=1800,
        showlegend=False,
        updatemenus=[
            {
                'type': 'buttons',
                'showactive': False,
                'x': 0.1,
                'y': 0,
                'xanchor': 'right',
                'yanchor': 'top',
                'buttons': [
                    {
                        'label': '▶️ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 500, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 100}
                        }]
                    },
                    {
                        'label': '⏸️ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate'
                        }]
                    },
                    {
                        'label': '⏭️ Fast',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 50}
                        }]
                    },
                    {
                        'label': '🐌 Slow',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 1000, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 200}
                        }]
                    }
                ]
            }
        ],
        sliders=[
            {
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 16, 'color': 'white'},
                    'prefix': 'Candles: ',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 100},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': [
                    {
                        'args': [[f'frame_{i}'], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate'
                        }],
                        'label': f'{frames_data[i]["n_candles"]}',
                        'method': 'animate'
                    }
                    for i in range(len(frames_data))
                ]
            }
        ]
    )
    
    # Update axes
    phase_space_plots = [(1,1), (1,2), (2,1), (2,2)]
    for row, col in phase_space_plots:
        fig.update_xaxes(title='Sentiment', range=[-1, 1], row=row, col=col)
        fig.update_yaxes(title='UWR', range=[0, 1], row=row, col=col)
    
    fig.update_xaxes(title='Time Index', row=1, col=3)
    fig.update_yaxes(title='Price', row=1, col=3)
    
    return fig


def main():
    """Create and save the animated evolution."""
    print("🎬 Creating Animated Phase Space Evolution...")
    
    # Generate market data (same as working visualization)
    ohlc_data = generate_rich_market_data(n_candles=150)  # More candles for better phase space filling
    print(f"📊 Generated {len(ohlc_data)} candles")
    
    # Create animation  
    fig = create_animated_evolution(ohlc_data, start_frame=15, end_frame=149)
    
    # Save as HTML
    fig.write_html("animated_phase_space_evolution.html")
    
    print("\n✅ Animated Phase Space Evolution Created!")
    print("   🎬 animated_phase_space_evolution.html")
    print("\n🎯 Features:")
    print("   ▶️ Play/Pause animation controls")
    print("   🎚️ Timeline slider to scrub through evolution")
    print("   🐌⏭️ Speed controls (Slow/Fast)")
    print("   📊 Watch ALL charts build up candle by candle!")
    print("\n🌟 You can see:")
    print("   • Density gradients evolving as clusters form")
    print("   • Time gradients building up the flow patterns")
    print("   • Phase space trajectory growing through the triangle")
    print("   • Candlestick chart extending with each new candle")


if __name__ == "__main__":
    main()