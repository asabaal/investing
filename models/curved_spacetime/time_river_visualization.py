"""
Time River + Volume Dams Visualization

Shows time flowing like a river where:
- River width varies with proper time intervals (wider = more time dilation)
- Volume creates "dams" that affect river flow
- Candles float down the river at positions determined by proper time
- High-volume candles create pools and eddies in the time flow
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc
from curvature_color_scheme import get_curvature_gradient_color


def create_river_path(proper_times, proper_time_intervals, volumes, river_width=20):
    """
    Create the path of the time river with variable width based on proper time intervals.
    
    Args:
        proper_times: Array of proper time coordinates
        proper_time_intervals: Array of time dilation factors
        volumes: Array of volume values (creates dams)
        river_width: Base width of the river
    
    Returns:
        x_coords, y_top, y_bottom for river boundaries
    """
    x_coords = []
    y_top = []
    y_bottom = []
    
    # Normalize intervals for river width variation
    max_interval = max(proper_time_intervals)
    min_interval = min(proper_time_intervals)
    interval_range = max_interval - min_interval if max_interval > min_interval else 1.0
    
    # Create smooth river path
    for i, (tau, dt_tau, volume) in enumerate(zip(proper_times, proper_time_intervals, volumes)):
        # Base river width varies with time dilation
        width_factor = 0.3 + 1.7 * (dt_tau - min_interval) / interval_range
        base_width = river_width * width_factor
        
        # Volume creates dam effect - wider "pools" behind high volume candles
        volume_factor = 1.0 + 0.5 * np.log1p(volume) / np.log1p(max(volumes))
        final_width = base_width * volume_factor
        
        # Add some flow turbulence
        turbulence = 2 * np.sin(tau * 0.5) * np.exp(-abs(tau - proper_times[len(proper_times)//2]) * 0.1)
        
        x_coords.append(tau)
        y_top.append(final_width / 2 + turbulence)
        y_bottom.append(-final_width / 2 + turbulence)
    
    return x_coords, y_top, y_bottom


def create_volume_dams(proper_times, volumes, candle_metrics):
    """
    Create visual representation of volume dams in the river.
    """
    dam_traces = []
    
    # Normalize volumes for dam size
    max_volume = max(volumes)
    
    for i, (tau, volume, candle) in enumerate(zip(proper_times, volumes, candle_metrics)):
        if volume > max_volume * 0.3:  # Only show significant dams
            # Dam height proportional to volume
            dam_height = 15 * (volume / max_volume)
            
            # Create dam shape (rectangular barrier)
            dam_x = [tau - 0.5, tau + 0.5, tau + 0.5, tau - 0.5, tau - 0.5]
            dam_y = [-dam_height/2, -dam_height/2, dam_height/2, dam_height/2, -dam_height/2]
            
            # Dam color based on curvature
            geometry = CurvedCandleGeometry([candle])
            curvature = geometry.compute_intrinsic_curvature(0)
            dam_color, _ = get_curvature_gradient_color(curvature)
            
            dam_traces.append(go.Scatter(
                x=dam_x,
                y=dam_y,
                fill='toself',
                fillcolor=dam_color.replace('0.8)', '0.6)'),  # Semi-transparent
                line=dict(color='white', width=2),
                mode='lines',
                name=f'Volume Dam {i}',
                showlegend=False,
                hovertemplate=f'Volume Dam<br>Volume: {volume:,}<br>Curvature: {curvature:.3f}<extra></extra>'
            ))
    
    return dam_traces


def create_floating_candles(proper_times, candle_metrics, curvatures):
    """
    Create candles floating down the river.
    """
    candle_traces = []
    
    for i, (tau, candle, curvature) in enumerate(zip(proper_times, candle_metrics, curvatures)):
        # Candle position (floating in river center, offset by turbulence)
        turbulence = 3 * np.sin(tau * 0.8 + i * 0.5)
        y_pos = turbulence
        
        # Candle size based on range and volume
        size = 8 + np.sqrt(candle.volume) * 0.001 + candle.range_value * 0.1
        
        # Candle color based on curvature
        color, description = get_curvature_gradient_color(curvature)
        
        candle_traces.append(go.Scatter(
            x=[tau],
            y=[y_pos],
            mode='markers',
            marker=dict(
                size=size,
                color=color,
                line=dict(color='white', width=2),
                symbol='diamond'
            ),
            name=f'Candle {i}',
            showlegend=False,
            hovertemplate=f'<b>Candle {i}</b><br>' +
                         f'Proper Time: {tau:.2f}<br>' +
                         f'Range: {candle.range_value:.1f}<br>' +
                         f'Volume: {candle.volume:,}<br>' +
                         f'Curvature: {curvature:.3f} ({description})<extra></extra>'
        ))
    
    return candle_traces


def create_time_river_visualization(ohlc_data: pd.DataFrame) -> go.Figure:
    """
    Create the Time River + Volume Dams visualization.
    """
    print("🌊 CREATING TIME RIVER + VOLUME DAMS VISUALIZATION")
    print("=" * 60)
    
    # Compute spacetime geometry with volume effects
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    geometry = CurvedCandleGeometry(candle_metrics)
    curvatures = geometry.compute_curvature_series()
    proper_times = geometry.compute_proper_time_series()
    proper_time_intervals = geometry.compute_proper_time_intervals()
    volumes = [candle.volume for candle in candle_metrics]
    
    print(f"Processing {len(candle_metrics)} candles floating down the time river...")
    
    # Create subplot layout
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Time River with Volume Dams", "River Flow Speed (Time Intervals)"),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Create river boundaries
    x_coords, y_top, y_bottom = create_river_path(proper_times, proper_time_intervals, volumes)
    
    # Add river surface (water)
    fig.add_trace(go.Scatter(
        x=x_coords + x_coords[::-1],  # Close the shape
        y=y_top + y_bottom[::-1],
        fill='toself',
        fillcolor='rgba(64, 156, 255, 0.3)',  # Blue water
        line=dict(color='rgba(64, 156, 255, 0.6)', width=1),
        mode='lines',
        name='Time River',
        showlegend=True,
        hovertemplate='Time River<br>Proper Time: %{x:.2f}<extra></extra>'
    ), row=1, col=1)
    
    # Add river banks
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_top,
        mode='lines',
        line=dict(color='#8b4513', width=3),  # Brown riverbank
        name='River Bank',
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_bottom,
        mode='lines',
        line=dict(color='#8b4513', width=3),  # Brown riverbank
        name='River Bank',
        showlegend=False
    ), row=1, col=1)
    
    # Add volume dams
    dam_traces = create_volume_dams(proper_times, volumes, candle_metrics)
    for trace in dam_traces:
        fig.add_trace(trace, row=1, col=1)
    
    # Add floating candles
    candle_traces = create_floating_candles(proper_times, candle_metrics, curvatures)
    for trace in candle_traces:
        fig.add_trace(trace, row=1, col=1)
    
    # Add river flow speed chart (bottom subplot)
    fig.add_trace(go.Scatter(
        x=proper_times,
        y=proper_time_intervals,
        mode='lines+markers',
        line=dict(color='#00ffff', width=3),
        marker=dict(
            size=8,
            color='#00ffff',
            line=dict(color='white', width=1)
        ),
        name='River Flow Speed',
        hovertemplate='Proper Time: %{x:.2f}<br>Flow Speed: %{y:.2f}<extra></extra>'
    ), row=2, col=1)
    
    # Add volume indicators on flow chart
    fig.add_trace(go.Scatter(
        x=proper_times,
        y=[max(proper_time_intervals) * 1.1] * len(proper_times),
        mode='markers',
        marker=dict(
            size=[np.sqrt(v) * 0.01 for v in volumes],
            color='rgba(255, 255, 255, 0.7)',
            line=dict(color='white', width=1)
        ),
        name='Volume Indicators',
        showlegend=False,
        hovertemplate='Volume: %{customdata:,}<extra></extra>',
        customdata=volumes
    ), row=2, col=1)
    
    # Layout
    fig.update_layout(
        title="Time River Visualization - How Volume Creates Dams in Time Flow",
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        height=900,
        width=1200,
        showlegend=True
    )
    
    # Update axes
    fig.update_yaxes(title_text="River Width (Time Dilation)", row=1, col=1)
    fig.update_yaxes(title_text="Flow Speed (dτ)", row=2, col=1)
    fig.update_xaxes(title_text="Proper Time (τ)", row=2, col=1)
    
    return fig


def main():
    """Test the Time River visualization."""
    
    # Create test data with volume
    dates = [datetime(2024, 1, 1, 9, 0) + timedelta(hours=i) for i in range(10)]
    
    ohlc_data = pd.DataFrame({
        'open':  [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0],
        'high':  [120.0, 108.0, 107.0, 106.5, 108.1, 115.0, 125.0, 115.0, 130.0, 118.05],
        'low':   [80.0,  98.0,  101.0, 105.8, 107.95, 105.0, 100.0, 113.5, 102.0, 117.98],
        'close': [110.0, 105.0, 104.5, 105.9, 107.96, 112.0, 105.0, 114.2, 125.0, 117.99],
        'volume': [50000, 10000, 25000, 5000, 2000, 30000, 80000, 8000, 60000, 3000]
    }, index=dates)
    
    # Create visualization
    fig = create_time_river_visualization(ohlc_data)
    fig.write_html("time_river_dams.html")
    
    print("\\n✅ Saved: time_river_dams.html")
    print("\\n🎯 Open the file to see:")
    print("   - Time flowing like a river (width = time dilation)")
    print("   - Volume dams creating pools and flow changes")
    print("   - Candles floating down river at proper time positions")
    print("   - River flow speed chart showing time interval variations")
    print("   - Visual metaphor for how mass affects time flow")


if __name__ == "__main__":
    main()