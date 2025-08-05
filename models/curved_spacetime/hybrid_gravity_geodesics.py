"""
Hybrid Gravity Wells + Geodesic Paths Visualization

Combines the 3D gravity well surface with flowing geodesic paths to show:
- Static spacetime curvature (gravity wells)
- Dynamic particle trajectories (geodesic paths)
- How massive candles affect both geometry and motion
"""

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc
from curvature_color_scheme import get_curvature_gradient_color


def create_simplified_gravity_surface(candle_metrics, proper_times, grid_resolution=30):
    """
    Create a simplified 3D surface for the hybrid visualization.
    Lower resolution for better performance with geodesic paths.
    """
    if len(candle_metrics) == 0:
        return None, None, None, None
    
    # Create coordinate grids (smaller for performance)
    time_min, time_max = proper_times[0], proper_times[-1]
    price_min = min(candle.low_value for candle in candle_metrics)
    price_max = max(candle.low_value + candle.range_value for candle in candle_metrics)
    
    # Calculate data ranges for proper scaling
    time_range = time_max - time_min
    price_range = price_max - price_min
    max_range = max(candle.range_value for candle in candle_metrics)
    
    # Expand ranges
    time_min -= time_range * 0.2
    time_max += time_range * 0.2
    price_min -= price_range * 0.2
    price_max += price_range * 0.2
    
    # Create meshgrid
    t_grid = np.linspace(time_min, time_max, grid_resolution)
    p_grid = np.linspace(price_min, price_max, grid_resolution)
    T, P = np.meshgrid(t_grid, p_grid)
    
    # Initialize surface
    Z = np.zeros_like(T)
    Colors = np.zeros_like(T)
    
    # Scale well depths relative to price range (shallower for hybrid view)
    depth_scale = price_range * 0.08  # Wells can be at most 8% of price range deep
    
    # Add gravitational wells
    for i, candle in enumerate(candle_metrics):
        candle_time = proper_times[i]
        candle_price = candle.low_value + candle.range_value / 2
        
        # Well parameters - scaled appropriately
        volatility_depth = (candle.range_value / max_range) * depth_scale  # Proportional depth
        mass_width = np.sqrt(candle.volume) * 0.15
        
        # Distance calculation
        time_dist = (T - candle_time) / (time_range * 0.3)
        price_dist = (P - candle_price) / (price_range * 0.3)
        spacetime_dist = np.sqrt(time_dist**2 + price_dist**2)
        
        # Gaussian wells
        well_depth = volatility_depth * np.exp(-spacetime_dist**2 / (2 * mass_width**2))
        Z -= well_depth
        Colors += well_depth
    
    # Normalize colors
    if Colors.max() > Colors.min():
        Colors = (Colors - Colors.min()) / (Colors.max() - Colors.min())
    
    return T, P, Z, Colors


def create_actual_data_geodesic(candle_metrics, proper_times, curvatures):
    """
    Create THE SINGLE ACTUAL GEODESIC PATH that the market data took through curved spacetime.
    
    This shows the real path of the data with:
    - X: Proper time coordinates
    - Y: Actual low prices (the price level)
    - Z: Spacetime depth (elevation above the curved surface)
    
    Simple, clear visualization with just one connected path.
    """
    if len(candle_metrics) < 2:
        return []
    
    # The actual path through spacetime
    path_times = proper_times
    path_prices = [candle.low_value for candle in candle_metrics]  # Use LOW as the price level
    
    # Calculate appropriate elevation for the path (scaled to data)
    price_range = max(candle.low_value + candle.range_value for candle in candle_metrics) - \
                 min(candle.low_value for candle in candle_metrics)
    path_elevation = price_range * 0.08  # 8% of price range above surface (more visible)
    path_z = [path_elevation] * len(candle_metrics)
    
    # Use a single color gradient along the path based on curvature
    # Create a smooth color gradient from start to end
    curvature_colors = []
    for i, curvature in enumerate(curvatures):
        color, _ = get_curvature_gradient_color(curvature)
        # Extract RGB values and make them full opacity
        rgba_color = color.replace('0.8)', '1.0)')
        curvature_colors.append(rgba_color)
    
    # Create the single actual geodesic path - ONLY THIS TRACE
    geodesic_trace = go.Scatter3d(
        x=path_times,
        y=path_prices,
        z=path_z,
        mode='lines+markers',
        line=dict(
            color='#00ffff',  # Bright cyan line for visibility
            width=6
        ),
        marker=dict(
            size=10,  # Larger, more visible markers
            color=curvature_colors,  # Each point colored by its curvature
            line=dict(color='white', width=2),
            opacity=1.0  # Full opacity
        ),
        name='Market Geodesic Path',
        hovertemplate='<b>Market Data Point %{pointNumber}</b><br>' +
                     'Proper Time: %{x:.2f}<br>' +
                     'Low Price: $%{y:.2f}<br>' +
                     'Curvature: %{customdata:.3f}<br>' +
                     'Volume: %{text:,}<extra></extra>',
        customdata=curvatures,
        text=[candle.volume for candle in candle_metrics],
        showlegend=True
    )
    
    return [geodesic_trace]


def create_particle_flow_arrows(candle_metrics, proper_times, curvatures):
    """
    Create arrows showing the direction of spacetime curvature effects.
    """
    if len(candle_metrics) < 2:
        return []
    
    arrow_traces = []
    
    # Calculate price range for proper arrow elevation
    price_range = max(candle.low_value + candle.range_value for candle in candle_metrics) - \
                  min(candle.low_value for candle in candle_metrics)
    arrow_elevation = price_range * 0.03  # 3% of price range above surface
    
    for i in range(len(candle_metrics) - 1):
        current_candle = candle_metrics[i]
        next_candle = candle_metrics[i + 1]
        
        # Arrow start position
        start_time = proper_times[i]
        start_price = current_candle.low_value + current_candle.range_value / 2
        start_z = arrow_elevation
        
        # Arrow end position (influenced by curvature)
        end_time = proper_times[i + 1]
        end_price = next_candle.low_value + next_candle.range_value / 2
        
        # Curvature effect on trajectory
        curvature = curvatures[i]
        curvature_deflection = curvature * (current_candle.range_value * 0.2)
        end_price += curvature_deflection
        end_z = arrow_elevation
        
        # Arrow color based on curvature
        arrow_color, _ = get_curvature_gradient_color(curvature)
        
        # Create arrow as line with annotation
        arrow_traces.append(go.Scatter3d(
            x=[start_time, end_time],
            y=[start_price, end_price],
            z=[start_z, end_z],
            mode='lines',
            line=dict(
                color=arrow_color.replace('0.8)', '0.6)'),  # Semi-transparent
                width=4
            ),
            name=f'Curvature Effect {i}',
            showlegend=False,
            hovertemplate=f'Curvature Effect<br>From: τ={start_time:.2f}<br>To: τ={end_time:.2f}<br>Curvature: {curvature:.3f}<extra></extra>'
        ))
    
    return arrow_traces


def create_hybrid_gravity_geodesics_visualization(ohlc_data: pd.DataFrame) -> go.Figure:
    """
    Create the hybrid gravity wells + geodesic paths visualization.
    """
    print("🌌🛤️  CREATING HYBRID GRAVITY WELLS + GEODESIC PATHS")
    print("=" * 60)
    
    # Compute spacetime geometry
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    geometry = CurvedCandleGeometry(candle_metrics)
    curvatures = geometry.compute_curvature_series()
    proper_times = geometry.compute_proper_time_series()
    
    print(f"Processing {len(candle_metrics)} candles with geodesic paths...")
    
    # Create figure
    fig = go.Figure()
    
    # Add simplified gravity well surface
    T, P, Z, Colors = create_simplified_gravity_surface(candle_metrics, proper_times)
    
    if T is not None:
        fig.add_trace(go.Surface(
            x=T,
            y=P,
            z=Z,
            surfacecolor=Colors,
            colorscale=[
                [0, 'rgba(13, 17, 23, 0.7)'],      # Semi-transparent dark
                [0.3, 'rgba(45, 27, 105, 0.7)'],   # Semi-transparent purple
                [0.6, 'rgba(100, 44, 138, 0.7)'],  # Semi-transparent purple
                [1, 'rgba(157, 78, 221, 0.7)']     # Semi-transparent light purple
            ],
            opacity=0.4,  # More transparent for hybrid view
            name='Spacetime Curvature',
            showscale=False,
            hovertemplate='Gravitational Field<br>Time: %{x:.2f}<br>Price: %{y:.2f}<extra></extra>'
        ))
    
    # Add ONLY the single actual geodesic path - no separate markers or arrows
    geodesic_traces = create_actual_data_geodesic(candle_metrics, proper_times, curvatures)
    for trace in geodesic_traces:
        fig.add_trace(trace)
    
    # Calculate appropriate Z-axis range based on actual well depths and paths
    if T is not None:
        z_min, z_max = Z.min(), Z.max()
        # Also consider the elevation of paths and markers
        price_range = max(candle.low_value + candle.range_value for candle in candle_metrics) - \
                     min(candle.low_value for candle in candle_metrics)
        path_elevation = price_range * 0.05
        
        # Set Z range to accommodate both wells and elevated elements
        z_range = z_max - z_min
        z_axis_min = z_min - z_range * 0.1
        z_axis_max = max(z_max, path_elevation) + z_range * 0.2
    else:
        z_axis_min, z_axis_max = -1, 1
    
    # Layout
    fig.update_layout(
        title="Hybrid Visualization: Gravity Wells + Geodesic Paths",
        scene=dict(
            xaxis=dict(
                title="Proper Time (τ)",
                backgroundcolor='rgba(13, 17, 23, 0.8)',
                gridcolor='#30363d',
                tickfont=dict(color='#c9d1d9')
            ),
            yaxis=dict(
                title="Price",
                backgroundcolor='rgba(13, 17, 23, 0.8)',
                gridcolor='#30363d',
                tickfont=dict(color='#c9d1d9')
            ),
            zaxis=dict(
                title="Spacetime Elevation",
                range=[z_axis_min, z_axis_max],  # Dynamic range based on data
                backgroundcolor='rgba(13, 17, 23, 0.8)',
                gridcolor='#30363d',
                tickfont=dict(color='#c9d1d9')
            ),
            bgcolor='rgba(13, 17, 23, 0.9)',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.5)  # Good angle to see both surface and paths
            )
        ),
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        height=900,
        width=1200
    )
    
    return fig


def main():
    """Test the hybrid visualization."""
    
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
    fig = create_hybrid_gravity_geodesics_visualization(ohlc_data)
    fig.write_html("hybrid_gravity_geodesics.html")
    
    print("\\n✅ Saved: hybrid_gravity_geodesics.html")
    print("\\n🎯 Open the file to see:")
    print("   - Semi-transparent gravity wells (background spacetime curvature)")
    print("   - ONE CLEAR GEODESIC PATH (bright cyan line with colored markers)")
    print("   - X-axis: Proper time, Y-axis: Low prices, Z-axis: Spacetime depth")
    print("   - Simple, clean visualization with just two elements:")
    print("     1. Curved spacetime surface (gravity wells)")
    print("     2. Single market data path floating above it")


if __name__ == "__main__":
    main()