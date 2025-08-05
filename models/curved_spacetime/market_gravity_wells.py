"""
Market Gravity Wells 3D Visualization

Creates a 3D landscape where:
- Height = Price level
- Depth = Volatility (creates wells)
- Width = Volume (mass creates larger wells)
- Surface curvature shows combined gravitational effects
"""

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc
from curvature_color_scheme import get_curvature_gradient_color


def create_gravity_well_surface(candle_metrics, proper_times, grid_resolution=50):
    """
    Create a 3D surface representing the gravitational field of the market.
    
    Args:
        candle_metrics: List of CandleMetric objects
        proper_times: Array of proper time coordinates
        grid_resolution: Resolution of the 3D surface grid
    
    Returns:
        x, y, z, colors for 3D surface plot
    """
    if len(candle_metrics) == 0:
        return None, None, None, None
    
    # Create coordinate grids
    time_min, time_max = proper_times[0], proper_times[-1]
    price_min = min(candle.low_value for candle in candle_metrics)
    price_max = max(candle.low_value + candle.range_value for candle in candle_metrics)
    
    # Calculate data ranges for proper scaling
    time_range = time_max - time_min
    price_range = price_max - price_min
    max_range = max(candle.range_value for candle in candle_metrics)
    
    # Expand ranges slightly for better visualization
    time_min -= time_range * 0.1
    time_max += time_range * 0.1
    price_min -= price_range * 0.1
    price_max += price_range * 0.1
    
    # Create meshgrid
    t_grid = np.linspace(time_min, time_max, grid_resolution)
    p_grid = np.linspace(price_min, price_max, grid_resolution)
    T, P = np.meshgrid(t_grid, p_grid)
    
    # Initialize the gravitational potential surface
    Z = np.zeros_like(T)  # Start with flat space
    Colors = np.zeros_like(T)  # Color based on curvature strength
    
    # Scale well depths relative to price range (make them reasonable)
    depth_scale = price_range * 0.15  # Wells can be at most 15% of price range deep
    
    # Add gravitational wells for each candle
    for i, candle in enumerate(candle_metrics):
        candle_time = proper_times[i]
        candle_price = candle.low_value + candle.range_value / 2  # Middle of candle
        
        # Well parameters - scaled appropriately
        volatility_depth = (candle.range_value / max_range) * depth_scale  # Proportional depth
        mass_width = np.sqrt(candle.volume) * 0.15  # Wider wells for higher volume
        
        # Distance from this candle in spacetime
        time_dist = (T - candle_time) / (time_range * 0.2)  # Normalize time distance
        price_dist = (P - candle_price) / (price_range * 0.2)  # Normalize price distance
        spacetime_dist = np.sqrt(time_dist**2 + price_dist**2)
        
        # Gaussian-like gravitational well
        # Deeper for higher volatility, wider for higher volume
        well_depth = volatility_depth * np.exp(-spacetime_dist**2 / (2 * mass_width**2))
        
        # Add this well to the surface (negative for depression)
        Z -= well_depth
        
        # Color intensity based on well strength
        Colors += well_depth
    
    # Normalize colors for better visualization
    if Colors.max() > Colors.min():
        Colors = (Colors - Colors.min()) / (Colors.max() - Colors.min())
    
    return T, P, Z, Colors


def create_candle_markers_3d(candle_metrics, proper_times, curvatures):
    """
    Create 3D markers showing candle positions on the gravity well surface.
    """
    x_markers = proper_times
    y_markers = [candle.low_value + candle.range_value / 2 for candle in candle_metrics]
    
    # Calculate price range for proper marker elevation
    price_range = max(candle.low_value + candle.range_value for candle in candle_metrics) - \
                  min(candle.low_value for candle in candle_metrics)
    
    # Place markers slightly above surface level (scaled to data)
    marker_elevation = price_range * 0.02  # 2% of price range above surface
    z_markers = [marker_elevation] * len(candle_metrics)
    
    # Size markers by volume (mass) - scaled appropriately
    max_volume = max(candle.volume for candle in candle_metrics)
    sizes = [8 + (candle.volume / max_volume) * 12 for candle in candle_metrics]  # 8-20 range
    
    # Color markers by curvature
    colors = [get_curvature_gradient_color(c)[0] for c in curvatures]
    
    return x_markers, y_markers, z_markers, sizes, colors


def create_market_gravity_wells_visualization(ohlc_data: pd.DataFrame) -> go.Figure:
    """
    Create the Market Gravity Wells 3D visualization.
    """
    print("🌌 CREATING MARKET GRAVITY WELLS VISUALIZATION")
    print("=" * 60)
    
    # Compute spacetime geometry with volume effects
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    geometry = CurvedCandleGeometry(candle_metrics)
    curvatures = geometry.compute_curvature_series()
    proper_times = geometry.compute_proper_time_series()
    
    print(f"Processing {len(candle_metrics)} candles with volume data...")
    
    # Create the gravitational field surface
    T, P, Z, Colors = create_gravity_well_surface(candle_metrics, proper_times)
    
    if T is None:
        print("❌ Failed to create gravity well surface")
        return go.Figure()
    
    # Create 3D surface plot
    fig = go.Figure()
    
    # Add the gravitational field surface
    fig.add_trace(go.Surface(
        x=T,
        y=P,
        z=Z,
        surfacecolor=Colors,
        colorscale=[
            [0, '#0d1117'],      # Dark background
            [0.2, '#2d1b69'],    # Deep purple (strong wells)
            [0.4, '#642c8a'],    # Purple
            [0.6, '#9d4edd'],    # Light purple
            [0.8, '#c77dff'],    # Lavender
            [1, '#e0aaff']       # Light lavender (weak fields)
        ],
        opacity=0.8,
        name='Gravitational Field',
        showscale=True,
        colorbar=dict(
            title="Field Strength",
            titlefont=dict(color='#c9d1d9'),
            tickfont=dict(color='#c9d1d9')
        )
    ))
    
    # Add candle markers
    x_markers, y_markers, z_markers, sizes, colors = create_candle_markers_3d(
        candle_metrics, proper_times, curvatures
    )
    
    fig.add_trace(go.Scatter3d(
        x=x_markers,
        y=y_markers,
        z=z_markers,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(color='white', width=2),
            opacity=0.9
        ),
        name='Market Candles',
        hovertemplate='<b>Candle %{customdata}</b><br>' +
                      'Proper Time: %{x:.2f}<br>' +
                      'Price: %{y:.2f}<br>' +
                      'Volume: %{customdata}<extra></extra>',
        customdata=[f"{i}: Vol={candle.volume:,}" for i, candle in enumerate(candle_metrics)]
    ))
    
    # Calculate appropriate Z-axis range based on actual well depths
    if Z is not None:
        z_min, z_max = Z.min(), Z.max()
        z_range = z_max - z_min
        # Add some padding to the Z range
        z_axis_min = z_min - z_range * 0.1
        z_axis_max = z_max + z_range * 0.3  # More space on top for markers
    else:
        z_axis_min, z_axis_max = -1, 1
    
    # Layout for dark mode 3D visualization
    fig.update_layout(
        title="Market Gravity Wells - 3D Spacetime Visualization",
        scene=dict(
            xaxis=dict(
                title="Proper Time (τ)",
                backgroundcolor='#0d1117',
                gridcolor='#30363d',
                tickfont=dict(color='#c9d1d9')
            ),
            yaxis=dict(
                title="Price",
                backgroundcolor='#0d1117',
                gridcolor='#30363d',
                tickfont=dict(color='#c9d1d9')
            ),
            zaxis=dict(
                title="Gravitational Potential",
                range=[z_axis_min, z_axis_max],  # Dynamic range based on data
                backgroundcolor='#0d1117',
                gridcolor='#30363d',
                tickfont=dict(color='#c9d1d9')
            ),
            bgcolor='#0d1117',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)  # Good viewing angle
            )
        ),
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        height=800,
        width=1200
    )
    
    return fig


def main():
    """Test the Market Gravity Wells visualization."""
    
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
    fig = create_market_gravity_wells_visualization(ohlc_data)
    fig.write_html("market_gravity_wells.html")
    
    print("\\n✅ Saved: market_gravity_wells.html")
    print("\\n🎯 Open the file to see:")
    print("   - 3D gravitational field created by volume + volatility")
    print("   - Deep wells for high-volume volatile candles")
    print("   - Shallow depressions for low-volume calm candles")
    print("   - Candle markers colored by spacetime curvature")
    print("   - Proper time coordinates showing time dilation effects")


if __name__ == "__main__":
    main()