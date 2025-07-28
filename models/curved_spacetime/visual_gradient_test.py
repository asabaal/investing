"""
Visual test to see if gradient colors are actually appearing.
Create a simple test with extreme curvature differences.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc
from shared_candle_visualization import create_curvature_candlestick_chart_shared

def create_extreme_gradient_test():
    """Create candles designed to show maximum gradient differences."""
    
    print("🎨 CREATING EXTREME GRADIENT TEST")
    print("=" * 50)
    
    dates = [
        datetime(2024, 1, 1, 9, 0),   # Extreme positive curvature
        datetime(2024, 1, 1, 10, 0),  # Moderate positive  
        datetime(2024, 1, 1, 11, 0),  # Neutral
        datetime(2024, 1, 1, 12, 0),  # Moderate negative
        datetime(2024, 1, 1, 13, 0),  # Extreme negative
    ]
    
    # Design candles to get specific curvature ranges
    ohlc_data = pd.DataFrame({
        # Candle 0: Large range + balanced pattern = HIGH positive curvature = CYAN
        # Candle 1: Medium range + balanced = moderate positive = PURPLE  
        # Candle 2: Medium range + neutral = neutral = GRAY
        # Candle 3: Small range + imbalanced = moderate negative = YELLOW
        # Candle 4: Very small range + very imbalanced = HIGH negative = RED
        'open': [100.0,  105.0,  107.0,  108.0,  109.0],
        'high': [120.0,  112.0,  110.0,  109.0,  109.1],  # Decreasing ranges: 20, 7, 3, 1, 0.1
        'low':  [80.0,   98.0,   104.0,  107.5,  108.9],  # Decreasing ranges
        'close': [110.0, 106.0,  107.5,  107.6,  108.9],  # Varying sentiment patterns
    }, index=dates)
    
    print("OHLC Data:")
    print(ohlc_data)
    
    # Debug curvature calculation
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    geometry = CurvedCandleGeometry(candle_metrics)
    curvatures = geometry.compute_curvature_series()
    
    print("\\nCandle Analysis:")
    for i, (candle, curvature) in enumerate(zip(candle_metrics, curvatures)):
        from curvature_color_scheme import get_curvature_gradient_color
        color, description = get_curvature_gradient_color(curvature)
        
        range_val = candle.range_value
        sentiment = candle.sentiment  
        uwr = candle.upper_wick_ratio
        
        print(f"Candle {i}:")
        print(f"  Range: {range_val:.1f}")
        print(f"  Sentiment: {sentiment:.3f}, UWR: {uwr:.3f}")
        print(f"  Curvature: {curvature:.4f}")
        print(f"  Color: {color} ({description})")
        print()
    
    # Create visualization
    fig = create_curvature_candlestick_chart_shared(
        ohlc_data=ohlc_data,
        curvatures=curvatures,
        title="EXTREME GRADIENT TEST - Should show Red→Yellow→Gray→Purple→Cyan",
        debug=True
    )
    
    return fig

def create_color_bar_reference():
    """Create a color bar showing the full gradient for reference."""
    
    print("\\n🎨 CREATING COLOR BAR REFERENCE")
    print("=" * 40)
    
    from curvature_color_scheme import get_curvature_gradient_color
    
    fig = go.Figure()
    
    # Create color bar with many curvature values
    curvature_values = np.linspace(-1, 1, 21)  # -1.0 to 1.0 in 0.1 steps
    
    for i, curvature in enumerate(curvature_values):
        color, description = get_curvature_gradient_color(curvature)
        
        # Create a small rectangle for each color
        x_start = i * 0.5
        x_end = x_start + 0.4
        
        fig.add_trace(go.Scatter(
            x=[x_start, x_end, x_end, x_start, x_start],
            y=[0, 0, 1, 1, 0],
            fill='toself',
            fillcolor=color,
            mode='lines',
            line=dict(width=1, color='white'),
            showlegend=False,
            hovertemplate=f'Curvature: {curvature:.2f}<br>{description}<extra></extra>'
        ))
        
        # Add text label
        fig.add_annotation(
            x=x_start + 0.2,
            y=-0.3,
            text=f"{curvature:.1f}",
            showarrow=False,
            font=dict(color='white', size=8),
            textangle=45
        )
    
    fig.update_layout(
        title="Curvature Color Gradient Reference",
        xaxis=dict(visible=False, range=[-0.5, len(curvature_values) * 0.5]),
        yaxis=dict(visible=False, range=[-0.5, 1.5]),
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        height=200,
        width=1200
    )
    
    return fig

def main():
    """Run visual gradient tests."""
    
    print("🎨 VISUAL GRADIENT TESTING")
    print("=" * 60)
    
    # Test 1: Extreme gradient differences
    fig1 = create_extreme_gradient_test()
    fig1.write_html("visual_gradient_test.html")
    print("✅ Saved: visual_gradient_test.html")
    
    # Test 2: Color bar reference
    fig2 = create_color_bar_reference()
    fig2.write_html("color_bar_reference.html")
    print("✅ Saved: color_bar_reference.html")
    
    print("\\n🎯 Check these files to see if gradient colors are working:")
    print("  - visual_gradient_test.html (should show 5 candles with very different colors)")
    print("  - color_bar_reference.html (should show full color gradient)")

if __name__ == "__main__":
    main()