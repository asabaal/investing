"""
Debug script to test curvature coloring issues.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc

def create_simple_test_data():
    """Create simple test data to debug coloring."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    
    # Create data with obvious curvature differences
    data = []
    for i, date in enumerate(dates):
        base_price = 100 + i * 2
        
        # Alternate between high and low volatility to create obvious curvature
        if i % 2 == 0:
            # High volatility = negative curvature (should be yellow)
            volatility = 5.0
        else:
            # Low volatility = positive curvature (should be purple)
            volatility = 0.5
            
        open_price = base_price
        close_price = base_price + np.random.uniform(-volatility, volatility)
        high_price = max(open_price, close_price) + volatility
        low_price = min(open_price, close_price) - volatility
        
        data.append({
            'open': open_price,
            'high': high_price, 
            'low': low_price,
            'close': close_price
        })
    
    return pd.DataFrame(data, index=dates)

def debug_curvature_calculation(ohlc_data):
    """Debug the curvature calculation."""
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    geometry = CurvedCandleGeometry(candle_metrics)
    
    curvatures = geometry.compute_curvature_series()
    curvature_normalized = np.clip(curvatures, -1, 1)
    
    print("=== CURVATURE DEBUG ===")
    for i, (idx, row) in enumerate(ohlc_data.iterrows()):
        print(f"Candle {i}: Date={idx}, O={row['open']:.2f}, H={row['high']:.2f}, L={row['low']:.2f}, C={row['close']:.2f}")
        print(f"  Range: {row['high'] - row['low']:.2f}")
        print(f"  Raw Curvature: {curvatures[i]:.6f}")
        print(f"  Normalized Curvature: {curvature_normalized[i]:.6f}")
        
        # Test color calculation
        curvature_val = curvature_normalized[i]
        if curvature_val > 0:
            intensity = min(curvature_val, 1.0)
            fill_color = f'rgba({int(128 + 127*intensity)}, 0, {int(255*intensity)}, 0.8)'
            color_type = "TRENDING (Purple)"
        elif curvature_val < 0:
            intensity = min(abs(curvature_val), 1.0)
            fill_color = f'rgba({int(255*intensity)}, {int(255*intensity)}, 0, 0.8)'
            color_type = "VOLATILE (Yellow)"
        else:
            fill_color = 'rgba(128, 128, 128, 0.8)'
            color_type = "NEUTRAL (Gray)"
            
        print(f"  Expected Color: {fill_color} ({color_type})")
        print()
    
    return geometry, curvature_normalized

def create_debug_chart(ohlc_data, geometry, curvature_normalized):
    """Create a debug chart to test curvature coloring."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.85, 0.15],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("DEBUG: Curvature-Filled Candlesticks", "Color Test")
    )
    
    print("\n=== CHART CREATION DEBUG ===")
    
    for i, (idx, row) in enumerate(ohlc_data.iterrows()):
        O, H, L, C = row['open'], row['high'], row['low'], row['close']
        
        # Determine directional color (outline)
        is_bullish = C >= O
        outline_color = '#00ff00' if is_bullish else '#ff0000'
        
        # Determine curvature color (fill)
        curvature_val = curvature_normalized[i]
        if curvature_val > 0:
            intensity = min(curvature_val, 1.0)
            fill_color = f'rgba({int(128 + 127*intensity)}, 0, {int(255*intensity)}, 0.8)'
        elif curvature_val < 0:
            intensity = min(abs(curvature_val), 1.0)
            fill_color = f'rgba({int(255*intensity)}, {int(255*intensity)}, 0, 0.8)'
        else:
            fill_color = 'rgba(128, 128, 128, 0.8)'
        
        print(f"Candle {i}: Outline={outline_color}, Fill={fill_color}, Curvature={curvature_val:.4f}")
        
        # Calculate body dimensions
        body_top = max(O, C)
        body_bottom = min(O, C)
        body_height = body_top - body_bottom
        
        # Add wick
        fig.add_trace(go.Scatter(
            x=[idx, idx],
            y=[L, H],
            mode='lines',
            line=dict(color=outline_color, width=1),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)
        
        # Add body using shape - TEST THIS CAREFULLY
        if body_height > 0:
            x_range = ohlc_data.index[-1] - ohlc_data.index[0]
            candle_half_width = x_range / len(ohlc_data) * 0.4
            
            print(f"  Shape: x0={idx - candle_half_width}, x1={idx + candle_half_width}, y0={body_bottom:.2f}, y1={body_top:.2f}")
            print(f"  Fill Color: {fill_color}")
            
            fig.add_shape(
                type="rect",
                x0=idx - candle_half_width,
                y0=body_bottom,
                x1=idx + candle_half_width,
                y1=body_top,
                fillcolor=fill_color,
                line=dict(color=outline_color, width=2),
                layer="above",
                row=1, col=1
            )
        else:
            # Doji - horizontal line with curvature color
            x_range = ohlc_data.index[-1] - ohlc_data.index[0]
            line_half_width = x_range / len(ohlc_data) * 0.4
            
            fig.add_trace(go.Scatter(
                x=[idx - line_half_width, idx + line_half_width],
                y=[O, C],
                mode='lines',
                line=dict(color=fill_color, width=8),
                showlegend=False
            ), row=1, col=1)
    
    # Add color test in bottom subplot
    test_colors = [
        'rgba(255, 0, 128, 0.8)',  # Purple
        'rgba(255, 255, 0, 0.8)',  # Yellow
        'rgba(128, 128, 128, 0.8)' # Gray
    ]
    test_labels = ['Trending Test', 'Volatile Test', 'Neutral Test']
    
    fig.add_trace(go.Scatter(
        x=[0, 1, 2],
        y=[0, 0, 0],
        mode='markers',
        marker=dict(size=20, color=test_colors),
        text=test_labels,
        showlegend=False
    ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title="DEBUG: Curvature Color Test",
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        height=700
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1, 
                     gridcolor='#30363d', zerolinecolor='#30363d', color='#c9d1d9')
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(gridcolor='#30363d', row=1, col=1)
    
    return fig

def main():
    print("🔍 DEBUGGING CURVATURE COLORS")
    print("=" * 40)
    
    # Create simple test data
    ohlc_data = create_simple_test_data()
    print(f"Created {len(ohlc_data)} test candles")
    
    # Debug curvature calculation
    geometry, curvature_normalized = debug_curvature_calculation(ohlc_data)
    
    # Create debug chart
    fig = create_debug_chart(ohlc_data, geometry, curvature_normalized)
    fig.write_html("debug_curvature_colors.html")
    print("\n✅ Saved debug_curvature_colors.html")
    
    print("\n🎯 If you see curvature colors in the debug chart, the issue is elsewhere.")
    print("🎯 If you still see only red/green, the issue is in the color application.")

if __name__ == "__main__":
    main()