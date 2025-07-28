"""
Minimal test to isolate curvature coloring issue.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

def create_minimal_test():
    """Create the most minimal test possible."""
    
    # Just 3 candles with obvious different curvature colors
    dates = pd.date_range(start='2024-01-01', periods=3, freq='D')
    
    # Simple OHLC data
    ohlc_data = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 102, 107],
        'low': [95, 100, 97],
        'close': [103, 101.5, 105]
    }, index=dates)
    
    # Create figure without any complexity
    fig = go.Figure()
    
    print("=== MINIMAL TEST ===")
    
    # Manually set curvature colors - no calculation
    curvature_colors = [
        'rgba(255, 0, 255, 0.8)',  # Bright magenta
        'rgba(255, 255, 0, 0.8)',  # Bright yellow  
        'rgba(0, 255, 0, 0.8)'     # Bright green
    ]
    
    for i, (idx, row) in enumerate(ohlc_data.iterrows()):
        O, H, L, C = row['open'], row['high'], row['low'], row['close']
        
        print(f"Candle {i}: {idx}, OHLC=({O}, {H}, {L}, {C})")
        
        # Determine directional color (outline)
        is_bullish = C >= O
        outline_color = '#00ff00' if is_bullish else '#ff0000'
        fill_color = curvature_colors[i]
        
        print(f"  Outline: {outline_color}, Fill: {fill_color}")
        
        # Calculate body dimensions
        body_top = max(O, C)
        body_bottom = min(O, C)
        body_height = body_top - body_bottom
        
        # Add wick
        fig.add_trace(go.Scatter(
            x=[idx, idx],
            y=[L, H],
            mode='lines',
            line=dict(color=outline_color, width=2),
            showlegend=False,
            name=f'Wick {i}'
        ))
        
        # Add body - try SIMPLEST approach first
        if body_height > 0:
            print(f"  Body: bottom={body_bottom:.2f}, top={body_top:.2f}, height={body_height:.2f}")
            
            # Calculate width in days
            candle_width_days = 0.8
            
            # Create filled rectangle using scatter
            x_coords = [
                idx - pd.Timedelta(days=candle_width_days/2),
                idx + pd.Timedelta(days=candle_width_days/2),
                idx + pd.Timedelta(days=candle_width_days/2),
                idx - pd.Timedelta(days=candle_width_days/2),
                idx - pd.Timedelta(days=candle_width_days/2)
            ]
            y_coords = [body_bottom, body_bottom, body_top, body_top, body_bottom]
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                fill='toself',
                fillcolor=fill_color,
                mode='lines',
                line=dict(color=outline_color, width=3),
                showlegend=False,
                name=f'Body {i}',
                hovertemplate=f'Candle {i}<br>Fill: {fill_color}<br>Outline: {outline_color}<extra></extra>'
            ))
        else:
            print(f"  Doji candle - using line")
            # Doji - horizontal line
            fig.add_trace(go.Scatter(
                x=[idx - pd.Timedelta(days=0.4), idx + pd.Timedelta(days=0.4)],
                y=[O, C],
                mode='lines',
                line=dict(color=fill_color, width=8),
                showlegend=False,
                name=f'Doji {i}'
            ))
    
    # Simple layout
    fig.update_layout(
        title="MINIMAL CURVATURE TEST - Should show MAGENTA, YELLOW, GREEN fills",
        xaxis_title="Date",
        yaxis_title="Price",
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400
    )
    
    return fig

if __name__ == "__main__":
    print("🔍 MINIMAL CURVATURE COLOR TEST")
    print("=" * 40)
    
    fig = create_minimal_test()
    fig.write_html("minimal_curvature_test.html")
    print("\n✅ Saved minimal_curvature_test.html")
    print("\n🎯 This should show 3 candles with BRIGHT MAGENTA, YELLOW, GREEN fills")
    print("🎯 If this doesn't work, the issue is with the basic fill approach")