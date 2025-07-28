"""
Test creating candlesticks with ONLY red/green outlines and empty centers.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def create_simple_ohlc():
    """Create simple OHLC data."""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
    
    data = []
    for i, date in enumerate(dates):
        base_price = 100 + i * 2
        
        open_price = base_price
        close_price = base_price + np.random.uniform(-2, 2)
        high_price = max(open_price, close_price) + 1
        low_price = min(open_price, close_price) - 1
        
        data.append({
            'open': open_price,
            'high': high_price, 
            'low': low_price,
            'close': close_price
        })
    
    return pd.DataFrame(data, index=dates)

def create_outline_only_candlesticks(ohlc_data):
    """Create candlesticks with ONLY outlines, NO fill."""
    
    fig = go.Figure()
    
    print("=== OUTLINE ONLY TEST ===")
    
    for i, (idx, row) in enumerate(ohlc_data.iterrows()):
        O, H, L, C = row['open'], row['high'], row['low'], row['close']
        
        # Determine directional color (outline only)
        is_bullish = C >= O
        outline_color = '#00ff00' if is_bullish else '#ff0000'
        
        print(f"Candle {i}: OHLC=({O:.2f}, {H:.2f}, {L:.2f}, {C:.2f}), Outline={outline_color}")
        
        # Body dimensions
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
            hoverinfo='skip'
        ))
        
        # Add body outline ONLY (no fill)
        if body_height > 0:
            # Calculate width
            x_range = ohlc_data.index[-1] - ohlc_data.index[0]
            candle_half_width = x_range / len(ohlc_data) * 0.4
            
            # Create rectangle outline using scatter - NO FILL
            x_coords = [
                idx - candle_half_width,
                idx + candle_half_width, 
                idx + candle_half_width,
                idx - candle_half_width,
                idx - candle_half_width
            ]
            y_coords = [body_bottom, body_bottom, body_top, body_top, body_bottom]
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',  # ONLY lines, NO fill
                line=dict(color=outline_color, width=3),
                showlegend=False,
                hovertemplate=f'Candle {i}<br>Direction: {"Bull" if is_bullish else "Bear"}<extra></extra>'
            ))
        else:
            # Doji - horizontal line
            x_range = ohlc_data.index[-1] - ohlc_data.index[0]
            line_half_width = x_range / len(ohlc_data) * 0.4
            
            fig.add_trace(go.Scatter(
                x=[idx - line_half_width, idx + line_half_width],
                y=[O, C],
                mode='lines',
                line=dict(color=outline_color, width=4),
                showlegend=False
            ))
    
    # Simple layout
    fig.update_layout(
        title="OUTLINE ONLY TEST - Should show RED/GREEN outlines with EMPTY centers",
        xaxis_title="Date",
        yaxis_title="Price",
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        height=500
    )
    
    return fig

if __name__ == "__main__":
    print("🔍 TESTING OUTLINE-ONLY CANDLESTICKS")
    print("=" * 40)
    
    ohlc_data = create_simple_ohlc()
    fig = create_outline_only_candlesticks(ohlc_data)
    fig.write_html("test_outline_only.html")
    print("\n✅ Saved test_outline_only.html")
    print("\n🎯 This should show candlesticks with RED/GREEN outlines and EMPTY centers")