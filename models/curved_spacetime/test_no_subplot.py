"""
Test curvature candlesticks WITHOUT subplots to isolate the issue.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc

def create_simple_ohlc():
    """Create simple OHLC data."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    
    data = []
    for i, date in enumerate(dates):
        base_price = 100 + i * 2
        
        # Alternate between high and low volatility
        if i % 2 == 0:
            volatility = 5.0  # High volatility = negative curvature
        else:
            volatility = 0.5  # Low volatility = positive curvature
            
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

def create_curvature_candlestick_no_subplot(ohlc_data):
    """Create curvature candlesticks WITHOUT subplots."""
    
    # Get curvature
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    geometry = CurvedCandleGeometry(candle_metrics)
    curvatures = geometry.compute_curvature_series()
    curvature_normalized = np.clip(curvatures, -1, 1)
    
    # Create simple figure (NO SUBPLOTS)
    fig = go.Figure()
    
    print("=== NO SUBPLOT TEST ===")
    
    for i, (idx, row) in enumerate(ohlc_data.iterrows()):
        O, H, L, C = row['open'], row['high'], row['low'], row['close']
        
        # Determine colors
        is_bullish = C >= O
        outline_color = '#00ff00' if is_bullish else '#ff0000'
        
        # Curvature color
        curvature_val = curvature_normalized[i]
        if curvature_val > 0:
            intensity = min(curvature_val, 1.0)
            fill_color = f'rgba({int(128 + 127*intensity)}, 0, {int(255*intensity)}, 0.8)'
        elif curvature_val < 0:
            intensity = min(abs(curvature_val), 1.0)
            fill_color = f'rgba({int(255*intensity)}, {int(255*intensity)}, 0, 0.8)'
        else:
            fill_color = 'rgba(128, 128, 128, 0.8)'
        
        print(f"Candle {i}: Curvature={curvature_val:.4f}, Fill={fill_color}")
        
        # Body dimensions
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
        ))
        
        # Add body
        if body_height > 0:
            # Calculate width
            x_range = ohlc_data.index[-1] - ohlc_data.index[0]
            candle_half_width = x_range / len(ohlc_data) * 0.4
            
            # Create filled rectangle
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
                fill='toself',
                fillcolor=fill_color,
                mode='lines',
                line=dict(color=outline_color, width=2),
                showlegend=False,
                hovertemplate=f'Curvature: {curvature_val:.4f}<br>Color: {fill_color}<extra></extra>'
            ))
    
    # Simple layout
    fig.update_layout(
        title="Curvature Candlesticks (No Subplot) - Should Show Purple/Yellow Fills",
        xaxis_title="Date",
        yaxis_title="Price",
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        height=500
    )
    
    return fig

if __name__ == "__main__":
    print("🔍 TESTING CURVATURE COLORS WITHOUT SUBPLOTS")
    print("=" * 50)
    
    ohlc_data = create_simple_ohlc()
    fig = create_curvature_candlestick_no_subplot(ohlc_data)
    fig.write_html("test_no_subplot.html")
    print("\n✅ Saved test_no_subplot.html")
    print("\n🎯 This should show curvature-filled candles WITHOUT subplot interference")