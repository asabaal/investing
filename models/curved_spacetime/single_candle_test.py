"""
Test with just ONE candle to get outline-only working.
"""

import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

def create_single_outline_candle():
    """Create ONE candle with outline only."""
    
    fig = go.Figure()
    
    # Single candle data
    date = datetime(2024, 1, 1)
    O, H, L, C = 100.0, 105.0, 95.0, 103.0  # Bullish candle
    
    print(f"Single candle: O={O}, H={H}, L={L}, C={C}")
    
    # Determine color
    is_bullish = C >= O
    outline_color = '#00ff00' if is_bullish else '#ff0000'
    print(f"Bullish: {is_bullish}, Outline color: {outline_color}")
    
    # Body dimensions
    body_top = max(O, C)  # 103.0
    body_bottom = min(O, C)  # 100.0
    body_height = body_top - body_bottom  # 3.0
    
    print(f"Body: bottom={body_bottom}, top={body_top}, height={body_height}")
    
    # Candle width (simple fixed width)
    candle_half_width = pd.Timedelta(days=0.3)
    left_x = date - candle_half_width
    right_x = date + candle_half_width
    
    print(f"Candle width: {left_x} to {right_x}")
    
    # 1. Add wick
    fig.add_trace(go.Scatter(
        x=[date, date],
        y=[L, H],
        mode='lines',
        line=dict(color=outline_color, width=2),
        showlegend=False,
        name='Wick'
    ))
    print("Added wick")
    
    # 2. Add body outline using 4 separate lines
    if body_height > 0:
        # Bottom line
        fig.add_trace(go.Scatter(
            x=[left_x, right_x],
            y=[body_bottom, body_bottom],
            mode='lines',
            line=dict(color=outline_color, width=3),
            showlegend=False,
            name='Bottom'
        ))
        print("Added bottom line")
        
        # Top line  
        fig.add_trace(go.Scatter(
            x=[left_x, right_x],
            y=[body_top, body_top],
            mode='lines',
            line=dict(color=outline_color, width=3),
            showlegend=False,
            name='Top'
        ))
        print("Added top line")
        
        # Left line
        fig.add_trace(go.Scatter(
            x=[left_x, left_x],
            y=[body_bottom, body_top],
            mode='lines',
            line=dict(color=outline_color, width=3),
            showlegend=False,
            name='Left'
        ))
        print("Added left line")
        
        # Right line
        fig.add_trace(go.Scatter(
            x=[right_x, right_x],
            y=[body_bottom, body_top],
            mode='lines',
            line=dict(color=outline_color, width=3),
            showlegend=False,
            name='Right'
        ))
        print("Added right line")
    
    # Simple layout
    fig.update_layout(
        title="SINGLE CANDLE TEST - Should be GREEN outline with EMPTY center",
        xaxis_title="Date",
        yaxis_title="Price",
        plot_bgcolor='white',  # White background to see clearly
        paper_bgcolor='white',
        height=400,
        showlegend=True  # Show legend to see all traces
    )
    
    return fig

if __name__ == "__main__":
    print("🔍 SINGLE CANDLE OUTLINE TEST")
    print("=" * 30)
    
    fig = create_single_outline_candle()
    fig.write_html("single_candle_test.html")
    print("\n✅ Saved single_candle_test.html")
    print("\n🎯 This should show ONE green-outlined candle with EMPTY center")
    print("🎯 If this still shows fill, the issue is fundamental")