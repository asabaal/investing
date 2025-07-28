"""
Test with shapes instead of scatter plots for outline-only candle.
"""

import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

def create_single_candle_with_shapes():
    """Create ONE candle using shapes for outline only."""
    
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
    
    # Candle width
    candle_half_width = pd.Timedelta(days=0.3)
    left_x = date - candle_half_width
    right_x = date + candle_half_width
    
    # 1. Add wick using scatter
    fig.add_trace(go.Scatter(
        x=[date, date],
        y=[L, H],
        mode='lines',
        line=dict(color=outline_color, width=2),
        showlegend=False,
        name='Wick'
    ))
    print("Added wick")
    
    # 2. Add body outline using shapes (NO FILL)
    if body_height > 0:
        # Add rectangle shape with NO fill
        fig.add_shape(
            type="rect",
            x0=left_x,
            y0=body_bottom,
            x1=right_x,
            y1=body_top,
            line=dict(color=outline_color, width=3),
            fillcolor="rgba(0,0,0,0)",  # Transparent fill
            layer="above"
        )
        print("Added rectangle shape with transparent fill")
    
    # Simple layout
    fig.update_layout(
        title="SINGLE CANDLE WITH SHAPES - Should be GREEN outline with EMPTY center",
        xaxis_title="Date",
        yaxis_title="Price",
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400
    )
    
    return fig

def create_single_candle_lines_only():
    """Create candle using individual line shapes."""
    
    fig = go.Figure()
    
    # Single candle data
    date = datetime(2024, 1, 1)
    O, H, L, C = 100.0, 105.0, 95.0, 103.0
    
    outline_color = '#00ff00'
    body_top = max(O, C)
    body_bottom = min(O, C)
    
    candle_half_width = pd.Timedelta(days=0.3)
    left_x = date - candle_half_width
    right_x = date + candle_half_width
    
    # Wick
    fig.add_shape(
        type="line",
        x0=date, y0=L,
        x1=date, y1=H,
        line=dict(color=outline_color, width=2)
    )
    
    # Body outline using 4 separate line shapes
    # Bottom line
    fig.add_shape(
        type="line",
        x0=left_x, y0=body_bottom,
        x1=right_x, y1=body_bottom,
        line=dict(color=outline_color, width=3)
    )
    
    # Top line
    fig.add_shape(
        type="line",
        x0=left_x, y0=body_top,
        x1=right_x, y1=body_top,
        line=dict(color=outline_color, width=3)
    )
    
    # Left line
    fig.add_shape(
        type="line",
        x0=left_x, y0=body_bottom,
        x1=left_x, y1=body_top,
        line=dict(color=outline_color, width=3)
    )
    
    # Right line
    fig.add_shape(
        type="line",
        x0=right_x, y0=body_bottom,
        x1=right_x, y1=body_top,
        line=dict(color=outline_color, width=3)
    )
    
    print("Added candle using individual line shapes")
    
    fig.update_layout(
        title="SINGLE CANDLE WITH LINE SHAPES - Should be GREEN outline ONLY",
        xaxis_title="Date",
        yaxis_title="Price",
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400
    )
    
    return fig

if __name__ == "__main__":
    print("🔍 TESTING DIFFERENT APPROACHES FOR OUTLINE-ONLY")
    print("=" * 50)
    
    # Test 1: Rectangle shape with transparent fill
    print("\n1. Testing rectangle shape with transparent fill...")
    fig1 = create_single_candle_with_shapes()
    fig1.write_html("single_candle_shapes.html")
    print("✅ Saved single_candle_shapes.html")
    
    # Test 2: Individual line shapes
    print("\n2. Testing individual line shapes...")
    fig2 = create_single_candle_lines_only()
    fig2.write_html("single_candle_lines.html")
    print("✅ Saved single_candle_lines.html")
    
    print("\n🎯 Check both files - one should show outline-only candle!")