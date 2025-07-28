"""
Simple test: Create ONLY 1 candle with outline only (no fill).
"""

import plotly.graph_objects as go
from datetime import datetime

def create_one_hollow_candle():
    """Create exactly ONE candle that should be hollow (outline only)."""
    
    fig = go.Figure()
    
    print("Creating ONE candle with hollow body...")
    
    # Simple candle data - use simple numbers instead of datetime for now
    center_x = 1  # Simple x coordinate
    open_price = 100.0
    high_price = 105.0
    low_price = 95.0
    close_price = 103.0  # Bullish (close > open)
    
    print(f"Candle data: O={open_price}, H={high_price}, L={low_price}, C={close_price}")
    print(f"Center X: {center_x}")
    
    # Green outline for bullish
    outline_color = '#00ff00'
    print(f"Using outline color: {outline_color}")
    
    # Body coordinates
    body_bottom = min(open_price, close_price)  # 100.0
    body_top = max(open_price, close_price)     # 103.0
    
    print(f"Body: {body_bottom} to {body_top}")
    
    # Candle width (simple numbers)
    half_width = 0.3
    left_x = center_x - half_width   # 0.7
    right_x = center_x + half_width  # 1.3
    
    print(f"Candle width: {left_x} to {right_x}")
    
    # METHOD 1: First try with scatter plots to see if anything shows up
    print("\nMethod 1: Using scatter plots first")
    
    # Wick using scatter
    fig.add_trace(go.Scatter(
        x=[center_x, center_x],
        y=[low_price, high_price],
        mode='lines',
        line=dict(color=outline_color, width=4),
        showlegend=False,
        name='Wick'
    ))
    print("  Added wick with scatter")
    
    # Body outline using 4 scatter lines
    # Bottom edge
    fig.add_trace(go.Scatter(
        x=[left_x, right_x],
        y=[body_bottom, body_bottom],
        mode='lines',
        line=dict(color=outline_color, width=4),
        showlegend=False,
        name='Bottom'
    ))
    print("  Added bottom edge")
    
    # Top edge
    fig.add_trace(go.Scatter(
        x=[left_x, right_x],
        y=[body_top, body_top],
        mode='lines',
        line=dict(color=outline_color, width=4),
        showlegend=False,
        name='Top'
    ))
    print("  Added top edge")
    
    # Left edge
    fig.add_trace(go.Scatter(
        x=[left_x, left_x],
        y=[body_bottom, body_top],
        mode='lines',
        line=dict(color=outline_color, width=4),
        showlegend=False,
        name='Left'
    ))
    print("  Added left edge")
    
    # Right edge
    fig.add_trace(go.Scatter(
        x=[right_x, right_x],
        y=[body_bottom, body_top],
        mode='lines',
        line=dict(color=outline_color, width=4),
        showlegend=False,
        name='Right'
    ))
    print("  Added right edge")
    
    # NOW ADD THE CURVATURE FILL!
    print("\nAdding curvature fill...")
    
    # Simulate a curvature value (let's say this candle has positive curvature = trending)
    curvature_value = 0.5  # Positive = trending
    print(f"Curvature value: {curvature_value}")
    
    # Calculate curvature fill color (using our vibrant scheme)
    if curvature_value > 0:
        # Positive curvature = trending (electric purple to magenta)
        intensity = min(curvature_value, 1.0)
        fill_color = f'rgba({int(128 + 127*intensity)}, 0, {int(255*intensity)}, 0.8)'
        fill_type = "TRENDING (Purple)"
    elif curvature_value < 0:
        # Negative curvature = volatile (electric cyan to yellow)
        intensity = min(abs(curvature_value), 1.0)
        fill_color = f'rgba({int(255*intensity)}, {int(255*intensity)}, 0, 0.8)'
        fill_type = "VOLATILE (Yellow)"
    else:
        # Zero curvature = neutral (gray)
        fill_color = 'rgba(128, 128, 128, 0.8)'
        fill_type = "NEUTRAL (Gray)"
    
    print(f"Fill color: {fill_color} ({fill_type})")
    
    # Add filled rectangle for curvature
    fig.add_trace(go.Scatter(
        x=[left_x, right_x, right_x, left_x, left_x],  # Rectangle coordinates
        y=[body_bottom, body_bottom, body_top, body_top, body_bottom],  # Close the shape
        fill='toself',
        fillcolor=fill_color,
        mode='lines',
        line=dict(width=0, color='rgba(0,0,0,0)'),  # Invisible outline for fill
        showlegend=False,
        name='Curvature Fill'
    ))
    print("  Added curvature fill")
    
    # Add a reference point to make sure something shows up
    fig.add_trace(go.Scatter(
        x=[center_x],
        y=[close_price],
        mode='markers',
        marker=dict(color='white', size=8),
        showlegend=False,
        name='Reference Point'
    ))
    print("  Added white reference point")
    
    # Update layout with dark mode
    fig.update_layout(
        title="CURVATURE-FILLED CANDLE TEST - GREEN outline with PURPLE curvature fill",
        xaxis=dict(
            title="X Position",
            range=[0, 2],
            gridcolor='#30363d',
            zerolinecolor='#30363d',
            color='#c9d1d9'
        ),
        yaxis=dict(
            title="Price",
            range=[90, 110],
            gridcolor='#30363d',
            zerolinecolor='#30363d',
            color='#c9d1d9'
        ),
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        width=800,
        height=500
    )
    
    print("\nLayout configured with dark mode")
    
    return fig

if __name__ == "__main__":
    print("=" * 50)
    print("SIMPLE SINGLE CANDLE TEST")
    print("=" * 50)
    
    fig = create_one_hollow_candle()
    
    # Save to new HTML file
    output_file = "simple_single_candle.html"
    fig.write_html(output_file)
    
    print(f"\n✅ Saved to: {output_file}")
    print("\n🎯 Expected result:")
    print("   - ONE green candle outline (bullish direction)")
    print("   - Wick from 95 to 105 (green)")
    print("   - Body outline from 100 to 103 (green)")
    print("   - PURPLE FILL inside the body (trending curvature)")
    print("   - White reference dot in center")
    print("\n🎉 This should be our FIRST successful curvature-filled candle!")