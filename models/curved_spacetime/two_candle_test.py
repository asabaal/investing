"""
Test with exactly 2 candles using datetime coordinates.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc
from curvature_color_scheme import get_curvature_gradient_color
from shared_candle_visualization import create_curvature_candlestick_chart_shared

def create_two_candle_test():
    """Create exactly 2 candlesticks with curvature fills."""
    
    print("🔧 TWO CANDLE TEST WITH DATETIME COORDINATES")
    print("=" * 50)
    
    # Create 2 candles with different Range values to produce different curvatures
    dates = [
        datetime(2024, 1, 1, 9, 0),   # 9 AM
        datetime(2024, 1, 1, 10, 0)   # 10 AM
    ]
    
    # First candle: Large range + balanced pattern = trending (positive curvature) = PURPLE
    # Second candle: Small range + very unbalanced pattern = volatile (negative curvature) = YELLOW
    ohlc_data = pd.DataFrame({
        'open': [100.0, 104.0],
        'high': [110.0, 106.0],     # Large range (20) vs small range (3)
        'low': [90.0, 103.0],       # Balanced vs extreme patterns
        'close': [105.0, 103.2]     # Balanced vs very bearish (sentiment = -0.8/3 = -0.27, UWR = 2.8/3 = 0.93)
    }, index=dates)
    
    print("OHLC Data:")
    print(ohlc_data)
    
    # DEBUG the curvature calculation step by step
    print("\n=== DEBUGGING CURVATURE CALCULATION ===")
    
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    print("Candle metrics created:")
    for i, candle in enumerate(candle_metrics):
        print(f"  Candle {i}: range_value={candle.range_value:.4f}, low_value={candle.low_value:.4f}")
        print(f"           Sentiment={candle.pattern_coordinates[0]:.4f}, UWR={candle.pattern_coordinates[1]:.4f}")
        print(f"           Metric tensor scale: {candle.range_value**2:.4f}")
    
    geometry = CurvedCandleGeometry(candle_metrics)
    print(f"Geometry created with {geometry.n_candles} candles")
    
    curvatures = geometry.compute_curvature_series()
    print(f"Raw curvatures: {curvatures}")
    print(f"Curvature types: {[type(c) for c in curvatures]}")
    
    curvature_normalized = np.clip(curvatures, -1, 1)
    print(f"Normalized curvatures: {curvature_normalized}")
    
    # Let's also manually check if the ranges are different
    ranges = [candle.range_value for candle in candle_metrics]
    print(f"Manual range_value check: {ranges}")
    print(f"Range difference: {ranges[1] - ranges[0] if len(ranges) > 1 else 'N/A'}")
    
    # Let's also check the actual curvature computation method
    print("Let's check what compute_curvature_series actually does...")
    
    print("=== END CURVATURE DEBUG ===\n")
    
    # Use shared visualization logic
    fig = create_curvature_candlestick_chart_shared(
        ohlc_data=ohlc_data,
        curvatures=curvatures,
        title="TWO CANDLE TEST: Should show Green/Red outlines + Gradient fills",
        debug=True
    )
    
    # Process each candle using EXACT single candle logic but with datetime
    for i, (idx, row) in enumerate(ohlc_data.iterrows()):
        O, H, L, C = row['open'], row['high'], row['low'], row['close']
        
        print(f"\nProcessing candle {i}:")
        print(f"  Date: {idx}")
        print(f"  OHLC: O={O}, H={H}, L={L}, C={C}")
        
        # Colors - EXACT from single candle test
        is_bullish = C >= O
        outline_color = '#00ff00' if is_bullish else '#ff0000'
        
        # Use gradient color scheme
        curvature_val = curvature_normalized[i]
        fill_color, fill_type = get_curvature_gradient_color(curvature_val)
        
        print(f"  Direction: {'Bullish' if is_bullish else 'Bearish'}")
        print(f"  Outline color: {outline_color}")
        print(f"  Curvature: {curvature_val:.4f}")
        print(f"  Fill color: {fill_color} ({fill_type})")
        
        # Body dimensions - EXACT from single candle test
        body_top = max(O, C)
        body_bottom = min(O, C)
        body_height = body_top - body_bottom
        
        print(f"  Body: {body_bottom} to {body_top} (height: {body_height})")
        
        # Width calculation - use timedelta like single candle test
        candle_half_width = timedelta(minutes=20)  # 20 minutes each side
        left_x = idx - candle_half_width
        right_x = idx + candle_half_width
        
        print(f"  Width: {left_x} to {right_x}")
        
        # STEP 1: Add curvature fill FIRST - EXACT from single candle test
        if body_height > 0:
            print("  Adding curvature fill...")
            fig.add_trace(go.Scatter(
                x=[left_x, right_x, right_x, left_x, left_x],  # Rectangle coordinates
                y=[body_bottom, body_bottom, body_top, body_top, body_bottom],  # Close the shape
                fill='toself',
                fillcolor=fill_color,
                mode='lines',
                line=dict(width=0, color='rgba(0,0,0,0)'),  # Invisible outline for fill
                showlegend=False,
                name=f'Fill {i}',
                hovertemplate=f'Candle {i}<br>OHLC: {O}, {H}, {L}, {C}<br>Curvature: {curvature_val:.4f}<extra></extra>'
            ))
            print("    ✓ Added curvature fill")
        else:
            print("  Adding doji line...")
            fig.add_trace(go.Scatter(
                x=[left_x, right_x],
                y=[O, C],
                mode='lines',
                line=dict(color=fill_color, width=6),
                showlegend=False,
                name=f'Doji {i}'
            ))
            print("    ✓ Added doji line")
        
        # STEP 2: Add wick - EXACT from single candle test
        print("  Adding wick...")
        fig.add_trace(go.Scatter(
            x=[idx, idx],
            y=[L, H],
            mode='lines',
            line=dict(color=outline_color, width=2),
            showlegend=False,
            name=f'Wick {i}'
        ))
        print("    ✓ Added wick")
        
        # STEP 3: Add body outline - EXACT from single candle test (4 separate lines)
        if body_height > 0:
            print("  Adding body outline (4 edges)...")
            
            # Bottom edge
            fig.add_trace(go.Scatter(
                x=[left_x, right_x],
                y=[body_bottom, body_bottom],
                mode='lines',
                line=dict(color=outline_color, width=3),
                showlegend=False,
                name=f'Bottom {i}'
            ))
            
            # Top edge
            fig.add_trace(go.Scatter(
                x=[left_x, right_x],
                y=[body_top, body_top],
                mode='lines',
                line=dict(color=outline_color, width=3),
                showlegend=False,
                name=f'Top {i}'
            ))
            
            # Left edge
            fig.add_trace(go.Scatter(
                x=[left_x, left_x],
                y=[body_bottom, body_top],
                mode='lines',
                line=dict(color=outline_color, width=3),
                showlegend=False,
                name=f'Left {i}'
            ))
            
            # Right edge
            fig.add_trace(go.Scatter(
                x=[right_x, right_x],
                y=[body_bottom, body_top],
                mode='lines',
                line=dict(color=outline_color, width=3),
                showlegend=False,
                name=f'Right {i}'
            ))
            print("    ✓ Added 4 outline edges")
        
        print(f"  ✅ Completed candle {i}")
    
    # Layout - EXACT from single candle test style
    fig.update_layout(
        title="TWO CANDLE TEST: Should show Green/Red outlines + Purple/Yellow fills",
        xaxis_title="Time",
        yaxis_title="Price",
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        height=500,
        width=800
    )
    
    print(f"\n✅ Created chart with {len(ohlc_data)} candles")
    return fig

if __name__ == "__main__":
    fig = create_two_candle_test()
    fig.write_html("two_candle_test.html")
    print("\n📁 Saved: two_candle_test.html")
    print("\n🎯 Expected: 2 candles with curvature fills and directional outlines!")