"""
Test with exactly 2 candles using shared visualization logic.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc
from shared_candle_visualization import create_curvature_candlestick_chart_shared

def create_two_candle_test():
    """Create exactly 2 candlesticks with curvature fills using shared logic."""
    
    print("🔧 TWO CANDLE TEST WITH SHARED VISUALIZATION LOGIC")
    print("=" * 60)
    
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
    print("\\n=== DEBUGGING CURVATURE CALCULATION ===")
    
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
    
    print("=== END CURVATURE DEBUG ===\\n")
    
    # Use shared visualization logic - EXACT SAME CODE as main example will use
    fig = create_curvature_candlestick_chart_shared(
        ohlc_data=ohlc_data,
        curvatures=curvatures,
        title="TWO CANDLE TEST: Shared Logic with Gradient Colors",
        debug=True
    )
    
    return fig

if __name__ == "__main__":
    fig = create_two_candle_test()
    fig.write_html("two_candle_test_new.html")
    print("\\n📁 Saved: two_candle_test_new.html")
    print("\\n🎯 Expected: 2 candles with curvature gradient fills and directional outlines!")