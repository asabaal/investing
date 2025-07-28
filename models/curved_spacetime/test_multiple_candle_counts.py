"""
Test the shared visualization logic with different numbers of candles.
This will help identify any issues with the gradient coloring.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc
from shared_candle_visualization import create_curvature_candlestick_chart_shared

def create_test_ohlc_data(n_candles: int, start_date: datetime = None) -> pd.DataFrame:
    """Create test OHLC data with varying ranges to produce different curvatures."""
    
    if start_date is None:
        start_date = datetime(2024, 1, 1, 9, 0)
    
    dates = [start_date + timedelta(hours=i) for i in range(n_candles)]
    
    np.random.seed(42)  # For reproducible results
    
    ohlc_data = []
    base_price = 100.0
    
    for i in range(n_candles):
        # Create different range patterns to get varied curvatures
        if i % 4 == 0:
            # Large range candles (should get positive curvature)
            range_size = 10.0 + np.random.random() * 5.0
            sentiment_bias = 0.2  # Slightly bullish
        elif i % 4 == 1:
            # Medium range candles
            range_size = 5.0 + np.random.random() * 3.0
            sentiment_bias = -0.1  # Slightly bearish
        elif i % 4 == 2:
            # Small range candles (should get negative curvature)
            range_size = 1.0 + np.random.random() * 2.0
            sentiment_bias = 0.8  # Very bullish (extreme pattern)
        else:
            # Variable range
            range_size = 3.0 + np.random.random() * 4.0
            sentiment_bias = -0.5  # Bearish
        
        # Random walk for base price
        base_price += (np.random.random() - 0.5) * 2.0
        
        # Create OHLC with desired range and sentiment
        low = base_price - range_size * 0.3
        high = low + range_size
        
        # Create open/close based on sentiment bias
        if sentiment_bias > 0:
            # Bullish
            open_price = low + range_size * (0.2 + np.random.random() * 0.3)
            close_price = open_price + range_size * sentiment_bias * (0.5 + np.random.random() * 0.3)
        else:
            # Bearish
            open_price = low + range_size * (0.5 + np.random.random() * 0.3)
            close_price = open_price + range_size * sentiment_bias * (0.3 + np.random.random() * 0.4)
        
        # Clamp to valid range
        close_price = max(low, min(high, close_price))
        
        ohlc_data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price
        })
    
    return pd.DataFrame(ohlc_data, index=dates)

def test_candle_count(n_candles: int) -> None:
    """Test visualization with specific number of candles."""
    
    print(f"\\n{'='*60}")
    print(f"🧪 TESTING WITH {n_candles} CANDLES")
    print(f"{'='*60}")
    
    # Create test data
    ohlc_data = create_test_ohlc_data(n_candles)
    print(f"Created OHLC data with {len(ohlc_data)} candles")
    
    # Create geometry and compute curvatures
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    geometry = CurvedCandleGeometry(candle_metrics)
    curvatures = geometry.compute_curvature_series()
    
    print(f"Curvature stats:")
    print(f"  - Min: {np.min(curvatures):.4f}")
    print(f"  - Max: {np.max(curvatures):.4f}")
    print(f"  - Mean: {np.mean(curvatures):.4f}")
    print(f"  - Std: {np.std(curvatures):.4f}")
    
    # Show first few curvature values and expected colors
    print(f"First few curvatures:")
    for i in range(min(5, len(curvatures))):
        from curvature_color_scheme import get_curvature_gradient_color
        color, description = get_curvature_gradient_color(curvatures[i])
        print(f"  Candle {i}: curvature={curvatures[i]:.4f} -> {color} ({description})")
    
    # Create visualization
    try:
        fig = create_curvature_candlestick_chart_shared(
            ohlc_data=ohlc_data,
            curvatures=curvatures,
            title=f"Test Chart: {n_candles} Candles with Gradient Colors",
            debug=False  # Don't spam debug for large counts
        )
        
        # Save to file
        filename = f"test_{n_candles}_candles.html"
        fig.write_html(filename)
        print(f"✅ Successfully created chart: {filename}")
        
        # Check if we have any traces (indication that something was drawn)
        print(f"Chart has {len(fig.data)} traces")
        
    except Exception as e:
        print(f"❌ Error creating chart: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Test multiple candle counts."""
    
    print("🧪 TESTING SHARED VISUALIZATION WITH DIFFERENT CANDLE COUNTS")
    print("="*80)
    
    candle_counts = [2, 5, 10, 20, 200]
    
    for count in candle_counts:
        test_candle_count(count)
    
    print(f"\\n{'='*80}")
    print("✅ All tests completed!")
    print("📁 Check the generated HTML files:")
    for count in candle_counts:
        print(f"  - test_{count}_candles.html")
    print("\\n🎯 Look for gradient color differences between candles in each chart!")

if __name__ == "__main__":
    main()