"""
Analyze the curvature values being produced by the main example
to understand why gradient colors might not be visible.
"""

import numpy as np
import pandas as pd
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc, analyze_market_curvature
from curvature_color_scheme import get_curvature_gradient_color
import matplotlib.pyplot as plt

def analyze_main_example():
    """Generate the same synthetic data as main example and analyze curvatures."""
    
    print("📊 ANALYZING MAIN EXAMPLE CURVATURE DISTRIBUTION")
    print("=" * 60)
    
    # Generate the EXACT same synthetic data as example_usage.py
    np.random.seed(42)
    n_candles = 200
    
    dates = pd.date_range(start='2025-07-20 04:00:00', periods=n_candles, freq='1H')
    base_price = 100.0
    volatility = 0.02
    
    # Generate OHLC data with varying characteristics
    ohlc_data = []
    
    for i in range(n_candles):
        # Random walk for base price
        base_price += np.random.normal(0, volatility)
        
        # Variable range based on market regime
        if i < 50:
            range_multiplier = 0.5  # Low volatility period
        elif i < 100:
            range_multiplier = 2.0  # High volatility period
        elif i < 150:
            range_multiplier = 0.8  # Medium volatility
        else:
            range_multiplier = 1.5  # Another high vol period
        
        daily_range = abs(np.random.normal(0.5, 0.3)) * range_multiplier
        
        # Create OHLC
        low = base_price - daily_range * np.random.uniform(0.3, 0.7)
        high = low + daily_range
        
        open_price = low + daily_range * np.random.uniform(0.2, 0.8)
        close_price = low + daily_range * np.random.uniform(0.2, 0.8)
        
        ohlc_data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price
        })
    
    ohlc_df = pd.DataFrame(ohlc_data, index=dates)
    
    print(f"Generated {len(ohlc_df)} candles")
    
    # Analyze curvatures
    analysis = analyze_market_curvature(ohlc_df)
    curvatures = analysis['curvatures']
    
    print(f"\\nCurvature Statistics:")
    print(f"  Min: {np.min(curvatures):.4f}")
    print(f"  Max: {np.max(curvatures):.4f}")
    print(f"  Mean: {np.mean(curvatures):.4f}")
    print(f"  Std: {np.std(curvatures):.4f}")
    print(f"  25th percentile: {np.percentile(curvatures, 25):.4f}")
    print(f"  75th percentile: {np.percentile(curvatures, 75):.4f}")
    
    # Count curvatures in different ranges
    highly_negative = np.sum(curvatures < -0.5)
    negative = np.sum((curvatures >= -0.5) & (curvatures < -0.1))
    neutral = np.sum((curvatures >= -0.1) & (curvatures <= 0.1))
    positive = np.sum((curvatures > 0.1) & (curvatures <= 0.5))
    highly_positive = np.sum(curvatures > 0.5)
    
    print(f"\\nCurvature Distribution:")
    print(f"  Highly Negative (< -0.5): {highly_negative} candles ({highly_negative/len(curvatures)*100:.1f}%)")
    print(f"  Negative (-0.5 to -0.1): {negative} candles ({negative/len(curvatures)*100:.1f}%)")
    print(f"  Neutral (-0.1 to 0.1): {neutral} candles ({neutral/len(curvatures)*100:.1f}%)")
    print(f"  Positive (0.1 to 0.5): {positive} candles ({positive/len(curvatures)*100:.1f}%)")
    print(f"  Highly Positive (> 0.5): {highly_positive} candles ({highly_positive/len(curvatures)*100:.1f}%)")
    
    # Show color distribution
    print(f"\\nColor Analysis (first 10 candles):")
    for i in range(min(10, len(curvatures))):
        color, description = get_curvature_gradient_color(curvatures[i])
        print(f"  Candle {i:2d}: curvature={curvatures[i]:6.4f} -> {description}")
    
    # Find most extreme examples
    min_idx = np.argmin(curvatures)
    max_idx = np.argmax(curvatures)
    
    print(f"\\nExtreme Examples:")
    print(f"  Most Volatile (curvature={curvatures[min_idx]:.4f}) at index {min_idx}:")
    min_color, min_desc = get_curvature_gradient_color(curvatures[min_idx])
    print(f"    Color: {min_color} ({min_desc})")
    
    print(f"  Most Trending (curvature={curvatures[max_idx]:.4f}) at index {max_idx}:")
    max_color, max_desc = get_curvature_gradient_color(curvatures[max_idx])
    print(f"    Color: {max_color} ({max_desc})")
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(curvatures, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Curvature')
    plt.ylabel('Frequency')
    plt.title('Distribution of Curvature Values')
    plt.axvline(np.mean(curvatures), color='red', linestyle='--', label=f'Mean: {np.mean(curvatures):.3f}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(curvatures)
    plt.xlabel('Candle Index')
    plt.ylabel('Curvature')
    plt.title('Curvature Over Time')
    plt.axhline(0, color='gray', linestyle='-', alpha=0.3)
    plt.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='High Positive')
    plt.axhline(-0.5, color='red', linestyle='--', alpha=0.5, label='High Negative')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('curvature_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\\n📊 Saved curvature analysis plot: curvature_analysis.png")
    
    return curvatures, ohlc_df

if __name__ == "__main__":
    curvatures, ohlc_df = analyze_main_example()
    
    print(f"\\n🎯 DIAGNOSIS:")
    print(f"If most curvatures are in a narrow range (e.g., 0.3-0.8), then all candles")
    print(f"will have similar colors and the gradient won't be visible.")
    print(f"\\nThe solution would be to:")
    print(f"1. Adjust the curvature calculation to spread values more")
    print(f"2. Create test data with more diverse curvature ranges")
    print(f"3. Use a different color mapping that's more sensitive to smaller differences")