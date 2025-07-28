"""
Example usage of the curved candle geometry framework.

This script demonstrates:
1. Loading market data
2. Computing curvature metrics
3. Visualizing curved spacetime properties
4. Predicting geodesic trajectories
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from curved_candle_geometry import (
    CurvedCandleGeometry, 
    create_candle_metrics_from_ohlc,
    analyze_market_curvature
)
from visualization import (
    plot_curvature_analysis,
    plot_pattern_space_with_metric,
    plot_geodesic_trajectory,
    plot_metric_field,
    create_curvature_heatmap
)
from combined_visualization import (
    plot_geodesic_with_timeseries,
    create_trajectory_comparison
)
from visualization_3d import (
    create_3d_geodesic_visualization,
    create_3d_multiple_trajectories,
    create_3d_curvature_field
)
from curvature_candlestick import (
    create_curvature_candlestick_chart,
    create_combined_curvature_chart
)


def generate_synthetic_data(n_candles: int = 100) -> pd.DataFrame:
    """
    Generate synthetic OHLC data for testing.
    """
    # Create time index
    dates = pd.date_range(end=datetime.now(), periods=n_candles, freq='H')
    
    # Generate price series with varying volatility
    np.random.seed(42)
    base_price = 100
    prices = [base_price]
    
    for i in range(1, n_candles):
        # Varying volatility creates curvature changes
        volatility = 0.02 * (1 + 0.5 * np.sin(i * 0.1))
        change = np.random.normal(0, volatility)
        prices.append(prices[-1] * (1 + change))
    
    # Create OHLC from price series
    ohlc_data = []
    for i in range(n_candles):
        base = prices[i]
        
        # Generate intrabar movement
        movement = np.random.uniform(0.002, 0.01) * base
        
        open_price = base + np.random.uniform(-movement/2, movement/2)
        close_price = base + np.random.uniform(-movement/2, movement/2)
        
        high_price = max(open_price, close_price) + np.random.uniform(0, movement)
        low_price = min(open_price, close_price) - np.random.uniform(0, movement)
        
        ohlc_data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price
        })
    
    df = pd.DataFrame(ohlc_data, index=dates)
    return df


def fetch_real_market_data(symbol: str = 'SPY', 
                         period: str = '1mo',
                         interval: str = '1h') -> pd.DataFrame:
    """
    Fetch real market data using yfinance.
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval)
    
    # Rename columns to lowercase
    data.columns = [col.lower() for col in data.columns]
    
    # Select OHLC columns
    ohlc_data = data[['open', 'high', 'low', 'close']].copy()
    
    return ohlc_data


def main():
    """
    Main demonstration of curved candle geometry.
    """
    print("🌌 Curved Candle Geometry Demonstration")
    print("=" * 50)
    
    # Choose data source
    use_real_data = False  # Set to True to use real market data
    
    if use_real_data:
        print("\n📊 Fetching real market data...")
        try:
            ohlc_data = fetch_real_market_data('SPY', period='1mo', interval='1h')
            print(f"Loaded {len(ohlc_data)} candles for SPY")
        except Exception as e:
            print(f"Error fetching data: {e}")
            print("Falling back to synthetic data...")
            ohlc_data = generate_synthetic_data(200)
    else:
        print("\n📊 Generating synthetic market data...")
        ohlc_data = generate_synthetic_data(200)
    
    # Perform curvature analysis
    print("\n🧮 Computing market curvature...")
    analysis = analyze_market_curvature(ohlc_data)
    
    print(f"\n📈 Curvature Analysis Results:")
    print(f"  - Mean absolute curvature: {analysis['mean_curvature']:.6f}")
    print(f"  - Maximum curvature: {analysis['max_curvature']:.6f}")
    print(f"  - Positive curvature ratio: {analysis['positive_curvature_ratio']:.2%}")
    print(f"  - Trending periods: {len(analysis['trending_periods'])}")
    print(f"  - Volatile periods: {len(analysis['volatile_periods'])}")
    print(f"  - Flat periods: {len(analysis['flat_periods'])}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    # 1. Curvature analysis plot
    fig1 = plot_curvature_analysis(ohlc_data, analysis['curvatures'])
    fig1.write_html("curvature_analysis.html")
    print("  ✓ Saved curvature_analysis.html")
    
    # 2. Pattern space visualization
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    geometry = CurvedCandleGeometry(candle_metrics)
    
    # Highlight high curvature candles
    high_curvature_indices = np.where(np.abs(analysis['curvatures']) > 
                                    2 * analysis['mean_curvature'])[0]
    
    fig2 = plot_pattern_space_with_metric(candle_metrics, high_curvature_indices)
    fig2.write_html("pattern_space.html")
    print("  ✓ Saved pattern_space.html")
    
    # 3. Geodesic trajectory prediction
    print("\n🚀 Computing geodesic trajectories...")
    
    # Start from early in the data to show more trajectory
    start_idx = 5  # Start early enough to show most of the data
    
    # Initial velocity in pattern space
    initial_velocity = np.array([0.5, -0.2])  # Bullish with decreasing upper wick
    
    # Use full trajectory length for pattern space view
    full_steps = len(ohlc_data) - start_idx - 1
    fig3 = plot_geodesic_trajectory(geometry, start_idx, initial_velocity, n_steps=full_steps)
    fig3.write_html("geodesic_trajectory.html")
    print("  ✓ Saved geodesic_trajectory.html")
    
    # 4. Curvature heatmap
    fig4 = create_curvature_heatmap(geometry)
    fig4.write_html("curvature_heatmap.html")
    print("  ✓ Saved curvature_heatmap.html")
    
    # 5. Metric field visualization (matplotlib)
    fig5 = plot_metric_field(candle_metrics)
    fig5.savefig("metric_field.png", dpi=150, bbox_inches='tight')
    print("  ✓ Saved metric_field.png")
    
    # 6. Combined geodesic and time series visualization
    print("\n📊 Creating combined geodesic-timeseries visualization...")
    
    # Full-length trajectory (to end of data) - HISTORICAL PATH
    fig6 = plot_geodesic_with_timeseries(
        ohlc_data, geometry, start_idx, initial_velocity=None, n_steps=None,
        title="Complete Historical Geodesic Evolution: Pattern Space & Time Series"
    )
    fig6.write_html("geodesic_combined_full.html")
    print("  ✓ Saved geodesic_combined_full.html")
    
    # Also create a shorter version for comparison
    fig6b = plot_geodesic_with_timeseries(
        ohlc_data, geometry, start_idx, initial_velocity, n_steps=50,
        title="Short Geodesic Evolution: Pattern Space & Time Series View"
    )
    fig6b.write_html("geodesic_combined_short.html")
    print("  ✓ Saved geodesic_combined_short.html")
    
    # 7. 3D Geodesic Visualization
    print("\n🚀 Creating 3D geodesic visualization...")
    
    fig_3d = create_3d_geodesic_visualization(
        ohlc_data, geometry, start_idx,
        title="3D Market Geodesic: Pattern Evolution Through Time"
    )
    fig_3d.write_html("geodesic_3d.html")
    print("  ✓ Saved geodesic_3d.html - 3D trajectory in (Sentiment, UWR, Time)")
    
    # 8. 3D Multiple Trajectories
    print("\n🌟 Creating 3D multiple trajectories...")
    
    # Different starting points for comparison
    trajectory_starts = [
        (start_idx, f"From Candle {start_idx}", "#00ff00"),
        (start_idx + 20, f"From Candle {start_idx + 20}", "#ff0000"), 
        (start_idx + 40, f"From Candle {start_idx + 40}", "#ffff00"),
        (start_idx + 60, f"From Candle {start_idx + 60}", "#ff00ff")
    ]
    
    fig_3d_multi = create_3d_multiple_trajectories(
        ohlc_data, geometry, trajectory_starts,
        title="3D Multiple Geodesic Trajectories"
    )
    fig_3d_multi.write_html("geodesic_3d_multiple.html")
    print("  ✓ Saved geodesic_3d_multiple.html - Multiple 3D trajectories")
    
    # 9. 3D Curvature Field
    print("\n🔮 Creating 3D curvature field...")
    
    fig_3d_curvature = create_3d_curvature_field(
        geometry, title="3D Market Curvature Field Through Time"
    )
    fig_3d_curvature.write_html("curvature_3d_field.html")
    print("  ✓ Saved curvature_3d_field.html - 3D curvature visualization")
    
    # 10. Curvature-Enhanced Candlestick Charts
    print("\n🕯️ Creating curvature-enhanced candlestick charts...")
    
    # Enhanced chart with integrated legend and vibrant colors
    fig_curv_candles = create_curvature_candlestick_chart(
        ohlc_data, geometry,
        title="Revolutionary Candlestick Analysis: Curved Spacetime Curvature"
    )
    fig_curv_candles.write_html("curvature_candlesticks.html")
    print("  ✓ Saved curvature_candlesticks.html - Candles with integrated legend!")
    
    # Demonstrate parallel transport
    print("\n🔄 Computing parallel transport...")
    test_vector = np.array([1.0, 0.0])  # Unit sentiment vector
    transported = geometry.compute_parallel_transport(test_vector, 0, len(candle_metrics)-1)
    
    print(f"  Original vector: {test_vector}")
    print(f"  Transported vector: {transported}")
    print(f"  Magnitude change: {np.linalg.norm(transported) / np.linalg.norm(test_vector):.3f}")
    print(f"  Angle change: {np.arccos(np.dot(test_vector, transported) / (np.linalg.norm(test_vector) * np.linalg.norm(transported))) * 180/np.pi:.1f}°")
    
    # Compute geodesic distances
    print("\n📏 Computing geodesic distances...")
    
    # Compare Euclidean vs geodesic distance
    idx1, idx2 = 10, 50
    pattern1 = candle_metrics[idx1].pattern_coordinates
    pattern2 = candle_metrics[idx2].pattern_coordinates
    
    euclidean_dist = np.linalg.norm(pattern2 - pattern1)
    geodesic_dist = geometry.curved_pattern_distance(idx1, idx2)
    
    print(f"  Between candles {idx1} and {idx2}:")
    print(f"    Euclidean distance: {euclidean_dist:.4f}")
    print(f"    Geodesic distance: {geodesic_dist:.4f}")
    print(f"    Curvature effect: {(geodesic_dist/euclidean_dist - 1)*100:.1f}% longer")
    
    print("\n✅ Analysis complete! Check the generated HTML files for interactive visualizations.")
    
    # Summary insights
    print("\n💡 Key Insights:")
    print("  1. Market curvature varies significantly over time")
    print("  2. High positive curvature indicates trending behavior")
    print("  3. Negative curvature suggests increased volatility")
    print("  4. Geodesic paths show how patterns evolve in curved space")
    print("  5. Pattern distances are distorted by market geometry")


if __name__ == "__main__":
    main()