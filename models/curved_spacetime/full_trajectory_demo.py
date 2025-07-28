"""
Demonstration of full-length geodesic trajectories with improved clarity.
"""

import numpy as np
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc
from combined_visualization import plot_geodesic_with_timeseries
from example_usage import generate_synthetic_data

def main():
    print("🚀 Full Trajectory Geodesic Demonstration")
    print("=" * 50)
    
    # Generate data
    print("\n📊 Generating market data...")
    ohlc_data = generate_synthetic_data(150)  # Medium-sized dataset
    
    # Create geometry
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    geometry = CurvedCandleGeometry(candle_metrics)
    
    # Find a good starting point (middle of data)
    start_idx = len(ohlc_data) // 3
    
    print(f"\n🎯 Starting analysis from candle {start_idx} (out of {len(ohlc_data)})")
    print(f"   Remaining trajectory length: {len(ohlc_data) - start_idx - 1} steps")
    
    # Different scenarios with full trajectories
    scenarios = [
        {
            "name": "Strong Bullish Evolution", 
            "velocity": np.array([0.7, -0.1]),
            "description": "Strong positive sentiment push with slight wick reduction"
        },
        {
            "name": "Volatile Expansion", 
            "velocity": np.array([0.0, 0.6]),
            "description": "Pure upper wick expansion (increased rejection/volatility)"
        },
        {
            "name": "Bearish Transition", 
            "velocity": np.array([-0.6, 0.2]),
            "description": "Strong negative sentiment with some upper rejection"
        },
        {
            "name": "Pattern Stabilization", 
            "velocity": np.array([0.1, -0.1]),
            "description": "Gentle drift toward stability"
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n📈 Creating Scenario {i+1}: {scenario['name']}")
        print(f"   {scenario['description']}")
        print(f"   Initial velocity: {scenario['velocity']}")
        
        # Create full trajectory visualization
        fig = plot_geodesic_with_timeseries(
            ohlc_data,
            geometry,
            start_idx,
            scenario['velocity'],
            n_steps=None,  # Full trajectory to end
            title=f"Full Geodesic: {scenario['name']}"
        )
        
        filename = f"full_trajectory_{i+1}_{scenario['name'].lower().replace(' ', '_')}.html"
        fig.write_html(filename)
        print(f"   ✓ Saved {filename}")
    
    print("\n✅ Full trajectory demonstrations complete!")
    print("\n🔍 What to look for in the visualizations:")
    print("   - Top Left: Complete geodesic path in pattern space (full curve)")
    print("   - Top Right: Cyan dashed line shows trajectory timeline on price chart")
    print("   - Bottom Left: How Sentiment and UWR evolve over the full trajectory")
    print("   - Bottom Right: Curvature plot with highlighted trajectory region")
    print("\n💡 Key differences from short trajectories:")
    print("   - See complete pattern evolution from start to end of data")
    print("   - Observe how curvature changes affect long-term trajectory")
    print("   - Notice boundary interactions over extended periods")
    print("   - Compare pattern space path with full price timeline")


if __name__ == "__main__":
    main()