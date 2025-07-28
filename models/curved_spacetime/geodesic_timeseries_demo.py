"""
Demonstration of geodesic trajectories with synchronized time series visualization.
Shows how patterns evolve in curved space alongside their time series representation.
"""

import numpy as np
import pandas as pd
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc
from combined_visualization import plot_geodesic_with_timeseries
from example_usage import generate_synthetic_data, fetch_real_market_data


def analyze_market_regime_transitions(ohlc_data: pd.DataFrame, geometry: CurvedCandleGeometry):
    """
    Find interesting starting points for geodesic trajectories based on market regimes.
    """
    curvatures = geometry.compute_curvature_series()
    
    # Find regime transitions
    transitions = []
    for i in range(2, len(curvatures) - 2):
        # Trending to volatile transition
        if curvatures[i-1] > 0.5 and curvatures[i+1] < -0.5:
            transitions.append((i, "Trend→Volatile", np.array([0.3, -0.3])))
        # Volatile to trending transition  
        elif curvatures[i-1] < -0.5 and curvatures[i+1] > 0.5:
            transitions.append((i, "Volatile→Trend", np.array([0.5, 0.1])))
        # High curvature points
        elif abs(curvatures[i]) > 0.9:
            if curvatures[i] > 0:
                transitions.append((i, "Strong Trend", np.array([0.7, -0.1])))
            else:
                transitions.append((i, "High Volatility", np.array([-0.3, 0.4])))
    
    return transitions


def main():
    print("🌌 Geodesic-TimeSeries Synchronized Visualization Demo")
    print("=" * 60)
    
    # Generate or load data
    print("\n📊 Loading market data...")
    ohlc_data = generate_synthetic_data(300)  # More data for interesting patterns
    
    # Create geometry
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    geometry = CurvedCandleGeometry(candle_metrics)
    
    # Find interesting regime transitions
    print("\n🔍 Analyzing market regime transitions...")
    transitions = analyze_market_regime_transitions(ohlc_data, geometry)
    
    print(f"Found {len(transitions)} interesting transition points")
    
    # Create visualizations for each interesting point
    for i, (start_idx, regime_type, velocity) in enumerate(transitions[:5]):  # First 5
        print(f"\n📈 Creating visualization {i+1}: {regime_type} at index {start_idx}")
        
        # Get current pattern coordinates
        current_pattern = candle_metrics[start_idx].pattern_coordinates
        print(f"   Starting pattern: Sentiment={current_pattern[0]:.3f}, UWR={current_pattern[1]:.3f}")
        print(f"   Initial velocity: {velocity}")
        
        # Create combined visualization with full trajectory
        title = f"Full Geodesic from {regime_type} Regime (Index {start_idx})"
        fig = plot_geodesic_with_timeseries(
            ohlc_data, 
            geometry, 
            start_idx, 
            velocity, 
            n_steps=None,  # Full trajectory to end of data
            title=title
        )
        
        filename = f"geodesic_regime_{i+1}_{regime_type.replace('→', '_to_').lower()}.html"
        fig.write_html(filename)
        print(f"   ✓ Saved {filename}")
    
    # Create a special visualization showing very different initial velocities
    print("\n🎯 Creating divergent trajectory visualization...")
    
    # Pick a central starting point
    center_idx = len(candle_metrics) // 2
    
    # Different market scenarios
    scenarios = [
        ("Strong Bull Push", np.array([0.8, -0.1])),
        ("Strong Bear Push", np.array([-0.8, -0.1])),
        ("Upper Rejection", np.array([0.0, 0.7])),
        ("Neutral Drift", np.array([0.0, 0.0])),
        ("Volatility Expansion", np.array([-0.2, 0.5]))
    ]
    
    for scenario_name, velocity in scenarios:
        fig = plot_geodesic_with_timeseries(
            ohlc_data,
            geometry,
            center_idx,
            velocity,
            n_steps=None,  # Full trajectory
            title=f"Full Geodesic Scenario: {scenario_name}"
        )
        
        filename = f"geodesic_scenario_{scenario_name.lower().replace(' ', '_')}.html"
        fig.write_html(filename)
        print(f"   ✓ Saved {filename}")
    
    print("\n✅ All visualizations created!")
    print("\n📊 Key Features of the Combined Visualizations:")
    print("   - Top Left: Geodesic path in 2D pattern space")
    print("   - Top Right: Candlestick chart with trajectory markers")
    print("   - Bottom Left: Pattern evolution (Sentiment & UWR over time)")
    print("   - Bottom Right: Gaussian curvature with trajectory region highlighted")
    print("\n💡 Notice how:")
    print("   - Geodesics curve based on local market geometry")
    print("   - Pattern evolution corresponds to price movements")
    print("   - Curvature influences trajectory direction")
    print("   - Boundaries are respected in pattern space")


if __name__ == "__main__":
    main()