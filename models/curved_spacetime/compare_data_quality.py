#!/usr/bin/env python3
"""
Compare data quality between the working static version and animated version.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phase_space_trajectory_3d import generate_rich_market_data as generate_static_data
from animated_phase_space_evolution import generate_rich_market_data as generate_animated_data, compute_evolutionary_frame
from curved_candle_geometry import create_candle_metrics_from_ohlc
import numpy as np

def compare_data_quality():
    """Compare the data generation between static and animated versions."""
    
    print("🔍 Comparing data quality between static and animated versions...")
    
    # Generate data from both functions
    static_data = generate_static_data(n_candles=200)
    animated_data = generate_animated_data(n_candles=200)
    
    print(f"\n📊 Static data: {len(static_data)} candles")
    print(f"📊 Animated data: {len(animated_data)} candles")
    
    # Compare phase space coverage
    static_metrics = create_candle_metrics_from_ohlc(static_data)
    animated_metrics = create_candle_metrics_from_ohlc(animated_data)
    
    static_sentiments = np.array([c.sentiment for c in static_metrics])
    static_uwrs = np.array([c.upper_wick_ratio for c in static_metrics])
    
    animated_sentiments = np.array([c.sentiment for c in animated_metrics])
    animated_uwrs = np.array([c.upper_wick_ratio for c in animated_metrics])
    
    print(f"\n🎯 Phase Space Coverage:")
    print(f"Static version:")
    print(f"   Sentiment range: {static_sentiments.min():.3f} to {static_sentiments.max():.3f}")
    print(f"   UWR range: {static_uwrs.min():.3f} to {static_uwrs.max():.3f}")
    print(f"   Phase space spread: {np.std(static_sentiments):.3f} (sentiment), {np.std(static_uwrs):.3f} (UWR)")
    
    print(f"Animated version:")
    print(f"   Sentiment range: {animated_sentiments.min():.3f} to {animated_sentiments.max():.3f}")
    print(f"   UWR range: {animated_uwrs.min():.3f} to {animated_uwrs.max():.3f}")
    print(f"   Phase space spread: {np.std(animated_sentiments):.3f} (sentiment), {np.std(animated_uwrs):.3f} (UWR)")
    
    # Test the final frame of animation
    print(f"\n🎬 Testing final frame computation...")
    final_frame = compute_evolutionary_frame(animated_data, 199)
    
    if final_frame:
        print(f"   ✅ Final frame: {final_frame['n_candles']} candles")
        print(f"   Sentiment range: {final_frame['sentiments'].min():.3f} to {final_frame['sentiments'].max():.3f}")
        print(f"   UWR range: {final_frame['uwrs'].min():.3f} to {final_frame['uwrs'].max():.3f}")
        
        # Check gradient coverage
        density_coverage = np.sum(~np.isnan(final_frame['density_grid'])) / final_frame['density_grid'].size
        time_coverage = np.sum(~np.isnan(final_frame['time_grid'])) / final_frame['time_grid'].size
        
        print(f"   Density grid coverage: {density_coverage*100:.1f}%")
        print(f"   Time grid coverage: {time_coverage*100:.1f}%")
    
    print(f"\n💡 Conclusion:")
    if np.allclose(static_sentiments, animated_sentiments) and np.allclose(static_uwrs, animated_uwrs):
        print("   ✅ Data generation is identical - should have same phase space filling")
    else:
        print("   ⚠️ Data generation differs - this might explain different phase space filling")
        print("   🔧 Consider using identical random seeds and parameters")

if __name__ == "__main__":
    compare_data_quality()