"""
Test script to verify geodesic trajectories respect triangular boundary.
"""

import numpy as np
from curved_candle_geometry import CandleMetric, CurvedCandleGeometry
from visualization import plot_geodesic_trajectory

# Create test candles near the boundary
test_candles = []
for i in range(10):
    # Create candles with varying properties
    metric = CandleMetric(
        range_value=5.0 + i * 0.5,
        low_value=100.0,
        sentiment=0.7,  # Near positive boundary
        upper_wick_ratio=0.2
    )
    test_candles.append(metric)

# Create geometry
geometry = CurvedCandleGeometry(test_candles)

# Test with velocity that would push outside boundary
test_velocities = [
    ([0.5, 0.5], "Pushing toward upper-right corner"),
    ([-0.5, 0.5], "Pushing toward upper-left corner"),
    ([0.8, 0.0], "Pushing right along sentiment"),
    ([0.0, 0.8], "Pushing up along UWR")
]

print("🧪 Testing Geodesic Boundary Constraints")
print("=" * 50)

for i, (velocity, description) in enumerate(test_velocities):
    print(f"\nTest {i+1}: {description}")
    print(f"Initial velocity: {velocity}")
    
    # Compute trajectory
    trajectory = geometry.predict_geodesic_path(
        start_index=5,
        initial_velocity=np.array(velocity),
        n_steps=30
    )
    
    # Check if all points are within boundary
    violations = 0
    for j, point in enumerate(trajectory):
        sentiment, uwr = point
        constraint = abs(sentiment) + uwr
        
        if constraint > 1.0001:  # Small tolerance for numerical errors
            violations += 1
            print(f"  ⚠️  Step {j}: Boundary violation! |{sentiment:.3f}| + {uwr:.3f} = {constraint:.3f}")
    
    if violations == 0:
        print(f"  ✅ All {len(trajectory)} points stay within boundary!")
        final_point = trajectory[-1]
        print(f"  Final position: Sentiment={final_point[0]:.3f}, UWR={final_point[1]:.3f}")
    
    # Create visualization
    fig = plot_geodesic_trajectory(geometry, 5, np.array(velocity), n_steps=30)
    fig.write_html(f"boundary_test_{i+1}.html")
    print(f"  📊 Saved visualization: boundary_test_{i+1}.html")

print("\n✅ Boundary testing complete!")
print("Check the generated HTML files to visually verify trajectories stay within the triangular region.")