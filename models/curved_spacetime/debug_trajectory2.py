"""
Detailed debug to understand trajectory termination.
"""

import numpy as np
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc
from example_usage import generate_synthetic_data

# Generate data
ohlc_data = generate_synthetic_data(200)
candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)

start_idx = 50
print(f"Starting candle pattern: {candle_metrics[start_idx].pattern_coordinates}")
print(f"Sentiment: {candle_metrics[start_idx].sentiment:.3f}")
print(f"UWR: {candle_metrics[start_idx].upper_wick_ratio:.3f}")
print(f"Constraint: |{candle_metrics[start_idx].sentiment:.3f}| + {candle_metrics[start_idx].upper_wick_ratio:.3f} = {abs(candle_metrics[start_idx].sentiment) + candle_metrics[start_idx].upper_wick_ratio:.3f}")

# Test with a very small velocity
small_velocity = np.array([0.01, 0.01])
print(f"\nTesting with small velocity: {small_velocity}")

geometry = CurvedCandleGeometry(candle_metrics)

# Test a few steps manually
position = candle_metrics[start_idx].pattern_coordinates.copy()
velocity = small_velocity.copy()
print(f"Initial position: {position}")
print(f"Initial velocity: {velocity}")

for step in range(5):
    print(f"\nStep {step + 1}:")
    
    # Simple update without curvature effects first
    new_position = position + velocity * 1.0
    print(f"  New position before boundary: {new_position}")
    
    # Check constraint
    sentiment, uwr = new_position
    constraint = abs(sentiment) + uwr
    print(f"  Constraint check: |{sentiment:.3f}| + {uwr:.3f} = {constraint:.3f} {'✓' if constraint <= 1.0 and uwr >= 0 else '✗'}")
    
    if constraint > 1.0 or uwr < 0:
        print(f"  Would violate constraint!")
        break
    
    position = new_position
    print(f"  Updated position: {position}")