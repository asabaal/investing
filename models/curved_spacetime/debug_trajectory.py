"""
Debug script to see what's happening with trajectory computation.
"""

import numpy as np
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc
from example_usage import generate_synthetic_data

# Generate data
ohlc_data = generate_synthetic_data(200)
candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
geometry = CurvedCandleGeometry(candle_metrics)

start_idx = 50
initial_velocity = np.array([0.5, -0.2])

print(f"Data length: {len(ohlc_data)}")
print(f"Start index: {start_idx}")
print(f"Max possible steps: {len(ohlc_data) - start_idx - 1}")

# Test trajectory computation
trajectory = geometry.predict_geodesic_path(start_idx, initial_velocity, n_steps=None)
print(f"Actual trajectory length: {len(trajectory)}")

# Check first few and last few points
print(f"First 5 points: {trajectory[:5]}")
print(f"Last 5 points: {trajectory[-5:]}")

# Test with explicit large number
trajectory2 = geometry.predict_geodesic_path(start_idx, initial_velocity, n_steps=100)
print(f"Trajectory with n_steps=100: {len(trajectory2)}")