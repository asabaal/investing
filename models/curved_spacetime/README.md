# 🌌 Curved Candle Geometry: Financial Markets as Curved Spacetime

## Overview

This implementation brings the revolutionary concept of curved spacetime to financial market analysis. Just as Einstein showed that mass curves spacetime, we demonstrate that market volatility (encoded in candle Range values) curves the pattern space of candlestick charts.

## Key Concepts

### The Mathematical Framework

1. **Local Metric Tensor**: Each candle has its own metric tensor g_ij proportional to Range²
2. **Christoffel Symbols**: Encode how the coordinate system changes between candles
3. **Riemann Curvature**: Measures the intrinsic curvature of market space
4. **Geodesics**: Optimal market trajectories through curved pattern space

### Physical Interpretation

- **Positive Curvature (K > 0)**: Trending markets where patterns converge
- **Negative Curvature (K < 0)**: Volatile markets where patterns diverge  
- **Zero Curvature (K = 0)**: Efficient markets following random walk

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from curved_candle_geometry import analyze_market_curvature
import pandas as pd

# Load your OHLC data
ohlc_data = pd.read_csv('market_data.csv', index_col='date')

# Analyze curvature
analysis = analyze_market_curvature(ohlc_data)

print(f"Mean curvature: {analysis['mean_curvature']}")
print(f"Trending periods: {len(analysis['trending_periods'])}")
```

### Advanced Geodesic Analysis

```python
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc

# Create geometry object
metrics = create_candle_metrics_from_ohlc(ohlc_data)
geometry = CurvedCandleGeometry(metrics)

# Predict geodesic path
initial_velocity = np.array([0.5, -0.2])  # Pattern space velocity
trajectory = geometry.predict_geodesic_path(
    start_index=50,
    initial_velocity=initial_velocity,
    n_steps=20
)
```

### Visualization

```python
from visualization import plot_curvature_analysis, plot_geodesic_trajectory

# Create curvature analysis plot
curvatures = geometry.compute_curvature_series()
fig = plot_curvature_analysis(ohlc_data, curvatures)
fig.show()

# Visualize geodesic
fig2 = plot_geodesic_trajectory(geometry, start_index=50, initial_velocity=[0.5, -0.2])
fig2.show()
```

## Core Components

### `curved_candle_geometry.py`
- `CandleMetric`: Data structure for metric properties at each candle
- `CurvedCandleGeometry`: Main calculator for geometric properties
- Curvature computation algorithms
- Geodesic solver
- Parallel transport implementation

### `visualization.py`
- Interactive Plotly visualizations
- Curvature heatmaps
- Pattern space with metric distortion
- Geodesic trajectory plots
- Animated parallel transport

### `example_usage.py`
- Complete demonstration script
- Synthetic and real data examples
- Full analysis pipeline

## Mathematical Details

### Metric Tensor
For each candle with Range R:
```
g_ij = R² * δ_ij
```

### Gaussian Curvature
```
K = R_1212 / det(g)
```

### Geodesic Equation
```
d²x^μ/dτ² + Γ^μ_νρ (dx^ν/dτ)(dx^ρ/dτ) = 0
```

## Applications

1. **Market Regime Detection**: Identify trending vs volatile periods via curvature
2. **Pattern Evolution**: Predict how patterns evolve along geodesics
3. **Risk Measurement**: Curvature-based volatility metrics
4. **Optimal Trading Paths**: Follow geodesics for efficient transitions

## Future Developments

- [ ] Higher-dimensional pattern spaces
- [ ] Non-diagonal metric tensors
- [ ] Curvature-based trading strategies
- [ ] Real-time curvature monitoring
- [ ] Multi-asset correlations in curved space

## References

- Initial Plan: `initial_plan.md`
- Mathematical Foundation: `initial_report.html`
- General Relativity: Einstein, A. (1915)
- Differential Geometry: do Carmo, M. (1992)

## License

This implementation is provided for research and educational purposes.