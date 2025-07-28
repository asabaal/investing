"""
Curved Candle Geometry: Implementing Spacetime Curvature for Financial Markets

This module implements the mathematical framework for treating candlestick patterns
as existing in a curved spacetime where the metric tensor varies with each candle's
Range and Low values.

Key Concepts:
- Each candle has its own local metric tensor g_ij
- Market trajectories follow geodesics in this curved space
- Curvature encodes volatility and market stress
- Pattern recognition must account for local geometry
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd


@dataclass
class CandleMetric:
    """Represents the metric properties at a single candle"""
    range_value: float  # H - L
    low_value: float    # L
    sentiment: float    # (C - O) / (H - L)
    upper_wick_ratio: float  # (H - max(O,C)) / (H - L)
    
    @property
    def metric_tensor(self) -> np.ndarray:
        """
        The local metric tensor at this candle.
        In the simplest case, we use Range^2 as the scale factor.
        
        g_ij = Range^2 * I_2
        """
        return self.range_value**2 * np.eye(2)
    
    @property
    def pattern_coordinates(self) -> np.ndarray:
        """Gauge-invariant pattern coordinates (Sentiment, UWR)"""
        return np.array([self.sentiment, self.upper_wick_ratio])


class CurvedCandleGeometry:
    """
    Main class for computing geometric properties of candlestick series
    in curved spacetime.
    """
    
    def __init__(self, candles: List[CandleMetric]):
        self.candles = candles
        self.n_candles = len(candles)
        
    def compute_metric_derivatives(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute first and second derivatives of the metric tensor at a given index.
        Uses finite differences.
        """
        if index < 1 or index >= self.n_candles - 1:
            # Boundary handling - return zero derivatives
            return np.zeros((2, 2)), np.zeros((2, 2))
        
        # Get metric tensors at neighboring points
        g_prev = self.candles[index - 1].metric_tensor
        g_curr = self.candles[index].metric_tensor
        g_next = self.candles[index + 1].metric_tensor
        
        # First derivative (centered difference)
        dg_dt = (g_next - g_prev) / 2.0
        
        # Second derivative
        d2g_dt2 = g_next - 2 * g_curr + g_prev
        
        return dg_dt, d2g_dt2
    
    def compute_christoffel_symbols(self, index: int) -> np.ndarray:
        """
        Compute Christoffel symbols of the second kind at a given candle.
        
        Γ^k_ij = (1/2) * g^kl * (∂g_il/∂x^j + ∂g_jl/∂x^i - ∂g_ij/∂x^l)
        
        For our 2D pattern space with time-varying metric.
        """
        if index < 1 or index >= self.n_candles - 1:
            return np.zeros((2, 2, 2))
        
        g = self.candles[index].metric_tensor
        g_inv = np.linalg.inv(g)
        dg_dt, _ = self.compute_metric_derivatives(index)
        
        # Initialize Christoffel symbols
        christoffel = np.zeros((2, 2, 2))  # Γ^k_ij
        
        # In our case, the metric only depends on time (candle index)
        # So spatial derivatives are zero, only time derivatives matter
        
        # For a diagonal metric that varies with time:
        # Γ^0_00 = (1/2) * g^00 * ∂g_00/∂t
        # Γ^1_11 = (1/2) * g^11 * ∂g_11/∂t
        
        for k in range(2):
            for i in range(2):
                for j in range(2):
                    if i == j == k:
                        christoffel[k, i, j] = 0.5 * g_inv[k, k] * dg_dt[i, i]
        
        return christoffel
    
    def compute_riemann_tensor(self, index: int) -> np.ndarray:
        """
        Compute the Riemann curvature tensor at a given candle.
        
        R^ρ_σμν = ∂_μ Γ^ρ_νσ - ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ - Γ^ρ_νλ Γ^λ_μσ
        """
        if index < 2 or index >= self.n_candles - 2:
            return np.zeros((2, 2, 2, 2))
        
        # Get Christoffel symbols at current and neighboring points
        gamma_prev = self.compute_christoffel_symbols(index - 1)
        gamma_curr = self.compute_christoffel_symbols(index)
        gamma_next = self.compute_christoffel_symbols(index + 1)
        
        # Compute derivatives of Christoffel symbols
        dgamma_dt = (gamma_next - gamma_prev) / 2.0
        
        # Initialize Riemann tensor
        riemann = np.zeros((2, 2, 2, 2))  # R^ρ_σμν
        
        # For our 2D manifold with time-varying metric, many components vanish
        # The main non-zero component for market curvature:
        # R^0_101 = -R^0_110 = ∂_1 Γ^0_10 - Γ^0_1k Γ^k_10
        
        # Simplified computation for diagonal time-varying metric
        for rho in range(2):
            for sigma in range(2):
                riemann[rho, sigma, 0, 1] = dgamma_dt[rho, 1, sigma]
                riemann[rho, sigma, 1, 0] = -riemann[rho, sigma, 0, 1]
        
        return riemann
    
    def compute_intrinsic_curvature(self, index: int) -> float:
        """
        Compute intrinsic spacetime curvature for a single candle as a differential window.
        
        Each candle represents a local spacetime patch with metric g_ij = Range² × I₂
        The curvature is intrinsic to this window based on how the geometry deviates
        from flat space within the candle's own OHLC structure.
        
        Physical interpretation:
        - The Range creates the "gravitational field strength" 
        - The internal OHLC geometry determines how space curves within this window
        - Larger deviations from equilibrium → higher curvature magnitude
        """
        if index < 0 or index >= self.n_candles:
            return 0.0
        
        candle = self.candles[index]
        
        # Extract candle properties
        O, H, L, C = candle.low_value + candle.range_value * candle.sentiment, \
                     candle.low_value + candle.range_value, \
                     candle.low_value, \
                     candle.low_value + candle.range_value * (candle.sentiment + candle.upper_wick_ratio)
        
        range_val = candle.range_value
        sentiment = candle.sentiment  
        uwr = candle.upper_wick_ratio
        
        # Avoid division by zero
        if range_val < 1e-10:
            return 0.0
        
        # Intrinsic curvature from the differential geometry within this candle window
        # The metric tensor g_ij = Range² × I₂ implies curvature scales with Range variations
        
        # Method: Treat the candle as a local coordinate patch where:
        # - High Range = strong gravitational field = high potential for curvature
        # - OHLC geometry determines the actual curvature sign and magnitude
        
        # 1. Base curvature strength from Range (like gravitational field strength)
        # Use log scaling since ranges vary over orders of magnitude
        range_curvature_strength = np.log1p(range_val)
        
        # 2. Geometric deviation from flat space equilibrium
        # In flat space: sentiment ≈ 0, UWR ≈ 0.5 (balanced candle)
        # Curvature increases with deviation from this equilibrium
        
        # Sentiment deviation: how far from balanced (0)
        sentiment_deviation = abs(sentiment)
        
        # UWR deviation: how far from balanced wicks (0.5)  
        uwr_deviation = abs(uwr - 0.5) * 2  # Scale to [0,1]
        
        # Combined geometric deviation
        total_deviation = np.sqrt(sentiment_deviation**2 + uwr_deviation**2)
        
        # 3. Curvature sign: determined by market regime
        # Positive curvature: trending markets (large range, moderate deviation)
        # Negative curvature: volatile markets (small range, high deviation)
        
        # Trending vs volatile classifier
        trending_factor = range_curvature_strength / (1.0 + total_deviation)
        volatile_factor = total_deviation / (1.0 + range_curvature_strength)
        
        if trending_factor > volatile_factor:
            # Trending regime: positive curvature
            curvature_magnitude = trending_factor
            curvature_sign = 1.0
        else:
            # Volatile regime: negative curvature  
            curvature_magnitude = volatile_factor
            curvature_sign = -1.0
        
        # 4. Final curvature value
        K = curvature_sign * np.tanh(curvature_magnitude)
        
        return K
    
    def compute_proper_time_interval(self, index: int) -> float:
        """
        Compute the proper time interval dτ for a single candle.
        
        In curved spacetime, proper time is related to coordinate time by:
        dτ² = g₀₀ dt²
        
        For our market spacetime with metric g_ij = Range² × I₂:
        - The Range acts as a gravitational potential affecting time dilation
        - Higher volatility (larger Range) → stronger "gravitational field" → more time dilation
        - Lower volatility (smaller Range) → weaker field → time passes more normally
        
        Physical interpretation:
        - Volatile periods feel "longer" in proper time (more market events per unit coordinate time)
        - Calm periods feel "shorter" in proper time (less market activity)
        """
        if index < 0 or index >= self.n_candles:
            return 1.0  # Default coordinate time
        
        candle = self.candles[index]
        range_val = candle.range_value
        
        # Avoid division by zero
        if range_val < 1e-10:
            return 1.0
        
        # Time dilation factor based on Range (gravitational potential)
        # Higher Range → stronger field → more time dilation
        # Use logarithmic scaling since ranges vary over orders of magnitude
        
        # Base time dilation from Range
        # Minimum dilation factor to ensure proper time is always positive
        min_dilation = 0.1
        max_dilation = 10.0
        
        # Normalize range for time dilation calculation
        # Use log scaling to handle wide range of values
        normalized_range = np.log1p(range_val)
        
        # Time dilation: higher volatility → more proper time per coordinate time
        # This means volatile periods are "stretched" in proper time
        time_dilation_factor = min_dilation + (max_dilation - min_dilation) * np.tanh(normalized_range)
        
        return time_dilation_factor
    
    def compute_proper_time_intervals(self) -> np.ndarray:
        """
        Compute proper time intervals dτ for each candle transition.
        
        Returns array where intervals[i] is the proper time interval from candle i-1 to candle i.
        intervals[0] is set to the interval for candle 0 itself.
        """
        intervals = np.zeros(self.n_candles)
        
        # For the first candle, use its own time interval
        if self.n_candles > 0:
            intervals[0] = self.compute_proper_time_interval(0)
        
        # For subsequent candles, compute interval from previous to current
        for i in range(1, self.n_candles):
            intervals[i] = self.compute_proper_time_interval(i - 1)
        
        return intervals
    
    def compute_proper_time_series(self) -> np.ndarray:
        """
        Compute proper time coordinates for all candles.
        
        Returns cumulative proper time τ at each candle, starting from τ=0.
        """
        proper_times = np.zeros(self.n_candles)
        
        # Start at proper time τ = 0
        if self.n_candles > 0:
            proper_times[0] = 0.0
        
        # Accumulate proper time intervals
        for i in range(1, self.n_candles):
            dt_proper = self.compute_proper_time_interval(i - 1)
            proper_times[i] = proper_times[i - 1] + dt_proper
        
        return proper_times

    def compute_curvature_series(self) -> np.ndarray:
        """
        Compute intrinsic curvature for all candles in the series.
        Each candle gets its own curvature based on its OHLC properties.
        """
        curvatures = np.zeros(self.n_candles)
        
        # Compute intrinsic curvature for each candle independently
        for i in range(self.n_candles):
            curvatures[i] = self.compute_intrinsic_curvature(i)
        
        return curvatures
    
    def get_historical_geodesic(self, 
                              start_index: int,
                              end_index: int = None) -> List[np.ndarray]:
        """
        Get the historical path through pattern space from start_index to end_index.
        
        This assumes the market already followed a geodesic path through the 
        curved pattern space defined by the varying Range/Low values.
        """
        if start_index < 0 or start_index >= self.n_candles:
            raise ValueError("Invalid start index")
        
        if end_index is None:
            end_index = self.n_candles - 1
        
        if end_index <= start_index:
            return [self.candles[start_index].pattern_coordinates]
        
        # Extract the actual historical pattern coordinates
        trajectory = []
        for i in range(start_index, min(end_index + 1, self.n_candles)):
            trajectory.append(self.candles[i].pattern_coordinates.copy())
        
        return trajectory
    
    def predict_geodesic_path(self, 
                            start_index: int,
                            initial_velocity: np.ndarray = None,
                            n_steps: int = None,
                            dt: float = 1.0,
                            use_historical: bool = True) -> List[np.ndarray]:
        """
        Compute geodesic path through pattern space.
        
        Args:
            start_index: Starting candle index
            initial_velocity: Initial velocity in pattern space (for prediction)
            n_steps: Number of steps (None = to end of data)
            dt: Time step size
            use_historical: If True, show actual historical path. If False, predict using geodesic equation.
        
        Returns:
            List of (sentiment, UWR) coordinates along the path
        """
        if start_index < 0 or start_index >= self.n_candles:
            raise ValueError("Invalid start index")
        
        if n_steps is None:
            n_steps = self.n_candles - start_index - 1
        
        # Option 1: Historical path (what actually happened)
        if use_historical:
            end_index = min(start_index + n_steps, self.n_candles - 1)
            return self.get_historical_geodesic(start_index, end_index)
        
        # Option 2: Predicted path using geodesic equation
        if initial_velocity is None:
            raise ValueError("initial_velocity required for prediction mode")
        
        # Initialize from actual starting position
        position = self.candles[start_index].pattern_coordinates.copy()
        velocity = initial_velocity.copy()
        trajectory = [position.copy()]
        
        for step in range(n_steps):
            # Use curvature from the historical data at this time point
            current_index = min(start_index + step, self.n_candles - 1)
            christoffel = self.compute_christoffel_symbols(current_index)
            
            # Compute acceleration from geodesic equation
            acceleration = np.zeros(2)
            for mu in range(2):
                for nu in range(2):
                    for rho in range(2):
                        # Clamp to avoid numerical issues
                        gamma_clamped = np.clip(christoffel[mu, nu, rho], -1.0, 1.0)
                        acceleration[mu] -= gamma_clamped * velocity[nu] * velocity[rho]
            
            # Update velocity and position
            velocity += acceleration * dt * 0.1  # Smaller step size for stability
            new_position = position + velocity * dt * 0.1
            
            # Enforce valid pattern space constraints
            sentiment = np.clip(new_position[0], -0.99, 0.99)
            uwr = np.clip(new_position[1], 0.01, 0.99)
            
            # Ensure triangular constraint: |sentiment| + uwr <= 1
            if abs(sentiment) + uwr > 1.0:
                # Scale back proportionally to stay on boundary
                total = abs(sentiment) + uwr
                sentiment = sentiment * 0.99 / total
                uwr = uwr * 0.99 / total
            
            position = np.array([sentiment, uwr])
            trajectory.append(position.copy())
        
        return trajectory
    
    def compute_parallel_transport(self, 
                                 vector: np.ndarray,
                                 start_index: int,
                                 end_index: int) -> np.ndarray:
        """
        Parallel transport a vector along the candle series.
        
        Parallel transport equation: ∇_v V = 0
        dV^μ/dτ + Γ^μ_νρ V^ν (dx^ρ/dτ) = 0
        """
        if start_index == end_index:
            return vector.copy()
        
        direction = 1 if end_index > start_index else -1
        current_vector = vector.copy()
        
        for i in range(start_index, end_index, direction):
            if i < 0 or i >= self.n_candles - 1:
                continue
                
            christoffel = self.compute_christoffel_symbols(i)
            
            # Simplified parallel transport for discrete steps
            # Assume unit velocity along time direction
            dV = np.zeros(2)
            for mu in range(2):
                for nu in range(2):
                    # Clamp Christoffel symbols to avoid overflow
                    gamma_clamped = np.clip(christoffel[mu, nu, 0], -10.0, 10.0)
                    dV[mu] -= gamma_clamped * current_vector[nu] * 0.01  # Small step size
            
            current_vector += dV * direction
            
            # Normalize to preserve magnitude (parallel transport preserves lengths)
            norm = np.linalg.norm(current_vector)
            if norm > 0:
                current_vector = current_vector / norm * np.linalg.norm(vector)
        
        return current_vector
    
    def curved_pattern_distance(self, index1: int, index2: int) -> float:
        """
        Compute the geodesic distance between two candle patterns
        accounting for the curved geometry.
        """
        if index1 == index2:
            return 0.0
        
        # Get pattern coordinates
        pattern1 = self.candles[index1].pattern_coordinates
        pattern2 = self.candles[index2].pattern_coordinates
        
        # For simplicity, compute path integral along the series
        # In practice, you'd solve for the actual geodesic
        distance = 0.0
        direction = 1 if index2 > index1 else -1
        
        for i in range(index1, index2, direction):
            if i < 0 or i >= self.n_candles - 1:
                continue
            
            g = self.candles[i].metric_tensor
            dp = self.candles[i + direction].pattern_coordinates - self.candles[i].pattern_coordinates
            
            # Infinitesimal distance: ds² = g_ij dx^i dx^j
            ds_squared = np.dot(dp, np.dot(g, dp))
            distance += np.sqrt(max(0, ds_squared))
        
        return distance
    
    def market_stress_energy_tensor(self, index: int) -> np.ndarray:
        """
        Compute the stress-energy tensor analogue for market dynamics.
        
        This encodes trading activity and volatility at each point.
        """
        if index < 1 or index >= self.n_candles - 1:
            return np.zeros((2, 2))
        
        # Volume would go here if available
        # For now, use rate of metric change as proxy for market stress
        dg_dt, d2g_dt2 = self.compute_metric_derivatives(index)
        
        # Stress-energy proportional to metric variations
        T = np.abs(dg_dt) + 0.5 * np.abs(d2g_dt2)
        
        return T


def create_candle_metrics_from_ohlc(ohlc_data: pd.DataFrame) -> List[CandleMetric]:
    """
    Convert OHLC data to CandleMetric objects.
    
    Expected columns: 'open', 'high', 'low', 'close'
    """
    metrics = []
    
    for _, row in ohlc_data.iterrows():
        O, H, L, C = row['open'], row['high'], row['low'], row['close']
        
        range_val = H - L
        if range_val > 0:
            sentiment = (C - O) / range_val
            upper_wick_ratio = (H - max(O, C)) / range_val
        else:
            sentiment = 0.0
            upper_wick_ratio = 0.0
        
        metric = CandleMetric(
            range_value=range_val,
            low_value=L,
            sentiment=sentiment,
            upper_wick_ratio=upper_wick_ratio
        )
        metrics.append(metric)
    
    return metrics


def analyze_market_curvature(ohlc_data: pd.DataFrame) -> dict:
    """
    Perform complete curvature analysis on a candlestick series.
    """
    # Create metric objects
    candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
    
    # Initialize geometry calculator
    geometry = CurvedCandleGeometry(candle_metrics)
    
    # Compute curvature series
    curvatures = geometry.compute_curvature_series()
    
    # Analyze curvature regimes
    analysis = {
        'curvatures': curvatures,
        'mean_curvature': np.mean(np.abs(curvatures)),
        'max_curvature': np.max(np.abs(curvatures)),
        'positive_curvature_ratio': np.sum(curvatures > 0) / len(curvatures),
        'trending_periods': np.where(curvatures > 0)[0],
        'volatile_periods': np.where(curvatures < 0)[0],
        'flat_periods': np.where(np.abs(curvatures) < 0.001)[0]
    }
    
    return analysis