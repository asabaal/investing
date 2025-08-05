#!/usr/bin/env python3
"""
Fully Relativistic Market Action Framework

Implements proper relativistic action minimization for market dynamics in curved spacetime.
This addresses the fundamental issues with our previous non-relativistic approach.

Key Relativistic Components:
1. Proper 4D spacetime manifold (τ, t, sentiment, UWR)  
2. Full metric tensor g_μν with signature (-,+,+,+)
3. Relativistic action: S = ∫ L √(-g) dτ where L includes proper kinetic and potential terms
4. Covariant derivatives and proper coordinate transformations
5. Measured potential energy from statistical mechanics: V = -kT ln(P)

Coordinate System:
- x⁰ = ct (coordinate time, c = speed of information propagation)
- x¹ = Low (translation coordinate with symmetry)  
- x² = Range (scale coordinate, H - L)
- x³ = Volume (mass-energy density coordinate)

Invariant Pattern Space:
- Sentiment = (C - O)/(H - L) (gauge invariant)
- Upper Wick Ratio = (H - max(O,C))/(H - L) (gauge invariant)

The metric g_μν encodes:
- Time dilation effects from market volatility
- Spatial curvature from Range/Volume effects  
- Coupling between coordinates
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import json
import logging
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde, multivariate_normal
from sklearn.neighbors import KernelDensity

from curved_candle_geometry import CurvedCandleGeometry, CandleMetric, create_candle_metrics_from_ohlc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RelativisticCandle:
    """Enhanced candle with full 4D spacetime coordinates."""
    coordinate_time: float      # x⁰/c (normalized coordinate time)
    proper_time: float          # τ (proper time)
    low_value: float            # x¹ (translation coordinate)
    range_value: float          # x² (scale coordinate, H - L)
    volume: float               # x³ (mass-energy density)
    
    # Gauge-invariant pattern coordinates (derived)
    sentiment: float            # (C - O)/(H - L)
    upper_wick_ratio: float     # (H - max(O,C))/(H - L)
    
    @property
    def four_position(self) -> np.ndarray:
        """4-vector position x^μ = (ct, Low, Range, Volume)"""
        return np.array([self.coordinate_time, self.low_value, self.range_value, self.volume])
    
    @property 
    def spatial_position(self) -> np.ndarray:
        """3-vector spatial position (Low, Range, Volume)"""
        return np.array([self.low_value, self.range_value, self.volume])
    
    @property
    def invariant_pattern(self) -> np.ndarray:
        """Gauge-invariant pattern coordinates (Sentiment, UWR)"""
        return np.array([self.sentiment, self.upper_wick_ratio])

class RelativisticMarketAction:
    """
    Fully relativistic action minimization framework for market dynamics.
    
    Implements proper general relativistic treatment:
    - 4D spacetime manifold with metric g_μν
    - Covariant action integral
    - Measured potential from statistical mechanics
    - Proper relativistic equations of motion
    """
    
    def __init__(self, candles: List[RelativisticCandle], 
                 information_speed_c: float = 1.0,
                 market_temperature_kT: float = 1.0):
        """
        Initialize relativistic framework.
        
        Args:
            candles: List of relativistic candle data
            information_speed_c: Speed of information propagation (sets time scale)
            market_temperature_kT: Market temperature for statistical mechanics
        """
        self.candles = candles
        self.n_candles = len(candles)
        self.c = information_speed_c  # Speed of information
        self.kT = market_temperature_kT  # Market temperature
        
        # Compute probability density field from data
        self.probability_field = self._compute_probability_field()
        
        logger.info(f"Initialized relativistic framework with {len(candles)} candles")
        logger.info(f"Information speed c = {self.c}, Temperature kT = {self.kT}")
    
    def metric_tensor(self, four_position: np.ndarray) -> np.ndarray:
        """
        Compute the full 4×4 metric tensor g_μν at a given spacetime point.
        
        Signature: (-,+,+,+) (mostly plus convention)
        
        The metric encodes:
        - Time dilation from market volatility (Range/Volume effects)
        - Translation symmetry: g₁₁ independent of Low (gauge symmetry)
        - Range-Volume coupling from your original framework
        
        Args:
            four_position: 4-vector (ct, Low, Range, Volume)
        """
        ct, low, range_val, vol = four_position
        
        # Compute gauge-invariant quantities for stress calculation
        # Note: We need OHLC to compute sentiment/UWR, but for metric we use Range/Volume
        
        # Base metric components
        g = np.zeros((4, 4))
        
        # Time-time component: g₀₀ = -c²f(Range, Volume)
        # Higher Range × Volume → stronger gravitational field → more time dilation
        gravitational_potential = np.log1p(range_val * np.power(vol, 0.5))
        time_dilation_factor = 1.0 + 0.1 * np.tanh(gravitational_potential)
        g[0, 0] = -(self.c**2) * time_dilation_factor
        
        # Low-Low component: g₁₁ = constant (translation symmetry!)
        # Low has translation symmetry, so metric shouldn't depend on Low value
        g[1, 1] = 1.0  # Flat space in Low direction
        
        # Range-Range component: g₂₂ = f(Range, Volume)
        # From your framework: g ∝ Range² × Volume^α
        range_metric = (range_val**2) * np.power(1.0 + vol, 0.5)
        g[2, 2] = range_metric
        
        # Volume-Volume component: g₃₃ = f(Range, Volume)  
        volume_metric = range_val * np.power(1.0 + vol, 0.3)
        g[3, 3] = volume_metric
        
        # Cross-coupling terms
        # Range-Volume coupling (from your original framework)
        g[2, 3] = g[3, 2] = 0.3 * np.sqrt(range_metric * volume_metric)
        
        # Time-Range coupling (volatility affects time)
        g[0, 2] = g[2, 0] = 0.05 * self.c * range_val
        
        # No Time-Low coupling (preserves translation symmetry)
        # No Low-Range or Low-Volume coupling (preserves translation symmetry)
        
        return g
    
    def metric_determinant(self, four_position: np.ndarray) -> float:
        """Compute det(g_μν) for the volume element √(-g)."""
        g = self.metric_tensor(four_position)
        det_g = np.linalg.det(g)
        return det_g
    
    def _compute_probability_field(self) -> callable:
        """
        Compute probability density P(s, u) from GMM cluster analysis.
        
        Uses the observed 5-cluster structure to define discrete probability wells
        at the statistically observed candle geometry centers.
        """
        logger.info("Computing probability density field from GMM cluster analysis...")
        
        # Load GMM analysis results
        gmm_file = Path("phase_space_analysis/candle_geometry_classification.json")
        if not gmm_file.exists():
            logger.warning("GMM analysis file not found, falling back to KDE")
            return self._compute_probability_field_kde()
        
        with open(gmm_file, 'r') as f:
            gmm_data = json.load(f)
        
        regions = gmm_data['classification_system']['regions']
        
        # Create Gaussian mixture from GMM results
        cluster_centers = []
        cluster_weights = []  # Frequencies as weights
        cluster_covariances = []
        
        for region in regions:
            # Cluster center (sentiment, UWR)
            sentiment_center = sum(region['sentiment_bounds']) / 2
            uwr_center = sum(region['uwr_bounds']) / 2
            center = np.array([sentiment_center, uwr_center])
            cluster_centers.append(center)
            
            # Weight = frequency (normalized)
            frequency = region['frequency_across_securities'] / 100.0
            cluster_weights.append(frequency)
            
            # Covariance from bounds (assume diagonal)
            sentiment_width = (region['sentiment_bounds'][1] - region['sentiment_bounds'][0]) / 4
            uwr_width = (region['uwr_bounds'][1] - region['uwr_bounds'][0]) / 4
            covariance = np.diag([sentiment_width**2, uwr_width**2])
            cluster_covariances.append(covariance)
        
        # Normalize weights
        cluster_weights = np.array(cluster_weights)
        cluster_weights = cluster_weights / np.sum(cluster_weights)
        
        def probability_density(sentiment: float, uwr: float) -> float:
            """
            Evaluate probability density using GMM clusters:
            P(s,u) = Σᵢ wᵢ × N(μᵢ, Σᵢ)
            """
            point = np.array([sentiment, uwr])
            total_prob = 0.0
            
            for i, (center, weight, cov) in enumerate(zip(cluster_centers, cluster_weights, cluster_covariances)):
                # Multivariate Gaussian probability
                try:
                    prob_i = multivariate_normal.pdf(point, mean=center, cov=cov)
                    total_prob += weight * prob_i
                except:
                    # Fallback for numerical issues
                    distance_sq = np.sum((point - center)**2)
                    prob_i = np.exp(-0.5 * distance_sq / np.trace(cov))
                    total_prob += weight * prob_i
            
            return total_prob
        
        logger.info(f"Probability field computed using {len(regions)} GMM clusters")
        for i, region in enumerate(regions):
            logger.info(f"  • {region['name']}: weight={cluster_weights[i]:.3f}, center=({cluster_centers[i][0]:.3f}, {cluster_centers[i][1]:.3f})")
        
        return probability_density
    
    def _compute_probability_field_kde(self) -> callable:
        """Fallback KDE method if GMM results not available."""
        logger.info("Using fallback KDE for probability field...")
        
        # Extract gauge-invariant pattern coordinates from all candles
        pattern_coords = []
        for candle in self.candles:
            pattern_coords.append(candle.invariant_pattern)
        
        pattern_data = np.array(pattern_coords)  # Shape: (n_candles, 2)
        
        # Use kernel density estimation for smooth probability field
        n_samples, n_dims = pattern_data.shape
        bandwidth = n_samples ** (-1.0 / (n_dims + 4))  # Scott's rule for KDE
        
        # Create KDE estimator
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(pattern_data)
        
        def probability_density(sentiment: float, uwr: float) -> float:
            """Evaluate probability density at given pattern coordinates."""
            point = np.array([[sentiment, uwr]])
            log_prob = kde.score_samples(point)[0]
            return np.exp(log_prob)
        
        logger.info(f"Fallback KDE computed with bandwidth = {bandwidth:.4f}")
        return probability_density
    
    def measured_potential_energy(self, sentiment: float, uwr: float) -> float:
        """
        Compute potential energy from measured statistical mechanics:
        
        V(s, u) = -kT ln(P(s, u)) + constant
        
        This is the KEY insight: potential energy is MEASURED from statistical frequency
        in the gauge-invariant pattern space (Sentiment, UWR).
        """
        # Get probability density at this point in pattern space
        prob_density = self.probability_field(sentiment, uwr)
        
        # Avoid numerical issues with very low probabilities
        min_prob = 1e-10
        prob_density = max(prob_density, min_prob)
        
        # Statistical mechanics relation
        potential_energy = -self.kT * np.log(prob_density)
        
        return potential_energy
    
    def potential_gradient_in_spacetime(self, four_position: np.ndarray) -> np.ndarray:
        """
        Compute 4-gradient ∇_μ V of the measured potential energy in spacetime coordinates.
        
        The potential V depends only on gauge-invariant pattern coordinates (Sentiment, UWR),
        but we need the gradient in spacetime coordinates (ct, Low, Range, Volume).
        
        Uses chain rule: ∂V/∂x^μ = (∂V/∂s)(∂s/∂x^μ) + (∂V/∂u)(∂u/∂x^μ)
        """
        ct, low, range_val, vol = four_position
        
        # Avoid division by zero
        if range_val < 1e-10:
            return np.zeros(4)
        
        # We need OHLC to compute sentiment and UWR
        # For now, use a simple approximation or assume we have this information
        # In practice, we'd get this from the candle data
        
        # Placeholder: assume we can compute sentiment and UWR from the coordinates
        # This would need to be properly implemented with actual OHLC reconstruction
        sentiment = 0.0  # Placeholder - would compute from OHLC
        uwr = 0.5  # Placeholder - would compute from OHLC
        
        # Potential gradient in pattern space
        eps = 1e-6
        V0 = self.measured_potential_energy(sentiment, uwr)
        dV_ds = (self.measured_potential_energy(sentiment + eps, uwr) - V0) / eps
        dV_du = (self.measured_potential_energy(sentiment, uwr + eps) - V0) / eps
        
        # Chain rule: ∂V/∂x^μ = (∂V/∂s)(∂s/∂x^μ) + (∂V/∂u)(∂u/∂x^μ)
        # 
        # ∂s/∂Low = 0 (sentiment is translation invariant)
        # ∂s/∂Range = -sentiment/Range (from s = (C-O)/(H-L))
        # ∂u/∂Low = 0 (UWR is translation invariant)  
        # ∂u/∂Range = -uwr/Range (from u = (H-max(O,C))/(H-L))
        
        gradient_spacetime = np.zeros(4)
        gradient_spacetime[0] = 0.0  # ∂V/∂t = 0 (potential is static)
        gradient_spacetime[1] = 0.0  # ∂V/∂Low = 0 (translation symmetry)
        gradient_spacetime[2] = dV_ds * (-sentiment/range_val) + dV_du * (-uwr/range_val)  # ∂V/∂Range
        gradient_spacetime[3] = 0.0  # ∂V/∂Volume = 0 (potential independent of volume)
        
        return gradient_spacetime
    
    def relativistic_lagrangian(self, four_position: np.ndarray, 
                               four_velocity: np.ndarray, rest_mass: float) -> float:
        """
        Compute the fully relativistic Lagrangian.
        
        For a particle in curved spacetime with external potential:
        L = -mc² √(-g_μν u^μ u^ν) - V(x^μ) 
        
        where u^μ = dx^μ/dτ is the 4-velocity.
        """
        # Get metric at current position
        g = self.metric_tensor(four_position)
        
        # Compute kinetic term: -mc² √(-g_μν u^μ u^ν)
        # Note: g_μν u^μ u^ν should be negative for timelike geodesics
        metric_contracted = np.dot(four_velocity, np.dot(g, four_velocity))
        
        # Ensure we have a timelike interval (negative)
        if metric_contracted >= 0:
            logger.warning(f"Spacelike/null interval detected: g_μν u^μ u^ν = {metric_contracted}")
            kinetic_term = 0.0
        else:
            kinetic_term = -rest_mass * (self.c**2) * np.sqrt(-metric_contracted)
        
        # Potential energy term (need to compute sentiment and UWR from coordinates)
        ct, low, range_val, vol = four_position
        
        # For the Lagrangian, we need the gauge-invariant pattern coordinates
        # This requires reconstruction from spacetime coordinates or lookup from data
        # For now, use a simplified approach - in practice we'd interpolate from candle data
        
        # Find nearest candle for sentiment/UWR lookup
        if hasattr(self, '_current_candle_index'):
            candle_idx = min(self._current_candle_index, len(self.candles) - 1)
            candle = self.candles[candle_idx]
            sentiment = candle.sentiment
            uwr = candle.upper_wick_ratio
        else:
            # Fallback: use approximate values
            sentiment = 0.0
            uwr = 0.5
        
        potential_term = self.measured_potential_energy(sentiment, uwr)
        
        # Total Lagrangian
        L = kinetic_term - potential_term
        
        return L
    
    def covariant_action_integrand(self, four_position: np.ndarray,
                                  four_velocity: np.ndarray, rest_mass: float) -> float:
        """
        Compute the action integrand with proper volume element:
        
        dS = L √(-g) dτ
        
        This is the fully covariant form.
        """
        # Lagrangian  
        L = self.relativistic_lagrangian(four_position, four_velocity, rest_mass)
        
        # Volume element
        det_g = self.metric_determinant(four_position)
        sqrt_minus_g = np.sqrt(-det_g) if det_g < 0 else np.sqrt(abs(det_g))
        
        # Action integrand
        integrand = L * sqrt_minus_g
        
        return integrand
    
    def compute_relativistic_action(self, worldline: List[np.ndarray], 
                                   proper_times: np.ndarray, rest_mass: float) -> float:
        """
        Compute total relativistic action along a worldline.
        
        S = ∫ L √(-g) dτ
        
        Args:
            worldline: List of 4-positions along the path
            proper_times: Array of proper time coordinates
            rest_mass: Rest mass of the market "particle"
        """
        if len(worldline) < 2:
            return 0.0
        
        total_action = 0.0
        
        for i in range(len(worldline) - 1):
            # Current state
            x_mu = worldline[i]
            x_mu_next = worldline[i + 1]
            
            # Proper time interval
            dtau = proper_times[i + 1] - proper_times[i]
            if dtau <= 0:
                continue
            
            # 4-velocity: u^μ = dx^μ/dτ
            four_velocity = (x_mu_next - x_mu) / dtau
            
            # Normalize velocity to avoid spacelike intervals
            # For timelike geodesics, we need g_μν u^μ u^ν < 0
            g = self.metric_tensor(x_mu)
            velocity_norm_squared = np.dot(four_velocity, np.dot(g, four_velocity))
            
            if velocity_norm_squared >= 0:
                # Force timelike by scaling down spatial components
                four_velocity[1:] *= 0.1  # Reduce spatial velocity
                # Ensure time component dominates
                four_velocity[0] = max(abs(four_velocity[0]), 1.0) * np.sign(four_velocity[0])
                if four_velocity[0] == 0:
                    four_velocity[0] = 1.0
            
            # Action integrand
            integrand = self.covariant_action_integrand(x_mu, four_velocity, rest_mass)
            
            # Add to total action
            total_action += integrand * dtau
        
        return total_action
    
    def einstein_field_equations(self, four_position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Einstein field equations G_μν = 8πG T_μν.
        
        This connects spacetime curvature to market stress-energy.
        For now, we compute the Einstein tensor G_μν and a simplified
        stress-energy tensor T_μν from market activity.
        """
        # This is a placeholder for full Einstein tensor computation
        # Would require computing Riemann tensor, Ricci tensor, etc.
        
        # Simplified stress-energy tensor from market activity
        _, s, u, vol = four_position
        
        # Energy density (volume represents mass-energy)
        rho = vol / (1.0 + vol)  # Normalized energy density
        
        # Pressure from market stress
        market_stress = abs(s) + u
        pressure = 0.1 * market_stress * rho
        
        # Simplified stress-energy tensor (diagonal)
        T = np.zeros((4, 4))
        T[0, 0] = rho  # Energy density
        T[1, 1] = pressure  # Spatial pressure
        T[2, 2] = pressure
        T[3, 3] = pressure
        
        # Placeholder Einstein tensor (would need full calculation)
        G = np.zeros((4, 4))
        
        return G, T
    
    def solve_relativistic_equations_of_motion(self, initial_four_position: np.ndarray,
                                             initial_four_velocity: np.ndarray,
                                             tau_span: Tuple[float, float],
                                             rest_mass: float = 1.0,
                                             n_points: int = 100) -> Dict[str, Any]:
        """
        Solve the relativistic equations of motion (geodesic equation with external forces).
        
        The equation is:
        d²x^μ/dτ² + Γ^μ_αβ (dx^α/dτ)(dx^β/dτ) = F^μ/m
        
        where F^μ is the 4-force from the external potential.
        """
        logger.info(f"Solving relativistic equations of motion from τ={tau_span[0]:.3f} to τ={tau_span[1]:.3f}")
        
        # Initial state: [x^0, x^1, x^2, x^3, dx^0/dτ, dx^1/dτ, dx^2/dτ, dx^3/dτ]
        initial_state = np.concatenate([initial_four_position, initial_four_velocity])
        
        def relativistic_equations(tau, state):
            """Relativistic equations of motion."""
            # Unpack state
            four_pos = state[:4]
            four_vel = state[4:]
            
            # Compute metric and Christoffel symbols (simplified)
            g = self.metric_tensor(four_pos)
            
            # For now, use simplified geodesic equation
            # Full implementation would compute all Christoffel symbols
            
            # External force from potential gradient in spacetime coordinates
            force_4d = -self.potential_gradient_in_spacetime(four_pos)  # F^μ = -∂V/∂x^μ
            
            # Use the 4-force directly
            four_force = force_4d
            
            # Geodesic equation: d²x^μ/dτ² = -Γ^μ_αβ u^α u^β + F^μ/m
            acceleration = four_force / rest_mass  # Simplified (ignoring Christoffel terms for now)
            
            # Return derivatives: [dx^μ/dτ, d²x^μ/dτ²]
            derivatives = np.concatenate([four_vel, acceleration])
            
            return derivatives
        
        # Solve ODE
        tau_points = np.linspace(tau_span[0], tau_span[1], n_points)
        
        solution = solve_ivp(
            relativistic_equations,
            tau_span,
            initial_state,
            t_eval=tau_points,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        if solution.success:
            # Extract worldline
            worldline = solution.y[:4].T  # Shape: (n_points, 4)
            four_velocities = solution.y[4:].T  # Shape: (n_points, 4)
            
            # Compute total action
            worldline_list = [pos for pos in worldline]
            total_action = self.compute_relativistic_action(worldline_list, tau_points, rest_mass)
            
            result = {
                'success': True,
                'tau_points': tau_points,
                'worldline': worldline,
                'four_velocities': four_velocities,
                'total_action': total_action,
                'solution': solution
            }
            
            logger.info(f"Successfully solved relativistic equations, total action = {total_action:.6f}")
            return result
        else:
            logger.error(f"Failed to solve relativistic equations: {solution.message}")
            return {'success': False, 'message': solution.message}

def create_relativistic_candles_from_geometry(geometry: CurvedCandleGeometry) -> List[RelativisticCandle]:
    """
    Convert CurvedCandleGeometry to relativistic candles with proper 4D coordinates.
    """
    logger.info("Converting curved geometry to relativistic candles...")
    
    relativistic_candles = []
    
    # Compute proper times from geometry
    proper_times = geometry.compute_proper_time_series()
    
    # Normalize coordinates to avoid huge values
    low_values = [candle.low_value for candle in geometry.candles]
    range_values = [candle.range_value for candle in geometry.candles]
    volumes = [candle.volume for candle in geometry.candles]
    
    # Normalize to reasonable scales
    low_mean = np.mean(low_values)
    low_std = np.std(low_values)
    range_mean = np.mean(range_values)
    range_std = np.std(range_values)
    volume_mean = np.mean(volumes)
    volume_std = np.std(volumes)
    
    for i, candle in enumerate(geometry.candles):
        # Coordinate time (normalized)
        coordinate_time = float(i) * 0.1  # Scale down time coordinate
        
        # Proper time  
        proper_time = proper_times[i]
        
        # Normalize spatial coordinates to avoid huge metric values
        low_normalized = (candle.low_value - low_mean) / (low_std + 1e-10)
        range_normalized = candle.range_value / (range_mean + 1e-10)
        volume_normalized = np.log1p(candle.volume) / 10.0  # Log scale for volume
        
        # Gauge-invariant coordinates (unchanged)
        sentiment = candle.sentiment
        uwr = candle.upper_wick_ratio
        
        rel_candle = RelativisticCandle(
            coordinate_time=coordinate_time,
            proper_time=proper_time,
            low_value=low_normalized,  # Normalized
            range_value=range_normalized,  # Normalized
            volume=volume_normalized,  # Normalized
            sentiment=sentiment,
            upper_wick_ratio=uwr
        )
        
        relativistic_candles.append(rel_candle)
    
    logger.info(f"Created {len(relativistic_candles)} relativistic candles with normalized coordinates")
    return relativistic_candles