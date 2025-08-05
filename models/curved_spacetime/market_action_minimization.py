#!/usr/bin/env python3
"""
Market Action Minimization Framework

Implements the principle of least action for market dynamics in curved spacetime.
Uses GMM cluster centers as energy minima in the potential energy landscape.

Key Components:
- Lagrangian formulation: L = T - V where T is kinetic energy, V is potential
- Action integral: S = ∫ L(q, q̇, τ) dτ over proper time τ
- Euler-Lagrange equations for finding stationary action paths
- GMM-based potential energy with cluster centers as energy wells
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import json
import logging
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

from curved_candle_geometry import CurvedCandleGeometry, CandleMetric, create_candle_metrics_from_ohlc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PotentialWell:
    """Represents a single potential energy well in pattern space."""
    center: np.ndarray  # (sentiment, upper_wick_ratio) coordinates
    depth: float        # Well depth Aᵢ (positive = attractive)
    width: float        # Characteristic width σᵢ
    frequency: float    # Statistical frequency from GMM
    name: str          # Human-readable name

class MarketActionMinimization:
    """
    Main class for market action minimization using curved spacetime geometry.
    
    Combines:
    - Curved spacetime metric from CurvedCandleGeometry
    - GMM-based potential energy landscape
    - Action minimization via Euler-Lagrange equations
    """
    
    def __init__(self, geometry: CurvedCandleGeometry, potential_wells: List[PotentialWell]):
        """
        Initialize the action minimization framework.
        
        Args:
            geometry: CurvedCandleGeometry instance with metric and curvature
            potential_wells: List of potential energy wells from GMM analysis
        """
        self.geometry = geometry
        self.potential_wells = potential_wells
        self.n_candles = len(geometry.candles)
        
        logger.info(f"Initialized action minimization with {len(potential_wells)} potential wells")
    
    def potential_energy(self, sentiment: float, upper_wick_ratio: float) -> float:
        """
        Compute potential energy V(s, u) at given coordinates.
        
        V(s, u) = -Σᵢ Aᵢ exp(-||r - rᵢ||²/2σᵢ²)
        
        Negative values indicate bound states (energy wells).
        """
        r = np.array([sentiment, upper_wick_ratio])
        energy = 0.0
        
        for well in self.potential_wells:
            # Distance from well center
            dr = r - well.center
            distance_squared = np.dot(dr, dr)
            
            # Gaussian well contribution
            well_contribution = -well.depth * np.exp(-distance_squared / (2 * well.width**2))
            energy += well_contribution
        
        return energy
    
    def potential_gradient(self, sentiment: float, upper_wick_ratio: float) -> np.ndarray:
        """
        Compute gradient of potential energy ∇V(s, u).
        
        Force field: F = -∇V
        """
        r = np.array([sentiment, upper_wick_ratio])
        gradient = np.zeros(2)
        
        for well in self.potential_wells:
            # Distance from well center
            dr = r - well.center
            distance_squared = np.dot(dr, dr)
            
            # Gaussian gradient contribution
            exp_factor = np.exp(-distance_squared / (2 * well.width**2))
            gradient_contribution = -(well.depth / well.width**2) * exp_factor * dr
            gradient += gradient_contribution
        
        return gradient
    
    def kinetic_energy(self, mass: float, velocity: np.ndarray, metric_tensor: np.ndarray) -> float:
        """
        Compute kinetic energy T = ½ m gᵢⱼ vⁱ vʲ in curved spacetime.
        
        Args:
            mass: Volume (trading mass) of the candle
            velocity: (ds/dτ, du/dτ) in proper time τ
            metric_tensor: Local metric tensor gᵢⱼ
        """
        # T = ½ m gᵢⱼ vⁱ vʲ
        kinetic = 0.5 * mass * np.dot(velocity, np.dot(metric_tensor, velocity))
        return kinetic
    
    def lagrangian(self, position: np.ndarray, velocity: np.ndarray, 
                  mass: float, metric_tensor: np.ndarray) -> float:
        """
        Compute Lagrangian L = T - V at given state.
        
        Args:
            position: (sentiment, upper_wick_ratio)
            velocity: (ds/dτ, du/dτ) 
            mass: Volume (gravitational mass)
            metric_tensor: Local metric gᵢⱼ
        """
        sentiment, uwr = position
        
        # Kinetic energy in curved spacetime
        T = self.kinetic_energy(mass, velocity, metric_tensor)
        
        # Potential energy from GMM wells
        V = self.potential_energy(sentiment, uwr)
        
        # Lagrangian
        L = T - V
        
        return L
    
    def euler_lagrange_equations(self, tau: float, state: np.ndarray, 
                                candle_index: int) -> np.ndarray:
        """
        Compute derivatives for Euler-Lagrange equations of motion.
        
        State vector: [s, u, ds/dτ, du/dτ]
        
        Euler-Lagrange equation:
        d/dτ(∂L/∂q̇ⁱ) - ∂L/∂qⁱ = 0
        
        In curved spacetime this becomes the geodesic equation with external forces.
        """
        # Unpack state
        s, u, ds_dtau, du_dtau = state
        position = np.array([s, u])
        velocity = np.array([ds_dtau, du_dtau])
        
        # Get current candle properties (interpolate if necessary)
        candle_idx = min(max(0, int(candle_index)), self.n_candles - 1)
        candle = self.geometry.candles[candle_idx]
        
        mass = candle.gravitational_mass
        metric = candle.metric_tensor
        
        # Compute Christoffel symbols for geodesic equation
        christoffel = self.geometry.compute_christoffel_symbols(candle_idx)
        
        # Potential gradient (external force)
        force = -self.potential_gradient(s, u)  # F = -∇V
        
        # Geodesic equation with external forces:
        # d²qⁱ/dτ² = -Γⁱⱼₖ (dqʲ/dτ)(dqᵏ/dτ) + (1/m) Fⁱ
        
        acceleration = np.zeros(2)
        
        # Christoffel symbol contribution (curvature)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    acceleration[i] -= christoffel[i, j, k] * velocity[j] * velocity[k]
        
        # External force contribution
        # Convert force to contravariant components: Fⁱ = gⁱʲ F_j
        try:
            metric_inv = np.linalg.inv(metric)
            force_contravariant = np.dot(metric_inv, force)
            acceleration += force_contravariant / mass
        except np.linalg.LinAlgError:
            # Handle singular metric
            acceleration += force / mass
        
        # Return derivative of state vector
        derivatives = np.array([
            ds_dtau,     # d/dτ(s) = ds/dτ
            du_dtau,     # d/dτ(u) = du/dτ  
            acceleration[0],  # d/dτ(ds/dτ) = d²s/dτ²
            acceleration[1]   # d/dτ(du/dτ) = d²u/dτ²
        ])
        
        return derivatives
    
    def compute_action_along_path(self, path: List[np.ndarray], 
                                 proper_times: np.ndarray) -> float:
        """
        Compute total action S = ∫ L dτ along a given path.
        
        Args:
            path: List of (sentiment, uwr) coordinates
            proper_times: Array of proper time coordinates τ
        """
        if len(path) < 2:
            return 0.0
        
        total_action = 0.0
        
        for i in range(len(path) - 1):
            # Current state
            position = path[i]
            next_position = path[i + 1] 
            
            # Proper time interval
            dtau = proper_times[i + 1] - proper_times[i]
            if dtau <= 0:
                continue
            
            # Velocity in proper time
            velocity = (next_position - position) / dtau
            
            # Get candle properties for this time
            candle_idx = min(i, self.n_candles - 1)
            candle = self.geometry.candles[candle_idx]
            
            # Compute Lagrangian
            L = self.lagrangian(position, velocity, candle.gravitational_mass, candle.metric_tensor)
            
            # Add to action integral
            total_action += L * dtau
        
        return total_action
    
    def find_action_minimizing_path(self, start_position: np.ndarray, end_position: np.ndarray,
                                   start_time: float, end_time: float, 
                                   n_intermediate_points: int = 10) -> Tuple[List[np.ndarray], float]:
        """
        Find the path that minimizes action between two points in pattern space.
        
        Uses variational calculus to find the optimal trajectory.
        """
        logger.info(f"Finding action-minimizing path from {start_position} to {end_position}")
        
        # Create initial guess path (straight line in pattern space)
        tau_points = np.linspace(start_time, end_time, n_intermediate_points + 2)
        
        def create_path_from_params(params):
            """Create path from optimization parameters."""
            # First and last points are fixed
            path = [start_position]
            
            # Intermediate points from parameters
            for i in range(n_intermediate_points):
                s = params[2*i]
                u = params[2*i + 1]
                path.append(np.array([s, u]))
            
            path.append(end_position)
            return path
        
        def action_objective(params):
            """Objective function: total action along path."""
            try:
                path = create_path_from_params(params)
                action = self.compute_action_along_path(path, tau_points)
                return action
            except:
                return 1e10  # Large penalty for invalid paths
        
        # Initial guess: straight line
        initial_params = []
        for i in range(1, n_intermediate_points + 1):
            alpha = i / (n_intermediate_points + 1)
            intermediate_point = (1 - alpha) * start_position + alpha * end_position
            
            # Ensure point is in valid phase space
            s = np.clip(intermediate_point[0], -0.99, 0.99)
            u = np.clip(intermediate_point[1], 0.01, 0.99)
            
            # Apply triangular constraint
            if abs(s) + u > 1.0:
                total = abs(s) + u
                s = s * 0.99 / total
                u = u * 0.99 / total
            
            initial_params.extend([s, u])
        
        # Bounds for pattern space
        bounds = []
        for i in range(n_intermediate_points):
            bounds.append((-0.99, 0.99))  # sentiment bounds
            bounds.append((0.01, 0.99))   # uwr bounds
        
        # Constraint: triangular phase space
        def triangular_constraint(params):
            violations = []
            for i in range(n_intermediate_points):
                s = params[2*i]
                u = params[2*i + 1]
                # |s| + u ≤ 1
                violations.append(1.0 - abs(s) - u)
            return np.array(violations)
        
        constraint = {'type': 'ineq', 'fun': triangular_constraint}
        
        # Optimize action
        result = minimize(
            action_objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            optimal_path = create_path_from_params(result.x)
            optimal_action = result.fun
            logger.info(f"Found optimal path with action = {optimal_action:.6f}")
            return optimal_path, optimal_action
        else:
            logger.warning(f"Optimization failed: {result.message}")
            # Return straight line fallback
            fallback_path = create_path_from_params(initial_params)
            fallback_action = action_objective(initial_params)
            return fallback_path, fallback_action
    
    def solve_equations_of_motion(self, initial_position: np.ndarray, initial_velocity: np.ndarray,
                                 tau_span: Tuple[float, float], n_points: int = 100) -> Dict[str, Any]:
        """
        Solve the Euler-Lagrange equations of motion numerically.
        
        Args:
            initial_position: Starting (sentiment, uwr)
            initial_velocity: Starting (ds/dτ, du/dτ)
            tau_span: (start_tau, end_tau) proper time interval
            n_points: Number of time points to evaluate
        """
        logger.info(f"Solving equations of motion from τ={tau_span[0]} to τ={tau_span[1]}")
        
        # Initial state vector [s, u, ds/dτ, du/dτ]
        initial_state = np.concatenate([initial_position, initial_velocity])
        
        # Time points
        tau_points = np.linspace(tau_span[0], tau_span[1], n_points)
        
        def equations_wrapper(tau, state):
            """Wrapper for equations of motion."""
            # Map proper time to candle index
            candle_fraction = (tau - tau_span[0]) / (tau_span[1] - tau_span[0])
            candle_index = candle_fraction * (self.n_candles - 1)
            return self.euler_lagrange_equations(tau, state, candle_index)
        
        # Solve ODE
        solution = solve_ivp(
            equations_wrapper,
            tau_span,
            initial_state,
            t_eval=tau_points,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        if solution.success:
            # Extract position and velocity trajectories
            positions = solution.y[:2].T  # Shape: (n_points, 2)
            velocities = solution.y[2:].T  # Shape: (n_points, 2)
            
            # Compute action along the trajectory
            path = [pos for pos in positions]
            total_action = self.compute_action_along_path(path, tau_points)
            
            result = {
                'success': True,
                'tau_points': tau_points,
                'positions': positions,
                'velocities': velocities,
                'total_action': total_action,
                'solution': solution
            }
            
            logger.info(f"Successfully solved equations of motion, total action = {total_action:.6f}")
            return result
        else:
            logger.error(f"Failed to solve equations of motion: {solution.message}")
            return {'success': False, 'message': solution.message}

@classmethod
def create_from_gmm_analysis(cls, geometry: CurvedCandleGeometry, 
                           gmm_file_path: str) -> 'MarketActionMinimization':
    """
    Create MarketActionMinimization instance from existing GMM analysis.
    
    Args:
        geometry: CurvedCandleGeometry instance
        gmm_file_path: Path to GMM classification JSON file
    """
    logger.info(f"Loading GMM analysis from {gmm_file_path}")
    
    # Load GMM results
    with open(gmm_file_path, 'r') as f:
        gmm_data = json.load(f)
    
    regions = gmm_data['classification_system']['regions']
    
    # Create potential wells from GMM clusters
    potential_wells = []
    
    for region in regions:
        # Extract cluster properties
        sentiment_center = sum(region['sentiment_bounds']) / 2
        uwr_center = sum(region['uwr_bounds']) / 2
        center = np.array([sentiment_center, uwr_center])
        
        # Well depth proportional to frequency (more common = deeper well)
        frequency = region['frequency_across_securities']
        depth = frequency / 100.0  # Scale to reasonable energy units
        
        # Well width from cluster bounds
        sentiment_width = (region['sentiment_bounds'][1] - region['sentiment_bounds'][0]) / 4
        uwr_width = (region['uwr_bounds'][1] - region['uwr_bounds'][0]) / 4
        width = np.sqrt(sentiment_width**2 + uwr_width**2)
        
        well = PotentialWell(
            center=center,
            depth=depth,
            width=max(width, 0.1),  # Minimum width for numerical stability
            frequency=frequency,
            name=region['name']
        )
        
        potential_wells.append(well)
    
    logger.info(f"Created {len(potential_wells)} potential wells from GMM analysis")
    
    return cls(geometry, potential_wells)

# Add the classmethod to the class
MarketActionMinimization.create_from_gmm_analysis = create_from_gmm_analysis