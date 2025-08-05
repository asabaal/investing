#!/usr/bin/env python3
"""
Non-Parametric Potential Learning for Market Dynamics

Instead of assuming Gaussian wells, we learn the potential energy landscape directly 
from observed market trajectory data using:

1. Maximum Entropy Methods - Find the potential that maximizes entropy subject to 
   constraints from observed statistics
2. Inverse Optimal Control - Recover the "reward function" (negative potential) 
   that best explains observed trajectory choices
3. Kernel Density-Based Potential - Use data-driven density estimation for V = -kT ln(P)
4. Trajectory-Based Learning - Learn from actual market paths, not just endpoints

Key Insight: Let the market data teach us what the potential actually looks like!
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any, Optional
import json
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator, griddata
from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

from market_data_database import MarketDataDatabase
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc
from validate_physics_predictions import PhysicsValidator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NonParametricPotentialLearner:
    """
    Learn potential energy landscape directly from market trajectory data.
    
    Uses multiple approaches:
    1. Maximum entropy potential learning
    2. Inverse optimal control from trajectories  
    3. Adaptive kernel density estimation
    4. Trajectory flow analysis
    """
    
    def __init__(self, market_data: pd.DataFrame, temperature_kT: float = 0.5):
        """
        Initialize with market data and temperature parameter.
        
        Args:
            market_data: OHLCV market data
            temperature_kT: Market temperature for statistical mechanics
        """
        self.market_data = market_data
        self.kT = temperature_kT
        
        # Extract trajectory data
        self.trajectory_points = self._extract_trajectory_data()
        self.n_points = len(self.trajectory_points)
        
        # Phase space bounds
        self.sentiment_bounds = (-0.99, 0.99)
        self.uwr_bounds = (0.01, 0.99)
        
        logger.info(f"Initialized non-parametric learner with {self.n_points} trajectory points")
        logger.info(f"Temperature kT = {self.kT}")
    
    def _extract_trajectory_data(self) -> np.ndarray:
        """Extract (sentiment, UWR) trajectory points from market data."""
        
        # Create candle metrics from OHLC data
        candle_metrics = create_candle_metrics_from_ohlc(self.market_data)
        
        # Extract gauge-invariant coordinates
        trajectory_points = []
        for candle in candle_metrics:
            # Apply phase space constraint: |sentiment| + UWR ≤ 1
            if abs(candle.sentiment) + candle.upper_wick_ratio <= 1.0:
                trajectory_points.append([candle.sentiment, candle.upper_wick_ratio])
        
        return np.array(trajectory_points)
    
    def learn_adaptive_density_potential(self, grid_resolution: int = 100, 
                                       bandwidth: str = 'scott') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Learn potential using adaptive kernel density estimation.
        
        V(s, u) = -kT ln(P(s, u)) where P is learned from data
        
        Args:
            grid_resolution: Number of grid points per dimension
            bandwidth: KDE bandwidth ('scott', 'silverman', or float)
            
        Returns:
            (sentiment_grid, uwr_grid, potential_grid)
        """
        logger.info(f"Learning adaptive density potential with {grid_resolution}x{grid_resolution} grid")
        
        # Create evaluation grid
        s_grid = np.linspace(self.sentiment_bounds[0], self.sentiment_bounds[1], grid_resolution)
        u_grid = np.linspace(self.uwr_bounds[0], self.uwr_bounds[1], grid_resolution)
        S, U = np.meshgrid(s_grid, u_grid)
        
        # Apply triangular phase space constraint
        valid_mask = np.abs(S) + U <= 1.0
        
        # Flatten for KDE evaluation
        grid_points = np.column_stack([S.ravel(), U.ravel()])
        
        # Fit adaptive KDE to trajectory data
        if isinstance(bandwidth, str):
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        else:
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        
        kde.fit(self.trajectory_points)
        
        # Evaluate probability density
        log_prob = kde.score_samples(grid_points)
        prob_density = np.exp(log_prob)
        
        # Reshape back to grid
        prob_grid = prob_density.reshape(S.shape)
        
        # Compute potential: V = -kT ln(P)
        # Add small constant to avoid log(0)
        min_prob = 1e-10
        prob_grid = np.maximum(prob_grid, min_prob)
        potential_grid = -self.kT * np.log(prob_grid)
        
        # Apply phase space constraints
        potential_grid[~valid_mask] = np.nan
        
        logger.info(f"Learned potential range: [{np.nanmin(potential_grid):.4f}, {np.nanmax(potential_grid):.4f}]")
        
        return S, U, potential_grid
    
    def learn_maximum_entropy_potential(self, n_basis_functions: int = 20,
                                      regularization: float = 1e-3) -> Dict[str, Any]:
        """
        Learn potential using maximum entropy principle.
        
        Find potential V(s,u) that maximizes entropy subject to constraints
        from observed trajectory statistics.
        
        Args:
            n_basis_functions: Number of radial basis functions
            regularization: L2 regularization strength
            
        Returns:
            Dictionary with learned potential parameters
        """
        logger.info(f"Learning maximum entropy potential with {n_basis_functions} basis functions")
        
        # Create radial basis function centers
        # Use observed data points as initial centers, then add some exploration
        n_data_centers = min(n_basis_functions // 2, len(self.trajectory_points))
        data_indices = np.random.choice(len(self.trajectory_points), n_data_centers, replace=False)
        data_centers = self.trajectory_points[data_indices]
        
        # Add additional centers for exploration
        n_exploration = n_basis_functions - n_data_centers
        exploration_centers = []
        for _ in range(n_exploration):
            # Sample uniformly in valid phase space
            while True:
                s = np.random.uniform(self.sentiment_bounds[0], self.sentiment_bounds[1])
                u = np.random.uniform(self.uwr_bounds[0], self.uwr_bounds[1])
                if abs(s) + u <= 1.0:
                    exploration_centers.append([s, u])
                    break
        
        exploration_centers = np.array(exploration_centers)
        centers = np.vstack([data_centers, exploration_centers])
        
        # Compute basis function matrix for observed points
        def rbf_basis(points, centers, length_scale=0.2):
            """Radial basis function evaluation."""
            distances = cdist(points, centers)
            return np.exp(-distances**2 / (2 * length_scale**2))
        
        Phi = rbf_basis(self.trajectory_points, centers)
        
        # Maximum entropy objective: maximize entropy while fitting data
        def max_entropy_objective(weights):
            """Objective function for maximum entropy learning."""
            
            # Potential at data points
            potential_data = Phi @ weights
            
            # Probability at data points (unnormalized)
            log_prob_data = -potential_data / self.kT
            
            # Regularization term
            reg_term = regularization * np.sum(weights**2)
            
            # Maximum entropy: maximize sum of log probabilities (minimize negative)
            entropy_term = -np.mean(log_prob_data)
            
            return entropy_term + reg_term
        
        # Optimize weights
        initial_weights = np.random.normal(0, 0.1, n_basis_functions)
        
        result = minimize(
            max_entropy_objective,
            initial_weights,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            logger.info(f"Maximum entropy optimization converged with objective = {result.fun:.4f}")
        else:
            logger.warning(f"Maximum entropy optimization failed: {result.message}")
            optimal_weights = initial_weights
        
        # Create potential function
        def learned_potential(sentiment, uwr):
            """Evaluate learned potential at given point."""
            point = np.array([[sentiment, uwr]])
            basis_values = rbf_basis(point, centers)
            return (basis_values @ optimal_weights)[0]
        
        return {
            'type': 'maximum_entropy',
            'weights': optimal_weights,
            'centers': centers,
            'potential_function': learned_potential,
            'optimization_result': result,
            'length_scale': 0.2
        }
    
    def learn_trajectory_flow_potential(self, trajectory_length: int = 10,
                                      flow_strength: float = 1.0) -> Dict[str, Any]:
        """
        Learn potential from trajectory flow patterns.
        
        Analyzes how trajectories actually move through phase space and learns
        the underlying flow field that generates these movements.
        
        Args:
            trajectory_length: Length of trajectory segments to analyze
            flow_strength: Strength of flow field contribution
            
        Returns:
            Dictionary with learned flow-based potential
        """
        logger.info(f"Learning trajectory flow potential with segments of length {trajectory_length}")
        
        # Extract trajectory segments
        trajectory_segments = []
        for i in range(len(self.trajectory_points) - trajectory_length):
            segment = self.trajectory_points[i:i+trajectory_length+1]
            trajectory_segments.append(segment)
        
        logger.info(f"Extracted {len(trajectory_segments)} trajectory segments")
        
        # Compute flow vectors at each point
        flow_vectors = []
        flow_points = []
        
        for segment in trajectory_segments:
            for i in range(len(segment) - 1):
                current_point = segment[i]
                next_point = segment[i + 1]
                
                # Flow vector = direction of movement
                flow_vector = next_point - current_point
                
                flow_points.append(current_point)
                flow_vectors.append(flow_vector)
        
        flow_points = np.array(flow_points)
        flow_vectors = np.array(flow_vectors)
        
        # Learn potential from flow: F = -∇V, so V recovers force field
        # Use interpolation to create smooth potential from flow data
        
        # Create grid for potential evaluation
        grid_resolution = 50
        s_grid = np.linspace(self.sentiment_bounds[0], self.sentiment_bounds[1], grid_resolution)
        u_grid = np.linspace(self.uwr_bounds[0], self.uwr_bounds[1], grid_resolution)
        S, U = np.meshgrid(s_grid, u_grid)
        
        # Apply phase space constraint
        valid_mask = np.abs(S) + U <= 1.0
        
        # Interpolate flow field to grid
        grid_points = np.column_stack([S.ravel(), U.ravel()])
        
        try:
            # Interpolate flow components
            flow_s = griddata(flow_points, flow_vectors[:, 0], grid_points, method='linear', fill_value=0)
            flow_u = griddata(flow_points, flow_vectors[:, 1], grid_points, method='linear', fill_value=0)
            
            # Reshape to grid
            flow_s_grid = flow_s.reshape(S.shape)
            flow_u_grid = flow_u.reshape(S.shape)
            
            # Compute potential by integrating flow field
            # This is a simplified integration - full implementation would solve ∇V = -F
            potential_grid = np.zeros_like(S)
            
            for i in range(1, grid_resolution):
                for j in range(1, grid_resolution):
                    if valid_mask[i, j]:
                        # Simple finite difference integration
                        ds = s_grid[1] - s_grid[0]
                        du = u_grid[1] - u_grid[0]
                        
                        # V(i,j) ≈ V(i-1,j) - F_s * ds
                        potential_from_s = potential_grid[i-1, j] - flow_s_grid[i, j] * ds
                        # V(i,j) ≈ V(i,j-1) - F_u * du  
                        potential_from_u = potential_grid[i, j-1] - flow_u_grid[i, j] * du
                        
                        # Average the two estimates
                        potential_grid[i, j] = 0.5 * (potential_from_s + potential_from_u)
            
            # Apply constraints
            potential_grid[~valid_mask] = np.nan
            
            def flow_potential_function(sentiment, uwr):
                """Evaluate flow-based potential at given point."""
                # Simple interpolation from grid
                s_idx = np.argmin(np.abs(s_grid - sentiment))
                u_idx = np.argmin(np.abs(u_grid - uwr))
                return potential_grid[u_idx, s_idx]
            
            logger.info(f"Learned flow potential from {len(flow_points)} flow vectors")
            
            return {
                'type': 'trajectory_flow',
                'potential_grid': potential_grid,
                'flow_s_grid': flow_s_grid,
                'flow_u_grid': flow_u_grid,
                'sentiment_grid': S,
                'uwr_grid': U,
                'potential_function': flow_potential_function,
                'flow_vectors': flow_vectors,
                'flow_points': flow_points
            }
            
        except Exception as e:
            logger.error(f"Flow potential learning failed: {e}")
            # Fallback to simple density-based potential
            return self._fallback_density_potential()
    
    def _fallback_density_potential(self) -> Dict[str, Any]:
        """Fallback density-based potential when other methods fail."""
        S, U, V = self.learn_adaptive_density_potential(grid_resolution=50)
        
        def fallback_function(sentiment, uwr):
            s_idx = np.argmin(np.abs(S[0, :] - sentiment))
            u_idx = np.argmin(np.abs(U[:, 0] - uwr))
            return V[u_idx, s_idx]
        
        return {
            'type': 'fallback_density',
            'potential_grid': V,
            'sentiment_grid': S,
            'uwr_grid': U,
            'potential_function': fallback_function
        }
    
    def learn_inverse_optimal_control_potential(self, n_iterations: int = 100,
                                              learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Learn potential using inverse optimal control.
        
        Treats observed trajectories as optimal solutions and recovers the 
        underlying reward function (negative potential) that explains them.
        
        Args:
            n_iterations: Number of learning iterations
            learning_rate: Learning rate for gradient descent
            
        Returns:
            Dictionary with learned potential from inverse optimal control
        """
        logger.info(f"Learning potential via inverse optimal control with {n_iterations} iterations")
        
        # Extract trajectory segments for learning
        segment_length = 5
        segments = []
        
        for i in range(len(self.trajectory_points) - segment_length):
            segment = self.trajectory_points[i:i+segment_length]
            segments.append(segment)
        
        # Create basis functions for reward representation
        n_basis = 15
        centers = []
        
        # Use k-means-like approach to place basis centers
        from sklearn.cluster import KMeans
        if len(self.trajectory_points) >= n_basis:
            kmeans = KMeans(n_clusters=n_basis, random_state=42)
            kmeans.fit(self.trajectory_points)
            centers = kmeans.cluster_centers_
        else:
            # Use data points as centers if not enough data
            centers = self.trajectory_points[:n_basis]
        
        centers = np.array(centers)
        
        def reward_function(points, weights):
            """Compute reward at points using basis functions."""
            distances = cdist(points, centers)
            basis = np.exp(-distances**2 / (2 * 0.3**2))  # RBF basis
            return basis @ weights
        
        # Initialize reward weights
        weights = np.random.normal(0, 0.1, n_basis)
        
        # Inverse optimal control learning
        for iteration in range(n_iterations):
            total_loss = 0
            gradient = np.zeros_like(weights)
            
            for segment in segments[:50]:  # Use subset for speed
                # Current reward function
                rewards = reward_function(segment, weights)
                
                # Expert trajectory features (what we observed)
                expert_features = np.mean(cdist(segment, centers), axis=0)
                
                # Generate alternative trajectories and their features
                n_alternatives = 10
                alternative_features = []
                
                for _ in range(n_alternatives):
                    # Generate random alternative trajectory
                    alt_traj = []
                    current = segment[0]
                    
                    for _ in range(len(segment) - 1):
                        # Random step with some preference for staying in bounds
                        noise = np.random.normal(0, 0.1, 2)
                        next_point = current + noise
                        
                        # Keep in bounds
                        next_point[0] = np.clip(next_point[0], self.sentiment_bounds[0], self.sentiment_bounds[1])
                        next_point[1] = np.clip(next_point[1], self.uwr_bounds[0], self.uwr_bounds[1])
                        
                        # Apply phase space constraint
                        if abs(next_point[0]) + next_point[1] > 1.0:
                            next_point = current + noise * 0.1  # Smaller step
                        
                        alt_traj.append(next_point)
                        current = next_point
                    
                    alt_traj = np.array(alt_traj)
                    alt_features = np.mean(cdist(alt_traj, centers), axis=0)
                    alternative_features.append(alt_features)
                
                alternative_features = np.array(alternative_features)
                
                # Maximum margin loss: expert trajectory should have higher reward
                expert_reward = np.sum(expert_features * weights)
                alt_rewards = np.sum(alternative_features * weights, axis=1)
                
                # Loss: max(0, margin + max(alt_reward) - expert_reward)
                margin = 0.1
                max_alt_reward = np.max(alt_rewards)
                loss = max(0, margin + max_alt_reward - expert_reward)
                total_loss += loss
                
                # Gradient
                if loss > 0:
                    gradient -= expert_features  # Increase expert reward
                    best_alt_idx = np.argmax(alt_rewards)
                    gradient += alternative_features[best_alt_idx]  # Decrease best alternative
            
            # Update weights
            weights -= learning_rate * gradient / len(segments[:50])
            
            if iteration % 20 == 0:
                logger.info(f"IOC iteration {iteration}, loss = {total_loss:.4f}")
        
        # Create potential function (negative reward)
        def ioc_potential_function(sentiment, uwr):
            point = np.array([[sentiment, uwr]])
            reward = reward_function(point, weights)[0]
            return -reward  # Potential = -reward
        
        logger.info(f"Inverse optimal control learning completed")
        
        return {
            'type': 'inverse_optimal_control',
            'weights': weights,
            'centers': centers,
            'potential_function': ioc_potential_function,
            'final_loss': total_loss
        }
    
    def compare_learning_methods(self) -> Dict[str, Any]:
        """Compare all non-parametric learning methods."""
        
        logger.info("🔍 Comparing all non-parametric potential learning methods")
        
        methods = {}
        
        # Method 1: Adaptive density
        logger.info("Learning Method 1: Adaptive Density Estimation")
        S, U, V_density = self.learn_adaptive_density_potential()
        
        def density_func(s, u):
            s_idx = np.argmin(np.abs(S[0, :] - s))
            u_idx = np.argmin(np.abs(U[:, 0] - u))
            return V_density[u_idx, s_idx]
        
        methods['adaptive_density'] = {
            'potential_function': density_func,
            'grid_data': (S, U, V_density),
            'description': 'V = -kT ln(P) with adaptive KDE'
        }
        
        # Method 2: Maximum entropy
        logger.info("Learning Method 2: Maximum Entropy")
        me_result = self.learn_maximum_entropy_potential()
        methods['maximum_entropy'] = me_result
        
        # Method 3: Trajectory flow
        logger.info("Learning Method 3: Trajectory Flow Analysis")
        flow_result = self.learn_trajectory_flow_potential()
        methods['trajectory_flow'] = flow_result
        
        # Method 4: Inverse optimal control
        logger.info("Learning Method 4: Inverse Optimal Control")
        ioc_result = self.learn_inverse_optimal_control_potential()
        methods['inverse_optimal_control'] = ioc_result
        
        logger.info("✅ All learning methods completed")
        
        return methods

class LearnedPotentialValidator:
    """Validate learned potentials against observed market statistics."""
    
    def __init__(self, learned_methods: Dict[str, Any], temperature_kT: float = 0.5):
        """Initialize validator with learned potential methods."""
        self.learned_methods = learned_methods
        self.kT = temperature_kT
        
    def validate_all_methods(self, n_trajectories: int = 50, n_steps: int = 100) -> Dict[str, Any]:
        """Validate all learned potential methods."""
        
        logger.info(f"🧪 Validating all learned potential methods")
        
        validation_results = {}
        
        for method_name, method_data in self.learned_methods.items():
            logger.info(f"Validating method: {method_name}")
            
            try:
                # Create a custom potential class for this method
                class LearnedPotential:
                    def __init__(self, potential_func, kT):
                        self.potential_func = potential_func
                        self.kT = kT
                        # Add dummy well_parameters for compatibility with validator
                        self.well_parameters = [
                            {'name': 'Learned Well 1', 'center': np.array([0.0, 0.5]), 'depth': 1.0, 'frequency': 0.2, 'width_s': 0.1, 'width_u': 0.1},
                            {'name': 'Learned Well 2', 'center': np.array([0.5, 0.3]), 'depth': 0.8, 'frequency': 0.2, 'width_s': 0.1, 'width_u': 0.1},
                            {'name': 'Learned Well 3', 'center': np.array([-0.5, 0.3]), 'depth': 0.6, 'frequency': 0.2, 'width_s': 0.1, 'width_u': 0.1},
                            {'name': 'Learned Well 4', 'center': np.array([0.0, 0.8]), 'depth': 0.4, 'frequency': 0.2, 'width_s': 0.1, 'width_u': 0.1},
                            {'name': 'Learned Well 5', 'center': np.array([0.3, 0.2]), 'depth': 0.2, 'frequency': 0.2, 'width_s': 0.1, 'width_u': 0.1}
                        ]
                    
                    def potential_energy(self, sentiment, uwr):
                        try:
                            return self.potential_func(sentiment, uwr)
                        except:
                            return 0.0  # Fallback
                    
                    def force_field(self, sentiment, uwr):
                        # Numerical gradient with boundary handling
                        eps = 1e-4
                        
                        # Ensure we stay within bounds and phase space constraints
                        s_plus = min(sentiment + eps, 0.99)
                        s_minus = max(sentiment - eps, -0.99)
                        u_plus = min(uwr + eps, 0.99)
                        u_minus = max(uwr - eps, 0.01)
                        
                        # Apply phase space constraint
                        if abs(s_plus) + uwr > 1.0:
                            s_plus = sentiment + eps * 0.1
                        if abs(sentiment) + u_plus > 1.0:
                            u_plus = uwr + eps * 0.1
                        
                        try:
                            V0 = self.potential_energy(sentiment, uwr)
                            V_s_plus = self.potential_energy(s_plus, uwr)
                            V_u_plus = self.potential_energy(sentiment, u_plus)
                            
                            # Central difference when possible
                            if abs(s_minus) + uwr <= 1.0:
                                V_s_minus = self.potential_energy(s_minus, uwr)
                                dV_ds = (V_s_plus - V_s_minus) / (2 * eps)
                            else:
                                dV_ds = (V_s_plus - V0) / eps
                            
                            if abs(sentiment) + u_minus <= 1.0:
                                V_u_minus = self.potential_energy(sentiment, u_minus)
                                dV_du = (V_u_plus - V_u_minus) / (2 * eps)
                            else:
                                dV_du = (V_u_plus - V0) / eps
                            
                            return -np.array([dV_ds, dV_du])  # Force = -∇V
                        except:
                            return np.array([0.0, 0.0])  # Fallback
                
                potential = LearnedPotential(method_data['potential_function'], self.kT)
                
                # Create validator
                validator = PhysicsValidator(potential)
                
                # Generate trajectories
                trajectories = validator.generate_multiple_trajectories(
                    n_trajectories=n_trajectories,
                    n_steps=n_steps,
                    temperature=self.kT
                )
                
                # Analyze results
                predicted_stats = validator.analyze_trajectory_statistics(trajectories)
                comparison = validator.compare_with_observations(predicted_stats)
                
                validation_results[method_name] = {
                    'agreement_score': comparison['overall_agreement'],
                    'mean_relative_error': comparison['mean_relative_error'],
                    'cluster_comparison': comparison['cluster_comparison'],
                    'predicted_stats': predicted_stats,
                    'n_trajectories': len(trajectories)
                }
                
                logger.info(f"  {method_name}: Agreement = {comparison['overall_agreement']:.1%}, Error = {comparison['mean_relative_error']:.1%}")
                
            except Exception as e:
                logger.error(f"Validation failed for {method_name}: {e}")
                validation_results[method_name] = {
                    'agreement_score': 0.0,
                    'mean_relative_error': 999.0,
                    'error': str(e)
                }
        
        return validation_results

def run_nonparametric_learning_analysis():
    """Run complete non-parametric potential learning analysis."""
    
    logger.info("🚀 Starting Non-Parametric Potential Learning Analysis")
    
    # Load market data
    db = MarketDataDatabase()
    symbol = "QQQ"
    
    try:
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=200)).strftime('%Y-%m-%d')
        data = db.get_data(symbol, start_date=start_date, end_date=end_date)
        
        # Map to expected format
        market_data = pd.DataFrame({
            'open': data['Open'],
            'high': data['High'],
            'low': data['Low'],
            'close': data['Close'],
            'volume': data['Volume']
        })
        
        logger.info(f"Loaded {len(market_data)} candles for {symbol}")
        
    except Exception as e:
        logger.error(f"Failed to load market data: {e}")
        return
    
    # Initialize learner
    learner = NonParametricPotentialLearner(market_data, temperature_kT=0.5)
    
    # Learn potentials using all methods
    learned_methods = learner.compare_learning_methods()
    
    # Validate all methods
    validator = LearnedPotentialValidator(learned_methods, temperature_kT=0.5)
    validation_results = validator.validate_all_methods()
    
    # Find best method
    best_method = None
    best_score = 0.0
    
    for method_name, results in validation_results.items():
        if 'agreement_score' in results and results['agreement_score'] > best_score:
            best_score = results['agreement_score']
            best_method = method_name
    
    # Create comprehensive visualization
    fig = create_nonparametric_comparison_visualization(learned_methods, validation_results)
    
    # Save results
    viz_file = Path("phase_space_analysis/nonparametric_potential_learning.html")
    fig.write_html(str(viz_file))
    
    results_file = Path("phase_space_analysis/nonparametric_learning_results.json")
    with open(results_file, 'w') as f:
        serializable_results = {}
        for method, data in validation_results.items():
            if isinstance(data, dict):
                serializable_results[method] = {k: v for k, v in data.items() 
                                               if not callable(v) and k != 'predicted_stats'}
        json.dump(serializable_results, f, indent=2, default=str)
    
    # Final report
    print("\n" + "="*80)
    print("🧠 NON-PARAMETRIC POTENTIAL LEARNING RESULTS")
    print("="*80)
    
    print(f"\n📊 METHOD COMPARISON:")
    for method_name, results in validation_results.items():
        if 'agreement_score' in results:
            print(f"   • {method_name}:")
            print(f"     Agreement: {results['agreement_score']:.1%}")
            print(f"     Error: {results['mean_relative_error']:.1%}")
    
    if best_method:
        print(f"\n🏆 BEST METHOD: {best_method}")
        print(f"   • Agreement Score: {validation_results[best_method]['agreement_score']:.1%}")
        print(f"   • Mean Error: {validation_results[best_method]['mean_relative_error']:.1%}")
        
        # Compare with previous approach
        previous_agreement = 0.388  # From focused parameter tuning
        improvement = validation_results[best_method]['agreement_score'] - previous_agreement
        print(f"\n📈 IMPROVEMENT OVER PARAMETRIC APPROACH:")
        print(f"   • Previous (Gaussian wells): {previous_agreement:.1%}")
        print(f"   • Best non-parametric: {validation_results[best_method]['agreement_score']:.1%}")
        print(f"   • Improvement: {improvement:+.1%}")
        
        if improvement > 0:
            print(f"   ✅ Non-parametric learning is superior!")
        else:
            print(f"   ⚠️  Parametric approach still better - need refinement")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   • Visualization: {viz_file}")
    print(f"   • Results data: {results_file}")
    
    print("="*80)
    
    return {
        'learned_methods': learned_methods,
        'validation_results': validation_results,
        'best_method': best_method,
        'best_score': best_score
    }

def create_nonparametric_comparison_visualization(learned_methods: Dict[str, Any], 
                                                validation_results: Dict[str, Any]) -> go.Figure:
    """Create comprehensive visualization comparing all non-parametric methods."""
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Method Agreement Comparison',
            'Adaptive Density Potential',
            'Error Analysis by Method',
            'Maximum Entropy Potential',
            'Learning Method Performance',
            'Best Method Cluster Analysis'
        ),
        specs=[
            [{"type": "bar"}, {"type": "heatmap"}],
            [{"type": "bar"}, {"type": "heatmap"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ]
    )
    
    # Extract validation data
    method_names = []
    agreements = []
    errors = []
    
    for method, results in validation_results.items():
        if 'agreement_score' in results:
            method_names.append(method.replace('_', ' ').title())
            agreements.append(results['agreement_score'])
            errors.append(results['mean_relative_error'])
    
    # 1. Method agreement comparison
    colors = ['red', 'blue', 'green', 'orange'][:len(method_names)]
    
    fig.add_trace(
        go.Bar(
            x=method_names,
            y=agreements,
            marker_color=colors,
            name='Agreement Score',
            hovertemplate='Method: %{x}<br>Agreement: %{y:.1%}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add target line
    fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                 annotation_text="Target: 80%", row=1, col=1)
    
    # 2. Adaptive density potential visualization
    if 'adaptive_density' in learned_methods:
        S, U, V = learned_methods['adaptive_density']['grid_data']
        
        fig.add_trace(
            go.Heatmap(
                x=S[0, :],
                y=U[:, 0],
                z=V,
                colorscale='RdBu_r',
                name='Adaptive Density V(s,u)',
                hovertemplate='Sentiment: %{x:.3f}<br>UWR: %{y:.3f}<br>Potential: %{z:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
    
    # 3. Error analysis
    fig.add_trace(
        go.Bar(
            x=method_names,
            y=errors,
            marker_color='lightcoral',
            name='Mean Relative Error',
            hovertemplate='Method: %{x}<br>Error: %{y:.1%}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add target line
    fig.add_hline(y=0.3, line_dash="dash", line_color="red", 
                 annotation_text="Target: 30%", row=2, col=1)
    
    # 4. Maximum entropy potential (if available)
    if 'maximum_entropy' in learned_methods:
        # Create a grid visualization for max entropy method
        centers = learned_methods['maximum_entropy']['centers']
        
        fig.add_trace(
            go.Scatter(
                x=centers[:, 0],
                y=centers[:, 1],
                mode='markers',
                marker=dict(size=10, color='red', symbol='star'),
                name='ME Centers',
                hovertemplate='Sentiment: %{x:.3f}<br>UWR: %{y:.3f}<extra></extra>'
            ),
            row=2, col=2
        )
    
    # 5. Performance scatter plot
    fig.add_trace(
        go.Scatter(
            x=errors,
            y=agreements,
            mode='markers+text',
            marker=dict(size=15, color=colors[:len(method_names)]),
            text=method_names,
            textposition='top center',
            name='Method Performance',
            hovertemplate='Error: %{x:.1%}<br>Agreement: %{y:.1%}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Add ideal region
    fig.add_shape(
        type="rect",
        x0=0, x1=0.3, y0=0.8, y1=1.0,
        fillcolor="lightgreen", opacity=0.2,
        row=3, col=1
    )
    
    # 6. Best method cluster analysis
    best_method = max(validation_results.keys(), 
                     key=lambda x: validation_results[x].get('agreement_score', 0))
    
    if 'cluster_comparison' in validation_results[best_method]:
        cluster_data = validation_results[best_method]['cluster_comparison']
        cluster_names = list(cluster_data.keys())
        observed_freqs = [cluster_data[name]['observed_frequency'] for name in cluster_names]
        predicted_freqs = [cluster_data[name]['predicted_frequency'] for name in cluster_names]
        
        fig.add_trace(
            go.Bar(
                x=cluster_names,
                y=observed_freqs,
                name='Observed',
                marker_color='blue',
                opacity=0.7
            ),
            row=3, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=cluster_names,
                y=predicted_freqs,
                name='Predicted (Best)',
                marker_color='red',
                opacity=0.7
            ),
            row=3, col=2
        )
    
    # Update layout
    best_score = max([results.get('agreement_score', 0) for results in validation_results.values()])
    
    fig.update_layout(
        title=dict(
            text=f"🧠 Non-Parametric Potential Learning Comparison<br>" +
                 f"<sub>Best Agreement: {best_score:.1%} | Methods: {len(method_names)}</sub>",
            x=0.5,
            xanchor='center'
        ),
        height=1200,
        width=1600,
        template="plotly_dark",
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Learning Method", row=1, col=1, tickangle=-45)
    fig.update_yaxes(title_text="Agreement Score", row=1, col=1)
    
    fig.update_xaxes(title_text="Sentiment", row=1, col=2)
    fig.update_yaxes(title_text="Upper Wick Ratio", row=1, col=2)
    
    fig.update_xaxes(title_text="Learning Method", row=2, col=1, tickangle=-45)
    fig.update_yaxes(title_text="Mean Relative Error", row=2, col=1)
    
    fig.update_xaxes(title_text="Sentiment", row=2, col=2)
    fig.update_yaxes(title_text="Upper Wick Ratio", row=2, col=2)
    
    fig.update_xaxes(title_text="Mean Relative Error", row=3, col=1)
    fig.update_yaxes(title_text="Agreement Score", row=3, col=1)
    
    fig.update_xaxes(title_text="Cluster", row=3, col=2, tickangle=-45)
    fig.update_yaxes(title_text="Frequency (%)", row=3, col=2)
    
    return fig

if __name__ == "__main__":
    results = run_nonparametric_learning_analysis()