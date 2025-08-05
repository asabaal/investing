"""
Comprehensive Trajectory Comparator

A unified system for comparing particle trajectories in curved vs flat spacetime.
This integrates all 6 comparison methods into a single, coherent framework that
provides intuitive insights into the geometric advantages of curved space modeling.

The comparator treats market evolution as particles moving through spacetime,
comparing how they behave in:
- Flat Space: Traditional price-time coordinates
- Curved Space: Geometric pattern space with dynamic metrics
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.metrics import mutual_info_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from curved_candle_geometry import create_candle_metrics_from_ohlc, CurvedCandleGeometry
from market_hamiltonian import MarketHamiltonian, analyze_market_energy


@dataclass
class TrajectoryMetrics:
    """Comprehensive metrics for a single trajectory"""
    # Path Properties
    path_length: float
    curvature_mean: float
    curvature_std: float
    smoothness_index: float
    
    # Information Content
    entropy: float
    complexity_measure: float
    predictability_index: float
    
    # Energetic Properties
    total_energy: float
    energy_efficiency: float
    stability_measure: float


@dataclass
class ComparisonResults:
    """Comprehensive results from trajectory comparison"""
    # Individual trajectory metrics
    flat_space_metrics: TrajectoryMetrics
    curved_space_metrics: TrajectoryMetrics
    
    # Cross-trajectory comparisons
    reconstruction_fidelity: float
    information_advantage: float
    forecast_superiority: float
    geometric_efficiency: float
    energy_predictive_power: float
    dimensional_coherence: float
    
    # Overall assessment
    curved_space_advantage: float
    confidence_level: float
    superiority_category: str
    detailed_analysis: Dict[str, Any]


class ComprehensiveTrajectoryComparator:
    """
    Unified system for comparing trajectories in flat vs curved spacetime.
    
    This class implements a particle physics approach to market trajectory analysis,
    treating price evolution as particles moving through spacetime geometries.
    """
    
    def __init__(self, ohlc_data: pd.DataFrame, lookback_window: int = 50):
        """
        Initialize the comparator with market data.
        
        Args:
            ohlc_data: Market OHLC data
            lookback_window: Window size for trajectory analysis
        """
        self.ohlc_data = ohlc_data
        self.lookback_window = lookback_window
        
        # Initialize geometric infrastructure
        self.candle_metrics = create_candle_metrics_from_ohlc(ohlc_data)
        self.geometry = CurvedCandleGeometry(self.candle_metrics)
        self.hamiltonian = MarketHamiltonian()
        
        # Extract trajectories
        self.flat_trajectory = self._extract_flat_space_trajectory()
        self.curved_trajectory = self._extract_curved_space_trajectory()
        
        # Cached results
        self._comparison_cache = {}
    
    def _extract_flat_space_trajectory(self) -> np.ndarray:
        """
        Extract flat spacetime trajectory: (time, price)
        
        This represents the traditional view where particles move through
        flat Euclidean space-time coordinates.
        """
        times = np.arange(len(self.ohlc_data))
        prices = self.ohlc_data['close'].values
        
        # Normalize for comparison
        times_norm = (times - times.min()) / (times.max() - times.min())
        prices_norm = (prices - prices.min()) / (prices.max() - prices.min())
        
        return np.column_stack([times_norm, prices_norm])
    
    def _extract_curved_space_trajectory(self) -> np.ndarray:
        """
        Extract curved spacetime trajectory: (proper_time, sentiment, UWR)
        
        This represents the geometric view where particles move through
        curved pattern space with dynamic metrics.
        """
        proper_times = self.geometry.compute_proper_time_series()
        sentiments = np.array([c.sentiment for c in self.candle_metrics])
        uwrs = np.array([c.upper_wick_ratio for c in self.candle_metrics])
        
        # Normalize proper time for comparison
        tau_norm = (proper_times - proper_times.min()) / (proper_times.max() - proper_times.min())
        
        return np.column_stack([tau_norm, sentiments, uwrs])
    
    def analyze_single_trajectory(self, trajectory: np.ndarray, is_curved: bool = False) -> TrajectoryMetrics:
        """
        Analyze metrics for a single trajectory.
        
        Args:
            trajectory: Trajectory coordinates
            is_curved: Whether this is a curved space trajectory
            
        Returns:
            Comprehensive trajectory metrics
        """
        # 1. PATH PROPERTIES
        path_length = self._calculate_path_length(trajectory, is_curved)
        curvature_series = self._calculate_trajectory_curvature(trajectory)
        curvature_mean = np.mean(np.abs(curvature_series))
        curvature_std = np.std(curvature_series)
        smoothness_index = self._calculate_smoothness(trajectory)
        
        # 2. INFORMATION CONTENT
        entropy = self._calculate_trajectory_entropy(trajectory)
        complexity_measure = self._calculate_complexity(trajectory)
        predictability_index = self._calculate_predictability(trajectory)
        
        # 3. ENERGETIC PROPERTIES (for curved space)
        if is_curved and len(trajectory) == len(self.candle_metrics):
            energy_analysis = analyze_market_energy(self.ohlc_data, self.hamiltonian)
            total_energy = np.sum(energy_analysis['energies'])
            energy_efficiency = total_energy / path_length if path_length > 0 else 0
            stability_measure = 1.0 / (1.0 + np.std(energy_analysis['energies']))
        else:
            # Proxy energetics for flat space
            velocity_changes = np.diff(trajectory, axis=0)
            kinetic_proxy = np.sum(np.linalg.norm(velocity_changes, axis=1)**2)
            total_energy = kinetic_proxy
            energy_efficiency = kinetic_proxy / path_length if path_length > 0 else 0
            stability_measure = 1.0 / (1.0 + np.std(np.linalg.norm(velocity_changes, axis=1)))
        
        return TrajectoryMetrics(
            path_length=path_length,
            curvature_mean=curvature_mean,
            curvature_std=curvature_std,
            smoothness_index=smoothness_index,
            entropy=entropy,
            complexity_measure=complexity_measure,
            predictability_index=predictability_index,
            total_energy=total_energy,
            energy_efficiency=energy_efficiency,
            stability_measure=stability_measure
        )
    
    def method_1_reconstruction_fidelity(self) -> Dict[str, float]:
        """
        METHOD 1: Measure how well curved space reconstructs flat space dynamics.
        
        This tests whether the geometric representation captures the essential
        dynamics of price evolution.
        """
        # Reconstruct flat space prices from curved space coordinates
        reconstructed_prices = []
        actual_prices = self.ohlc_data['close'].values
        
        for i, candle in enumerate(self.candle_metrics):
            # Use geometric coordinates to reconstruct price
            sentiment = candle.sentiment
            uwr = candle.upper_wick_ratio
            range_val = candle.range_value
            low = candle.low_value
            
            # Reconstruct close price from phase coordinates
            high = low + range_val
            max_oc = high - uwr * range_val
            
            if sentiment >= 0:
                # Bullish candle
                open_price = max_oc - sentiment * range_val
                close_price = max_oc
            else:
                # Bearish candle  
                open_price = max_oc
                close_price = max_oc + sentiment * range_val
            
            reconstructed_prices.append(close_price)
        
        # Calculate fidelity metrics
        r2 = r2_score(actual_prices, reconstructed_prices)
        correlation = np.corrcoef(actual_prices, reconstructed_prices)[0, 1]
        rmse = np.sqrt(np.mean((actual_prices - np.array(reconstructed_prices))**2))
        
        # Normalized reconstruction error
        price_range = actual_prices.max() - actual_prices.min()
        normalized_rmse = rmse / price_range if price_range > 0 else 0
        
        fidelity_score = r2 * (1 - normalized_rmse)
        
        return {
            'reconstruction_r2': r2,
            'reconstruction_correlation': correlation,
            'normalized_rmse': normalized_rmse,
            'fidelity_score': fidelity_score
        }
    
    def method_2_information_advantage(self) -> Dict[str, float]:
        """
        METHOD 2: Compare information content between trajectory spaces.
        
        Curved space should contain more information if it captures
        additional market structure.
        """
        # Flat space information (1D price evolution)
        price_changes = np.diff(self.ohlc_data['close'].values)
        flat_entropy = self._calculate_entropy_1d(price_changes, bins=20)
        
        # Curved space information (2D pattern evolution)
        sentiments = np.array([c.sentiment for c in self.candle_metrics])
        uwrs = np.array([c.upper_wick_ratio for c in self.candle_metrics])
        curved_entropy = self._calculate_entropy_2d(sentiments, uwrs, bins=15)
        
        # Mutual information between spaces
        price_changes_binned = np.digitize(price_changes, 
                                         np.histogram_bin_edges(price_changes, bins=10))
        sentiments_binned = np.digitize(sentiments[1:], 
                                      np.histogram_bin_edges(sentiments, bins=10))
        
        mutual_info = mutual_info_score(price_changes_binned, sentiments_binned)
        
        # Information advantage metrics
        entropy_ratio = curved_entropy / flat_entropy if flat_entropy > 0 else 0
        information_gain = curved_entropy - flat_entropy
        redundancy = 1 - (mutual_info / min(flat_entropy, curved_entropy) if min(flat_entropy, curved_entropy) > 0 else 0)
        
        return {
            'flat_entropy': flat_entropy,
            'curved_entropy': curved_entropy,
            'entropy_ratio': entropy_ratio,
            'mutual_information': mutual_info,
            'information_gain': information_gain,
            'redundancy': redundancy,
            'advantage_score': entropy_ratio * (1 - redundancy)
        }
    
    def method_3_forecast_superiority(self, horizon: int = 5) -> Dict[str, float]:
        """
        METHOD 3: Compare forecasting performance between spaces.
        
        If curved space captures better dynamics, it should forecast better.
        """
        # Split data for out-of-sample testing
        split_point = int(len(self.ohlc_data) * 0.8)
        
        # Flat space forecasting (linear extrapolation)
        flat_forecast_errors = []
        curved_forecast_errors = []
        
        for i in range(split_point, len(self.ohlc_data) - horizon):
            # Actual future prices
            actual_future = self.ohlc_data['close'].iloc[i:i+horizon].values
            
            # Flat space forecast (simple trend)
            recent_prices = self.ohlc_data['close'].iloc[i-5:i].values
            if len(recent_prices) > 1:
                trend = np.mean(np.diff(recent_prices))
                flat_forecast = recent_prices[-1] + trend * np.arange(1, horizon+1)
            else:
                flat_forecast = np.full(horizon, recent_prices[-1])
            
            # Curved space forecast (geodesic-based)
            if i < len(self.candle_metrics):
                try:
                    curved_forecast = self._forecast_from_curved_space(i, horizon)
                except:
                    curved_forecast = flat_forecast  # Fallback
            else:
                curved_forecast = flat_forecast
            
            # Calculate errors
            flat_error = np.mean(np.abs(actual_future - flat_forecast))
            curved_error = np.mean(np.abs(actual_future - curved_forecast))
            
            flat_forecast_errors.append(flat_error)
            curved_forecast_errors.append(curved_error)
        
        if len(flat_forecast_errors) > 0:
            avg_flat_error = np.mean(flat_forecast_errors)
            avg_curved_error = np.mean(curved_forecast_errors)
            
            # Forecast improvement metrics
            error_reduction = (avg_flat_error - avg_curved_error) / avg_flat_error if avg_flat_error > 0 else 0
            forecast_accuracy_flat = 1 / (1 + avg_flat_error)
            forecast_accuracy_curved = 1 / (1 + avg_curved_error)
            
            superiority_score = forecast_accuracy_curved - forecast_accuracy_flat
        else:
            avg_flat_error = 0
            avg_curved_error = 0
            error_reduction = 0
            forecast_accuracy_flat = 0
            forecast_accuracy_curved = 0
            superiority_score = 0
        
        return {
            'flat_forecast_error': avg_flat_error,
            'curved_forecast_error': avg_curved_error,
            'error_reduction': error_reduction,
            'forecast_accuracy_flat': forecast_accuracy_flat,
            'forecast_accuracy_curved': forecast_accuracy_curved,
            'superiority_score': superiority_score
        }
    
    def method_4_geometric_efficiency(self) -> Dict[str, float]:
        """
        METHOD 4: Compare geometric properties of trajectory paths.
        
        Curved space should show more efficient paths that respect
        the natural geometry of market evolution.
        """
        flat_metrics = self.analyze_single_trajectory(self.flat_trajectory, is_curved=False)
        curved_metrics = self.analyze_single_trajectory(self.curved_trajectory, is_curved=True)
        
        # Path efficiency comparisons
        path_length_ratio = curved_metrics.path_length / flat_metrics.path_length if flat_metrics.path_length > 0 else 1
        curvature_ratio = curved_metrics.curvature_mean / flat_metrics.curvature_mean if flat_metrics.curvature_mean > 0 else 1
        smoothness_ratio = curved_metrics.smoothness_index / flat_metrics.smoothness_index if flat_metrics.smoothness_index > 0 else 1
        
        # Energy efficiency (curved space should be more efficient)
        energy_efficiency_ratio = curved_metrics.energy_efficiency / flat_metrics.energy_efficiency if flat_metrics.energy_efficiency > 0 else 1
        
        # Stability comparison
        stability_advantage = curved_metrics.stability_measure - flat_metrics.stability_measure
        
        # Overall geometric efficiency score
        efficiency_score = (
            (1 / path_length_ratio) * 0.3 +  # Shorter paths are better
            smoothness_ratio * 0.3 +         # Smoother paths are better
            energy_efficiency_ratio * 0.2 +   # More energy efficient is better
            (1 + stability_advantage) * 0.2   # More stable is better
        )
        
        return {
            'path_length_ratio': path_length_ratio,
            'curvature_ratio': curvature_ratio,
            'smoothness_ratio': smoothness_ratio,
            'energy_efficiency_ratio': energy_efficiency_ratio,
            'stability_advantage': stability_advantage,
            'efficiency_score': efficiency_score
        }
    
    def method_5_energy_predictive_power(self) -> Dict[str, float]:
        """
        METHOD 5: Assess energy states' ability to predict price movements.
        
        In curved space, energy should be a leading indicator of price changes.
        """
        try:
            energy_analysis = analyze_market_energy(self.ohlc_data, self.hamiltonian)
            energies = np.array(energy_analysis['energies'])
            price_changes = np.diff(self.ohlc_data['close'].values)
            
            # Align arrays
            min_len = min(len(energies), len(price_changes))
            energies = energies[:min_len]
            price_changes = price_changes[:min_len]
            
            if len(energies) > 1:
                # Contemporaneous correlation
                contemp_correlation = np.corrcoef(energies, price_changes)[0, 1] if not np.isnan(np.corrcoef(energies, price_changes)[0, 1]) else 0
                
                # Predictive correlation (energy leads price)
                if len(energies) > 2:
                    predictive_correlation = np.corrcoef(energies[:-1], price_changes[1:])[0, 1] if not np.isnan(np.corrcoef(energies[:-1], price_changes[1:])[0, 1]) else 0
                else:
                    predictive_correlation = 0
                
                # Energy regime detection
                energy_volatility = np.std(energies)
                price_volatility = np.std(price_changes)
                volatility_correlation = energy_volatility / price_volatility if price_volatility > 0 else 0
                
                # Predictive power score
                predictive_power = (abs(contemp_correlation) + 2 * abs(predictive_correlation)) / 3
            else:
                contemp_correlation = 0
                predictive_correlation = 0
                volatility_correlation = 0
                predictive_power = 0
                
        except Exception as e:
            print(f"Warning: Energy analysis failed: {e}")
            contemp_correlation = 0
            predictive_correlation = 0
            volatility_correlation = 0
            predictive_power = 0
        
        return {
            'energy_price_correlation': contemp_correlation,
            'predictive_correlation': predictive_correlation,
            'volatility_correlation': volatility_correlation,
            'predictive_power': predictive_power
        }
    
    def method_6_dimensional_coherence(self) -> Dict[str, float]:
        """
        METHOD 6: Assess coherence between trajectory embeddings.
        
        Both spaces should contain similar underlying structure,
        but curved space should organize it more coherently.
        """
        # Prepare trajectory data for PCA
        flat_data = self.flat_trajectory
        curved_data_2d = self.curved_trajectory[:, 1:]  # Remove time dimension for fair comparison
        
        # Standardize data
        scaler = StandardScaler()
        flat_data_scaled = scaler.fit_transform(flat_data)
        curved_data_scaled = scaler.fit_transform(curved_data_2d)
        
        # PCA analysis
        pca_flat = PCA(n_components=2)
        pca_curved = PCA(n_components=2)
        
        flat_pca = pca_flat.fit_transform(flat_data_scaled)
        curved_pca = pca_curved.fit_transform(curved_data_scaled)
        
        # Compare principal components
        flat_variance_explained = np.sum(pca_flat.explained_variance_ratio_)
        curved_variance_explained = np.sum(pca_curved.explained_variance_ratio_)
        
        # Alignment between spaces (correlation of first principal components)
        if flat_pca.shape[0] == curved_pca.shape[0]:
            pc1_correlation = np.corrcoef(flat_pca[:, 0], curved_pca[:, 0])[0, 1] if not np.isnan(np.corrcoef(flat_pca[:, 0], curved_pca[:, 0])[0, 1]) else 0
        else:
            pc1_correlation = 0
        
        # Dimensional coherence score
        coherence_score = (curved_variance_explained * abs(pc1_correlation)) / flat_variance_explained if flat_variance_explained > 0 else 0
        
        return {
            'flat_variance_explained': flat_variance_explained,
            'curved_variance_explained': curved_variance_explained,
            'pc1_correlation': pc1_correlation,
            'coherence_score': coherence_score,
            'dimensional_advantage': curved_variance_explained - flat_variance_explained
        }
    
    def comprehensive_comparison(self) -> ComparisonResults:
        """
        Run all comparison methods and synthesize results.
        
        Returns:
            Comprehensive comparison results with overall assessment
        """
        print("🌌 Running Comprehensive Trajectory Comparison...")
        print("=" * 60)
        
        # Run all methods
        method1 = self.method_1_reconstruction_fidelity()
        method2 = self.method_2_information_advantage()
        method3 = self.method_3_forecast_superiority()
        method4 = self.method_4_geometric_efficiency()
        method5 = self.method_5_energy_predictive_power()
        method6 = self.method_6_dimensional_coherence()
        
        print(f"✅ Method 1 - Reconstruction Fidelity: {method1['fidelity_score']:.3f}")
        print(f"✅ Method 2 - Information Advantage: {method2['advantage_score']:.3f}")
        print(f"✅ Method 3 - Forecast Superiority: {method3['superiority_score']:.3f}")
        print(f"✅ Method 4 - Geometric Efficiency: {method4['efficiency_score']:.3f}")
        print(f"✅ Method 5 - Energy Predictive Power: {method5['predictive_power']:.3f}")
        print(f"✅ Method 6 - Dimensional Coherence: {method6['coherence_score']:.3f}")
        
        # Calculate individual trajectory metrics
        flat_metrics = self.analyze_single_trajectory(self.flat_trajectory, is_curved=False)
        curved_metrics = self.analyze_single_trajectory(self.curved_trajectory, is_curved=True)
        
        # Synthesize overall curved space advantage
        advantage_components = [
            method1['fidelity_score'],
            method2['advantage_score'],
            method3['superiority_score'],
            method4['efficiency_score'],
            method5['predictive_power'],
            method6['coherence_score']
        ]
        
        # Remove any NaN values
        valid_components = [x for x in advantage_components if not np.isnan(x)]
        
        if valid_components:
            curved_space_advantage = np.mean(valid_components)
            confidence_level = len(valid_components) / len(advantage_components)
        else:
            curved_space_advantage = 0
            confidence_level = 0
        
        # Categorize superiority
        if curved_space_advantage > 0.7:
            superiority_category = "Strong Curved Space Advantage"
        elif curved_space_advantage > 0.4:
            superiority_category = "Moderate Curved Space Advantage"
        elif curved_space_advantage > 0.1:
            superiority_category = "Weak Curved Space Advantage"
        elif curved_space_advantage > -0.1:
            superiority_category = "Equivalent Performance"
        else:
            superiority_category = "Flat Space Advantage"
        
        # Detailed analysis
        detailed_analysis = {
            'method_results': {
                'reconstruction_fidelity': method1,
                'information_advantage': method2,
                'forecast_superiority': method3,
                'geometric_efficiency': method4,
                'energy_predictive_power': method5,
                'dimensional_coherence': method6
            },
            'trajectory_properties': {
                'flat_space': flat_metrics,
                'curved_space': curved_metrics
            },
            'advantage_breakdown': {
                'reconstruction': method1['fidelity_score'],
                'information': method2['advantage_score'],
                'forecasting': method3['superiority_score'],
                'geometry': method4['efficiency_score'],
                'energy': method5['predictive_power'],
                'coherence': method6['coherence_score']
            }
        }
        
        return ComparisonResults(
            flat_space_metrics=flat_metrics,
            curved_space_metrics=curved_metrics,
            reconstruction_fidelity=method1['fidelity_score'],
            information_advantage=method2['advantage_score'],
            forecast_superiority=method3['superiority_score'],
            geometric_efficiency=method4['efficiency_score'],
            energy_predictive_power=method5['predictive_power'],
            dimensional_coherence=method6['coherence_score'],
            curved_space_advantage=curved_space_advantage,
            confidence_level=confidence_level,
            superiority_category=superiority_category,
            detailed_analysis=detailed_analysis
        )
    
    # Helper methods for calculations
    def _calculate_path_length(self, trajectory: np.ndarray, is_curved: bool = False) -> float:
        """Calculate path length accounting for metric tensor in curved space."""
        if len(trajectory) < 2:
            return 0.0
        
        total_length = 0.0
        
        for i in range(len(trajectory) - 1):
            if is_curved and i < len(self.candle_metrics):
                # Use metric tensor for curved space distance
                metric = self.candle_metrics[i].metric_tensor
                dp = trajectory[i+1, 1:] - trajectory[i, 1:]  # Spatial components only
                ds_squared = np.dot(dp, np.dot(metric, dp))
                ds = np.sqrt(max(0, ds_squared))
            else:
                # Euclidean distance for flat space
                dp = trajectory[i+1] - trajectory[i]
                ds = np.linalg.norm(dp)
            
            total_length += ds
        
        return total_length
    
    def _calculate_trajectory_curvature(self, trajectory: np.ndarray) -> np.ndarray:
        """Calculate curvature along trajectory path."""
        if len(trajectory) < 3:
            return np.zeros(len(trajectory))
        
        curvatures = np.zeros(len(trajectory))
        
        for i in range(1, len(trajectory) - 1):
            # Three consecutive points
            p1, p2, p3 = trajectory[i-1], trajectory[i], trajectory[i+1]
            
            # Calculate curvature using discrete approximation
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Avoid division by zero
            if np.linalg.norm(v1) > 1e-10 and np.linalg.norm(v2) > 1e-10:
                # Discrete curvature formula
                cross_product = np.cross(v1, v2) if v1.shape[0] == 2 else np.linalg.norm(np.cross(v1, v2))
                curvature = cross_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
                curvatures[i] = curvature
        
        return curvatures
    
    def _calculate_smoothness(self, trajectory: np.ndarray) -> float:
        """Calculate trajectory smoothness index."""
        if len(trajectory) < 3:
            return 1.0
        
        # Calculate second derivatives (acceleration)
        velocities = np.diff(trajectory, axis=0)
        accelerations = np.diff(velocities, axis=0)
        
        # Smoothness is inverse of acceleration magnitude
        accel_magnitudes = np.linalg.norm(accelerations, axis=1)
        smoothness = 1.0 / (1.0 + np.mean(accel_magnitudes))
        
        return smoothness
    
    def _calculate_trajectory_entropy(self, trajectory: np.ndarray) -> float:
        """Calculate Shannon entropy of trajectory."""
        # Discretize trajectory into bins
        if trajectory.shape[1] == 2:
            hist, _, _ = np.histogram2d(trajectory[:, 0], trajectory[:, 1], bins=10)
        else:
            # For higher dimensions, use 1D projection
            projected = np.linalg.norm(trajectory, axis=1)
            hist, _ = np.histogram(projected, bins=20)
        
        # Normalize and calculate entropy
        hist = hist.flatten()
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return entropy
    
    def _calculate_complexity(self, trajectory: np.ndarray) -> float:
        """Calculate trajectory complexity measure."""
        # Complexity based on path length vs direct distance
        if len(trajectory) < 2:
            return 0.0
        
        path_length = self._calculate_path_length(trajectory)
        direct_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
        
        complexity = path_length / direct_distance if direct_distance > 0 else 1.0
        return complexity
    
    def _calculate_predictability(self, trajectory: np.ndarray) -> float:
        """Calculate trajectory predictability index."""
        if len(trajectory) < 4:
            return 0.0
        
        # Use autocorrelation as predictability measure
        velocities = np.diff(trajectory, axis=0)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        # Calculate autocorrelation
        if len(velocity_magnitudes) > 1:
            autocorr = np.corrcoef(velocity_magnitudes[:-1], velocity_magnitudes[1:])[0, 1]
            predictability = abs(autocorr) if not np.isnan(autocorr) else 0
        else:
            predictability = 0
        
        return predictability
    
    def _calculate_entropy_1d(self, data: np.ndarray, bins: int = 20) -> float:
        """Calculate 1D Shannon entropy."""
        hist, _ = np.histogram(data, bins=bins)
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    def _calculate_entropy_2d(self, x_data: np.ndarray, y_data: np.ndarray, bins: int = 10) -> float:
        """Calculate 2D Shannon entropy."""
        hist, _, _ = np.histogram2d(x_data, y_data, bins=bins)
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    def _forecast_from_curved_space(self, start_index: int, horizon: int) -> np.ndarray:
        """Generate forecast using curved space geodesics."""
        if start_index >= len(self.candle_metrics) - 1:
            return np.array([self.ohlc_data['close'].iloc[-1]] * horizon)
        
        try:
            # Use geodesic prediction from geometry
            initial_velocity = np.array([0.1, 0.0])  # Small initial velocity
            geodesic_path = self.geometry.predict_geodesic_path(
                start_index, initial_velocity, horizon, use_historical=False
            )
            
            # Convert phase coordinates back to prices
            forecasted_prices = []
            for coords in geodesic_path:
                if len(coords) >= 2:
                    sentiment, uwr = coords[0], coords[1]
                    
                    # Use recent candle properties for reconstruction
                    recent_candle = self.candle_metrics[min(start_index, len(self.candle_metrics)-1)]
                    range_val = recent_candle.range_value
                    low = recent_candle.low_value
                    
                    # Reconstruct price
                    high = low + range_val
                    max_oc = high - uwr * range_val
                    
                    if sentiment >= 0:
                        close_price = max_oc
                    else:
                        close_price = max_oc + sentiment * range_val
                    
                    forecasted_prices.append(close_price)
            
            if len(forecasted_prices) == 0:
                return np.array([self.ohlc_data['close'].iloc[-1]] * horizon)
            
            return np.array(forecasted_prices[:horizon])
            
        except Exception as e:
            print(f"Geodesic forecast failed: {e}")
            # Fallback to last price
            return np.array([self.ohlc_data['close'].iloc[-1]] * horizon)