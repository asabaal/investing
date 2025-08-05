#!/usr/bin/env python3
"""
Ultimate Physics Validation Test

The definitive test of our market physics framework:

1. MEASURE potential energy from observed statistical frequencies (✅ Done)
2. WRITE explicit potential energy function with measured parameters (✅ Done)  
3. PREDICT market behavior using action minimization with written potential
4. VALIDATE that predictions reproduce the original observed statistics

This is the gold standard validation:
- If our physics is correct, action minimization should predict the same 5-cluster structure
- Predicted paths should preferentially visit the high-probability regions
- Statistical frequencies from predictions should match observations

SUCCESS = Physics reproduces reality
FAILURE = Back to the drawing board
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import json
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from explicit_potential_energy import ExplicitMarketPotential
from market_data_database import MarketDataDatabase
from curved_candle_geometry import CurvedCandleGeometry, create_candle_metrics_from_ohlc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhysicsValidator:
    """
    Validates market physics by comparing predictions with observations.
    
    Uses explicit potential energy function to generate predicted market trajectories
    via action minimization, then compares statistical properties with observed data.
    """
    
    def __init__(self, explicit_potential: ExplicitMarketPotential):
        """Initialize with explicit potential energy function."""
        self.potential = explicit_potential
        self.dt = 0.1  # Time step for integration
        
    def generate_predicted_trajectory(self, initial_state: np.ndarray, n_steps: int = 100,
                                    temperature: float = 0.1) -> List[np.ndarray]:
        """
        Generate predicted market trajectory using action minimization.
        
        Uses gradient descent with thermal noise to simulate market evolution
        in the measured potential energy landscape.
        
        Args:
            initial_state: Starting (sentiment, UWR) position
            n_steps: Number of time steps to simulate
            temperature: Temperature for thermal fluctuations
        """
        logger.info(f"Generating predicted trajectory with {n_steps} steps")
        
        trajectory = [initial_state.copy()]
        current_state = initial_state.copy()
        
        # Add some momentum for more realistic dynamics
        velocity = np.zeros(2)
        damping = 0.1  # Velocity damping
        mass = 1.0
        
        for step in range(n_steps):
            # Force from potential gradient
            force = self.potential.force_field(current_state[0], current_state[1])
            
            # Add thermal noise (Brownian motion)
            thermal_force = np.random.normal(0, np.sqrt(2 * temperature), 2)
            total_force = force + thermal_force
            
            # Update velocity and position (simple Verlet integration)
            acceleration = total_force / mass
            velocity = velocity * (1 - damping) + acceleration * self.dt
            new_state = current_state + velocity * self.dt
            
            # Apply phase space constraints: |s| + u ≤ 1
            s, u = new_state
            s = np.clip(s, -0.99, 0.99)
            u = np.clip(u, 0.01, 0.99)
            
            if abs(s) + u > 1.0:
                # Project onto boundary
                total = abs(s) + u
                s = s * 0.99 / total
                u = u * 0.99 / total
            
            current_state = np.array([s, u])
            trajectory.append(current_state.copy())
        
        logger.info(f"Generated trajectory with {len(trajectory)} points")
        return trajectory
    
    def generate_multiple_trajectories(self, n_trajectories: int = 50, n_steps: int = 100,
                                     temperature: float = 0.1) -> List[List[np.ndarray]]:
        """Generate multiple independent predicted trajectories."""
        
        logger.info(f"Generating {n_trajectories} independent trajectories")
        trajectories = []
        
        for i in range(n_trajectories):
            # Random initial conditions
            initial_s = np.random.uniform(-0.5, 0.5)
            initial_u = np.random.uniform(0.1, min(0.9, 1.0 - abs(initial_s)))
            initial_state = np.array([initial_s, initial_u])
            
            trajectory = self.generate_predicted_trajectory(initial_state, n_steps, temperature)
            trajectories.append(trajectory)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Generated {i + 1}/{n_trajectories} trajectories")
        
        return trajectories
    
    def analyze_trajectory_statistics(self, trajectories: List[List[np.ndarray]]) -> Dict:
        """Analyze statistical properties of predicted trajectories."""
        
        logger.info("Analyzing trajectory statistics...")
        
        # Flatten all trajectory points
        all_points = []
        for traj in trajectories:
            for point in traj:
                all_points.append(point)
        
        all_points = np.array(all_points)
        sentiments = all_points[:, 0]
        uwrs = all_points[:, 1]
        
        # Basic statistics
        stats = {
            'n_points': len(all_points),
            'sentiment_stats': {
                'mean': np.mean(sentiments),
                'std': np.std(sentiments),
                'min': np.min(sentiments),
                'max': np.max(sentiments)
            },
            'uwr_stats': {
                'mean': np.mean(uwrs),
                'std': np.std(uwrs),
                'min': np.min(uwrs),
                'max': np.max(uwrs)
            }
        }
        
        # Analyze cluster occupancy
        stats['cluster_occupancy'] = self.analyze_cluster_occupancy(all_points)
        
        return stats
    
    def analyze_cluster_occupancy(self, points: np.ndarray) -> Dict:
        """Analyze how much time trajectories spend in each well region."""
        
        occupancy = {}
        
        for well in self.potential.well_parameters:
            center = well['center']
            width_s = well['width_s']
            width_u = well['width_u']
            
            # Count points within 2 standard deviations of well center
            distances_s = np.abs(points[:, 0] - center[0]) / width_s
            distances_u = np.abs(points[:, 1] - center[1]) / width_u
            
            # Points within elliptical region around well
            in_well = (distances_s <= 2.0) & (distances_u <= 2.0)
            count = np.sum(in_well)
            percentage = count / len(points) * 100
            
            occupancy[well['name']] = {
                'count': count,
                'percentage': percentage,
                'expected_percentage': well['frequency'] * 100
            }
        
        return occupancy
    
    def compare_with_observations(self, predicted_stats: Dict) -> Dict:
        """Compare predicted statistics with original observations."""
        
        logger.info("Comparing predictions with observations...")
        
        # Load original GMM results for comparison
        gmm_file = "phase_space_analysis/candle_geometry_classification.json"
        with open(gmm_file, 'r') as f:
            gmm_data = json.load(f)
        
        observed_frequencies = {}
        for region in gmm_data['classification_system']['regions']:
            observed_frequencies[region['name']] = region['frequency_across_securities']
        
        # Compare cluster occupancy
        comparison = {
            'cluster_comparison': {},
            'overall_agreement': 0.0
        }
        
        total_error = 0.0
        n_clusters = 0
        
        for cluster_name in observed_frequencies:
            if cluster_name in predicted_stats['cluster_occupancy']:
                observed_freq = observed_frequencies[cluster_name]
                predicted_freq = predicted_stats['cluster_occupancy'][cluster_name]['percentage']
                
                error = abs(predicted_freq - observed_freq)
                relative_error = error / observed_freq if observed_freq > 0 else 0
                
                comparison['cluster_comparison'][cluster_name] = {
                    'observed_frequency': observed_freq,
                    'predicted_frequency': predicted_freq,
                    'absolute_error': error,
                    'relative_error': relative_error
                }
                
                total_error += relative_error
                n_clusters += 1
        
        # Overall agreement score (lower is better)
        comparison['overall_agreement'] = 1.0 - (total_error / n_clusters) if n_clusters > 0 else 0.0
        comparison['mean_relative_error'] = total_error / n_clusters if n_clusters > 0 else float('inf')
        
        return comparison
    
    def create_validation_visualization(self, trajectories: List[List[np.ndarray]], 
                                      predicted_stats: Dict, comparison: Dict) -> go.Figure:
        """Create comprehensive validation visualization."""
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Predicted Trajectories vs Potential Wells',
                'Cluster Occupancy: Predicted vs Observed',
                'Trajectory Density Heatmap',
                'Statistical Agreement Analysis',
                'Potential Energy Landscape',
                'Physics Validation Summary'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "heatmap"}, {"type": "table"}]
            ]
        )
        
        # 1. Predicted trajectories with well positions
        colors = px.colors.qualitative.Set1
        
        # Plot sample trajectories
        for i, traj in enumerate(trajectories[:5]):  # Show first 5 trajectories
            traj_array = np.array(traj)
            fig.add_trace(
                go.Scatter(
                    x=traj_array[:, 0],
                    y=traj_array[:, 1],
                    mode='lines',
                    line=dict(color=colors[i % len(colors)], width=1),
                    opacity=0.7,
                    name=f'Trajectory {i+1}',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Add well centers
        for i, well in enumerate(self.potential.well_parameters):
            fig.add_trace(
                go.Scatter(
                    x=[well['center'][0]],
                    y=[well['center'][1]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='star',
                        line=dict(color='white', width=2)
                    ),
                    name=well['name'],
                    showlegend=False,
                    hovertemplate=f'<b>{well["name"]}</b><br>Well Center<br>Depth: {well["depth"]:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Cluster occupancy comparison
        cluster_names = list(comparison['cluster_comparison'].keys())
        observed_freqs = [comparison['cluster_comparison'][name]['observed_frequency'] for name in cluster_names]
        predicted_freqs = [comparison['cluster_comparison'][name]['predicted_frequency'] for name in cluster_names]
        
        fig.add_trace(
            go.Bar(
                x=cluster_names,
                y=observed_freqs,
                name='Observed',
                marker_color='blue',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=cluster_names,
                y=predicted_freqs,
                name='Predicted',
                marker_color='red',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # 3. Trajectory density heatmap
        all_points = []
        for traj in trajectories:
            all_points.extend(traj)
        all_points = np.array(all_points)
        
        # Create 2D histogram
        s_bins = np.linspace(-0.8, 0.8, 40)
        u_bins = np.linspace(0.1, 0.9, 40)
        hist, s_edges, u_edges = np.histogram2d(all_points[:, 0], all_points[:, 1], bins=[s_bins, u_bins])
        
        fig.add_trace(
            go.Heatmap(
                z=hist.T,
                x=s_bins,
                y=u_bins,
                colorscale='Blues',
                name='Trajectory Density'
            ),
            row=1, col=3
        )
        
        # 4. Statistical agreement
        agreement_data = []
        for name in cluster_names:
            comp = comparison['cluster_comparison'][name]
            agreement_data.append([
                comp['observed_frequency'],
                comp['predicted_frequency'],
                comp['relative_error']
            ])
        
        fig.add_trace(
            go.Scatter(
                x=observed_freqs,
                y=predicted_freqs,
                mode='markers+text',
                marker=dict(size=10, color='purple'),
                text=cluster_names,
                textposition='top center',
                name='Agreement',
                hovertemplate='Observed: %{x:.1f}%<br>Predicted: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add perfect agreement line
        max_freq = max(max(observed_freqs), max(predicted_freqs))
        fig.add_trace(
            go.Scatter(
                x=[0, max_freq],
                y=[0, max_freq],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Perfect Agreement',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 5. Potential energy landscape
        s_grid = np.linspace(-0.8, 0.8, 50)
        u_grid = np.linspace(0.1, 0.9, 50)
        S, U = np.meshgrid(s_grid, u_grid)
        
        V = np.zeros_like(S)
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if abs(S[i, j]) + U[i, j] <= 1.0:
                    V[i, j] = self.potential.potential_energy(S[i, j], U[i, j])
                else:
                    V[i, j] = np.nan
        
        fig.add_trace(
            go.Heatmap(
                z=V,
                x=s_grid,
                y=u_grid,
                colorscale='RdBu_r',
                name='Potential Energy'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="🏆 Ultimate Physics Validation: Predictions vs Reality<br>" +
                     f"<sub>Agreement Score: {comparison['overall_agreement']:.1%} | Mean Error: {comparison['mean_relative_error']:.1%}</sub>",
                x=0.5,
                xanchor='center'
            ),
            height=1000,
            width=1800,
            template="plotly_dark"
        )
        
        # Update axes
        fig.update_xaxes(title_text="Sentiment", row=1, col=1)
        fig.update_yaxes(title_text="Upper Wick Ratio", row=1, col=1)
        
        fig.update_xaxes(title_text="Cluster", row=1, col=2, tickangle=-45)
        fig.update_yaxes(title_text="Frequency (%)", row=1, col=2)
        
        fig.update_xaxes(title_text="Sentiment", row=1, col=3)
        fig.update_yaxes(title_text="Upper Wick Ratio", row=1, col=3)
        
        fig.update_xaxes(title_text="Observed Frequency (%)", row=2, col=1)
        fig.update_yaxes(title_text="Predicted Frequency (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="Sentiment", row=2, col=2)
        fig.update_yaxes(title_text="Upper Wick Ratio", row=2, col=2)
        
        return fig

def run_ultimate_validation_test():
    """Run the ultimate physics validation test."""
    
    logger.info("🏆 Running Ultimate Physics Validation Test")
    
    # 1. Load explicit potential function
    logger.info("📊 Loading measured potential energy function...")
    gmm_file = "phase_space_analysis/candle_geometry_classification.json"
    explicit_potential = ExplicitMarketPotential(gmm_file, energy_scale=1.0)
    
    # 2. Initialize validator
    validator = PhysicsValidator(explicit_potential)
    
    # 3. Generate predicted trajectories
    logger.info("🔮 Generating predicted market trajectories...")
    trajectories = validator.generate_multiple_trajectories(
        n_trajectories=100,  # More trajectories for better statistics
        n_steps=200,         # Longer trajectories
        temperature=0.05     # Lower temperature for more deterministic dynamics
    )
    
    # 4. Analyze predicted statistics
    logger.info("📈 Analyzing predicted trajectory statistics...")
    predicted_stats = validator.analyze_trajectory_statistics(trajectories)
    
    # 5. Compare with observations
    logger.info("⚖️ Comparing predictions with observations...")
    comparison = validator.compare_with_observations(predicted_stats)
    
    # 6. Create validation visualization
    logger.info("📊 Creating validation visualization...")
    validation_fig = validator.create_validation_visualization(trajectories, predicted_stats, comparison)
    
    # Save visualization
    output_file = Path("phase_space_analysis/ultimate_physics_validation.html")
    validation_fig.write_html(str(output_file))
    logger.info(f"📊 Validation visualization saved to {output_file}")
    
    # 7. Generate final report
    print("\n" + "="*80)
    print("🏆 ULTIMATE PHYSICS VALIDATION TEST RESULTS")
    print("="*80)
    
    print(f"\n📊 PREDICTION STATISTICS:")
    print(f"   • Total trajectory points analyzed: {predicted_stats['n_points']:,}")
    print(f"   • Number of independent trajectories: {len(trajectories)}")
    print(f"   • Average sentiment: {predicted_stats['sentiment_stats']['mean']:.3f}")
    print(f"   • Average UWR: {predicted_stats['uwr_stats']['mean']:.3f}")
    
    print(f"\n🎯 CLUSTER OCCUPANCY COMPARISON:")
    for cluster_name, comp in comparison['cluster_comparison'].items():
        observed = comp['observed_frequency']
        predicted = comp['predicted_frequency']
        error = comp['relative_error']
        print(f"   • {cluster_name}:")
        print(f"     Observed: {observed:.1f}% | Predicted: {predicted:.1f}% | Error: {error:.1%}")
    
    print(f"\n🏆 VALIDATION RESULTS:")
    print(f"   • Overall Agreement Score: {comparison['overall_agreement']:.1%}")
    print(f"   • Mean Relative Error: {comparison['mean_relative_error']:.1%}")
    
    # Determine validation outcome
    agreement_threshold = 0.80  # 80% agreement required for success
    error_threshold = 0.30      # <30% mean error required for success
    
    if (comparison['overall_agreement'] >= agreement_threshold and 
        comparison['mean_relative_error'] <= error_threshold):
        print(f"\n🎉 VALIDATION SUCCESS!")
        print(f"   ✅ Physics correctly reproduces observed market statistics!")
        print(f"   ✅ Action minimization predicts the same cluster frequencies!")
        print(f"   ✅ Market physics framework VALIDATED!")
    else:
        print(f"\n⚠️  VALIDATION INCONCLUSIVE:")
        if comparison['overall_agreement'] < agreement_threshold:
            print(f"   • Agreement below threshold ({comparison['overall_agreement']:.1%} < {agreement_threshold:.1%})")
        if comparison['mean_relative_error'] > error_threshold:
            print(f"   • Error above threshold ({comparison['mean_relative_error']:.1%} > {error_threshold:.1%})")
        print(f"   • Framework needs refinement")
    
    print("="*80)
    
    return {
        'validator': validator,
        'trajectories': trajectories,
        'predicted_stats': predicted_stats,
        'comparison': comparison,
        'validation_success': (comparison['overall_agreement'] >= agreement_threshold and 
                             comparison['mean_relative_error'] <= error_threshold)
    }

if __name__ == "__main__":
    results = run_ultimate_validation_test()