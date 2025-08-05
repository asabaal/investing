#!/usr/bin/env python3
"""
Physics Framework Parameter Tuning System

Our validation test showed 18.4% agreement with 81.6% error - we need parameter optimization!

Key Issues Identified:
1. Temperature too low (0.05) - system too deterministic, not exploring all regions
2. Force field needs calibration - gradient dynamics may be wrong scale
3. Well depth scaling - frequency->depth relationship needs optimization  
4. Trajectory generation parameters - damping, mass, integration

This system will:
1. Define parameter search space
2. Run systematic grid search and optimization
3. Use validation metrics as objective function
4. Find optimal parameter combination
5. Generate tuned physics framework

Target: Achieve >80% agreement, <30% mean error
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
import json
from itertools import product
from scipy.optimize import minimize, differential_evolution
import time

from explicit_potential_energy import ExplicitMarketPotential
from validate_physics_predictions import PhysicsValidator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhysicsParameterTuner:
    """
    Systematic parameter tuning for the market physics framework.
    
    Optimizes physics parameters to maximize agreement between predictions and observations.
    Uses validation metrics as objective function for optimization.
    """
    
    def __init__(self, gmm_results_file: str):
        """Initialize parameter tuner with GMM data."""
        self.gmm_file = gmm_results_file
        self.validation_results = []
        self.best_params = None
        self.best_score = 0.0
        
        # Load reference data for validation
        with open(gmm_results_file, 'r') as f:
            self.gmm_data = json.load(f)
        
        logger.info("Initialized parameter tuner for physics framework optimization")
    
    def define_parameter_space(self) -> Dict[str, Tuple[float, float]]:
        """
        Define the parameter search space based on validation analysis.
        
        Returns parameter bounds for optimization.
        """
        parameter_bounds = {
            # Temperature parameters - key issue from validation
            'temperature': (0.01, 1.0),        # Was 0.05 - too low, try wider range
            
            # Energy scaling parameters  
            'energy_scale': (0.1, 5.0),        # Was 1.0 - try scaling the well depths
            
            # Trajectory generation parameters
            'damping': (0.01, 0.5),             # Was 0.1 - affects dynamics
            'mass': (0.1, 10.0),                # Was 1.0 - particle mass
            'force_scale': (0.1, 10.0),         # New - scale the force field
            
            # Integration parameters
            'n_trajectories': (50, 200),        # Was 100 - more trajectories for better stats
            'n_steps': (100, 500),               # Was 200 - longer trajectories
            'dt': (0.05, 0.2),                  # Was 0.1 - integration time step
            
            # Well parameters
            'well_depth_exponent': (0.5, 2.0),  # New - frequency^exponent -> depth
            'min_well_width': (0.02, 0.1),      # Was 0.05 - minimum well width
        }
        
        logger.info(f"Defined parameter space with {len(parameter_bounds)} parameters")
        for param, (min_val, max_val) in parameter_bounds.items():
            logger.info(f"  • {param}: [{min_val:.3f}, {max_val:.3f}]")
        
        return parameter_bounds
    
    def create_tuned_potential(self, params: Dict[str, float]) -> ExplicitMarketPotential:
        """Create potential energy function with tuned parameters."""
        
        # Create modified potential with tuned parameters
        class TunedMarketPotential(ExplicitMarketPotential):
            def __init__(self, gmm_file: str, tuning_params: Dict[str, float]):
                self.tuning_params = tuning_params
                energy_scale = tuning_params.get('energy_scale', 1.0)
                super().__init__(gmm_file, energy_scale)
                
                # Apply parameter tuning to well parameters
                self._apply_parameter_tuning()
            
            def _apply_parameter_tuning(self):
                """Apply tuning parameters to well structure."""
                depth_exp = self.tuning_params.get('well_depth_exponent', 1.0)
                min_width = self.tuning_params.get('min_well_width', 0.05)
                
                for well in self.well_parameters:
                    # Tune well depth with exponent
                    frequency = well['frequency']
                    well['depth'] = (frequency ** depth_exp) * self.energy_scale
                    
                    # Tune well widths
                    well['width_s'] = max(well['width_s'], min_width)
                    well['width_u'] = max(well['width_u'], min_width)
            
            def force_field(self, sentiment: float, uwr: float) -> np.ndarray:
                """Apply force scaling to the gradient."""
                force = super().force_field(sentiment, uwr)
                force_scale = self.tuning_params.get('force_scale', 1.0)
                return force * force_scale
        
        return TunedMarketPotential(self.gmm_file, params)
    
    def create_tuned_validator(self, params: Dict[str, float]) -> PhysicsValidator:
        """Create physics validator with tuned parameters."""
        
        # Create tuned potential
        tuned_potential = self.create_tuned_potential(params)
        
        # Create modified validator with tuned parameters
        class TunedPhysicsValidator(PhysicsValidator):
            def __init__(self, potential, tuning_params: Dict[str, float]):
                super().__init__(potential)
                self.tuning_params = tuning_params
                
                # Apply tuned parameters
                self.dt = tuning_params.get('dt', 0.1)
            
            def generate_predicted_trajectory(self, initial_state: np.ndarray, 
                                            n_steps: int = None, temperature: float = None):
                """Override with tuned parameters."""
                if n_steps is None:
                    n_steps = int(self.tuning_params.get('n_steps', 200))
                if temperature is None:
                    temperature = self.tuning_params.get('temperature', 0.1)
                
                # Use tuned dynamics parameters
                damping = self.tuning_params.get('damping', 0.1)
                mass = self.tuning_params.get('mass', 1.0)
                
                # Modified trajectory generation with tuned parameters
                trajectory = [initial_state.copy()]
                current_state = initial_state.copy()
                velocity = np.zeros(2)
                
                for step in range(n_steps):
                    # Force from potential gradient
                    force = self.potential.force_field(current_state[0], current_state[1])
                    
                    # Add thermal noise
                    thermal_force = np.random.normal(0, np.sqrt(2 * temperature), 2)
                    total_force = force + thermal_force
                    
                    # Update with tuned parameters
                    acceleration = total_force / mass
                    velocity = velocity * (1 - damping) + acceleration * self.dt
                    new_state = current_state + velocity * self.dt
                    
                    # Apply phase space constraints
                    s, u = new_state
                    s = np.clip(s, -0.99, 0.99)
                    u = np.clip(u, 0.01, 0.99)
                    
                    if abs(s) + u > 1.0:
                        total = abs(s) + u
                        s = s * 0.99 / total
                        u = u * 0.99 / total
                    
                    current_state = np.array([s, u])
                    trajectory.append(current_state.copy())
                
                return trajectory
        
        return TunedPhysicsValidator(tuned_potential, params)
    
    def evaluate_parameter_set(self, params: Dict[str, float], quick_eval: bool = False) -> Dict[str, float]:
        """
        Evaluate a parameter set and return validation metrics.
        
        Args:
            params: Dictionary of parameter values
            quick_eval: If True, use fewer trajectories for faster evaluation
            
        Returns:
            Dictionary with validation metrics
        """
        try:
            # Create tuned validator
            validator = self.create_tuned_validator(params)
            
            # Generate trajectories with tuned parameters
            n_traj = int(params.get('n_trajectories', 100))
            if quick_eval:
                n_traj = min(n_traj, 20)  # Faster evaluation
            
            n_steps = int(params.get('n_steps', 200))
            if quick_eval:
                n_steps = min(n_steps, 50)  # Shorter trajectories
            
            temperature = params.get('temperature', 0.1)
            
            # Generate trajectories
            trajectories = validator.generate_multiple_trajectories(
                n_trajectories=n_traj,
                n_steps=n_steps,
                temperature=temperature
            )
            
            # Analyze statistics
            predicted_stats = validator.analyze_trajectory_statistics(trajectories)
            
            # Compare with observations
            comparison = validator.compare_with_observations(predicted_stats)
            
            # Extract key metrics
            agreement_score = comparison['overall_agreement']
            mean_error = comparison['mean_relative_error']
            
            # Combined objective function (higher is better)
            objective = agreement_score - 0.5 * mean_error  # Balance agreement and error
            
            metrics = {
                'agreement_score': agreement_score,
                'mean_relative_error': mean_error,
                'objective': objective,
                'n_trajectories_actual': len(trajectories),
                'n_points_total': predicted_stats['n_points']
            }
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Parameter evaluation failed: {e}")
            return {
                'agreement_score': 0.0,
                'mean_relative_error': 999.0,
                'objective': -999.0,
                'n_trajectories_actual': 0,
                'n_points_total': 0
            }
    
    def grid_search(self, n_samples_per_param: int = 5) -> Dict[str, Any]:
        """
        Perform systematic grid search over parameter space.
        
        Args:
            n_samples_per_param: Number of values to test per parameter
            
        Returns:
            Grid search results
        """
        logger.info(f"🔍 Starting grid search with {n_samples_per_param} samples per parameter")
        
        param_bounds = self.define_parameter_space()
        
        # Create grid of parameter values
        param_grids = {}
        for param, (min_val, max_val) in param_bounds.items():
            if param in ['n_trajectories', 'n_steps']:
                # Integer parameters
                param_grids[param] = np.linspace(min_val, max_val, n_samples_per_param, dtype=int)
            else:
                # Float parameters
                param_grids[param] = np.linspace(min_val, max_val, n_samples_per_param)
        
        # Generate all combinations
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        total_combinations = np.prod([len(values) for values in param_values])
        logger.info(f"Total parameter combinations to evaluate: {total_combinations}")
        
        if total_combinations > 1000:
            logger.warning(f"Large search space ({total_combinations} combinations). Consider reducing n_samples_per_param.")
        
        # Evaluate each combination
        results = []
        start_time = time.time()
        
        for i, param_combination in enumerate(product(*param_values)):
            params = dict(zip(param_names, param_combination))
            
            # Quick evaluation for grid search
            metrics = self.evaluate_parameter_set(params, quick_eval=True)
            
            result = {
                'combination_id': i,
                'parameters': params.copy(),
                **metrics
            }
            results.append(result)
            
            # Track best parameters
            if metrics['objective'] > self.best_score:
                self.best_score = metrics['objective']
                self.best_params = params.copy()
                
                logger.info(f"🎯 New best parameters found!")
                logger.info(f"   Agreement: {metrics['agreement_score']:.1%}")
                logger.info(f"   Error: {metrics['mean_relative_error']:.1%}")
                logger.info(f"   Objective: {metrics['objective']:.4f}")
            
            # Progress reporting
            if (i + 1) % max(1, total_combinations // 20) == 0:
                elapsed = time.time() - start_time
                progress = (i + 1) / total_combinations
                eta = elapsed / progress - elapsed
                logger.info(f"Grid search progress: {i+1}/{total_combinations} ({progress:.1%}) - ETA: {eta/60:.1f}min")
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Grid search completed in {elapsed_time/60:.1f} minutes")
        
        # Sort results by objective function
        results.sort(key=lambda x: x['objective'], reverse=True)
        
        grid_search_results = {
            'results': results,
            'best_parameters': self.best_params,
            'best_score': self.best_score,
            'total_evaluations': len(results),
            'search_time_minutes': elapsed_time / 60
        }
        
        return grid_search_results
    
    def optimize_parameters(self, initial_params: Dict[str, float] = None, 
                          method: str = 'differential_evolution') -> Dict[str, Any]:
        """
        Perform optimization-based parameter tuning.
        
        Args:
            initial_params: Starting parameters (uses best from grid search if None)
            method: Optimization method ('differential_evolution', 'minimize')
            
        Returns:
            Optimization results
        """
        logger.info(f"🎯 Starting parameter optimization using {method}")
        
        param_bounds = self.define_parameter_space()
        
        # Prepare bounds for scipy
        bounds_list = []
        param_names = []
        for param, (min_val, max_val) in param_bounds.items():
            bounds_list.append((min_val, max_val))
            param_names.append(param)
        
        def objective_function(param_array):
            """Objective function for optimization (minimize negative objective)."""
            params = dict(zip(param_names, param_array))
            
            # Handle integer parameters
            params['n_trajectories'] = int(params['n_trajectories'])
            params['n_steps'] = int(params['n_steps'])
            
            metrics = self.evaluate_parameter_set(params, quick_eval=False)
            
            # Return negative objective for minimization
            return -metrics['objective']
        
        start_time = time.time()
        
        if method == 'differential_evolution':
            # Global optimization
            result = differential_evolution(
                objective_function,
                bounds=bounds_list,
                maxiter=50,  # Limit iterations for reasonable time
                popsize=10,   # Population size
                seed=42,
                disp=True
            )
            
            optimal_params = dict(zip(param_names, result.x))
            
        elif method == 'minimize':
            # Local optimization
            if initial_params is None:
                if self.best_params is not None:
                    initial_params = self.best_params
                else:
                    # Use middle of parameter ranges
                    initial_params = {}
                    for param, (min_val, max_val) in param_bounds.items():
                        initial_params[param] = (min_val + max_val) / 2
            
            x0 = [initial_params[param] for param in param_names]
            
            result = minimize(
                objective_function,
                x0=x0,
                bounds=bounds_list,
                method='L-BFGS-B',
                options={'maxiter': 100}
            )
            
            optimal_params = dict(zip(param_names, result.x))
        
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Handle integer parameters
        optimal_params['n_trajectories'] = int(optimal_params['n_trajectories'])
        optimal_params['n_steps'] = int(optimal_params['n_steps'])
        
        # Evaluate final parameters thoroughly
        final_metrics = self.evaluate_parameter_set(optimal_params, quick_eval=False)
        
        elapsed_time = time.time() - start_time
        
        optimization_results = {
            'optimal_parameters': optimal_params,
            'final_metrics': final_metrics,
            'optimization_result': result,
            'optimization_time_minutes': elapsed_time / 60,
            'method': method
        }
        
        # Update best parameters if better
        if final_metrics['objective'] > self.best_score:
            self.best_score = final_metrics['objective']
            self.best_params = optimal_params
            
            logger.info(f"🏆 Optimization found better parameters!")
            logger.info(f"   Agreement: {final_metrics['agreement_score']:.1%}")
            logger.info(f"   Error: {final_metrics['mean_relative_error']:.1%}")
            logger.info(f"   Objective: {final_metrics['objective']:.4f}")
        
        logger.info(f"✅ Parameter optimization completed in {elapsed_time/60:.1f} minutes")
        
        return optimization_results
    
    def comprehensive_parameter_tuning(self) -> Dict[str, Any]:
        """
        Run comprehensive parameter tuning: grid search followed by optimization.
        
        Returns:
            Complete tuning results
        """
        logger.info("🚀 Starting comprehensive parameter tuning")
        
        # Stage 1: Grid search for initial exploration
        logger.info("📋 Stage 1: Grid search for parameter space exploration")
        grid_results = self.grid_search(n_samples_per_param=3)  # Coarse grid
        
        # Stage 2: Optimization from best grid search result
        logger.info("🎯 Stage 2: Fine-tuning with optimization")
        optimization_results = self.optimize_parameters(
            initial_params=grid_results['best_parameters'],
            method='differential_evolution'
        )
        
        # Stage 3: Local refinement
        logger.info("🔧 Stage 3: Local refinement")
        local_results = self.optimize_parameters(
            initial_params=optimization_results['optimal_parameters'],
            method='minimize'
        )
        
        comprehensive_results = {
            'stage1_grid_search': grid_results,
            'stage2_global_optimization': optimization_results,
            'stage3_local_refinement': local_results,
            'final_best_parameters': self.best_params,
            'final_best_score': self.best_score
        }
        
        return comprehensive_results
    
    def validate_tuned_framework(self, tuned_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Run complete validation with tuned parameters.
        
        Args:
            tuned_params: Optimized parameter set
            
        Returns:
            Complete validation results
        """
        logger.info("🏆 Running final validation with tuned parameters")
        
        # Create tuned validator
        validator = self.create_tuned_validator(tuned_params)
        
        # Generate trajectories with full parameters (no quick eval)
        n_traj = int(tuned_params.get('n_trajectories', 100))
        n_steps = int(tuned_params.get('n_steps', 200))
        temperature = tuned_params.get('temperature', 0.1)
        
        trajectories = validator.generate_multiple_trajectories(
            n_trajectories=n_traj,
            n_steps=n_steps,
            temperature=temperature
        )
        
        # Full analysis
        predicted_stats = validator.analyze_trajectory_statistics(trajectories)
        comparison = validator.compare_with_observations(predicted_stats)
        
        # Create validation visualization
        validation_fig = validator.create_validation_visualization(
            trajectories, predicted_stats, comparison
        )
        
        # Save results
        output_file = Path("phase_space_analysis/tuned_physics_validation.html")
        validation_fig.write_html(str(output_file))
        
        validation_results = {
            'trajectories': trajectories,
            'predicted_stats': predicted_stats,
            'comparison': comparison,
            'visualization_file': output_file,
            'tuned_parameters': tuned_params
        }
        
        logger.info(f"🎯 Tuned validation results:")
        logger.info(f"   Agreement: {comparison['overall_agreement']:.1%}")
        logger.info(f"   Mean Error: {comparison['mean_relative_error']:.1%}")
        logger.info(f"   Visualization saved to: {output_file}")
        
        return validation_results
    
    def create_tuning_report(self, comprehensive_results: Dict[str, Any]) -> go.Figure:
        """Create comprehensive visualization of parameter tuning results."""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Parameter Search Progress',
                'Best Parameters vs Defaults',
                'Validation Metrics Comparison',
                'Parameter Sensitivity Analysis',
                'Tuning Timeline',
                'Final Results Summary'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "table"}]
            ]
        )
        
        # Extract results for visualization
        grid_results = comprehensive_results['stage1_grid_search']['results']
        final_params = comprehensive_results['final_best_parameters']
        
        # 1. Parameter search progress
        objectives = [r['objective'] for r in grid_results]
        agreements = [r['agreement_score'] for r in grid_results]
        
        fig.add_trace(
            go.Scatter(
                y=objectives,
                mode='lines+markers',
                name='Objective Function',
                line=dict(color='blue'),
                hovertemplate='Evaluation: %{x}<br>Objective: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Best parameters vs defaults
        default_params = {
            'temperature': 0.05,
            'energy_scale': 1.0,
            'damping': 0.1,
            'mass': 1.0,
            'force_scale': 1.0
        }
        
        param_names = []
        default_values = []
        tuned_values = []
        
        for param in default_params:
            if param in final_params:
                param_names.append(param)
                default_values.append(default_params[param])
                tuned_values.append(final_params[param])
        
        fig.add_trace(
            go.Bar(
                x=param_names,
                y=default_values,
                name='Default',
                marker_color='lightcoral',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=param_names,
                y=tuned_values,
                name='Tuned',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # 3. Validation metrics comparison
        stages = ['Initial', 'Grid Search', 'Global Opt', 'Local Opt']
        initial_agreement = 0.184  # From our validation
        agreements_progress = [
            initial_agreement,
            comprehensive_results['stage1_grid_search']['best_score'],
            comprehensive_results['stage2_global_optimization']['final_metrics']['agreement_score'],
            comprehensive_results['stage3_local_refinement']['final_metrics']['agreement_score']
        ]
        
        fig.add_trace(
            go.Bar(
                x=stages,
                y=agreements_progress,
                marker_color=['red', 'orange', 'yellow', 'green'],
                name='Agreement Score',
                hovertemplate='Stage: %{x}<br>Agreement: %{y:.1%}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add target line
        fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                     annotation_text="Target: 80%", row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="🎯 Physics Framework Parameter Tuning Results<br>" +
                     f"<sub>Final Agreement: {comprehensive_results['final_best_score']:.1%} | " +
                     f"Target: 80% Agreement</sub>",
                x=0.5,
                xanchor='center'
            ),
            height=1200,
            width=1600,
            template="plotly_dark",
            showlegend=True
        )
        
        return fig

def run_comprehensive_parameter_tuning():
    """Run the complete parameter tuning pipeline."""
    
    logger.info("🚀 Starting Comprehensive Physics Parameter Tuning")
    
    # Initialize tuner
    gmm_file = "phase_space_analysis/candle_geometry_classification.json" 
    tuner = PhysicsParameterTuner(gmm_file)
    
    # Run comprehensive tuning
    results = tuner.comprehensive_parameter_tuning()
    
    # Final validation with best parameters
    final_validation = tuner.validate_tuned_framework(results['final_best_parameters'])
    
    # Create tuning report
    tuning_report = tuner.create_tuning_report(results)
    report_file = Path("phase_space_analysis/parameter_tuning_report.html")
    tuning_report.write_html(str(report_file))
    
    # Save results
    results_file = Path("phase_space_analysis/parameter_tuning_results.json")
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {k: float(v) if isinstance(v, (np.float64, np.int64)) else v 
                                           for k, v in value.items() if not callable(v)}
            else:
                serializable_results[key] = value
        json.dump(serializable_results, f, indent=2, default=str)
    
    # Final report
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE PARAMETER TUNING RESULTS")
    print("="*80)
    
    final_metrics = results['stage3_local_refinement']['final_metrics']
    print(f"\n🏆 FINAL RESULTS:")
    print(f"   • Agreement Score: {final_metrics['agreement_score']:.1%}")
    print(f"   • Mean Relative Error: {final_metrics['mean_relative_error']:.1%}")
    print(f"   • Objective Function: {final_metrics['objective']:.4f}")
    
    success = (final_metrics['agreement_score'] >= 0.8 and 
              final_metrics['mean_relative_error'] <= 0.3)
    
    if success:
        print(f"\n🎉 PARAMETER TUNING SUCCESS!")
        print(f"   ✅ Agreement ≥ 80%: {final_metrics['agreement_score']:.1%}")
        print(f"   ✅ Error ≤ 30%: {final_metrics['mean_relative_error']:.1%}")
        print(f"   ✅ Physics framework validated!")
    else:
        print(f"\n⚠️  PARTIAL SUCCESS - Further tuning may be needed")
        if final_metrics['agreement_score'] < 0.8:
            print(f"   • Agreement below target: {final_metrics['agreement_score']:.1%} < 80%")
        if final_metrics['mean_relative_error'] > 0.3:
            print(f"   • Error above target: {final_metrics['mean_relative_error']:.1%} > 30%")
    
    print(f"\n📊 OPTIMIZED PARAMETERS:")
    for param, value in results['final_best_parameters'].items():
        print(f"   • {param}: {value:.4f}")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   • Tuning report: {report_file}")
    print(f"   • Validation: {final_validation['visualization_file']}")
    print(f"   • Results data: {results_file}")
    
    print("="*80)
    
    return {
        'tuning_results': results,
        'final_validation': final_validation,
        'success': success
    }

if __name__ == "__main__":
    results = run_comprehensive_parameter_tuning()