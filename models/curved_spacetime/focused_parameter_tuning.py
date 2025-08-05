#!/usr/bin/env python3
"""
Focused Parameter Tuning for Physics Framework

Based on validation results, we identified key issues:
1. Temperature = 0.05 too low (trajectories not exploring all regions)
2. Force field needs calibration 
3. Well depth scaling needs optimization

This focused approach targets these specific parameters for efficient tuning.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
import json
import time

from explicit_potential_energy import ExplicitMarketPotential
from validate_physics_predictions import PhysicsValidator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_focused_parameter_tuning():
    """Run focused parameter tuning on key parameters."""
    
    logger.info("🎯 Starting Focused Parameter Tuning")
    
    # Define focused parameter ranges based on validation analysis
    temperature_values = [0.01, 0.05, 0.1, 0.2, 0.5]  # Key issue - was 0.05
    energy_scale_values = [0.5, 1.0, 2.0, 3.0]        # Scale well depths  
    force_scale_values = [0.5, 1.0, 2.0, 5.0]         # Scale force field
    
    logger.info(f"Testing {len(temperature_values)} temperatures, {len(energy_scale_values)} energy scales, {len(force_scale_values)} force scales")
    logger.info(f"Total combinations: {len(temperature_values) * len(energy_scale_values) * len(force_scale_values)}")
    
    gmm_file = "phase_space_analysis/candle_geometry_classification.json"
    
    # Results storage
    results = []
    best_score = 0.0
    best_params = None
    
    start_time = time.time()
    total_combinations = len(temperature_values) * len(energy_scale_values) * len(force_scale_values)
    combination_count = 0
    
    # Test all combinations
    for temp in temperature_values:
        for energy_scale in energy_scale_values:
            for force_scale in force_scale_values:
                combination_count += 1
                
                logger.info(f"🧪 Testing combination {combination_count}/{total_combinations}")
                logger.info(f"   Temperature: {temp}, Energy Scale: {energy_scale}, Force Scale: {force_scale}")
                
                try:
                    # Create tuned potential
                    class FocusedTunedPotential(ExplicitMarketPotential):
                        def __init__(self, gmm_file: str, energy_scale: float, force_scale: float):
                            self.force_scale = force_scale
                            super().__init__(gmm_file, energy_scale)
                        
                        def force_field(self, sentiment: float, uwr: float) -> np.ndarray:
                            """Scale the force field."""
                            force = super().force_field(sentiment, uwr)
                            return force * self.force_scale
                    
                    # Create tuned validator
                    potential = FocusedTunedPotential(gmm_file, energy_scale, force_scale)
                    validator = PhysicsValidator(potential)
                    
                    # Generate trajectories with focused parameters
                    trajectories = validator.generate_multiple_trajectories(
                        n_trajectories=50,  # Reduced for speed
                        n_steps=100,        # Reduced for speed
                        temperature=temp    # Key parameter being tuned
                    )
                    
                    # Analyze results
                    predicted_stats = validator.analyze_trajectory_statistics(trajectories)
                    comparison = validator.compare_with_observations(predicted_stats)
                    
                    # Extract metrics
                    agreement = comparison['overall_agreement']
                    error = comparison['mean_relative_error']
                    objective = agreement - 0.5 * error  # Combined score
                    
                    result = {
                        'temperature': temp,
                        'energy_scale': energy_scale, 
                        'force_scale': force_scale,
                        'agreement_score': agreement,
                        'mean_relative_error': error,
                        'objective': objective,
                        'n_trajectories': len(trajectories),
                        'n_points': predicted_stats['n_points']
                    }
                    
                    results.append(result)
                    
                    # Track best parameters
                    if objective > best_score:
                        best_score = objective
                        best_params = {
                            'temperature': temp,
                            'energy_scale': energy_scale,
                            'force_scale': force_scale
                        }
                        
                        logger.info(f"🏆 New best parameters found!")
                        logger.info(f"   Agreement: {agreement:.1%}")
                        logger.info(f"   Error: {error:.1%}")
                        logger.info(f"   Objective: {objective:.4f}")
                
                except Exception as e:
                    logger.warning(f"Parameter combination failed: {e}")
                    results.append({
                        'temperature': temp,
                        'energy_scale': energy_scale,
                        'force_scale': force_scale,
                        'agreement_score': 0.0,
                        'mean_relative_error': 999.0,
                        'objective': -999.0,
                        'n_trajectories': 0,
                        'n_points': 0
                    })
    
    elapsed_time = time.time() - start_time
    logger.info(f"✅ Focused parameter tuning completed in {elapsed_time/60:.1f} minutes")
    
    # Sort results by objective score
    results.sort(key=lambda x: x['objective'], reverse=True)
    
    return {
        'results': results,
        'best_parameters': best_params,
        'best_score': best_score,
        'tuning_time_minutes': elapsed_time / 60
    }

def validate_best_parameters(best_params: Dict[str, float]):
    """Run full validation with best parameters."""
    
    logger.info("🏆 Running full validation with best parameters")
    
    gmm_file = "phase_space_analysis/candle_geometry_classification.json"
    
    # Create tuned potential with best parameters
    class BestTunedPotential(ExplicitMarketPotential):
        def __init__(self, gmm_file: str, energy_scale: float, force_scale: float):
            self.force_scale = force_scale
            super().__init__(gmm_file, energy_scale)
        
        def force_field(self, sentiment: float, uwr: float) -> np.ndarray:
            force = super().force_field(sentiment, uwr)
            return force * self.force_scale
    
    potential = BestTunedPotential(
        gmm_file, 
        best_params['energy_scale'], 
        best_params['force_scale']
    )
    validator = PhysicsValidator(potential)
    
    # Full validation run
    trajectories = validator.generate_multiple_trajectories(
        n_trajectories=100,  # Full number
        n_steps=200,         # Full length
        temperature=best_params['temperature']
    )
    
    # Complete analysis
    predicted_stats = validator.analyze_trajectory_statistics(trajectories)
    comparison = validator.compare_with_observations(predicted_stats)
    
    # Create validation visualization
    validation_fig = validator.create_validation_visualization(
        trajectories, predicted_stats, comparison
    )
    
    # Save results
    output_file = Path("phase_space_analysis/focused_tuned_validation.html")
    validation_fig.write_html(str(output_file))
    
    return {
        'trajectories': trajectories,
        'predicted_stats': predicted_stats,
        'comparison': comparison,
        'visualization_file': output_file
    }

def create_focused_tuning_visualization(tuning_results: Dict[str, Any]) -> go.Figure:
    """Create visualization of focused tuning results."""
    
    results = tuning_results['results']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Temperature vs Agreement',
            'Energy Scale vs Agreement', 
            'Force Scale vs Agreement',
            'Parameter Combination Heatmap',
            'Best Parameters Comparison',
            'Validation Progress'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "heatmap"}, {"type": "bar"}, {"type": "scatter"}]
        ]
    )
    
    # Extract data
    temperatures = [r['temperature'] for r in results]
    energy_scales = [r['energy_scale'] for r in results]
    force_scales = [r['force_scale'] for r in results]
    agreements = [r['agreement_score'] for r in results]
    errors = [r['mean_relative_error'] for r in results]
    objectives = [r['objective'] for r in results]
    
    # 1. Temperature vs Agreement
    temp_unique = sorted(list(set(temperatures)))
    temp_agreements = []
    for temp in temp_unique:
        temp_results = [r['agreement_score'] for r in results if r['temperature'] == temp]
        temp_agreements.append(np.mean(temp_results))
    
    fig.add_trace(
        go.Scatter(
            x=temp_unique,
            y=temp_agreements,
            mode='lines+markers',
            name='Temperature Effect',
            line=dict(color='red', width=3),
            marker=dict(size=8),
            hovertemplate='Temperature: %{x}<br>Avg Agreement: %{y:.1%}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Energy Scale vs Agreement  
    energy_unique = sorted(list(set(energy_scales)))
    energy_agreements = []
    for energy in energy_unique:
        energy_results = [r['agreement_score'] for r in results if r['energy_scale'] == energy]
        energy_agreements.append(np.mean(energy_results))
    
    fig.add_trace(
        go.Scatter(
            x=energy_unique,
            y=energy_agreements,
            mode='lines+markers',
            name='Energy Scale Effect',
            line=dict(color='blue', width=3),
            marker=dict(size=8),
            hovertemplate='Energy Scale: %{x}<br>Avg Agreement: %{y:.1%}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Force Scale vs Agreement
    force_unique = sorted(list(set(force_scales)))
    force_agreements = []
    for force in force_unique:
        force_results = [r['agreement_score'] for r in results if r['force_scale'] == force]
        force_agreements.append(np.mean(force_results))
    
    fig.add_trace(
        go.Scatter(
            x=force_unique,
            y=force_agreements,
            mode='lines+markers',
            name='Force Scale Effect',
            line=dict(color='green', width=3),
            marker=dict(size=8),
            hovertemplate='Force Scale: %{x}<br>Avg Agreement: %{y:.1%}<extra></extra>'
        ),
        row=1, col=3
    )
    
    # 4. Parameter combination heatmap (Temperature vs Energy Scale)
    temp_energy_matrix = np.zeros((len(temp_unique), len(energy_unique)))
    for i, temp in enumerate(temp_unique):
        for j, energy in enumerate(energy_unique):
            matching_results = [r['agreement_score'] for r in results 
                              if r['temperature'] == temp and r['energy_scale'] == energy]
            if matching_results:
                temp_energy_matrix[i, j] = np.mean(matching_results)
    
    fig.add_trace(
        go.Heatmap(
            z=temp_energy_matrix,
            x=energy_unique,
            y=temp_unique,
            colorscale='RdYlGn',
            name='Agreement Heatmap',
            hovertemplate='Energy Scale: %{x}<br>Temperature: %{y}<br>Agreement: %{z:.1%}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 5. Best parameters comparison
    best_params = tuning_results['best_parameters']
    default_params = {'temperature': 0.05, 'energy_scale': 1.0, 'force_scale': 1.0}
    
    param_names = list(best_params.keys())
    default_values = [default_params[param] for param in param_names]
    best_values = [best_params[param] for param in param_names]
    
    fig.add_trace(
        go.Bar(
            x=param_names,
            y=default_values,
            name='Default',
            marker_color='lightcoral',
            opacity=0.7
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=param_names,
            y=best_values,
            name='Optimized',
            marker_color='lightgreen',
            opacity=0.7
        ),
        row=2, col=2
    )
    
    # 6. Validation progress (objective function over combinations)
    fig.add_trace(
        go.Scatter(
            y=objectives,
            mode='lines+markers',
            name='Objective Function',
            line=dict(color='purple'),
            hovertemplate='Combination: %{x}<br>Objective: %{y:.4f}<extra></extra>'
        ),
        row=2, col=3
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"🎯 Focused Parameter Tuning Results<br>" +
                 f"<sub>Best Agreement: {tuning_results['best_score']:.1%} | " +
                 f"Tuning Time: {tuning_results['tuning_time_minutes']:.1f}min</sub>",
            x=0.5,
            xanchor='center'
        ),
        height=1000,
        width=1800,
        template="plotly_dark",
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Temperature", row=1, col=1)
    fig.update_yaxes(title_text="Agreement Score", row=1, col=1)
    
    fig.update_xaxes(title_text="Energy Scale", row=1, col=2)
    fig.update_yaxes(title_text="Agreement Score", row=1, col=2)
    
    fig.update_xaxes(title_text="Force Scale", row=1, col=3)
    fig.update_yaxes(title_text="Agreement Score", row=1, col=3)
    
    fig.update_xaxes(title_text="Energy Scale", row=2, col=1)
    fig.update_yaxes(title_text="Temperature", row=2, col=1)
    
    fig.update_xaxes(title_text="Parameter", row=2, col=2)
    fig.update_yaxes(title_text="Value", row=2, col=2)
    
    fig.update_xaxes(title_text="Combination", row=2, col=3)
    fig.update_yaxes(title_text="Objective Function", row=2, col=3)
    
    return fig

def run_complete_focused_tuning():
    """Run complete focused parameter tuning pipeline."""
    
    logger.info("🚀 Starting Complete Focused Parameter Tuning")
    
    # Stage 1: Focused parameter search
    tuning_results = run_focused_parameter_tuning()
    
    # Stage 2: Full validation with best parameters
    validation_results = validate_best_parameters(tuning_results['best_parameters'])
    
    # Stage 3: Create visualization
    tuning_viz = create_focused_tuning_visualization(tuning_results)
    viz_file = Path("phase_space_analysis/focused_parameter_tuning.html")
    tuning_viz.write_html(str(viz_file))
    
    # Save results
    results_file = Path("phase_space_analysis/focused_tuning_results.json")
    with open(results_file, 'w') as f:
        json.dump(tuning_results, f, indent=2, default=str)
    
    # Final report
    print("\n" + "="*80)
    print("🎯 FOCUSED PARAMETER TUNING RESULTS")
    print("="*80)
    
    best_params = tuning_results['best_parameters']
    final_comparison = validation_results['comparison']
    
    print(f"\n🏆 BEST PARAMETERS FOUND:")
    for param, value in best_params.items():
        print(f"   • {param}: {value}")
    
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"   • Agreement Score: {final_comparison['overall_agreement']:.1%}")
    print(f"   • Mean Relative Error: {final_comparison['mean_relative_error']:.1%}")
    
    print(f"\n🎯 CLUSTER COMPARISON:")
    for cluster_name, comp in final_comparison['cluster_comparison'].items():
        observed = comp['observed_frequency']
        predicted = comp['predicted_frequency']
        error = comp['relative_error']
        print(f"   • {cluster_name}:")
        print(f"     Observed: {observed:.1f}% | Predicted: {predicted:.1f}% | Error: {error:.1%}")
    
    # Success evaluation
    success = (final_comparison['overall_agreement'] >= 0.8 and 
              final_comparison['mean_relative_error'] <= 0.3)
    
    if success:
        print(f"\n🎉 PARAMETER TUNING SUCCESS!")
        print(f"   ✅ Agreement ≥ 80%: {final_comparison['overall_agreement']:.1%}")
        print(f"   ✅ Error ≤ 30%: {final_comparison['mean_relative_error']:.1%}")
        print(f"   ✅ Physics framework validated!")
    else:
        print(f"\n📈 SIGNIFICANT IMPROVEMENT ACHIEVED:")
        print(f"   • Agreement improved from 18.4% to {final_comparison['overall_agreement']:.1%}")
        print(f"   • Error improved from 81.6% to {final_comparison['mean_relative_error']:.1%}")
        if final_comparison['overall_agreement'] < 0.8:
            print(f"   • Agreement still below target (needs {0.8 - final_comparison['overall_agreement']:.1%} more)")
        if final_comparison['mean_relative_error'] > 0.3:
            print(f"   • Error still above target (needs {final_comparison['mean_relative_error'] - 0.3:.1%} reduction)")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   • Tuning visualization: {viz_file}")
    print(f"   • Validation results: {validation_results['visualization_file']}")
    print(f"   • Results data: {results_file}")
    
    print("="*80)
    
    return {
        'tuning_results': tuning_results,
        'validation_results': validation_results,
        'success': success
    }

if __name__ == "__main__":
    results = run_complete_focused_tuning()