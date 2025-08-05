#!/usr/bin/env python3
"""
Explicit Potential Energy Function for Market Dynamics

Based on our measured potential energy from GMM cluster analysis, we now write
an explicit potential energy function with well-defined parameters:

V(s, u) = Σᵢ Aᵢ exp(-[(s-sᵢ)²/2σₛᵢ² + (u-uᵢ)²/2σᵤᵢ²])

Where the parameters are derived from our statistical measurements:
- Well positions (sᵢ, uᵢ) from GMM cluster centers  
- Well depths Aᵢ from observed frequencies
- Well widths σᵢ from cluster spreads

This explicit form allows us to:
1. Test action minimization predictions
2. Validate that predicted paths match observed statistics
3. Explore parameter sensitivity
4. Make quantitative predictions
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplicitMarketPotential:
    """
    Explicit potential energy function based on measured GMM cluster structure.
    
    V(s, u) = -Σᵢ Aᵢ exp(-[(s-sᵢ)²/2σₛᵢ² + (u-uᵢ)²/2σᵤᵢ²])
    
    The negative sign ensures that high-probability regions are low-energy wells.
    """
    
    def __init__(self, gmm_results_file: str, energy_scale: float = 1.0):
        """
        Initialize explicit potential from GMM measurements.
        
        Args:
            gmm_results_file: Path to GMM classification JSON
            energy_scale: Overall energy scale factor
        """
        self.energy_scale = energy_scale
        
        # Load measured GMM parameters
        self.well_parameters = self._load_well_parameters(gmm_results_file)
        
        logger.info(f"Initialized explicit potential with {len(self.well_parameters)} wells")
        for i, params in enumerate(self.well_parameters):
            logger.info(f"  Well {i+1}: {params['name']}")
            logger.info(f"    Center: ({params['center'][0]:.3f}, {params['center'][1]:.3f})")
            logger.info(f"    Depth: {params['depth']:.4f}")
            logger.info(f"    Width: ({params['width_s']:.3f}, {params['width_u']:.3f})")
    
    def _load_well_parameters(self, gmm_file: str) -> List[Dict]:
        """Load well parameters from GMM analysis results."""
        
        with open(gmm_file, 'r') as f:
            gmm_data = json.load(f)
        
        regions = gmm_data['classification_system']['regions']
        well_parameters = []
        
        for region in regions:
            # Well center from cluster center
            sentiment_center = sum(region['sentiment_bounds']) / 2
            uwr_center = sum(region['uwr_bounds']) / 2
            
            # Well depth from frequency (more frequent = deeper well)
            frequency = region['frequency_across_securities'] / 100.0
            depth = frequency * self.energy_scale  # Proportional to frequency
            
            # Well widths from cluster spreads
            sentiment_width = (region['sentiment_bounds'][1] - region['sentiment_bounds'][0]) / 4
            uwr_width = (region['uwr_bounds'][1] - region['uwr_bounds'][0]) / 4
            
            well_params = {
                'name': region['name'],
                'center': np.array([sentiment_center, uwr_center]),
                'depth': depth,
                'width_s': max(sentiment_width, 0.05),  # Minimum width for stability
                'width_u': max(uwr_width, 0.05),
                'frequency': frequency
            }
            
            well_parameters.append(well_params)
        
        # Sort by depth (deepest first)
        well_parameters.sort(key=lambda x: x['depth'], reverse=True)
        
        return well_parameters
    
    def potential_energy(self, sentiment: float, uwr: float) -> float:
        """
        Compute explicit potential energy V(s, u).
        
        V(s, u) = -Σᵢ Aᵢ exp(-[(s-sᵢ)²/2σₛᵢ² + (u-uᵢ)²/2σᵤᵢ²])
        """
        total_energy = 0.0
        
        for well in self.well_parameters:
            # Distance from well center
            s_center, u_center = well['center']
            ds = sentiment - s_center
            du = uwr - u_center
            
            # Gaussian well contribution
            exponent = -(ds**2 / (2 * well['width_s']**2) + du**2 / (2 * well['width_u']**2))
            well_contribution = -well['depth'] * np.exp(exponent)
            
            total_energy += well_contribution
        
        return total_energy
    
    def potential_gradient(self, sentiment: float, uwr: float) -> np.ndarray:
        """
        Compute gradient ∇V(s, u) = [∂V/∂s, ∂V/∂u].
        
        Force field: F = -∇V
        """
        gradient = np.zeros(2)
        
        for well in self.well_parameters:
            s_center, u_center = well['center']
            ds = sentiment - s_center
            du = uwr - u_center
            
            # Gaussian parameters
            sigma_s_sq = well['width_s']**2
            sigma_u_sq = well['width_u']**2
            
            # Exponential factor
            exponent = -(ds**2 / (2 * sigma_s_sq) + du**2 / (2 * sigma_u_sq))
            exp_factor = np.exp(exponent)
            
            # Gradient components
            dV_ds = -well['depth'] * exp_factor * (-ds / sigma_s_sq)
            dV_du = -well['depth'] * exp_factor * (-du / sigma_u_sq)
            
            gradient[0] += dV_ds
            gradient[1] += dV_du
        
        return gradient
    
    def force_field(self, sentiment: float, uwr: float) -> np.ndarray:
        """
        Compute force field F = -∇V.
        
        Positive force points toward lower potential energy (attractive).
        """
        return -self.potential_gradient(sentiment, uwr)
    
    def find_equilibrium_points(self) -> List[Dict]:
        """
        Find equilibrium points where ∇V = 0.
        
        These should correspond to the well centers (stable equilibria)
        and saddle points between wells (unstable equilibria).
        """
        from scipy.optimize import minimize
        
        equilibrium_points = []
        
        # Search for equilibria starting near each well center
        for well in self.well_parameters:
            center = well['center']
            
            def gradient_magnitude(point):
                """Objective: minimize |∇V|²"""
                grad = self.potential_gradient(point[0], point[1])
                return np.dot(grad, grad)
            
            # Constraint: stay in valid phase space
            def phase_space_constraint(point):
                return 1.0 - abs(point[0]) - point[1]  # |s| + u ≤ 1
            
            constraint = {'type': 'ineq', 'fun': phase_space_constraint}
            bounds = [(-0.99, 0.99), (0.01, 0.99)]
            
            result = minimize(
                gradient_magnitude,
                x0=center,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint,
                options={'ftol': 1e-12}
            )
            
            if result.success and result.fun < 1e-8:
                equilibrium = {
                    'position': result.x,
                    'potential': self.potential_energy(result.x[0], result.x[1]),
                    'type': 'stable',  # Assume stable (would need Hessian analysis)
                    'nearest_well': well['name']
                }
                equilibrium_points.append(equilibrium)
        
        return equilibrium_points
    
    def probability_from_potential(self, sentiment: float, uwr: float, temperature: float = 1.0) -> float:
        """
        Convert potential energy back to probability using Boltzmann distribution.
        
        P(s, u) = exp(-V(s, u)/kT) / Z
        
        This allows us to test consistency: if we derived V from P,
        then converting back should give the original P.
        """
        energy = self.potential_energy(sentiment, uwr)
        prob = np.exp(-energy / temperature)
        return prob
    
    def get_well_summary(self) -> Dict:
        """Return summary of all potential wells."""
        summary = {
            'n_wells': len(self.well_parameters),
            'total_depth': sum(well['depth'] for well in self.well_parameters),
            'deepest_well': max(self.well_parameters, key=lambda x: x['depth']),
            'shallowest_well': min(self.well_parameters, key=lambda x: x['depth']),
            'well_centers': [well['center'].tolist() for well in self.well_parameters],
            'well_depths': [well['depth'] for well in self.well_parameters],
            'well_names': [well['name'] for well in self.well_parameters]
        }
        return summary

def test_explicit_potential():
    """Test the explicit potential energy function."""
    
    logger.info("🧪 Testing Explicit Potential Energy Function")
    
    # Initialize with measured parameters
    gmm_file = "phase_space_analysis/candle_geometry_classification.json"
    potential = ExplicitMarketPotential(gmm_file, energy_scale=1.0)
    
    # Get well summary
    summary = potential.get_well_summary()
    
    print("\n" + "="*60)
    print("🎯 EXPLICIT POTENTIAL ENERGY FUNCTION")
    print("="*60)
    
    print(f"\n📊 WELL STRUCTURE:")
    print(f"   • Number of wells: {summary['n_wells']}")
    print(f"   • Total depth: {summary['total_depth']:.4f}")
    print(f"   • Deepest well: {summary['deepest_well']['name']} (depth={summary['deepest_well']['depth']:.4f})")
    print(f"   • Shallowest well: {summary['shallowest_well']['name']} (depth={summary['shallowest_well']['depth']:.4f})")
    
    print(f"\n⚡ WELL DETAILS:")
    for i, well in enumerate(potential.well_parameters):
        print(f"   {i+1}. {well['name']}")
        print(f"      Center: ({well['center'][0]:.3f}, {well['center'][1]:.3f})")
        print(f"      Depth: {well['depth']:.4f}")
        print(f"      Frequency: {well['frequency']:.1%}")
    
    # Test potential evaluation
    print(f"\n🧮 POTENTIAL EVALUATION TESTS:")
    
    # Evaluate at well centers
    for well in potential.well_parameters:
        s, u = well['center']
        V = potential.potential_energy(s, u)
        print(f"   • V({s:.3f}, {u:.3f}) = {V:.4f} [{well['name']}]")
    
    # Find equilibrium points
    print(f"\n⚖️ EQUILIBRIUM ANALYSIS:")
    equilibria = potential.find_equilibrium_points()
    
    for i, eq in enumerate(equilibria):
        print(f"   Equilibrium {i+1}: ({eq['position'][0]:.3f}, {eq['position'][1]:.3f})")
        print(f"     Potential: {eq['potential']:.4f}")
        print(f"     Near: {eq['nearest_well']}")
    
    print("="*60)
    
    return potential

if __name__ == "__main__":
    explicit_potential = test_explicit_potential()