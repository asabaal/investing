"""
Equilibrium State Finder

Discovers natural low-energy market configurations and energy minima.
These equilibrium states represent stable market conditions where the
system naturally settles without external interventions.

Key capabilities:
1. Find global energy minima (ground states)
2. Identify metastable states (local minima)
3. Calculate energy barriers between states
4. Predict state transition probabilities
5. Estimate equilibrium lifetimes
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import pandas as pd
from market_hamiltonian import MarketHamiltonian, MarketState
from curved_candle_geometry import CandleMetric
from market_lie_group import MarketLieGroup


@dataclass
class EquilibriumState:
    """Represents an equilibrium configuration."""
    candle: CandleMetric
    energy: float
    state_type: str  # 'global', 'local', 'saddle'
    stability: float  # Eigenvalue analysis result
    basin_size: float  # Size of attraction basin
    lifetime_estimate: float  # Expected lifetime in time units
    neighboring_states: List['EquilibriumState'] = None
    transition_barriers: Dict[int, float] = None  # barriers to other states


class EquilibriumStateFinder:
    """
    Finds and analyzes equilibrium states in the market energy landscape.
    
    Uses optimization techniques to locate energy minima and analyze
    their stability properties.
    """
    
    def __init__(self, 
                 hamiltonian: Optional[MarketHamiltonian] = None,
                 n_global_searches: int = 50,
                 n_local_searches: int = 100,
                 energy_tolerance: float = 1e-6,
                 gradient_tolerance: float = 1e-6,
                 cluster_threshold: float = 0.1):
        """
        Initialize the equilibrium finder.
        
        Args:
            hamiltonian: Energy function (default creates new one)
            n_global_searches: Number of global optimization attempts
            n_local_searches: Number of local optimization attempts
            energy_tolerance: Convergence tolerance for energy
            gradient_tolerance: Convergence tolerance for gradients
            cluster_threshold: Distance threshold for clustering similar states
        """
        self.hamiltonian = hamiltonian or MarketHamiltonian()
        self.n_global_searches = n_global_searches
        self.n_local_searches = n_local_searches
        self.energy_tolerance = energy_tolerance
        self.gradient_tolerance = gradient_tolerance
        self.cluster_threshold = cluster_threshold
        
        # Cached results
        self._equilibrium_states = None
        self._energy_landscape = None
    
    def find_global_minimum(self, 
                          initial_candle: Optional[CandleMetric] = None,
                          bounds: Optional[Dict] = None) -> EquilibriumState:
        """
        Find the global energy minimum (ground state).
        
        Args:
            initial_candle: Starting point for optimization
            bounds: Parameter bounds for optimization
            
        Returns:
            Ground state equilibrium
        """
        if bounds is None:
            bounds = {
                'sentiment': (-0.99, 0.99),
                'upper_wick_ratio': (0.01, 0.99),
                'range_value': (0.01, 10.0),
                'volume': (1.0, 10000.0)
            }
        
        def objective(params):
            """Objective function for optimization."""
            sentiment, uwr, range_val, volume = params
            
            # Enforce triangular constraint
            if abs(sentiment) + uwr > 1.0:
                return 1e10  # Large penalty
            
            candle = CandleMetric(
                range_value=range_val,
                low_value=100,  # Fixed reference
                sentiment=sentiment,
                upper_wick_ratio=uwr,
                volume=volume
            )
            
            state = MarketState(
                candle=candle,
                velocity=np.zeros(2),
                acceleration=np.zeros(2)
            )
            
            return self.hamiltonian.total_energy(state)
        
        # Parameter bounds for scipy
        scipy_bounds = [
            bounds['sentiment'],
            bounds['upper_wick_ratio'],
            bounds['range_value'],
            bounds['volume']
        ]
        
        # Global optimization
        best_result = None
        best_energy = float('inf')
        
        for _ in range(self.n_global_searches):
            result = differential_evolution(
                objective,
                scipy_bounds,
                maxiter=1000,
                tol=self.energy_tolerance,
                seed=np.random.randint(0, 10000)
            )
            
            if result.success and result.fun < best_energy:
                best_energy = result.fun
                best_result = result
        
        if best_result is None:
            raise RuntimeError("Failed to find global minimum")
        
        # Create equilibrium state
        sentiment, uwr, range_val, volume = best_result.x
        
        ground_state_candle = CandleMetric(
            range_value=range_val,
            low_value=100,
            sentiment=sentiment,
            upper_wick_ratio=uwr,
            volume=volume
        )
        
        # Analyze stability
        stability = self._analyze_stability(ground_state_candle)
        basin_size = self._estimate_basin_size(ground_state_candle)
        
        return EquilibriumState(
            candle=ground_state_candle,
            energy=best_energy,
            state_type='global',
            stability=stability,
            basin_size=basin_size,
            lifetime_estimate=float('inf')  # Ground state is permanent
        )
    
    def find_all_equilibria(self, 
                          bounds: Optional[Dict] = None,
                          reference_candles: Optional[List[CandleMetric]] = None) -> List[EquilibriumState]:
        """
        Find all equilibrium states (global and local minima).
        
        Args:
            bounds: Parameter bounds
            reference_candles: Starting points for local searches
            
        Returns:
            List of all equilibrium states
        """
        if bounds is None:
            bounds = {
                'sentiment': (-0.99, 0.99),
                'upper_wick_ratio': (0.01, 0.99),
                'range_value': (0.01, 10.0),
                'volume': (1.0, 10000.0)
            }
        
        equilibria = []
        
        # Find global minimum
        try:
            global_min = self.find_global_minimum(bounds=bounds)
            equilibria.append(global_min)
        except RuntimeError:
            pass
        
        # Find local minima using multiple starting points
        if reference_candles is None:
            # Generate random starting points
            reference_candles = self._generate_reference_candles(
                self.n_local_searches, bounds
            )
        
        for ref_candle in reference_candles:
            try:
                local_min = self._find_local_minimum(ref_candle, bounds)
                if local_min is not None:
                    equilibria.append(local_min)
            except:
                continue
        
        # Remove duplicates and cluster similar states
        equilibria = self._cluster_equilibria(equilibria)
        
        # Analyze transitions between states
        equilibria = self._analyze_state_transitions(equilibria)
        
        self._equilibrium_states = equilibria
        return equilibria
    
    def _find_local_minimum(self, 
                           initial_candle: CandleMetric,
                           bounds: Dict) -> Optional[EquilibriumState]:
        """Find local minimum starting from given candle."""
        
        def objective(params):
            sentiment, uwr, range_val, volume = params
            
            # Enforce constraints
            if abs(sentiment) + uwr > 1.0:
                return 1e10
            
            candle = CandleMetric(
                range_value=range_val,
                low_value=100,
                sentiment=sentiment,
                upper_wick_ratio=uwr,
                volume=volume
            )
            
            state = MarketState(
                candle=candle,
                velocity=np.zeros(2),
                acceleration=np.zeros(2)
            )
            
            return self.hamiltonian.total_energy(state)
        
        # Starting point
        x0 = [
            initial_candle.sentiment,
            initial_candle.upper_wick_ratio,
            initial_candle.range_value,
            initial_candle.volume
        ]
        
        # Bounds
        scipy_bounds = [
            bounds['sentiment'],
            bounds['upper_wick_ratio'],
            bounds['range_value'],
            bounds['volume']
        ]
        
        # Local optimization
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=scipy_bounds,
            options={'ftol': self.energy_tolerance, 'gtol': self.gradient_tolerance}
        )
        
        if not result.success:
            return None
        
        # Create equilibrium state
        sentiment, uwr, range_val, volume = result.x
        
        equilibrium_candle = CandleMetric(
            range_value=range_val,
            low_value=100,
            sentiment=sentiment,
            upper_wick_ratio=uwr,
            volume=volume
        )
        
        # Analyze properties
        stability = self._analyze_stability(equilibrium_candle)
        basin_size = self._estimate_basin_size(equilibrium_candle)
        lifetime = self._estimate_lifetime(equilibrium_candle, stability)
        
        # Determine if this is global or local minimum
        state_type = 'local'  # Will be updated by clustering if it's actually global
        
        return EquilibriumState(
            candle=equilibrium_candle,
            energy=result.fun,
            state_type=state_type,
            stability=stability,
            basin_size=basin_size,
            lifetime_estimate=lifetime
        )
    
    def _analyze_stability(self, candle: CandleMetric) -> float:
        """
        Analyze stability of equilibrium via Hessian eigenvalues.
        
        Returns:
            Stability measure (positive = stable, negative = unstable)
        """
        # Compute numerical Hessian
        h = 1e-6  # Step size
        
        def energy_func(params):
            sentiment, uwr = params
            test_candle = CandleMetric(
                range_value=candle.range_value,
                low_value=candle.low_value,
                sentiment=sentiment,
                upper_wick_ratio=uwr,
                volume=candle.volume
            )
            
            state = MarketState(
                candle=test_candle,
                velocity=np.zeros(2),
                acceleration=np.zeros(2)
            )
            
            return self.hamiltonian.total_energy(state)
        
        # Current point
        x0 = np.array([candle.sentiment, candle.upper_wick_ratio])
        
        # Compute Hessian numerically
        hessian = np.zeros((2, 2))
        
        for i in range(2):
            for j in range(2):
                # Second partial derivative
                x_pp = x0.copy()
                x_pm = x0.copy()
                x_mp = x0.copy()  
                x_mm = x0.copy()
                
                x_pp[i] += h
                x_pp[j] += h
                
                x_pm[i] += h
                x_pm[j] -= h
                
                x_mp[i] -= h
                x_mp[j] += h
                
                x_mm[i] -= h
                x_mm[j] -= h
                
                # Check bounds
                def check_bounds(x):
                    if x[0] < -0.99 or x[0] > 0.99:
                        return False
                    if x[1] < 0.01 or x[1] > 0.99:
                        return False
                    if abs(x[0]) + x[1] > 1.0:
                        return False
                    return True
                
                if all(check_bounds(x) for x in [x_pp, x_pm, x_mp, x_mm]):
                    f_pp = energy_func(x_pp)
                    f_pm = energy_func(x_pm)
                    f_mp = energy_func(x_mp)
                    f_mm = energy_func(x_mm)
                    
                    hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
        
        # Analyze eigenvalues
        try:
            eigenvals = np.linalg.eigvals(hessian)
            
            # Stability measure: minimum eigenvalue
            # Positive = stable (local minimum)
            # Negative = unstable (saddle point)
            stability = np.min(eigenvals)
            
            return float(stability)
        except:
            return 0.0  # Neutral if computation fails
    
    def _estimate_basin_size(self, candle: CandleMetric) -> float:
        """
        Estimate the size of the attraction basin around this equilibrium.
        
        Uses random sampling to find the region from which optimization
        converges to this equilibrium.
        """
        n_samples = 100
        convergence_count = 0
        
        reference_energy = self.hamiltonian.total_energy(MarketState(
            candle=candle,
            velocity=np.zeros(2),
            acceleration=np.zeros(2)
        ))
        
        # Sample points in a sphere around the equilibrium
        max_radius = 0.5  # Maximum radius to test
        
        for _ in range(n_samples):
            # Random point in pattern space
            radius = np.random.uniform(0.01, max_radius)
            angle = np.random.uniform(0, 2*np.pi)
            
            # Offset from equilibrium
            delta_sentiment = radius * np.cos(angle)
            delta_uwr = radius * np.sin(angle)
            
            # New point
            new_sentiment = candle.sentiment + delta_sentiment
            new_uwr = candle.upper_wick_ratio + delta_uwr
            
            # Check constraints
            if (abs(new_sentiment) <= 0.99 and 
                0.01 <= new_uwr <= 0.99 and
                abs(new_sentiment) + new_uwr <= 1.0):
                
                test_candle = CandleMetric(
                    range_value=candle.range_value,
                    low_value=candle.low_value,
                    sentiment=new_sentiment,
                    upper_wick_ratio=new_uwr,
                    volume=candle.volume
                )
                
                # Try to optimize from this point
                try:
                    optimized = self._find_local_minimum(test_candle, {
                        'sentiment': (-0.99, 0.99),
                        'upper_wick_ratio': (0.01, 0.99),
                        'range_value': (0.01, 10.0),
                        'volume': (1.0, 10000.0)
                    })
                    
                    if optimized is not None:
                        # Check if it converged to the same equilibrium
                        distance = np.linalg.norm([
                            optimized.candle.sentiment - candle.sentiment,
                            optimized.candle.upper_wick_ratio - candle.upper_wick_ratio
                        ])
                        
                        if distance < self.cluster_threshold:
                            convergence_count += 1
                
                except:
                    pass
        
        # Basin size estimate
        basin_size = convergence_count / n_samples
        return basin_size
    
    def _estimate_lifetime(self, candle: CandleMetric, stability: float) -> float:
        """
        Estimate how long the system stays in this equilibrium.
        
        Based on stability and thermal noise considerations.
        """
        if stability <= 0:
            return 0.1  # Unstable states decay quickly
        
        # Arrhenius-like formula: lifetime ∝ exp(barrier_height / temperature)
        # Higher stability = higher barriers = longer lifetime
        
        # Estimate "temperature" from market volatility
        temperature = candle.range_value * 0.1  # Heuristic
        
        if temperature > 0:
            lifetime = np.exp(stability / temperature)
            # Cap at reasonable values
            lifetime = min(lifetime, 1000.0)  # Max 1000 time units
        else:
            lifetime = 1000.0  # Cold market = long lifetimes
        
        return lifetime
    
    def _generate_reference_candles(self, n_candles: int, bounds: Dict) -> List[CandleMetric]:
        """Generate random reference candles for local optimization."""
        candles = []
        
        for _ in range(n_candles):
            # Random parameters within bounds
            sentiment = np.random.uniform(*bounds['sentiment'])
            uwr = np.random.uniform(*bounds['upper_wick_ratio'])
            range_val = np.random.uniform(*bounds['range_value'])
            volume = np.random.uniform(*bounds['volume'])
            
            # Enforce triangular constraint
            if abs(sentiment) + uwr > 1.0:
                # Scale to boundary
                scale = 0.99 / (abs(sentiment) + uwr)
                sentiment *= scale
                uwr *= scale
            
            candle = CandleMetric(
                range_value=range_val,
                low_value=100,
                sentiment=sentiment,
                upper_wick_ratio=uwr,
                volume=volume
            )
            
            candles.append(candle)
        
        return candles
    
    def _cluster_equilibria(self, equilibria: List[EquilibriumState]) -> List[EquilibriumState]:
        """Remove duplicate equilibria that are too close together."""
        if len(equilibria) <= 1:
            return equilibria
        
        # Extract coordinates for clustering
        coords = np.array([
            [eq.candle.sentiment, eq.candle.upper_wick_ratio] 
            for eq in equilibria
        ])
        
        # Compute pairwise distances
        distances = cdist(coords, coords)
        
        # Greedy clustering
        clustered = []
        used = set()
        
        for i, eq in enumerate(equilibria):
            if i in used:
                continue
            
            # Find all equilibria close to this one
            close_indices = np.where(distances[i] < self.cluster_threshold)[0]
            close_equilibria = [equilibria[j] for j in close_indices if j not in used]
            
            if close_equilibria:
                # Keep the one with lowest energy
                best_eq = min(close_equilibria, key=lambda x: x.energy)
                clustered.append(best_eq)
                
                # Mark all as used
                for j in close_indices:
                    used.add(j)
        
        # Update state types - lowest energy is global
        if clustered:
            global_eq = min(clustered, key=lambda x: x.energy)
            global_eq.state_type = 'global'
            
            for eq in clustered:
                if eq != global_eq:
                    eq.state_type = 'local'
        
        return clustered
    
    def _analyze_state_transitions(self, equilibria: List[EquilibriumState]) -> List[EquilibriumState]:
        """
        Analyze transition barriers between equilibrium states.
        """
        n_states = len(equilibria)
        
        for i, eq_i in enumerate(equilibria):
            eq_i.neighboring_states = []
            eq_i.transition_barriers = {}
            
            for j, eq_j in enumerate(equilibria):
                if i == j:
                    continue
                
                # Estimate transition barrier
                barrier = self._estimate_transition_barrier(eq_i.candle, eq_j.candle)
                
                eq_i.transition_barriers[j] = barrier
                
                # Consider as neighbor if barrier is reasonable
                if barrier < 10.0:  # Threshold for "reachable" states
                    eq_i.neighboring_states.append(eq_j)
        
        return equilibria
    
    def _estimate_transition_barrier(self, 
                                   candle1: CandleMetric, 
                                   candle2: CandleMetric) -> float:
        """
        Estimate energy barrier between two equilibrium states.
        
        Uses the nudged elastic band method approximation.
        """
        # Simple approximation: sample along straight line path
        n_points = 20
        max_energy = float('-inf')
        
        for i in range(n_points + 1):
            t = i / n_points
            
            # Interpolate between states
            sentiment = (1-t) * candle1.sentiment + t * candle2.sentiment
            uwr = (1-t) * candle1.upper_wick_ratio + t * candle2.upper_wick_ratio
            range_val = (1-t) * candle1.range_value + t * candle2.range_value
            volume = (1-t) * candle1.volume + t * candle2.volume
            
            # Check constraints
            if abs(sentiment) + uwr > 1.0:
                continue
            
            test_candle = CandleMetric(
                range_value=range_val,
                low_value=100,
                sentiment=sentiment,
                upper_wick_ratio=uwr,
                volume=volume
            )
            
            state = MarketState(
                candle=test_candle,
                velocity=np.zeros(2),
                acceleration=np.zeros(2)
            )
            
            energy = self.hamiltonian.total_energy(state)
            max_energy = max(max_energy, energy)
        
        # Barrier height relative to starting state
        start_energy = self.hamiltonian.total_energy(MarketState(
            candle=candle1,
            velocity=np.zeros(2),
            acceleration=np.zeros(2)
        ))
        
        barrier = max_energy - start_energy
        return max(0.0, barrier)  # Barriers are non-negative


def analyze_equilibrium_landscape(ohlc_data: pd.DataFrame,
                                hamiltonian: Optional[MarketHamiltonian] = None) -> Dict:
    """
    Analyze the equilibrium landscape for given market data.
    
    Returns comprehensive analysis including:
    - All equilibrium states
    - State transition network
    - Stability analysis
    - Lifetime predictions
    """
    # Use market data to inform bounds and starting points
    from curved_candle_geometry import create_candle_metrics_from_ohlc
    candles = create_candle_metrics_from_ohlc(ohlc_data)
    
    # Determine reasonable bounds from data
    sentiments = [c.sentiment for c in candles]
    uwrs = [c.upper_wick_ratio for c in candles]
    ranges = [c.range_value for c in candles]
    volumes = [c.volume for c in candles]
    
    bounds = {
        'sentiment': (min(sentiments) - 0.1, max(sentiments) + 0.1),
        'upper_wick_ratio': (max(0.01, min(uwrs) - 0.1), min(0.99, max(uwrs) + 0.1)),
        'range_value': (max(0.01, min(ranges) * 0.5), max(ranges) * 2.0),
        'volume': (max(1.0, min(volumes) * 0.5), max(volumes) * 2.0)
    }
    
    # Create finder and analyze
    finder = EquilibriumStateFinder(hamiltonian=hamiltonian)
    equilibria = finder.find_all_equilibria(bounds=bounds, reference_candles=candles[:10])
    
    # Additional analysis
    analysis = {
        'equilibria': equilibria,
        'n_equilibria': len(equilibria),
        'global_minimum': min(equilibria, key=lambda x: x.energy) if equilibria else None,
        'stable_states': [eq for eq in equilibria if eq.stability > 0],
        'unstable_states': [eq for eq in equilibria if eq.stability <= 0],
        'long_lived_states': [eq for eq in equilibria if eq.lifetime_estimate > 100],
        'transition_network': {},
        'bounds_used': bounds
    }
    
    # Build transition network
    for i, eq in enumerate(equilibria):
        analysis['transition_network'][i] = {
            'state': eq,
            'neighbors': eq.neighboring_states,
            'barriers': eq.transition_barriers
        }
    
    return analysis