"""
Market Hamiltonian Energy System

Implements the complete energy function H = T + V + I for financial markets where:
- T: Kinetic energy from price velocity and volume
- V: Potential energy from sentiment strain and wick tension
- I: Interaction energy from pattern correlations

The Hamiltonian allows us to:
1. Calculate total market energy states
2. Find natural equilibrium points (energy minima)
3. Detect interventions (unnatural energy injections)
4. Predict energy-driven price movements
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import pandas as pd
from curved_candle_geometry import CandleMetric, CurvedCandleGeometry


@dataclass
class MarketState:
    """Complete state of the market at a given time"""
    candle: CandleMetric
    velocity: np.ndarray  # Price velocity in pattern space (sentiment, UWR)
    acceleration: np.ndarray  # Price acceleration
    neighboring_states: Optional[List['MarketState']] = None  # For interaction energy
    

class MarketHamiltonian:
    """
    Complete energy function H = T + V + I for market dynamics.
    
    The Hamiltonian encodes all energy contributions:
    - Kinetic: Motion energy from price changes and volume
    - Potential: Stored energy in market tensions and strains
    - Interaction: Energy from correlations with nearby patterns
    """
    
    def __init__(self, 
                 mass_scaling: float = 0.5,
                 wick_tension_strength: float = 1.0,
                 sentiment_strain_strength: float = 1.0,
                 interaction_range: int = 5):
        """
        Initialize the Hamiltonian with coupling constants.
        
        Args:
            mass_scaling: How volume affects kinetic energy (α parameter)
            wick_tension_strength: Coupling for wick potential energy
            sentiment_strain_strength: Coupling for sentiment potential energy
            interaction_range: Number of neighboring candles for interaction energy
        """
        self.mass_scaling = mass_scaling
        self.wick_tension_strength = wick_tension_strength
        self.sentiment_strain_strength = sentiment_strain_strength
        self.interaction_range = interaction_range
    
    def kinetic_energy(self, state: MarketState) -> float:
        """
        Calculate kinetic energy T = (1/2) * m * v²
        
        where:
        - m = volume^α (effective mass)
        - v = velocity in pattern space
        
        Physical interpretation:
        - High volume + high velocity = high kinetic energy
        - Markets with momentum carry more energy
        """
        # Effective mass from volume
        mass = np.power(state.candle.volume, self.mass_scaling)
        
        # Velocity magnitude in pattern space
        velocity_magnitude = np.linalg.norm(state.velocity)
        
        # Classical kinetic energy
        T = 0.5 * mass * velocity_magnitude**2
        
        return T
    
    def potential_energy(self, state: MarketState) -> float:
        """
        Calculate potential energy V = V_wick + V_sentiment + V_range
        
        Components:
        1. Wick tension: Energy stored in upper/lower wicks
        2. Sentiment strain: Energy from deviation from equilibrium
        3. Range potential: Energy from volatility compression/expansion
        
        Physical interpretation:
        - Extreme wicks store energy (like stretched springs)
        - Large sentiment deviations are high-energy states
        - Compressed/expanded ranges store potential energy
        """
        candle = state.candle
        
        # 1. Wick Tension Energy
        # Wicks represent unfulfilled price attempts - store energy
        # Balanced wicks (UWR ≈ 0.5) have minimum energy
        wick_deviation = abs(candle.upper_wick_ratio - 0.5)
        V_wick = self.wick_tension_strength * wick_deviation**2
        
        # 2. Sentiment Strain Energy
        # Extreme sentiment (near ±1) represents strained market state
        # Zero sentiment is the natural equilibrium
        V_sentiment = self.sentiment_strain_strength * candle.sentiment**2
        
        # 3. Range Potential Energy
        # High volatility (large range) stores energy
        # Use log scaling since range varies over orders of magnitude
        normalized_range = np.log1p(candle.range_value)
        V_range = 0.5 * normalized_range**2
        
        # Total potential energy
        V = V_wick + V_sentiment + V_range
        
        return V
    
    def interaction_energy(self, state: MarketState) -> float:
        """
        Calculate interaction energy I from pattern correlations.
        
        Energy contributions from:
        1. Pattern similarity with neighbors (cohesion)
        2. Volume-weighted interactions (massive objects interact more)
        3. Temporal decay (recent neighbors matter more)
        
        Physical interpretation:
        - Similar patterns attract (negative energy)
        - Dissimilar patterns repel (positive energy)
        - High-volume candles influence neighbors more
        """
        if not state.neighboring_states:
            return 0.0
        
        I_total = 0.0
        current_pattern = state.candle.pattern_coordinates
        
        for i, neighbor in enumerate(state.neighboring_states):
            if neighbor is None:
                continue
                
            # Pattern similarity (dot product in pattern space)
            neighbor_pattern = neighbor.candle.pattern_coordinates
            similarity = np.dot(current_pattern, neighbor_pattern)
            
            # Volume-weighted interaction strength
            # Massive objects interact more strongly
            mass_factor = np.sqrt(state.candle.volume * neighbor.candle.volume)
            mass_factor = np.power(mass_factor, 0.25)  # Moderate the effect
            
            # Temporal decay - recent neighbors matter more
            time_decay = np.exp(-i / 3.0)  # Decay constant of 3 time steps
            
            # Interaction energy contribution
            # Negative for attraction (similar patterns), positive for repulsion
            I_pair = -similarity * mass_factor * time_decay
            
            I_total += I_pair
        
        # Normalize by number of neighbors
        if len(state.neighboring_states) > 0:
            I_total /= len(state.neighboring_states)
        
        return I_total
    
    def total_energy(self, state: MarketState) -> float:
        """
        Calculate total Hamiltonian H = T + V + I
        """
        T = self.kinetic_energy(state)
        V = self.potential_energy(state)
        I = self.interaction_energy(state)
        
        H = T + V + I
        
        return H
    
    def energy_components(self, state: MarketState) -> Dict[str, float]:
        """
        Get detailed breakdown of all energy components.
        """
        T = self.kinetic_energy(state)
        V = self.potential_energy(state)
        I = self.interaction_energy(state)
        H = T + V + I
        
        # Also compute individual potential components
        candle = state.candle
        wick_deviation = abs(candle.upper_wick_ratio - 0.5)
        V_wick = self.wick_tension_strength * wick_deviation**2
        V_sentiment = self.sentiment_strain_strength * candle.sentiment**2
        normalized_range = np.log1p(candle.range_value)
        V_range = 0.5 * normalized_range**2
        
        return {
            'total': H,
            'kinetic': T,
            'potential': V,
            'interaction': I,
            'potential_wick': V_wick,
            'potential_sentiment': V_sentiment,
            'potential_range': V_range
        }
    
    def compute_energy_gradient(self, states: List[MarketState], index: int) -> np.ndarray:
        """
        Compute the energy gradient ∇H at a given state.
        
        Used for:
        1. Finding force direction F = -∇H
        2. Detecting energy flow direction
        3. Identifying intervention points (gradient anomalies)
        """
        if index < 1 or index >= len(states) - 1:
            return np.zeros(2)  # No gradient at boundaries
        
        # Finite difference approximation
        h = 0.01  # Small perturbation
        current_energy = self.total_energy(states[index])
        gradient = np.zeros(2)
        
        # Perturb in sentiment direction
        perturbed_state = MarketState(
            candle=CandleMetric(
                range_value=states[index].candle.range_value,
                low_value=states[index].candle.low_value,
                sentiment=states[index].candle.sentiment + h,
                upper_wick_ratio=states[index].candle.upper_wick_ratio,
                volume=states[index].candle.volume
            ),
            velocity=states[index].velocity,
            acceleration=states[index].acceleration,
            neighboring_states=states[index].neighboring_states
        )
        gradient[0] = (self.total_energy(perturbed_state) - current_energy) / h
        
        # Perturb in UWR direction
        perturbed_state = MarketState(
            candle=CandleMetric(
                range_value=states[index].candle.range_value,
                low_value=states[index].candle.low_value,
                sentiment=states[index].candle.sentiment,
                upper_wick_ratio=states[index].candle.upper_wick_ratio + h,
                volume=states[index].candle.volume
            ),
            velocity=states[index].velocity,
            acceleration=states[index].acceleration,
            neighboring_states=states[index].neighboring_states
        )
        gradient[1] = (self.total_energy(perturbed_state) - current_energy) / h
        
        return gradient
    
    def compute_force(self, states: List[MarketState], index: int) -> np.ndarray:
        """
        Compute the force F = -∇H acting on the market.
        
        This force drives the market toward lower energy states.
        """
        return -self.compute_energy_gradient(states, index)
    
    def find_local_minimum(self, 
                          initial_state: MarketState,
                          learning_rate: float = 0.01,
                          max_iterations: int = 1000,
                          tolerance: float = 1e-6) -> Tuple[MarketState, float]:
        """
        Find local energy minimum using gradient descent.
        
        Returns:
            Equilibrium state and its energy
        """
        current_state = initial_state
        current_energy = self.total_energy(current_state)
        
        for iteration in range(max_iterations):
            # Create temporary states list for gradient computation
            temp_states = [None, current_state, None]
            gradient = self.compute_energy_gradient(temp_states, 1)
            
            # Update state in direction of negative gradient
            new_sentiment = current_state.candle.sentiment - learning_rate * gradient[0]
            new_uwr = current_state.candle.upper_wick_ratio - learning_rate * gradient[1]
            
            # Enforce constraints
            new_sentiment = np.clip(new_sentiment, -0.99, 0.99)
            new_uwr = np.clip(new_uwr, 0.01, 0.99)
            
            # Ensure triangular constraint
            if abs(new_sentiment) + new_uwr > 1.0:
                total = abs(new_sentiment) + new_uwr
                new_sentiment = new_sentiment * 0.99 / total
                new_uwr = new_uwr * 0.99 / total
            
            # Create new state
            new_candle = CandleMetric(
                range_value=current_state.candle.range_value,
                low_value=current_state.candle.low_value,
                sentiment=new_sentiment,
                upper_wick_ratio=new_uwr,
                volume=current_state.candle.volume
            )
            
            new_state = MarketState(
                candle=new_candle,
                velocity=np.zeros(2),  # At equilibrium, velocity is zero
                acceleration=np.zeros(2),
                neighboring_states=current_state.neighboring_states
            )
            
            new_energy = self.total_energy(new_state)
            
            # Check convergence
            if abs(new_energy - current_energy) < tolerance:
                return new_state, new_energy
            
            current_state = new_state
            current_energy = new_energy
        
        return current_state, current_energy


def create_market_states_from_candles(candles: List[CandleMetric], 
                                    interaction_range: int = 5) -> List[MarketState]:
    """
    Convert candle metrics to market states with velocities and neighbors.
    """
    states = []
    n_candles = len(candles)
    
    for i in range(n_candles):
        # Compute velocity (finite difference)
        if i == 0:
            velocity = np.zeros(2)
        else:
            dp = candles[i].pattern_coordinates - candles[i-1].pattern_coordinates
            velocity = dp  # Assuming unit time steps
        
        # Compute acceleration
        if i < 2:
            acceleration = np.zeros(2)
        else:
            v_prev = candles[i-1].pattern_coordinates - candles[i-2].pattern_coordinates
            acceleration = velocity - v_prev
        
        # Get neighboring states for interaction energy
        neighbors = []
        for j in range(max(0, i - interaction_range), min(n_candles, i + interaction_range + 1)):
            if j != i and j < len(states):
                neighbors.append(states[j])
            elif j != i:
                neighbors.append(None)  # Placeholder for future states
        
        state = MarketState(
            candle=candles[i],
            velocity=velocity,
            acceleration=acceleration,
            neighboring_states=neighbors
        )
        
        states.append(state)
    
    # Second pass to update neighbors that weren't available in first pass
    for i in range(n_candles):
        neighbors = []
        for j in range(max(0, i - interaction_range), min(n_candles, i + interaction_range + 1)):
            if j != i:
                neighbors.append(states[j])
        states[i].neighboring_states = neighbors
    
    return states


def analyze_market_energy(ohlc_data: pd.DataFrame, 
                         hamiltonian: Optional[MarketHamiltonian] = None) -> Dict:
    """
    Perform complete energy analysis on market data.
    
    Returns dictionary with:
    - Energy time series
    - Energy components breakdown
    - High/low energy periods
    - Energy gradients
    """
    if hamiltonian is None:
        hamiltonian = MarketHamiltonian()
    
    # Create candle metrics
    from curved_candle_geometry import create_candle_metrics_from_ohlc
    candles = create_candle_metrics_from_ohlc(ohlc_data)
    
    # Convert to market states
    states = create_market_states_from_candles(candles)
    
    # Compute energy for each state
    energies = []
    energy_components = []
    gradients = []
    
    for i, state in enumerate(states):
        # Total energy
        H = hamiltonian.total_energy(state)
        energies.append(H)
        
        # Component breakdown
        components = hamiltonian.energy_components(state)
        energy_components.append(components)
        
        # Energy gradient
        gradient = hamiltonian.compute_energy_gradient(states, i)
        gradients.append(gradient)
    
    # Identify high/low energy periods
    energies_array = np.array(energies)
    mean_energy = np.mean(energies_array)
    std_energy = np.std(energies_array)
    
    high_energy_mask = energies_array > mean_energy + std_energy
    low_energy_mask = energies_array < mean_energy - std_energy
    
    return {
        'energies': energies,
        'energy_components': energy_components,
        'gradients': gradients,
        'mean_energy': mean_energy,
        'std_energy': std_energy,
        'high_energy_periods': np.where(high_energy_mask)[0],
        'low_energy_periods': np.where(low_energy_mask)[0],
        'states': states
    }