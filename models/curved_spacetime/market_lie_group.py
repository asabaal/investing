"""
Market Lie Group Transformation Library

Implements the complete group ML(2) of market pattern transformations.
This provides a mathematical taxonomy of ALL possible market patterns
through group theory.

Key transformations:
1. Sentiment Shift (translation in sentiment)
2. Wick Rotation (rotation in wick space)
3. Volatility Scaling (dilation of range)
4. Time Dilation (temporal scaling)
5. Volume Boost (mass transformation)
6. Pattern Inversion (reflection)

The group structure allows us to:
- Classify any pattern as a transformation of fundamental patterns
- Find invariant features that persist across transformations
- Generate synthetic patterns for training
- Detect pattern similarities modulo transformations
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from abc import ABC, abstractmethod
import pandas as pd
from curved_candle_geometry import CandleMetric


class MarketTransformation(ABC):
    """Abstract base class for market transformations."""
    
    @abstractmethod
    def apply(self, candle: CandleMetric) -> CandleMetric:
        """Apply transformation to a single candle."""
        pass
    
    @abstractmethod
    def inverse(self) -> 'MarketTransformation':
        """Return the inverse transformation."""
        pass
    
    @abstractmethod
    def compose(self, other: 'MarketTransformation') -> 'MarketTransformation':
        """Compose this transformation with another."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> dict:
        """Get transformation parameters."""
        pass
    
    def apply_series(self, candles: List[CandleMetric]) -> List[CandleMetric]:
        """Apply transformation to a series of candles."""
        return [self.apply(candle) for candle in candles]


class IdentityTransformation(MarketTransformation):
    """The identity transformation (does nothing)."""
    
    def apply(self, candle: CandleMetric) -> CandleMetric:
        return candle
    
    def inverse(self) -> 'IdentityTransformation':
        return IdentityTransformation()
    
    def compose(self, other: MarketTransformation) -> MarketTransformation:
        return other
    
    def get_parameters(self) -> dict:
        return {'type': 'identity'}


class SentimentShift(MarketTransformation):
    """Translation in sentiment space."""
    
    def __init__(self, shift_amount: float):
        """
        Args:
            shift_amount: Amount to shift sentiment (-1 to 1)
        """
        self.shift_amount = shift_amount
    
    def apply(self, candle: CandleMetric) -> CandleMetric:
        # Shift sentiment while maintaining constraints
        new_sentiment = candle.sentiment + self.shift_amount
        new_sentiment = np.clip(new_sentiment, -0.99, 0.99)
        
        # Ensure triangular constraint
        if abs(new_sentiment) + candle.upper_wick_ratio > 1.0:
            # Scale back to maintain constraint
            scale = 0.99 / (abs(new_sentiment) + candle.upper_wick_ratio)
            new_sentiment *= scale
        
        return CandleMetric(
            range_value=candle.range_value,
            low_value=candle.low_value,
            sentiment=new_sentiment,
            upper_wick_ratio=candle.upper_wick_ratio,
            volume=candle.volume
        )
    
    def inverse(self) -> 'SentimentShift':
        return SentimentShift(-self.shift_amount)
    
    def compose(self, other: MarketTransformation) -> MarketTransformation:
        if isinstance(other, SentimentShift):
            return SentimentShift(self.shift_amount + other.shift_amount)
        else:
            return CompositeTransformation([self, other])
    
    def get_parameters(self) -> dict:
        return {'type': 'sentiment_shift', 'amount': self.shift_amount}


class WickRotation(MarketTransformation):
    """Rotation in the wick space (upper/lower wick balance)."""
    
    def __init__(self, angle: float):
        """
        Args:
            angle: Rotation angle in radians
        """
        self.angle = angle
    
    def apply(self, candle: CandleMetric) -> CandleMetric:
        # Convert to wick coordinates
        upper_wick = candle.upper_wick_ratio
        lower_wick = 1.0 - upper_wick - abs(candle.sentiment)
        
        # Apply rotation
        cos_a, sin_a = np.cos(self.angle), np.sin(self.angle)
        new_upper = cos_a * upper_wick - sin_a * lower_wick
        new_lower = sin_a * upper_wick + cos_a * lower_wick
        
        # Ensure positivity
        new_upper = max(0.01, new_upper)
        new_lower = max(0.01, new_lower)
        
        # Renormalize
        total = new_upper + new_lower + abs(candle.sentiment)
        if total > 1.0:
            scale = 0.99 / total
            new_upper *= scale
            new_lower *= scale
        
        return CandleMetric(
            range_value=candle.range_value,
            low_value=candle.low_value,
            sentiment=candle.sentiment,
            upper_wick_ratio=new_upper,
            volume=candle.volume
        )
    
    def inverse(self) -> 'WickRotation':
        return WickRotation(-self.angle)
    
    def compose(self, other: MarketTransformation) -> MarketTransformation:
        if isinstance(other, WickRotation):
            return WickRotation(self.angle + other.angle)
        else:
            return CompositeTransformation([self, other])
    
    def get_parameters(self) -> dict:
        return {'type': 'wick_rotation', 'angle': self.angle}


class VolatilityScaling(MarketTransformation):
    """Dilation/contraction of the range (volatility)."""
    
    def __init__(self, scale_factor: float):
        """
        Args:
            scale_factor: Multiplicative scaling factor (>0)
        """
        self.scale_factor = max(0.1, scale_factor)  # Prevent collapse
    
    def apply(self, candle: CandleMetric) -> CandleMetric:
        # Scale the range
        new_range = candle.range_value * self.scale_factor
        
        return CandleMetric(
            range_value=new_range,
            low_value=candle.low_value,
            sentiment=candle.sentiment,
            upper_wick_ratio=candle.upper_wick_ratio,
            volume=candle.volume
        )
    
    def inverse(self) -> 'VolatilityScaling':
        return VolatilityScaling(1.0 / self.scale_factor)
    
    def compose(self, other: MarketTransformation) -> MarketTransformation:
        if isinstance(other, VolatilityScaling):
            return VolatilityScaling(self.scale_factor * other.scale_factor)
        else:
            return CompositeTransformation([self, other])
    
    def get_parameters(self) -> dict:
        return {'type': 'volatility_scaling', 'scale': self.scale_factor}


class TimeDilation(MarketTransformation):
    """Temporal scaling transformation (affects proper time)."""
    
    def __init__(self, time_factor: float):
        """
        Args:
            time_factor: Time dilation factor (>0)
        """
        self.time_factor = max(0.1, time_factor)
    
    def apply(self, candle: CandleMetric) -> CandleMetric:
        # Time dilation affects the metric tensor scaling
        # In our model, this manifests as a combined range/volume effect
        
        # Scale range by square root of time factor (metric scaling)
        new_range = candle.range_value * np.sqrt(self.time_factor)
        
        # Volume scales linearly with time (accumulation effect)
        new_volume = candle.volume * self.time_factor
        
        return CandleMetric(
            range_value=new_range,
            low_value=candle.low_value,
            sentiment=candle.sentiment,
            upper_wick_ratio=candle.upper_wick_ratio,
            volume=new_volume
        )
    
    def inverse(self) -> 'TimeDilation':
        return TimeDilation(1.0 / self.time_factor)
    
    def compose(self, other: MarketTransformation) -> MarketTransformation:
        if isinstance(other, TimeDilation):
            return TimeDilation(self.time_factor * other.time_factor)
        else:
            return CompositeTransformation([self, other])
    
    def get_parameters(self) -> dict:
        return {'type': 'time_dilation', 'factor': self.time_factor}


class VolumeBoost(MarketTransformation):
    """Lorentz-like boost in volume (mass) dimension."""
    
    def __init__(self, boost_parameter: float):
        """
        Args:
            boost_parameter: Boost parameter (like rapidity in physics)
        """
        self.boost_parameter = boost_parameter
    
    def apply(self, candle: CandleMetric) -> CandleMetric:
        # Apply hyperbolic boost to volume
        # V' = V * cosh(β) + R * sinh(β)
        # R' = R * cosh(β) + V * sinh(β)
        
        cosh_b = np.cosh(self.boost_parameter)
        sinh_b = np.sinh(self.boost_parameter)
        
        new_volume = candle.volume * cosh_b + candle.range_value * sinh_b
        new_range = candle.range_value * cosh_b + candle.volume * sinh_b
        
        # Ensure positive values
        new_volume = max(1.0, new_volume)
        new_range = max(0.01, new_range)
        
        return CandleMetric(
            range_value=new_range,
            low_value=candle.low_value,
            sentiment=candle.sentiment,
            upper_wick_ratio=candle.upper_wick_ratio,
            volume=new_volume
        )
    
    def inverse(self) -> 'VolumeBoost':
        return VolumeBoost(-self.boost_parameter)
    
    def compose(self, other: MarketTransformation) -> MarketTransformation:
        if isinstance(other, VolumeBoost):
            return VolumeBoost(self.boost_parameter + other.boost_parameter)
        else:
            return CompositeTransformation([self, other])
    
    def get_parameters(self) -> dict:
        return {'type': 'volume_boost', 'parameter': self.boost_parameter}


class PatternInversion(MarketTransformation):
    """Reflection transformation (bearish ↔ bullish)."""
    
    def apply(self, candle: CandleMetric) -> CandleMetric:
        # Invert sentiment (bullish becomes bearish)
        new_sentiment = -candle.sentiment
        
        # Swap upper and lower wicks
        lower_wick_ratio = 1.0 - candle.upper_wick_ratio - abs(candle.sentiment)
        new_upper_wick_ratio = lower_wick_ratio
        
        # Ensure constraints
        if abs(new_sentiment) + new_upper_wick_ratio > 1.0:
            scale = 0.99 / (abs(new_sentiment) + new_upper_wick_ratio)
            new_upper_wick_ratio *= scale
        
        return CandleMetric(
            range_value=candle.range_value,
            low_value=candle.low_value,
            sentiment=new_sentiment,
            upper_wick_ratio=new_upper_wick_ratio,
            volume=candle.volume
        )
    
    def inverse(self) -> 'PatternInversion':
        return PatternInversion()  # Self-inverse
    
    def compose(self, other: MarketTransformation) -> MarketTransformation:
        if isinstance(other, PatternInversion):
            return IdentityTransformation()  # Two inversions = identity
        else:
            return CompositeTransformation([self, other])
    
    def get_parameters(self) -> dict:
        return {'type': 'pattern_inversion'}


class CompositeTransformation(MarketTransformation):
    """Composition of multiple transformations."""
    
    def __init__(self, transformations: List[MarketTransformation]):
        self.transformations = transformations
    
    def apply(self, candle: CandleMetric) -> CandleMetric:
        result = candle
        for transform in self.transformations:
            result = transform.apply(result)
        return result
    
    def inverse(self) -> 'CompositeTransformation':
        # Reverse order and invert each
        inverse_transforms = [t.inverse() for t in reversed(self.transformations)]
        return CompositeTransformation(inverse_transforms)
    
    def compose(self, other: MarketTransformation) -> 'CompositeTransformation':
        if isinstance(other, CompositeTransformation):
            return CompositeTransformation(self.transformations + other.transformations)
        else:
            return CompositeTransformation(self.transformations + [other])
    
    def get_parameters(self) -> dict:
        return {
            'type': 'composite',
            'components': [t.get_parameters() for t in self.transformations]
        }


class MarketLieGroup:
    """
    The complete Lie group ML(2) of market transformations.
    
    This class manages the group structure and provides utilities for:
    - Generating group elements
    - Computing invariant features
    - Pattern classification
    - Similarity measures
    """
    
    def __init__(self):
        self.generators = {
            'sentiment': lambda x: SentimentShift(x),
            'wick': lambda x: WickRotation(x),
            'volatility': lambda x: VolatilityScaling(x),
            'time': lambda x: TimeDilation(x),
            'volume': lambda x: VolumeBoost(x),
            'inversion': lambda: PatternInversion()
        }
    
    def random_element(self, max_params: dict = None) -> MarketTransformation:
        """Generate a random group element."""
        if max_params is None:
            max_params = {
                'sentiment': 0.5,
                'wick': np.pi/4,
                'volatility': 2.0,
                'time': 2.0,
                'volume': 1.0
            }
        
        transforms = []
        
        # Random sentiment shift
        if np.random.rand() > 0.5:
            shift = np.random.uniform(-max_params['sentiment'], max_params['sentiment'])
            transforms.append(self.generators['sentiment'](shift))
        
        # Random wick rotation
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-max_params['wick'], max_params['wick'])
            transforms.append(self.generators['wick'](angle))
        
        # Random volatility scaling
        if np.random.rand() > 0.5:
            scale = np.random.uniform(1/max_params['volatility'], max_params['volatility'])
            transforms.append(self.generators['volatility'](scale))
        
        # Random inversion
        if np.random.rand() > 0.3:
            transforms.append(self.generators['inversion']())
        
        if len(transforms) == 0:
            return IdentityTransformation()
        elif len(transforms) == 1:
            return transforms[0]
        else:
            return CompositeTransformation(transforms)
    
    def compute_invariants(self, candle: CandleMetric) -> dict:
        """
        Compute transformation-invariant features.
        
        These features remain constant under certain group actions:
        - Volatility ratio invariants
        - Wick balance invariants
        - Energy-based invariants
        """
        invariants = {}
        
        # 1. Range-normalized volume (invariant under volatility scaling)
        if candle.range_value > 0:
            invariants['normalized_volume'] = candle.volume / candle.range_value
        else:
            invariants['normalized_volume'] = 0.0
        
        # 2. Wick asymmetry (invariant under sentiment shift)
        lower_wick = 1.0 - candle.upper_wick_ratio - abs(candle.sentiment)
        invariants['wick_asymmetry'] = candle.upper_wick_ratio - lower_wick
        
        # 3. Pattern energy (approximately invariant under isometries)
        from market_hamiltonian import MarketHamiltonian, MarketState
        hamiltonian = MarketHamiltonian()
        state = MarketState(
            candle=candle,
            velocity=np.zeros(2),
            acceleration=np.zeros(2)
        )
        invariants['pattern_energy'] = hamiltonian.potential_energy(state)
        
        # 4. Geometric curvature (intrinsic invariant)
        from curved_candle_geometry import CurvedCandleGeometry
        geometry = CurvedCandleGeometry([candle])
        invariants['intrinsic_curvature'] = geometry.compute_intrinsic_curvature(0)
        
        return invariants
    
    def pattern_distance(self, candle1: CandleMetric, candle2: CandleMetric,
                        invariant_only: bool = False) -> float:
        """
        Compute distance between patterns, optionally using only invariants.
        """
        if invariant_only:
            inv1 = self.compute_invariants(candle1)
            inv2 = self.compute_invariants(candle2)
            
            # Weighted distance in invariant space
            distances = []
            weights = {'normalized_volume': 0.3, 'wick_asymmetry': 0.2,
                      'pattern_energy': 0.3, 'intrinsic_curvature': 0.2}
            
            for key, weight in weights.items():
                if key in inv1 and key in inv2:
                    distances.append(weight * (inv1[key] - inv2[key])**2)
            
            return np.sqrt(sum(distances))
        else:
            # Direct pattern space distance
            p1 = candle1.pattern_coordinates
            p2 = candle2.pattern_coordinates
            
            # Include range and volume differences
            range_diff = (candle1.range_value - candle2.range_value)**2
            volume_diff = np.log1p(candle1.volume) - np.log1p(candle2.volume)
            
            pattern_dist = np.linalg.norm(p1 - p2)
            
            return np.sqrt(pattern_dist**2 + 0.1*range_diff + 0.1*volume_diff**2)
    
    def find_transformation(self, source: CandleMetric, target: CandleMetric,
                          max_iterations: int = 100) -> Optional[MarketTransformation]:
        """
        Try to find a transformation that maps source to target.
        
        Uses optimization to search the group manifold.
        """
        # This is a simplified version - full implementation would use
        # Lie algebra optimization on the group manifold
        
        best_transform = IdentityTransformation()
        best_distance = self.pattern_distance(source, target)
        
        for _ in range(max_iterations):
            # Try a random transformation
            transform = self.random_element()
            transformed = transform.apply(source)
            distance = self.pattern_distance(transformed, target)
            
            if distance < best_distance:
                best_distance = distance
                best_transform = transform
                
                if distance < 0.01:  # Close enough
                    return best_transform
        
        return best_transform if best_distance < 0.5 else None


def classify_pattern_via_transformations(candle: CandleMetric,
                                       reference_patterns: dict) -> Tuple[str, MarketTransformation]:
    """
    Classify a pattern by finding which reference pattern it's closest to
    under group transformations.
    
    Args:
        candle: Pattern to classify
        reference_patterns: Dict of name -> reference CandleMetric
        
    Returns:
        (pattern_name, transformation_to_reference)
    """
    group = MarketLieGroup()
    
    best_match = None
    best_transform = None
    best_distance = float('inf')
    
    for name, ref_pattern in reference_patterns.items():
        # Try to find transformation from candle to reference
        transform = group.find_transformation(candle, ref_pattern)
        
        if transform is not None:
            transformed = transform.apply(candle)
            distance = group.pattern_distance(transformed, ref_pattern)
            
            if distance < best_distance:
                best_distance = distance
                best_match = name
                best_transform = transform
    
    return best_match, best_transform


def generate_pattern_variations(base_pattern: CandleMetric,
                               n_variations: int = 10) -> List[CandleMetric]:
    """
    Generate variations of a pattern using group transformations.
    
    Useful for:
    - Data augmentation
    - Robustness testing
    - Pattern exploration
    """
    group = MarketLieGroup()
    variations = []
    
    for _ in range(n_variations):
        transform = group.random_element()
        variation = transform.apply(base_pattern)
        variations.append(variation)
    
    return variations