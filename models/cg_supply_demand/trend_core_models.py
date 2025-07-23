"""
Final fixed core data models for the Trend Detection Algorithm
Added missing 'candles' field to TrendAnalysisResult
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
from datetime import datetime


class TrendDirection(Enum):
    """Direction of market trend"""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


class SwingType(Enum):
    """Type of swing point in price action"""
    HIGH = "high"
    LOW = "low"


class TrendSignificance(Enum):
    """Significance level of detected trend"""
    MAJOR = "major"
    MINOR = "minor"
    CONSOLIDATION = "consolidation"


@dataclass(frozen=True)
class SwingPoint:
    """
    Immutable swing point in market structure.
    Responsibility: Hold swing point data only.
    """
    candle_index: int
    price: float
    swing_type: SwingType
    timestamp: datetime
    
    def __post_init__(self):
        """Validate swing point data"""
        if self.candle_index < 0:
            raise ValueError("Candle index must be non-negative")
        if self.price <= 0:
            raise ValueError("Price must be positive")
    
    def __repr__(self):
        return f"Swing{self.swing_type.value.upper()}({self.candle_index}:{self.price:.1f})"
    
    # Comparison operators - compare by price
    def __lt__(self, other):
        if not isinstance(other, SwingPoint):
            return NotImplemented
        return self.price < other.price
    
    def __le__(self, other):
        if not isinstance(other, SwingPoint):
            return NotImplemented
        return self.price <= other.price
    
    def __gt__(self, other):
        if not isinstance(other, SwingPoint):
            return NotImplemented
        return self.price > other.price
    
    def __ge__(self, other):
        if not isinstance(other, SwingPoint):
            return NotImplemented
        return self.price >= other.price
    
    def __eq__(self, other):
        if not isinstance(other, SwingPoint):
            return NotImplemented
        return (self.candle_index == other.candle_index and 
                self.price == other.price and 
                self.swing_type == other.swing_type)
    
    def __hash__(self):
        return hash((self.candle_index, self.price, self.swing_type))


@dataclass(frozen=True)
class TrendPattern:
    """
    Immutable representation of a detected trend pattern.
    Responsibility: Hold pattern formation data only.
    """
    formation_swings: Tuple[SwingPoint, ...]
    pattern_type: TrendDirection
    start_index: int
    end_index: int
    
    def __post_init__(self):
        """Validate pattern data"""
        if len(self.formation_swings) < 2:
            raise ValueError("Pattern must have at least 2 swings")
        if self.start_index >= self.end_index:
            raise ValueError("Start index must be less than end index")
        if self.formation_swings != tuple(sorted(self.formation_swings, key=lambda s: s.candle_index)):
            raise ValueError("Formation swings must be sorted by candle index")


@dataclass(frozen=True)
class Trend:
    """
    Immutable trend representation.
    Responsibility: Hold complete trend data only.
    """
    trend_id: int
    direction: TrendDirection
    start_index: int
    end_index: Optional[int]
    controlling_swing: Optional[SwingPoint]
    formation_pattern: TrendPattern
    significance: TrendSignificance = TrendSignificance.MINOR
    is_active: bool = True
    
    # Trend metrics
    price_range: float = 0.0
    duration: int = 0
    moveout_confirmed: bool = False
    breakout_confirmed: bool = False
    
    # Sideways-specific fields
    range_high: Optional[float] = None
    range_low: Optional[float] = None
    
    # Multi-timeframe fields
    genesis_point: Optional[SwingPoint] = None
    dominance_score: float = 0.0
    
    def __post_init__(self):
        """Validate trend data"""
        if self.trend_id <= 0:
            raise ValueError("Trend ID must be positive")
        if self.start_index < 0:
            raise ValueError("Start index must be non-negative")
        if self.end_index is not None and self.start_index >= self.end_index:
            raise ValueError("Start index must be less than end index")
        if self.price_range < 0:
            raise ValueError("Price range must be non-negative")
        if self.duration < 0:
            raise ValueError("Duration must be non-negative")
        if self.direction == TrendDirection.SIDEWAYS:
            if self.range_high is not None and self.range_low is not None:
                if self.range_high <= self.range_low:
                    raise ValueError("Range high must be greater than range low")
    
    def __repr__(self):
        status = "ACTIVE" if self.is_active else "TERMINATED"
        sig = self.significance.value.upper()
        if self.direction == TrendDirection.SIDEWAYS and self.range_high and self.range_low:
            controlling = f"Range:{self.range_low:.1f}-{self.range_high:.1f}"
        else:
            controlling = f"Ctrl:{self.controlling_swing.price:.1f}" if self.controlling_swing else "No-Ctrl"
        return f"Trend{self.trend_id}({self.direction.value.upper()}-{sig}, {controlling}, {status})"


@dataclass(frozen=True)
class TrendMetrics:
    """
    Immutable trend analysis metrics.
    Responsibility: Hold calculated metrics for a trend.
    """
    duration: int
    price_range: float
    volatility: float
    strength: float
    dominance_score: float
    
    def __post_init__(self):
        """Validate metrics"""
        if self.duration < 0:
            raise ValueError("Duration must be non-negative")
        if self.price_range < 0:
            raise ValueError("Price range must be non-negative")
        if self.volatility < 0:
            raise ValueError("Volatility must be non-negative")


@dataclass(frozen=True)
class TrendAnalysisConfig:
    """
    Immutable configuration for trend analysis.
    Responsibility: Hold algorithm parameters.
    """
    sideways_range_threshold: float = 0.15
    moveout_threshold: float = 2.0
    major_trend_min_duration: int = 4
    major_trend_min_range: float = 6.0
    swing_lookback_period: int = 2
    max_trend_overlap: float = 0.3
    
    def __post_init__(self):
        """Validate configuration"""
        if self.sideways_range_threshold <= 0:
            raise ValueError("Sideways range threshold must be positive")
        if self.moveout_threshold <= 0:
            raise ValueError("Moveout threshold must be positive")
        if self.major_trend_min_duration < 1:
            raise ValueError("Major trend minimum duration must be at least 1")
        if self.major_trend_min_range <= 0:
            raise ValueError("Major trend minimum range must be positive")
        if self.swing_lookback_period < 1:
            raise ValueError("Swing lookback period must be at least 1")
        if not (0 <= self.max_trend_overlap <= 1):
            raise ValueError("Max trend overlap must be between 0 and 1")


@dataclass(frozen=True)
class TrendAnalysisResult:
    """
    Immutable complete trend analysis result.
    Responsibility: Hold all analysis results.
    """
    candles: Tuple['Candle', ...]  # Added missing field
    swings: Tuple[SwingPoint, ...]
    detected_patterns: Tuple[TrendPattern, ...]
    trends: Tuple[Trend, ...]
    active_trends: Tuple[Trend, ...]
    major_trends: Tuple[Trend, ...]
    current_trend: Optional[Trend]
    analysis_window: Tuple[int, int]
    total_candles_analyzed: int
    
    def __post_init__(self):
        """Validate result consistency"""
        # Validate candles field
        if len(self.candles) != self.total_candles_analyzed:
            raise ValueError("Candles length must match total_candles_analyzed")
        
        # Fix for single candle case
        if self.total_candles_analyzed == 1:
            # Allow (0, 0) for single candle case
            if self.analysis_window != (0, 0):
                raise ValueError("Single candle analysis should have window (0, 0)")
        else:
            # Normal validation for multiple candles
            if self.analysis_window[0] >= self.analysis_window[1]:
                raise ValueError("Analysis window start must be less than end")
        
        if self.total_candles_analyzed <= 0:
            raise ValueError("Total candles analyzed must be positive")
        
        # Verify trend classifications
        active_count = sum(1 for t in self.trends if t.is_active)
        if active_count != len(self.active_trends):
            raise ValueError("Active trends count mismatch")
        
        major_count = sum(1 for t in self.trends if t.significance == TrendSignificance.MAJOR)
        if major_count != len(self.major_trends):
            raise ValueError("Major trends count mismatch")


@dataclass(frozen=True)
class GenesisPoint:
    """
    Immutable genesis point for multi-timeframe analysis.
    Responsibility: Hold genesis point data for trend initiation.
    """
    swing: SwingPoint
    candle_index: int
    terminated_trend_id: Optional[int]
    genesis_type: SwingType
    created_trends: Tuple[int, ...] = ()  # IDs of trends created from this genesis
    
    def __post_init__(self):
        """Validate genesis point data"""
        if self.candle_index < 0:
            raise ValueError("Candle index must be non-negative")
        if self.terminated_trend_id is not None and self.terminated_trend_id <= 0:
            raise ValueError("Terminated trend ID must be positive if provided")


@dataclass(frozen=True)
class VisualizationData:
    """
    Immutable data structure for trend visualization.
    Responsibility: Hold all data needed for visualization.
    """
    analysis_result: TrendAnalysisResult
    candles: Tuple['Candle', ...]  # From utilities or external
    trend_colors: dict
    swing_annotations: Tuple[dict, ...]
    trend_annotations: Tuple[dict, ...]
    
    def __post_init__(self):
        """Validate visualization data"""
        if len(self.candles) != self.analysis_result.total_candles_analyzed:
            raise ValueError("Candles count must match analysis result")