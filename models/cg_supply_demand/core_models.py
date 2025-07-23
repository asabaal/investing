"""
Enhanced Candle class with body_to_wick_ratio property
Add this to your core_models.py file
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
from datetime import datetime


class CandleType(Enum):
    """Classification of candle types"""
    BASE = "base"
    LEG = "leg"
    UNCERTAIN = "uncertain"


class ZoneType(Enum):
    """Type of supply/demand zone"""
    SUPPLY = "supply"
    DEMAND = "demand"


@dataclass(frozen=True)
class Candle:
    """
    Immutable candlestick data model with trend detection support.
    Responsibility: Hold candlestick data only, no business logic.
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    
    def __post_init__(self):
        """Validate candle data integrity"""
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("High price must be >= open, close, and low prices")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("Low price must be <= open, close, and high prices")
        if self.volume is not None and self.volume < 0:
            raise ValueError("Volume cannot be negative")
    
    @property
    def body_to_wick_ratio(self) -> float:
        """
        Calculate body-to-wick ratio for trend detection.
        Returns float('inf') if total wick size is 0.
        """
        body_size = abs(self.close - self.open)
        upper_wick = self.high - max(self.open, self.close)
        lower_wick = min(self.open, self.close) - self.low
        total_wick = upper_wick + lower_wick
        
        if total_wick == 0:
            return float('inf')
        
        return body_size / total_wick


# Rest of the core_models.py file remains the same...
@dataclass(frozen=True)
class Zone:
    """
    Immutable supply/demand zone model.
    Responsibility: Hold zone data only, no business logic.
    """
    zone_type: ZoneType
    start_index: int
    end_index: int
    high: float
    low: float
    base_candles: tuple[int, ...]  # Immutable tuple instead of list
    entry_leg_index: int
    exit_leg_index: int
    
    def __post_init__(self):
        """Validate zone data integrity"""
        if self.high <= self.low:
            raise ValueError("Zone high must be greater than low")
        if self.start_index >= self.end_index:
            raise ValueError("Start index must be less than end index")
        if not self.base_candles:
            raise ValueError("Zone must have at least one base candle")


@dataclass(frozen=True)
class CandleMetrics:
    """
    Immutable candle analysis metrics.
    Responsibility: Hold calculated metrics for a candle.
    """
    body_size: float
    upper_wick: float
    lower_wick: float
    total_range: float
    body_to_wick_ratio: float
    is_bullish: bool
    
    def __post_init__(self):
        """Validate metrics"""
        if any(value < 0 for value in [self.body_size, self.upper_wick, 
                                      self.lower_wick, self.total_range]):
            raise ValueError("All size metrics must be non-negative")


@dataclass(frozen=True)
class ZoneMetrics:
    """
    Immutable zone analysis metrics.
    Responsibility: Hold calculated metrics for a zone.
    """
    price_range: float
    midpoint: float
    base_candle_count: int
    
    def __post_init__(self):
        """Validate zone metrics"""
        if self.price_range < 0:
            raise ValueError("Price range must be non-negative")
        if self.base_candle_count < 1:
            raise ValueError("Must have at least one base candle")


@dataclass(frozen=True)
class AnalysisResult:
    """
    Immutable analysis result container.
    Responsibility: Hold complete analysis results.
    """
    candles: tuple[Candle, ...]
    classifications: tuple[CandleType, ...]
    zones: tuple[Zone, ...]
    supply_zones: tuple[Zone, ...]
    demand_zones: tuple[Zone, ...]
    total_zones: int
    
    def __post_init__(self):
        """Validate result consistency"""
        if len(self.candles) != len(self.classifications):
            raise ValueError("Candles and classifications must have same length")
        if self.total_zones != len(self.zones):
            raise ValueError("Total zones count must match zones length")
        if len(self.supply_zones) + len(self.demand_zones) != self.total_zones:
            raise ValueError("Supply and demand zones must sum to total zones")


@dataclass(frozen=True)
class AlgorithmConfig:
    """
    Immutable configuration for the algorithm.
    Responsibility: Hold algorithm parameters.
    """
    body_ratio_threshold: float = 1.0
    min_base_candles: int = 1
    max_base_candles: int = 5
    
    def __post_init__(self):
        """Validate configuration"""
        if self.body_ratio_threshold <= 0:
            raise ValueError("Body ratio threshold must be positive")
        if self.min_base_candles < 1:
            raise ValueError("Minimum base candles must be at least 1")
        if self.max_base_candles < self.min_base_candles:
            raise ValueError("Max base candles must be >= min base candles")