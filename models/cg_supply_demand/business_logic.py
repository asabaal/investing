"""
Business logic components for the Supply & Demand Zone Detection Algorithm
Each class follows Single Responsibility Principle
"""

from abc import ABC, abstractmethod
from typing import List, Protocol, Tuple
import numpy as np

from core_models import (
    Candle, CandleType, Zone, ZoneType, CandleMetrics, ZoneMetrics,
    AnalysisResult, AlgorithmConfig
)


class CandleAnalyzer:
    """
    Responsibility: Calculate metrics for individual candles.
    """
    
    @staticmethod
    def analyze_candle(candle: Candle) -> CandleMetrics:
        """Calculate all metrics for a single candle"""
        body_size = abs(candle.close - candle.open)
        upper_wick = candle.high - max(candle.open, candle.close)
        lower_wick = min(candle.open, candle.close) - candle.low
        total_range = candle.high - candle.low
        
        # Handle division by zero
        total_wick = upper_wick + lower_wick
        body_to_wick_ratio = float('inf') if total_wick == 0 else body_size / total_wick
        
        is_bullish = candle.close > candle.open
        
        return CandleMetrics(
            body_size=body_size,
            upper_wick=upper_wick,
            lower_wick=lower_wick,
            total_range=total_range,
            body_to_wick_ratio=body_to_wick_ratio,
            is_bullish=is_bullish
        )
    
    @classmethod
    def analyze_candles(cls, candles: List[Candle]) -> List[CandleMetrics]:
        """Calculate metrics for multiple candles"""
        return [cls.analyze_candle(candle) for candle in candles]


class CandleClassificationStrategy(Protocol):
    """Protocol for candle classification strategies"""
    
    def classify(self, candle: Candle, metrics: CandleMetrics) -> CandleType:
        """Classify a single candle"""
        ...


class BodyRatioClassificationStrategy:
    """
    Responsibility: Classify candles based on body-to-wick ratio.
    """
    
    def __init__(self, threshold: float):
        self.threshold = threshold
    
    def classify(self, candle: Candle, metrics: CandleMetrics) -> CandleType:
        """Classify candle based on body-to-wick ratio"""
        if metrics.body_to_wick_ratio > self.threshold:
            return CandleType.LEG
        else:
            return CandleType.BASE


class CandleClassifier:
    """
    Responsibility: Classify candles using a pluggable strategy.
    """
    
    def __init__(self, strategy: CandleClassificationStrategy):
        self._strategy = strategy
    
    def classify_candles(self, candles: List[Candle], 
                        metrics: List[CandleMetrics]) -> List[CandleType]:
        """Classify all candles using the configured strategy"""
        if len(candles) != len(metrics):
            raise ValueError("Candles and metrics lists must have same length")
        
        return [
            self._strategy.classify(candle, metric)
            for candle, metric in zip(candles, metrics)
        ]


class ZoneAnalyzer:
    """
    Responsibility: Calculate metrics for zones.
    """
    
    @staticmethod
    def analyze_zone(zone: Zone) -> ZoneMetrics:
        """Calculate metrics for a single zone"""
        price_range = zone.high - zone.low
        midpoint = (zone.high + zone.low) / 2
        base_candle_count = len(zone.base_candles)
        
        return ZoneMetrics(
            price_range=price_range,
            midpoint=midpoint,
            base_candle_count=base_candle_count
        )
    
    @classmethod
    def analyze_zones(cls, zones: List[Zone]) -> List[ZoneMetrics]:
        """Calculate metrics for multiple zones"""
        return [cls.analyze_zone(zone) for zone in zones]


class ZonePatternDetector:
    """
    Responsibility: Detect LEG->BASE(s)->LEG patterns in classified candles.
    """
    
    def __init__(self, min_base_candles: int = 1, max_base_candles: int = 5):
        self.min_base_candles = min_base_candles
        self.max_base_candles = max_base_candles
    
    def find_patterns(self, candles: List[Candle], 
                     classifications: List[CandleType]) -> List[Tuple[int, List[int], int]]:
        """
        Find LEG->BASE(s)->LEG patterns.
        Returns: List of (entry_leg_index, base_indices, exit_leg_index)
        """
        if len(candles) != len(classifications):
            raise ValueError("Candles and classifications must have same length")
        
        patterns = []
        i = 0
        
        while i < len(candles) - 2:  # Need at least LEG + BASE + LEG
            if classifications[i] == CandleType.LEG:
                base_indices = self._find_consecutive_base_candles(
                    classifications, i + 1
                )
                
                if self._is_valid_base_sequence(base_indices):
                    exit_leg_index = base_indices[-1] + 1
                    if (exit_leg_index < len(classifications) and 
                        classifications[exit_leg_index] == CandleType.LEG):
                        
                        patterns.append((i, base_indices, exit_leg_index))
                        i = exit_leg_index + 1
                        continue
            
            i += 1
        
        return patterns
    
    def _find_consecutive_base_candles(self, classifications: List[CandleType], 
                                     start_index: int) -> List[int]:
        """Find consecutive BASE candles starting from start_index"""
        base_indices = []
        j = start_index
        
        while (j < len(classifications) and 
               classifications[j] == CandleType.BASE and 
               len(base_indices) < self.max_base_candles):
            base_indices.append(j)
            j += 1
        
        return base_indices
    
    def _is_valid_base_sequence(self, base_indices: List[int]) -> bool:
        """Check if base sequence meets minimum requirements"""
        return len(base_indices) >= self.min_base_candles


class ZoneTypeClassifier:
    """
    Responsibility: Determine zone type (SUPPLY/DEMAND) from leg candles.
    """
    
    @staticmethod
    def classify_zone_type(entry_leg: Candle, exit_leg: Candle) -> ZoneType | None:
        """
        Classify zone type based on entry and exit leg directions.
        Returns None if pattern is ambiguous.
        """
        entry_bullish = entry_leg.close > entry_leg.open
        exit_bullish = exit_leg.close > exit_leg.open
        
        if entry_bullish and not exit_bullish:
            return ZoneType.SUPPLY
        elif not entry_bullish and exit_bullish:
            return ZoneType.DEMAND
        else:
            return None  # Ambiguous pattern


class ZoneBuilder:
    """
    Responsibility: Build Zone objects from detected patterns.
    """
    
    @staticmethod
    def build_zone(candles: List[Candle], entry_leg_index: int, 
                  base_indices: List[int], exit_leg_index: int,
                  zone_type: ZoneType) -> Zone:
        """Build a Zone object from pattern components"""
        # Calculate zone boundaries using only BASE candles
        base_candles = [candles[idx] for idx in base_indices]
        zone_high = max(candle.high for candle in base_candles)
        zone_low = min(candle.low for candle in base_candles)
        
        return Zone(
            zone_type=zone_type,
            start_index=entry_leg_index,
            end_index=exit_leg_index,
            high=zone_high,
            low=zone_low,
            base_candles=tuple(base_indices),
            entry_leg_index=entry_leg_index,
            exit_leg_index=exit_leg_index
        )


class ZoneDetector:
    """
    Responsibility: Orchestrate zone detection from classified candles.
    """
    
    def __init__(self, config: AlgorithmConfig):
        self.pattern_detector = ZonePatternDetector(
            config.min_base_candles, config.max_base_candles
        )
        self.type_classifier = ZoneTypeClassifier()
        self.zone_builder = ZoneBuilder()
    
    def detect_zones(self, candles: List[Candle], 
                    classifications: List[CandleType]) -> List[Zone]:
        """Detect all valid zones from classified candles"""
        patterns = self.pattern_detector.find_patterns(candles, classifications)
        zones = []
        
        for entry_idx, base_indices, exit_idx in patterns:
            entry_leg = candles[entry_idx]
            exit_leg = candles[exit_idx]
            
            zone_type = self.type_classifier.classify_zone_type(entry_leg, exit_leg)
            if zone_type is not None:
                zone = self.zone_builder.build_zone(
                    candles, entry_idx, base_indices, exit_idx, zone_type
                )
                zones.append(zone)
        
        return zones


class ResultProcessor:
    """
    Responsibility: Process and format analysis results.
    """
    
    @staticmethod
    def create_analysis_result(candles: List[Candle], 
                             classifications: List[CandleType],
                             zones: List[Zone]) -> AnalysisResult:
        """Create a complete analysis result object"""
        supply_zones = [z for z in zones if z.zone_type == ZoneType.SUPPLY]
        demand_zones = [z for z in zones if z.zone_type == ZoneType.DEMAND]
        
        return AnalysisResult(
            candles=tuple(candles),
            classifications=tuple(classifications),
            zones=tuple(zones),
            supply_zones=tuple(supply_zones),
            demand_zones=tuple(demand_zones),
            total_zones=len(zones)
        )
    
    @staticmethod
    def create_zone_summary(zones: List[Zone]) -> dict:
        """Create summary statistics for zones"""
        if not zones:
            return {'count': 0}
        
        supply_zones = [z for z in zones if z.zone_type == ZoneType.SUPPLY]
        demand_zones = [z for z in zones if z.zone_type == ZoneType.DEMAND]
        
        # Calculate zone metrics
        zone_analyzer = ZoneAnalyzer()
        zone_metrics = zone_analyzer.analyze_zones(zones)
        
        return {
            'count': len(zones),
            'supply_count': len(supply_zones),
            'demand_count': len(demand_zones),
            'avg_range': np.mean([m.price_range for m in zone_metrics]),
            'avg_base_candles': np.mean([m.base_candle_count for m in zone_metrics])
        }


class SupplyDemandAlgorithm:
    """
    Responsibility: Orchestrate the complete analysis process.
    """
    
    def __init__(self, config: AlgorithmConfig | None = None):
        self.config = config or AlgorithmConfig()
        
        # Initialize components
        self.candle_analyzer = CandleAnalyzer()
        self.candle_classifier = CandleClassifier(
            BodyRatioClassificationStrategy(self.config.body_ratio_threshold)
        )
        self.zone_detector = ZoneDetector(self.config)
        self.result_processor = ResultProcessor()
    
    def analyze_market_data(self, candles: List[Candle]) -> AnalysisResult:
        """
        Perform complete supply and demand analysis.
        
        Args:
            candles: List of market candles to analyze
            
        Returns:
            Complete analysis results
        """
        if not candles:
            raise ValueError("Cannot analyze empty candle list")
        
        # Step 1: Analyze candles
        candle_metrics = self.candle_analyzer.analyze_candles(candles)
        
        # Step 2: Classify candles
        classifications = self.candle_classifier.classify_candles(
            candles, candle_metrics
        )
        
        # Step 3: Detect zones
        zones = self.zone_detector.detect_zones(candles, classifications)
        
        # Step 4: Process results
        return self.result_processor.create_analysis_result(
            candles, classifications, zones
        )
    
    def get_zone_summary(self, zones: List[Zone]) -> dict:
        """Get summary statistics for zones"""
        return self.result_processor.create_zone_summary(zones)