"""
Test utilities for the Supply & Demand Zone Detection Algorithm
Contains ONLY testing-specific utilities that aren't useful in production

All production utilities (CandleBuilder, ZoneBuilder, etc.) are in utilities.py
"""

from datetime import datetime, timedelta
from typing import List
from dataclasses import replace
from unittest.mock import Mock

from core_models import (
    Candle, Zone, ZoneType, CandleType, CandleMetrics, ZoneMetrics,
    AlgorithmConfig, AnalysisResult
)
from utilities import CandleBuilder, ZoneBuilder  # Import production utilities


class TestDataFactory:
    """
    Factory for creating predefined test scenarios and patterns.
    Responsibility: Generate known test patterns for algorithm validation.
    
    This is test-specific because it creates hardcoded patterns designed
    to test specific algorithm behaviors, not for general production use.
    """
    
    @staticmethod
    def create_simple_supply_pattern() -> List[Candle]:
        """Create a simple LEG->BASE->LEG supply pattern for testing"""
        builder = CandleBuilder()
        base_time = datetime(2024, 1, 1, 9, 0)
        
        return [
            # Strong bullish LEG
            builder.reset().with_timestamp(base_time).with_ohlc(100, 115, 99, 113).build(),
            # BASE candle (consolidation)
            builder.reset().with_timestamp(base_time + timedelta(minutes=5)).with_ohlc(113, 115, 111, 113).build(),
            # Strong bearish LEG
            builder.reset().with_timestamp(base_time + timedelta(minutes=10)).with_ohlc(113, 114, 105, 106).build(),
        ]
    
    @staticmethod
    def create_simple_demand_pattern() -> List[Candle]:
        """Create a simple LEG->BASE->LEG demand pattern for testing"""
        builder = CandleBuilder()
        base_time = datetime(2024, 1, 1, 9, 0)
        
        return [
            # Strong bearish LEG
            builder.reset().with_timestamp(base_time).with_ohlc(100, 101, 85, 87).build(),
            # BASE candle (consolidation)
            builder.reset().with_timestamp(base_time + timedelta(minutes=5)).with_ohlc(87, 90, 85, 88).build(),
            # Strong bullish LEG
            builder.reset().with_timestamp(base_time + timedelta(minutes=10)).with_ohlc(88, 102, 87, 100).build(),
        ]
    
    @staticmethod
    def create_multiple_base_pattern() -> List[Candle]:
        """Create pattern with multiple consecutive BASE candles for testing"""
        builder = CandleBuilder()
        base_time = datetime(2024, 1, 1, 9, 0)
        
        return [
            # Strong bullish LEG (large body, small wicks)
            builder.reset().with_timestamp(base_time).bullish(12).large_body(12, 0.5).build(),
            
            # Multiple BASE candles (small bodies, large wicks)
            builder.reset().with_timestamp(base_time + timedelta(minutes=5)).bullish(1).large_wicks(1, 3).build(),
            builder.reset().with_timestamp(base_time + timedelta(minutes=10)).bullish(1).large_wicks(1, 3).build(),
            builder.reset().with_timestamp(base_time + timedelta(minutes=15)).bullish(1).large_wicks(1, 3).build(),
            
            # Strong bearish LEG (large body, small wicks)
            builder.reset().with_timestamp(base_time + timedelta(minutes=20)).bearish(10).large_body(10, 0.5).build(),
        ]
    
    @staticmethod
    def create_ambiguous_pattern() -> List[Candle]:
        """Create pattern that should not form a valid zone (same direction legs) - for testing edge cases"""
        builder = CandleBuilder()
        base_time = datetime(2024, 1, 1, 9, 0)
        
        return [
            # Bullish LEG
            builder.reset().with_timestamp(base_time).bullish(8).large_body(8, 0.5).build(),
            # BASE candle
            builder.reset().with_timestamp(base_time + timedelta(minutes=5)).large_wicks(1, 3).build(),
            # Another bullish LEG (same direction - should be rejected)
            builder.reset().with_timestamp(base_time + timedelta(minutes=10)).bullish(7).large_body(7, 0.5).build(),
        ]
    
    @staticmethod
    def create_complex_market_data() -> List[Candle]:
        """Create complex market data with multiple patterns for integration testing"""
        patterns = [
            TestDataFactory.create_simple_supply_pattern(),
            TestDataFactory._create_noise_candles(3),
            TestDataFactory.create_simple_demand_pattern(),
            TestDataFactory._create_noise_candles(2),
            TestDataFactory.create_multiple_base_pattern(),
        ]
        
        # Flatten and adjust timestamps
        all_candles = []
        current_time = datetime(2024, 1, 1, 9, 0)
        
        for pattern in patterns:
            for candle in pattern:
                adjusted_candle = replace(candle, timestamp=current_time)
                all_candles.append(adjusted_candle)
                current_time += timedelta(minutes=5)
        
        return all_candles
    
    @staticmethod
    def _create_noise_candles(count: int) -> List[Candle]:
        """Create noise candles that don't form patterns - helper for testing"""
        builder = CandleBuilder()
        candles = []
        
        for i in range(count):
            # Create candles with characteristics that don't form clear patterns
            candle = (builder.reset()
                     .with_ohlc(100 + i, 102 + i, 98 + i, 101 + i)
                     .build())
            candles.append(candle)
        
        return candles
    
    @staticmethod
    def create_edge_case_data() -> dict:
        """Create various edge case scenarios for boundary testing"""
        return {
            'empty_list': [],
            'single_candle': [CandleBuilder().build()],
            'two_candles': [CandleBuilder().build(), CandleBuilder().bullish().build()],
            'all_base_candles': [CandleBuilder().large_wicks().build() for _ in range(5)],
            'all_leg_candles': [CandleBuilder().large_body().build() for _ in range(5)],
        }


class MockObjects:
    """
    Factory for creating mock objects for unit testing.
    Responsibility: Provide consistent mock objects for isolated unit testing.
    
    This is test-specific because mocks are only used in testing scenarios.
    """
    
    @staticmethod
    def create_mock_candle_analyzer():
        """Create mock CandleAnalyzer with predefined behavior for testing"""
        mock = Mock()
        mock.analyze_candle.return_value = CandleMetrics(
            body_size=5.0,
            upper_wick=1.0,
            lower_wick=1.0,
            total_range=7.0,
            body_to_wick_ratio=2.5,
            is_bullish=True
        )
        mock.analyze_candles.return_value = [mock.analyze_candle.return_value]
        return mock
    
    @staticmethod
    def create_mock_classifier():
        """Create mock CandleClassifier with predefined behavior for testing"""
        mock = Mock()
        mock.classify_candles.return_value = [CandleType.LEG, CandleType.BASE, CandleType.LEG]
        return mock
    
    @staticmethod
    def create_mock_zone_detector():
        """Create mock ZoneDetector with predefined behavior for testing"""
        mock = Mock()
        mock.detect_zones.return_value = [ZoneBuilder().build()]  # Uses production ZoneBuilder
        return mock


class TestAssertions:
    """
    Custom assertions for testing supply/demand algorithm components.
    Responsibility: Provide domain-specific test assertions.
    
    This is test-specific because these assertions are designed specifically
    for validating algorithm behavior in test scenarios.
    """
    
    @staticmethod
    def assert_valid_candle(candle: Candle) -> None:
        """Assert that a candle has valid OHLC data"""
        assert candle.high >= max(candle.open, candle.close), \
            f"High {candle.high} must be >= max(open {candle.open}, close {candle.close})"
        assert candle.low <= min(candle.open, candle.close), \
            f"Low {candle.low} must be <= min(open {candle.open}, close {candle.close})"
        if candle.volume is not None:
            assert candle.volume >= 0, f"Volume {candle.volume} must be non-negative"
    
    @staticmethod
    def assert_valid_zone(zone: Zone) -> None:
        """Assert that a zone has valid structure"""
        assert zone.high > zone.low, f"Zone high {zone.high} must be > low {zone.low}"
        assert zone.start_index <= zone.end_index, \
            f"Start index {zone.start_index} must be <= end index {zone.end_index}"
        assert len(zone.base_candles) > 0, "Zone must have at least one base candle"
        assert zone.entry_leg_index <= min(zone.base_candles), \
            "Entry leg must come before base candles"
        assert zone.exit_leg_index >= max(zone.base_candles), \
            "Exit leg must come after base candles"
    
    @staticmethod
    def assert_supply_zone_logic(zone: Zone, candles: List[Candle]) -> None:
        """Assert that a supply zone follows correct logic"""
        assert zone.zone_type == ZoneType.SUPPLY
        entry_leg = candles[zone.entry_leg_index]
        exit_leg = candles[zone.exit_leg_index]
        
        # Supply zone: bullish entry leg, bearish exit leg
        assert entry_leg.close > entry_leg.open, "Supply zone entry leg should be bullish"
        assert exit_leg.close < exit_leg.open, "Supply zone exit leg should be bearish"
    
    @staticmethod
    def assert_demand_zone_logic(zone: Zone, candles: List[Candle]) -> None:
        """Assert that a demand zone follows correct logic"""
        assert zone.zone_type == ZoneType.DEMAND
        entry_leg = candles[zone.entry_leg_index]
        exit_leg = candles[zone.exit_leg_index]
        
        # Demand zone: bearish entry leg, bullish exit leg
        assert entry_leg.close < entry_leg.open, "Demand zone entry leg should be bearish"
        assert exit_leg.close > exit_leg.open, "Demand zone exit leg should be bullish"
    
    @staticmethod
    def assert_analysis_result_consistency(result: AnalysisResult) -> None:
        """Assert that analysis result is internally consistent"""
        assert len(result.candles) == len(result.classifications), \
            "Candles and classifications must have same length"
        assert result.total_zones == len(result.zones), \
            "Total zones must match zones list length"
        assert len(result.supply_zones) + len(result.demand_zones) == result.total_zones, \
            "Supply and demand zones must sum to total zones"
        
        # Verify zone classifications
        supply_count = sum(1 for z in result.zones if z.zone_type == ZoneType.SUPPLY)
        demand_count = sum(1 for z in result.zones if z.zone_type == ZoneType.DEMAND)
        assert supply_count == len(result.supply_zones), "Supply zone count mismatch"
        assert demand_count == len(result.demand_zones), "Demand zone count mismatch"