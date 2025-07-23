"""
Comprehensive test suite for the Supply & Demand Zone Detection Algorithm
Organized by component with unit, integration, and property-based tests
"""

import pytest
import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import numpy as np
from typing import List

# Import the refactored modules
from core_models import (
    Candle, Zone, ZoneType, CandleType, CandleMetrics, ZoneMetrics,
    AlgorithmConfig, AnalysisResult
)
from business_logic import (
    CandleAnalyzer, BodyRatioClassificationStrategy, CandleClassifier,
    ZoneAnalyzer, ZonePatternDetector, ZoneTypeClassifier, ZoneBuilder,
    ZoneDetector, ResultProcessor, SupplyDemandAlgorithm
)
from utilities import CandleBuilder, ZoneBuilder, RandomDataGenerator
from test_utilities import TestDataFactory, MockObjects, TestAssertions


class TestCandleBuilder(unittest.TestCase):
    """Test the CandleBuilder utility class"""
    
    def setUp(self):
        self.builder = CandleBuilder()
        self.analyzer = CandleAnalyzer()
    
    def test_basic_building(self):
        """Test basic candle building with explicit OHLC"""
        candle = self.builder.with_ohlc(100, 105, 95, 102).build()
        
        self.assertEqual(candle.open, 100)
        self.assertEqual(candle.high, 105)
        self.assertEqual(candle.low, 95)
        self.assertEqual(candle.close, 102)
        TestAssertions.assert_valid_candle(candle)
    
    def test_bullish_method(self):
        """Test bullish() method creates bullish candles"""
        candle = self.builder.reset().bullish(5).build()
        metrics = self.analyzer.analyze_candle(candle)
        
        self.assertTrue(metrics.is_bullish)
        self.assertEqual(metrics.body_size, 5.0)
        TestAssertions.assert_valid_candle(candle)
    
    def test_bearish_method(self):
        """Test bearish() method creates bearish candles"""
        candle = self.builder.reset().bearish(7).build()
        metrics = self.analyzer.analyze_candle(candle)
        
        self.assertFalse(metrics.is_bullish)
        self.assertEqual(metrics.body_size, 7.0)
        TestAssertions.assert_valid_candle(candle)
    
    def test_doji_method(self):
        """Test doji() method creates doji candles"""
        candle = self.builder.reset().doji(3).build()
        metrics = self.analyzer.analyze_candle(candle)
        
        self.assertEqual(candle.open, candle.close)
        self.assertEqual(metrics.body_size, 0.0)
        TestAssertions.assert_valid_candle(candle)
    
    def test_bearish_large_body_composition(self):
        """Test that bearish().large_body() preserves bearish direction"""
        candle = self.builder.reset().bearish(8).large_body(8, 0.5).build()
        metrics = self.analyzer.analyze_candle(candle)
        
        # Should be bearish with large body and small wicks
        self.assertFalse(metrics.is_bullish, "Should remain bearish after large_body()")
        self.assertEqual(metrics.body_size, 8.0)
        self.assertGreater(metrics.body_to_wick_ratio, 1.0, "Should have high body/wick ratio")
        TestAssertions.assert_valid_candle(candle)
    
    def test_bullish_large_body_composition(self):
        """Test that bullish().large_body() preserves bullish direction"""
        candle = self.builder.reset().bullish(10).large_body(10, 0.5).build()
        metrics = self.analyzer.analyze_candle(candle)
        
        # Should be bullish with large body and small wicks
        self.assertTrue(metrics.is_bullish, "Should remain bullish after large_body()")
        self.assertEqual(metrics.body_size, 10.0)
        self.assertGreater(metrics.body_to_wick_ratio, 1.0, "Should have high body/wick ratio")
        TestAssertions.assert_valid_candle(candle)
    
    def test_bearish_large_wicks_composition(self):
        """Test that bearish().large_wicks() preserves bearish direction"""
        candle = self.builder.reset().bearish(1).large_wicks(1, 5).build()
        metrics = self.analyzer.analyze_candle(candle)
        
        # Should be bearish with small body and large wicks
        self.assertFalse(metrics.is_bullish, "Should remain bearish after large_wicks()")
        self.assertEqual(metrics.body_size, 1.0)
        self.assertLess(metrics.body_to_wick_ratio, 1.0, "Should have low body/wick ratio")
        TestAssertions.assert_valid_candle(candle)
    
    def test_bullish_large_wicks_composition(self):
        """Test that bullish().large_wicks() preserves bullish direction"""
        candle = self.builder.reset().bullish(1).large_wicks(1, 5).build()
        metrics = self.analyzer.analyze_candle(candle)
        
        # Should be bullish with small body and large wicks
        self.assertTrue(metrics.is_bullish, "Should remain bullish after large_wicks()")
        self.assertEqual(metrics.body_size, 1.0)
        self.assertLess(metrics.body_to_wick_ratio, 1.0, "Should have low body/wick ratio")
        TestAssertions.assert_valid_candle(candle)
    
    def test_method_chaining_order_independence(self):
        """Test that method order doesn't matter for final result"""
        # These should produce equivalent candles
        candle1 = self.builder.reset().bearish(5).large_body(5, 0.5).build()
        candle2 = self.builder.reset().large_body(5, 0.5).bearish(5).build()
        
        metrics1 = self.analyzer.analyze_candle(candle1)
        metrics2 = self.analyzer.analyze_candle(candle2)
        
        # Both should be bearish with same body size
        self.assertFalse(metrics1.is_bullish)
        self.assertFalse(metrics2.is_bullish)
        self.assertEqual(metrics1.body_size, metrics2.body_size)
    
    def test_reset_clears_state(self):
        """Test that reset() properly clears builder state"""
        # Build a complex candle
        self.builder.bearish(10).large_body(10, 1).with_volume(5000)
        
        # Reset and build a simple candle
        candle = self.builder.reset().build()
        
        # Should have default values, not previous state
        self.assertEqual(candle.open, 100.0)  # Default open
        self.assertEqual(candle.volume, 1000.0)  # Default volume
        TestAssertions.assert_valid_candle(candle)
    
    def test_with_timestamp(self):
        """Test timestamp setting"""
        test_time = datetime(2023, 5, 15, 14, 30)
        candle = self.builder.reset().with_timestamp(test_time).build()
        
        self.assertEqual(candle.timestamp, test_time)
    
    def test_with_volume(self):
        """Test volume setting"""
        candle = self.builder.reset().with_volume(2500).build()
        
        self.assertEqual(candle.volume, 2500)
    
    def test_edge_case_zero_body_large_body(self):
        """Test large_body() with zero-sized body"""
        # Start with doji (zero body)
        candle = self.builder.reset().doji().large_body(0, 1).build()
        metrics = self.analyzer.analyze_candle(candle)
        
        self.assertEqual(metrics.body_size, 0.0)
        TestAssertions.assert_valid_candle(candle)
    
    def test_builder_produces_classifiable_candles(self):
        """Test that builder produces candles that classify correctly"""
        classifier = CandleClassifier(BodyRatioClassificationStrategy(1.0))
        
        # Create LEG candle (high body/wick ratio)
        leg_candle = self.builder.reset().bullish(10).large_body(10, 0.5).build()
        leg_metrics = self.analyzer.analyze_candle(leg_candle)
        leg_classification = classifier.classify_candles([leg_candle], [leg_metrics])
        
        self.assertEqual(leg_classification[0], CandleType.LEG)
        
        # Create BASE candle (low body/wick ratio)
        base_candle = self.builder.reset().bullish(1).large_wicks(1, 5).build()
        base_metrics = self.analyzer.analyze_candle(base_candle)
        base_classification = classifier.classify_candles([base_candle], [base_metrics])
        
        self.assertEqual(base_classification[0], CandleType.BASE)


class TestZoneBuilder(unittest.TestCase):
    """Test the ZoneBuilder utility class"""
    
    def setUp(self):
        self.builder = ZoneBuilder()
    
    def test_basic_zone_building(self):
        """Test basic zone building with explicit parameters"""
        zone = (self.builder
                .supply_zone()
                .with_price_range(115.0, 110.0)
                .with_base_candles(1, 2)
                .with_legs(0, 3)
                .with_indices(0, 3)
                .build())
        
        self.assertEqual(zone.zone_type, ZoneType.SUPPLY)
        self.assertEqual(zone.high, 115.0)
        self.assertEqual(zone.low, 110.0)
        self.assertEqual(zone.base_candles, (1, 2))
        TestAssertions.assert_valid_zone(zone)
    
    def test_demand_zone_creation(self):
        """Test creating demand zones"""
        zone = (self.builder.reset()
                .demand_zone()
                .with_price_range(100.0, 95.0)
                .with_base_candles(2)
                .with_legs(1, 3)
                .with_indices(1, 3)
                .build())
        
        self.assertEqual(zone.zone_type, ZoneType.DEMAND)
        TestAssertions.assert_valid_zone(zone)
    
    def test_from_candles_method(self):
        """Test building zone from actual candle data"""
        candles = TestDataFactory.create_simple_supply_pattern()
        
        zone = (self.builder.reset()
                .from_candles(candles, entry_idx=0, base_indices=[1], 
                             exit_idx=2, zone_type=ZoneType.SUPPLY)
                .build())
        
        # Should calculate price range from base candles
        base_candle = candles[1]
        self.assertEqual(zone.high, base_candle.high)
        self.assertEqual(zone.low, base_candle.low)
        self.assertEqual(zone.zone_type, ZoneType.SUPPLY)
        TestAssertions.assert_valid_zone(zone)
    
    def test_from_candles_multiple_base(self):
        """Test building zone from multiple base candles"""
        candles = TestDataFactory.create_multiple_base_pattern()
        
        zone = (self.builder.reset()
                .from_candles(candles, entry_idx=0, base_indices=[1, 2, 3], 
                             exit_idx=4, zone_type=ZoneType.SUPPLY)
                .build())
        
        # Should use min/max from all base candles
        base_candles = [candles[i] for i in [1, 2, 3]]
        expected_high = max(c.high for c in base_candles)
        expected_low = min(c.low for c in base_candles)
        
        self.assertEqual(zone.high, expected_high)
        self.assertEqual(zone.low, expected_low)
        self.assertEqual(zone.base_candles, (1, 2, 3))
        TestAssertions.assert_valid_zone(zone)
    
    def test_base_candles_sorting(self):
        """Test that base candles are sorted regardless of input order"""
        zone = (self.builder.reset()
                .with_base_candles(3, 1, 2)  # Unsorted input
                .with_legs(0, 4)
                .with_indices(0, 4)
                .build())
        
        self.assertEqual(zone.base_candles, (1, 2, 3))  # Should be sorted
    
    def test_validation_invalid_price_range(self):
        """Test validation catches invalid price ranges"""
        with self.assertRaises(ValueError):
            self.builder.reset().with_price_range(100.0, 110.0)  # high < low
    
    def test_validation_no_base_candles(self):
        """Test validation catches missing base candles"""
        with self.assertRaises(ValueError):
            self.builder.reset().with_base_candles().build()  # No base candles
    
    def test_validation_from_candles_no_base(self):
        """Test from_candles validation with no base indices"""
        candles = TestDataFactory.create_simple_supply_pattern()
        
        with self.assertRaises(ValueError):
            self.builder.reset().from_candles(
                candles, entry_idx=0, base_indices=[], 
                exit_idx=2, zone_type=ZoneType.SUPPLY
            )
    
    def test_validation_invalid_indices(self):
        """Test validation catches invalid index relationships"""
        with self.assertRaises(ValueError):
            # Start >= end
            (self.builder.reset()
             .with_indices(3, 2)  # start >= end
             .with_base_candles(1)
             .with_legs(0, 4)
             .build())
    
    def test_validation_entry_leg_position(self):
        """Test validation of entry leg position"""
        with self.assertRaises(ValueError):
            # Entry leg after base candles
            (self.builder.reset()
             .with_base_candles(1, 2)
             .with_legs(2, 4)  # entry_leg_index >= min(base_candles)
             .with_indices(0, 4)
             .build())
    
    def test_validation_exit_leg_position(self):
        """Test validation of exit leg position"""
        with self.assertRaises(ValueError):
            # Exit leg before base candles
            (self.builder.reset()
             .with_base_candles(2, 3)
             .with_legs(0, 2)  # exit_leg_index <= max(base_candles)
             .with_indices(0, 4)
             .build())
    
    def test_reset_clears_state(self):
        """Test that reset properly clears builder state"""
        # Build complex zone
        self.builder.demand_zone().with_price_range(200.0, 150.0).with_base_candles(5, 6, 7)
        
        # Reset and build simple zone
        zone = self.builder.reset().build()
        
        # Should have default values
        self.assertEqual(zone.zone_type, ZoneType.SUPPLY)  # Default
        self.assertEqual(zone.base_candles, (1,))  # Default
        TestAssertions.assert_valid_zone(zone)
    
    def test_method_chaining_fluency(self):
        """Test that all methods return self for fluent chaining"""
        result = (self.builder
                 .reset()
                 .supply_zone()
                 .with_price_range(120.0, 115.0)
                 .with_base_candles(1, 2)
                 .with_legs(0, 3)
                 .with_indices(0, 3))
        
        self.assertIs(result, self.builder)
    
    def test_builder_produces_valid_zones_for_algorithm(self):
        """Test that builder produces zones compatible with algorithm logic"""
        # Create a supply zone
        supply_zone = (self.builder.reset()
                      .supply_zone()
                      .with_price_range(115.0, 110.0)
                      .with_base_candles(1, 2)
                      .with_legs(0, 3)
                      .with_indices(0, 3)
                      .build())
        
        # Create matching candles for logic validation
        candles = [
            CandleBuilder().bullish(5).build(),   # Entry leg (bullish)
            CandleBuilder().doji().build(),       # Base candle
            CandleBuilder().doji().build(),       # Base candle
            CandleBuilder().bearish(5).build(),   # Exit leg (bearish)
        ]
        
        # Should pass supply zone logic validation
        TestAssertions.assert_supply_zone_logic(supply_zone, candles)


class TestCoreModels(unittest.TestCase):
    """Test the core data models"""
    
    def setUp(self):
        self.candle_builder = CandleBuilder()
        self.zone_builder = ZoneBuilder()
    
    def test_candle_creation_valid(self):
        """Test creating valid candles"""
        candle = self.candle_builder.with_ohlc(100, 105, 95, 102).build()
        
        self.assertEqual(candle.open, 100)
        self.assertEqual(candle.high, 105)
        self.assertEqual(candle.low, 95)
        self.assertEqual(candle.close, 102)
        TestAssertions.assert_valid_candle(candle)
    
    def test_candle_validation_invalid_high(self):
        """Test candle validation catches invalid high price"""
        with self.assertRaises(ValueError):
            Candle(
                timestamp=datetime.now(),
                open=100, high=95, low=90, close=102  # high < close
            )
    
    def test_candle_validation_invalid_low(self):
        """Test candle validation catches invalid low price"""
        with self.assertRaises(ValueError):
            Candle(
                timestamp=datetime.now(),
                open=100, high=105, low=102, close=99  # low > close
            )
    
    def test_candle_validation_negative_volume(self):
        """Test candle validation catches negative volume"""
        with self.assertRaises(ValueError):
            Candle(
                timestamp=datetime.now(),
                open=100, high=105, low=95, close=102,
                volume=-100  # negative volume
            )
    
    def test_zone_creation_valid(self):
        """Test creating valid zones"""
        zone = self.zone_builder.supply_zone().build()
        
        self.assertEqual(zone.zone_type, ZoneType.SUPPLY)
        TestAssertions.assert_valid_zone(zone)
    
    def test_zone_validation_invalid_price_range(self):
        """Test zone validation catches invalid price range"""
        with self.assertRaises(ValueError):
            Zone(
                zone_type=ZoneType.SUPPLY,
                start_index=0, end_index=2,
                high=100, low=105,  # high < low
                base_candles=(1,),
                entry_leg_index=0, exit_leg_index=2
            )
    
    def test_algorithm_config_validation(self):
        """Test algorithm configuration validation"""
        # Valid config
        config = AlgorithmConfig(body_ratio_threshold=1.5, min_base_candles=2, max_base_candles=5)
        self.assertEqual(config.body_ratio_threshold, 1.5)
        
        # Invalid configs
        with self.assertRaises(ValueError):
            AlgorithmConfig(body_ratio_threshold=-1.0)  # negative threshold
        
        with self.assertRaises(ValueError):
            AlgorithmConfig(min_base_candles=0)  # zero minimum
        
        with self.assertRaises(ValueError):
            AlgorithmConfig(min_base_candles=5, max_base_candles=3)  # max < min


class TestCandleAnalyzer(unittest.TestCase):
    """Test candle analysis functionality"""
    
    def setUp(self):
        self.analyzer = CandleAnalyzer()
        self.candle_builder = CandleBuilder()
    
    def test_analyze_bullish_candle(self):
        """Test analysis of bullish candle"""
        candle = self.candle_builder.with_ohlc(100, 108, 98, 105).build()
        metrics = self.analyzer.analyze_candle(candle)
        
        self.assertEqual(metrics.body_size, 5.0)  # |105 - 100|
        self.assertEqual(metrics.upper_wick, 3.0)  # 108 - 105
        self.assertEqual(metrics.lower_wick, 2.0)  # 100 - 98
        self.assertEqual(metrics.total_range, 10.0)  # 108 - 98
        self.assertEqual(metrics.body_to_wick_ratio, 1.0)  # 5 / (3 + 2)
        self.assertTrue(metrics.is_bullish)
    
    def test_analyze_bearish_candle(self):
        """Test analysis of bearish candle"""
        candle = self.candle_builder.with_ohlc(105, 108, 98, 100).build()
        metrics = self.analyzer.analyze_candle(candle)
        
        self.assertEqual(metrics.body_size, 5.0)  # |100 - 105|
        self.assertEqual(metrics.upper_wick, 3.0)  # 108 - 105
        self.assertEqual(metrics.lower_wick, 2.0)  # 100 - 98
        self.assertFalse(metrics.is_bullish)
    
    def test_analyze_doji_candle(self):
        """Test analysis of doji candle (open = close)"""
        candle = self.candle_builder.doji(wick_size=3.0).build()
        metrics = self.analyzer.analyze_candle(candle)
        
        self.assertEqual(metrics.body_size, 0.0)
        self.assertEqual(metrics.body_to_wick_ratio, 0.0)  # 0 / (3 + 3)
    
    def test_analyze_no_wicks_candle(self):
        """Test analysis of candle with no wicks"""
        candle = self.candle_builder.with_ohlc(100, 105, 100, 105).build()
        metrics = self.analyzer.analyze_candle(candle)
        
        self.assertEqual(metrics.upper_wick, 0.0)
        self.assertEqual(metrics.lower_wick, 0.0)
        self.assertEqual(metrics.body_to_wick_ratio, float('inf'))  # division by zero case
    
    def test_analyze_multiple_candles(self):
        """Test analyzing multiple candles"""
        candles = [
            self.candle_builder.bullish().build(),
            self.candle_builder.bearish().build(),
            self.candle_builder.doji().build()
        ]
        
        metrics_list = self.analyzer.analyze_candles(candles)
        
        self.assertEqual(len(metrics_list), 3)
        self.assertTrue(metrics_list[0].is_bullish)
        self.assertFalse(metrics_list[1].is_bullish)
        self.assertEqual(metrics_list[2].body_size, 0.0)


class TestCandleClassifier(unittest.TestCase):
    """Test candle classification functionality"""
    
    def setUp(self):
        self.strategy = BodyRatioClassificationStrategy(threshold=1.0)
        self.classifier = CandleClassifier(self.strategy)
        self.candle_builder = CandleBuilder()
        self.analyzer = CandleAnalyzer()
    
    def test_classify_leg_candle(self):
        """Test classification of LEG candle (high body-to-wick ratio)"""
        candle = self.candle_builder.large_body(body_size=10, small_wicks=1).build()
        metrics = self.analyzer.analyze_candle(candle)
        
        classification = self.strategy.classify(candle, metrics)
        self.assertEqual(classification, CandleType.LEG)
    
    def test_classify_base_candle(self):
        """Test classification of BASE candle (low body-to-wick ratio)"""
        candle = self.candle_builder.large_wicks(body_size=1, wick_size=5).build()
        metrics = self.analyzer.analyze_candle(candle)
        
        classification = self.strategy.classify(candle, metrics)
        self.assertEqual(classification, CandleType.BASE)
    
    def test_classify_multiple_candles(self):
        """Test classifying multiple candles"""
        candles = [
            self.candle_builder.large_body().build(),  # LEG
            self.candle_builder.large_wicks().build(),  # BASE
            self.candle_builder.large_body().build(),  # LEG
        ]
        
        metrics = self.analyzer.analyze_candles(candles)
        classifications = self.classifier.classify_candles(candles, metrics)
        
        expected = [CandleType.LEG, CandleType.BASE, CandleType.LEG]
        self.assertEqual(classifications, expected)
    
    def test_classify_mismatched_lengths(self):
        """Test error handling for mismatched candles and metrics lengths"""
        candles = [self.candle_builder.build()]
        metrics = []  # empty list
        
        with self.assertRaises(ValueError):
            self.classifier.classify_candles(candles, metrics)


class TestZoneDetection(unittest.TestCase):
    """Test zone detection functionality"""
    
    def setUp(self):
        self.pattern_detector = ZonePatternDetector(min_base_candles=1, max_base_candles=5)
        self.type_classifier = ZoneTypeClassifier()
        self.zone_builder = ZoneBuilder()
        self.candle_builder = CandleBuilder()
    
    def test_find_simple_pattern(self):
        """Test finding simple LEG->BASE->LEG pattern"""
        classifications = [CandleType.LEG, CandleType.BASE, CandleType.LEG]
        candles = [self.candle_builder.build() for _ in range(3)]
        
        patterns = self.pattern_detector.find_patterns(candles, classifications)
        
        self.assertEqual(len(patterns), 1)
        entry_idx, base_indices, exit_idx = patterns[0]
        self.assertEqual(entry_idx, 0)
        self.assertEqual(base_indices, [1])
        self.assertEqual(exit_idx, 2)
    
    def test_find_multiple_base_pattern(self):
        """Test finding pattern with multiple BASE candles"""
        classifications = [CandleType.LEG, CandleType.BASE, CandleType.BASE, CandleType.LEG]
        candles = [self.candle_builder.build() for _ in range(4)]
        
        patterns = self.pattern_detector.find_patterns(candles, classifications)
        
        self.assertEqual(len(patterns), 1)
        entry_idx, base_indices, exit_idx = patterns[0]
        self.assertEqual(entry_idx, 0)
        self.assertEqual(base_indices, [1, 2])
        self.assertEqual(exit_idx, 3)
    
    def test_no_patterns_found(self):
        """Test when no valid patterns exist"""
        classifications = [CandleType.BASE, CandleType.BASE, CandleType.BASE]
        candles = [self.candle_builder.build() for _ in range(3)]
        
        patterns = self.pattern_detector.find_patterns(candles, classifications)
        
        self.assertEqual(len(patterns), 0)
    
    def test_classify_supply_zone_type(self):
        """Test supply zone type classification"""
        entry_leg = self.candle_builder.bullish().build()  # bullish entry
        exit_leg = self.candle_builder.bearish().build()   # bearish exit
        
        zone_type = self.type_classifier.classify_zone_type(entry_leg, exit_leg)
        
        self.assertEqual(zone_type, ZoneType.SUPPLY)
    
    def test_classify_demand_zone_type(self):
        """Test demand zone type classification"""
        entry_leg = self.candle_builder.bearish().build()  # bearish entry
        exit_leg = self.candle_builder.bullish().build()   # bullish exit
        
        zone_type = self.type_classifier.classify_zone_type(entry_leg, exit_leg)
        
        self.assertEqual(zone_type, ZoneType.DEMAND)
    
    def test_classify_ambiguous_zone_type(self):
        """Test ambiguous zone type (same direction legs)"""
        entry_leg = self.candle_builder.bullish().build()  # bullish entry
        exit_leg = self.candle_builder.bullish().build()   # bullish exit
        
        zone_type = self.type_classifier.classify_zone_type(entry_leg, exit_leg)
        
        self.assertIsNone(zone_type)


class TestIntegratedAlgorithm(unittest.TestCase):
    """Integration tests for the complete algorithm"""
    
    def setUp(self):
        self.algorithm = SupplyDemandAlgorithm()
    
    def test_analyze_simple_supply_pattern(self):
        """Test complete analysis of simple supply pattern"""
        candles = TestDataFactory.create_simple_supply_pattern()
        
        result = self.algorithm.analyze_market_data(candles)
        
        TestAssertions.assert_analysis_result_consistency(result)
        self.assertEqual(result.total_zones, 1)
        self.assertEqual(len(result.supply_zones), 1)
        self.assertEqual(len(result.demand_zones), 0)
        
        # Verify the supply zone
        supply_zone = result.supply_zones[0]
        TestAssertions.assert_supply_zone_logic(supply_zone, list(result.candles))
    
    def test_analyze_simple_demand_pattern(self):
        """Test complete analysis of simple demand pattern"""
        candles = TestDataFactory.create_simple_demand_pattern()
        
        result = self.algorithm.analyze_market_data(candles)
        
        TestAssertions.assert_analysis_result_consistency(result)
        self.assertEqual(result.total_zones, 1)
        self.assertEqual(len(result.supply_zones), 0)
        self.assertEqual(len(result.demand_zones), 1)
        
        # Verify the demand zone
        demand_zone = result.demand_zones[0]
        TestAssertions.assert_demand_zone_logic(demand_zone, list(result.candles))
    
    def test_analyze_multiple_base_pattern(self):
        """Test analysis of pattern with multiple base candles"""
        candles = TestDataFactory.create_multiple_base_pattern()
        
        result = self.algorithm.analyze_market_data(candles)
        
        TestAssertions.assert_analysis_result_consistency(result)
        self.assertEqual(result.total_zones, 1)
        
        zone = result.zones[0]
        self.assertEqual(len(zone.base_candles), 3)  # Three base candles
    
    def test_analyze_complex_data(self):
        """Test analysis of complex market data with multiple patterns"""
        candles = TestDataFactory.create_complex_market_data()
        
        result = self.algorithm.analyze_market_data(candles)
        
        TestAssertions.assert_analysis_result_consistency(result)
        self.assertGreater(result.total_zones, 0)
        
        # Verify all zones are valid
        for zone in result.zones:
            TestAssertions.assert_valid_zone(zone)
    
    def test_analyze_ambiguous_pattern(self):
        """Test that ambiguous patterns are correctly rejected"""
        candles = TestDataFactory.create_ambiguous_pattern()
        
        result = self.algorithm.analyze_market_data(candles)
        
        # Should detect no zones due to ambiguous pattern
        self.assertEqual(result.total_zones, 0)
    
    def test_get_zone_summary(self):
        """Test zone summary generation"""
        candles = TestDataFactory.create_complex_market_data()
        result = self.algorithm.analyze_market_data(candles)
        
        summary = self.algorithm.get_zone_summary(list(result.zones))
        
        self.assertIn('count', summary)
        self.assertIn('supply_count', summary)
        self.assertIn('demand_count', summary)
        self.assertEqual(summary['count'], result.total_zones)
        
        if result.total_zones > 0:
            self.assertIn('avg_range', summary)
            self.assertIn('avg_base_candles', summary)
    
    def test_empty_candle_list(self):
        """Test error handling for empty candle list"""
        with self.assertRaises(ValueError):
            self.algorithm.analyze_market_data([])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        self.algorithm = SupplyDemandAlgorithm()
    
    def test_single_candle(self):
        """Test analysis with single candle"""
        candles = [CandleBuilder().build()]
        
        result = self.algorithm.analyze_market_data(candles)
        
        self.assertEqual(result.total_zones, 0)  # Cannot form patterns with one candle
    
    def test_two_candles(self):
        """Test analysis with two candles"""
        candles = [CandleBuilder().build(), CandleBuilder().bullish().build()]
        
        result = self.algorithm.analyze_market_data(candles)
        
        self.assertEqual(result.total_zones, 0)  # Need at least 3 for LEG->BASE->LEG
    
    def test_all_base_candles(self):
        """Test analysis with all BASE candles"""
        edge_cases = TestDataFactory.create_edge_case_data()
        candles = edge_cases['all_base_candles']
        
        result = self.algorithm.analyze_market_data(candles)
        
        self.assertEqual(result.total_zones, 0)  # No LEG candles to form patterns
    
    def test_all_leg_candles(self):
        """Test analysis with all LEG candles"""
        edge_cases = TestDataFactory.create_edge_case_data()
        candles = edge_cases['all_leg_candles']
        
        result = self.algorithm.analyze_market_data(candles)
        
        self.assertEqual(result.total_zones, 0)  # No BASE candles to form patterns
    
    def test_custom_configuration(self):
        """Test algorithm with custom configuration"""
        config = AlgorithmConfig(
            body_ratio_threshold=2.0,
            min_base_candles=2,
            max_base_candles=3
        )
        algorithm = SupplyDemandAlgorithm(config)
        
        candles = TestDataFactory.create_simple_supply_pattern()
        result = algorithm.analyze_market_data(candles)
        
        # With stricter requirements, may find fewer zones
        TestAssertions.assert_analysis_result_consistency(result)


class TestPropertyBased(unittest.TestCase):
    """Property-based tests using randomly generated data"""
    
    def test_random_candles_produce_valid_results(self):
        """Test that random valid candles always produce valid results"""
        algorithm = SupplyDemandAlgorithm()
        
        for _ in range(10):  # Run multiple iterations
            candles = RandomDataGenerator.generate_random_candles(
                count=20, min_price=50, max_price=200
            )
            
            result = algorithm.analyze_market_data(candles)
            
            # Properties that should always hold
            TestAssertions.assert_analysis_result_consistency(result)
            self.assertEqual(len(result.candles), 20)
            self.assertGreaterEqual(result.total_zones, 0)
            
            # All detected zones should be valid
            for zone in result.zones:
                TestAssertions.assert_valid_zone(zone)
    
    def test_random_configs_work(self):
        """Test that random valid configurations work correctly"""
        candles = TestDataFactory.create_simple_supply_pattern()
        
        for _ in range(5):
            config = RandomDataGenerator.generate_random_config()
            algorithm = SupplyDemandAlgorithm(config)
            
            result = algorithm.analyze_market_data(candles)
            TestAssertions.assert_analysis_result_consistency(result)
    
    def test_invariant_zone_count_relationship(self):
        """Test invariant: supply_zones + demand_zones = total_zones"""
        algorithm = SupplyDemandAlgorithm()
        
        for _ in range(10):
            candles = RandomDataGenerator.generate_random_candles(count=15)
            result = algorithm.analyze_market_data(candles)
            
            # This should always be true
            self.assertEqual(
                len(result.supply_zones) + len(result.demand_zones),
                result.total_zones
            )


class TestPerformance(unittest.TestCase):
    """Performance tests for the algorithm"""
    
    def test_large_dataset_performance(self):
        """Test performance with large dataset"""
        import time
        
        # Generate large dataset
        candles = RandomDataGenerator.generate_random_candles(count=1000)
        algorithm = SupplyDemandAlgorithm()
        
        start_time = time.time()
        result = algorithm.analyze_market_data(candles)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(execution_time, 5.0, "Algorithm should complete within 5 seconds")
        TestAssertions.assert_analysis_result_consistency(result)
    
    def test_memory_usage_reasonable(self):
        """Test that memory usage is reasonable"""
        import sys
        
        candles = RandomDataGenerator.generate_random_candles(count=500)
        algorithm = SupplyDemandAlgorithm()
        
        # Measure approximate memory usage
        initial_size = sys.getsizeof(candles)
        result = algorithm.analyze_market_data(candles)
        result_size = sys.getsizeof(result)
        
        # Result should not be disproportionately larger than input
        # (This is a rough heuristic, adjust as needed)
        self.assertLess(result_size, initial_size * 10)


class TestMockIntegration(unittest.TestCase):
    """Test integration with mock objects"""
    
    def test_algorithm_with_mocked_components(self):
        """Test that algorithm works correctly with mocked components"""
        # This tests the architecture's dependency injection capability
        
        mock_analyzer = MockObjects.create_mock_candle_analyzer()
        mock_classifier = MockObjects.create_mock_classifier()
        mock_detector = MockObjects.create_mock_zone_detector()
        
        # These tests would require dependency injection in the main algorithm
        # which isn't fully implemented in the current version but demonstrates
        # how proper SRP enables easy testing with mocks
        
        candles = [CandleBuilder().build()]
        
        # Verify mocks are called appropriately
        # mock_analyzer.analyze_candles.assert_called_once_with(candles)
        # This would work if we had proper dependency injection


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)