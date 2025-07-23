"""
Comprehensive test suite for the Trend Detection Algorithm
Organized by component with unit, integration, and property-based tests
"""

import unittest
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np

# Core trend detection imports
from trend_business_logic import (
    TrendAnalysisEngine, SwingDetector, BasicSwingDetectionStrategy,
    PatternMatcher, TrendFactory, TrendBreakoutValidator,
    TrendTerminationDetector, TrendClassifier, TrendManager
)

from trend_core_models import (
    SwingPoint, SwingType, Trend, TrendDirection, TrendSignificance,
    TrendPattern, TrendAnalysisConfig, TrendAnalysisResult
)

from trend_utilities import (
    TrendScenarioBuilder, SwingPointBuilder, TrendPatternBuilder,
    TrendTestDataFactory, MarketConditionGenerator,
    TrendDataProcessor, TrendValidationUtils  # ← ADD THESE TWO
)

# Import test-specific utilities (KEPT IN TEST FILE)
from trend_test_utilities import (
    TrendTestAssertions, TrendMockObjects, TrendTestScenarios
)

# Use existing Candle from supply/demand refactoring
from utilities import Candle, CandleBuilder


class TestTrendCoreModels(unittest.TestCase):
    """Test the core data models"""
    
    def setUp(self):
        self.swing_builder = SwingPointBuilder()
        self.pattern_builder = TrendPatternBuilder()
    
    def test_swing_point_creation_valid(self):
        """Test creating valid swing points"""
        swing = (self.swing_builder
                .at_candle(5)
                .with_price(105.5)
                .swing_high()
                .build())
        
        self.assertEqual(swing.candle_index, 5)
        self.assertEqual(swing.price, 105.5)
        self.assertEqual(swing.swing_type, SwingType.HIGH)
        TrendTestAssertions.assert_valid_swing_point(swing)
    
    def test_swing_point_validation_negative_index(self):
        """Test swing point validation catches negative candle index"""
        with self.assertRaises(ValueError):
            SwingPoint(
                candle_index=-1,
                price=100.0,
                swing_type=SwingType.HIGH,
                timestamp=datetime.now()
            )
    
    def test_swing_point_validation_zero_price(self):
        """Test swing point validation catches zero/negative price"""
        with self.assertRaises(ValueError):
            SwingPoint(
                candle_index=1,
                price=0.0,
                swing_type=SwingType.LOW,
                timestamp=datetime.now()
            )
    
    def test_trend_pattern_creation_valid(self):
        """Test creating valid trend patterns"""
        swings = [
            self.swing_builder.reset().at_candle(1).with_price(98).swing_low().build(),
            self.swing_builder.reset().at_candle(3).with_price(108).swing_high().build(),
            self.swing_builder.reset().at_candle(5).with_price(102).swing_low().build()
        ]
        
        pattern = (self.pattern_builder
                  .uptrend_pattern()
                  .with_swings(*swings)
                  .build())
        
        self.assertEqual(pattern.pattern_type, TrendDirection.UP)
        self.assertEqual(len(pattern.formation_swings), 3)
        TrendTestAssertions.assert_valid_trend_pattern(pattern)
    
    def test_trend_pattern_validation_insufficient_swings(self):
        """Test pattern validation catches insufficient swings"""
        with self.assertRaises(ValueError):
            TrendPattern(
                formation_swings=tuple([self.swing_builder.build()]),  # Only 1 swing
                pattern_type=TrendDirection.UP,
                start_index=0,
                end_index=2
            )
    
    def test_trend_pattern_validation_unsorted_swings(self):
        """Test pattern validation catches unsorted swings"""
        swings = [
            self.swing_builder.reset().at_candle(5).swing_low().build(),  # Later candle first
            self.swing_builder.reset().at_candle(1).swing_high().build()  # Earlier candle second
        ]
        
        with self.assertRaises(ValueError):
            TrendPattern(
                formation_swings=tuple(swings),
                pattern_type=TrendDirection.UP,
                start_index=1,
                end_index=5
            )
    
    def test_trend_analysis_config_validation(self):
        """Test analysis configuration validation"""
        # Valid config
        config = TrendAnalysisConfig(
            sideways_range_threshold=0.12,
            moveout_threshold=1.5,
            major_trend_min_duration=5
        )
        self.assertEqual(config.sideways_range_threshold, 0.12)
        
        # Invalid configs
        with self.assertRaises(ValueError):
            TrendAnalysisConfig(sideways_range_threshold=-0.1)  # Negative threshold
        
        with self.assertRaises(ValueError):
            TrendAnalysisConfig(major_trend_min_duration=0)  # Zero duration
    
    def test_trend_creation_valid(self):
        """Test creating valid trends"""
        pattern = self.pattern_builder.uptrend_pattern().build()
        
        trend = Trend(
            trend_id=1,
            direction=TrendDirection.UP,
            start_index=0,
            end_index=5,
            controlling_swing=self.swing_builder.build(),
            formation_pattern=pattern,
            significance=TrendSignificance.MAJOR
        )
        
        self.assertEqual(trend.direction, TrendDirection.UP)
        self.assertEqual(trend.significance, TrendSignificance.MAJOR)
        TrendTestAssertions.assert_valid_trend(trend)


class TestSwingDetection(unittest.TestCase):
    """Test swing detection functionality"""
    
    def setUp(self):
        self.strategy = BasicSwingDetectionStrategy()
        self.detector = SwingDetector(self.strategy)
    
    def test_detect_clear_swings(self):
        """Test detection of clear swing points"""
        candles = TrendTestDataFactory.create_simple_uptrend_scenario()
        
        swings = self.detector.detect_swings(candles)
        
        self.assertGreater(len(swings), 0)
        for swing in swings:
            TrendTestAssertions.assert_valid_swing_point(swing)
    
    def test_detect_no_swings_identical_candles(self):
        """Test no swings detected in flat market"""
        edge_cases = TrendTestDataFactory.create_edge_case_scenarios()
        identical_candles = edge_cases['identical_candles']
        
        swings = self.detector.detect_swings(identical_candles)
        
        # Should find no swings in identical candles
        self.assertEqual(len(swings), 0)
    
    def test_detect_swings_insufficient_data(self):
        """Test swing detection with insufficient data"""
        edge_cases = TrendTestDataFactory.create_edge_case_scenarios()
        
        # Single candle
        swings = self.detector.detect_swings(edge_cases['single_candle'])
        self.assertEqual(len(swings), 0)
        
        # Two candles
        swings = self.detector.detect_swings(edge_cases['two_candles'])
        self.assertEqual(len(swings), 0)
    
    def test_swing_alternation_pattern(self):
        """Test that detected swings alternate HIGH-LOW or LOW-HIGH"""
        candles = TrendTestDataFactory.create_multi_trend_scenario()
        
        swings = self.detector.detect_swings(candles)
        
        if len(swings) >= 2:
            # Check that consecutive swings are different types
            for i in range(1, len(swings)):
                self.assertNotEqual(swings[i].swing_type, swings[i-1].swing_type,
                                  "Consecutive swings should be different types")
    
    def test_swing_price_extremes(self):
        """Test that swing highs/lows are actual local extremes"""
        candles = TrendTestDataFactory.create_simple_uptrend_scenario()
        
        swings = self.detector.detect_swings(candles)
        
        for swing in swings:
            candle = candles[swing.candle_index]
            
            if swing.swing_type == SwingType.HIGH:
                self.assertEqual(swing.price, candle.high,
                               "Swing high should use candle's high price")
            else:
                self.assertEqual(swing.price, candle.low,
                               "Swing low should use candle's low price")


class TestPatternMatching(unittest.TestCase):
    """Test pattern matching functionality"""
    
    def setUp(self):
        self.config = TrendAnalysisConfig()
        self.pattern_matcher = PatternMatcher(self.config)
    
    def test_find_uptrend_patterns(self):
        """Test uptrend pattern (L-H-L) detection"""
        # Create L-H-L swing sequence
        swings = [
            SwingPointBuilder().at_candle(1).with_price(95).swing_low().build(),
            SwingPointBuilder().at_candle(3).with_price(105).swing_high().build(),
            SwingPointBuilder().at_candle(5).with_price(100).swing_low().build(),  # Higher low
            SwingPointBuilder().at_candle(7).with_price(110).swing_high().build()
        ]
        
        patterns = self.pattern_matcher.find_uptrend_patterns(swings)
        
        self.assertGreater(len(patterns), 0)
        for pattern in patterns:
            self.assertEqual(pattern.pattern_type, TrendDirection.UP)
            TrendTestAssertions.assert_valid_trend_pattern(pattern)
    
    def test_find_downtrend_patterns(self):
        """Test downtrend pattern (H-L-H) detection"""
        # Create H-L-H swing sequence
        swings = [
            SwingPointBuilder().at_candle(1).with_price(105).swing_high().build(),
            SwingPointBuilder().at_candle(3).with_price(95).swing_low().build(),
            SwingPointBuilder().at_candle(5).with_price(100).swing_high().build(),  # Lower high
            SwingPointBuilder().at_candle(7).with_price(90).swing_low().build()
        ]
        
        patterns = self.pattern_matcher.find_downtrend_patterns(swings)
        
        self.assertGreater(len(patterns), 0)
        for pattern in patterns:
            self.assertEqual(pattern.pattern_type, TrendDirection.DOWN)
            TrendTestAssertions.assert_valid_trend_pattern(pattern)
    
    def test_reject_invalid_uptrend_pattern(self):
        """Test rejection of invalid uptrend (lower low)"""
        # Create L-H-L with lower low (should be rejected)
        swings = [
            SwingPointBuilder().at_candle(1).with_price(100).swing_low().build(),
            SwingPointBuilder().at_candle(3).with_price(105).swing_high().build(),
            SwingPointBuilder().at_candle(5).with_price(95).swing_low().build()  # Lower low
        ]
        
        patterns = self.pattern_matcher.find_uptrend_patterns(swings)
        
        # Should find no valid uptrend patterns
        self.assertEqual(len(patterns), 0)
    
    def test_reject_invalid_downtrend_pattern(self):
        """Test rejection of invalid downtrend (higher high)"""
        # Create H-L-H with higher high (should be rejected)
        swings = [
            SwingPointBuilder().at_candle(1).with_price(100).swing_high().build(),
            SwingPointBuilder().at_candle(3).with_price(95).swing_low().build(),
            SwingPointBuilder().at_candle(5).with_price(105).swing_high().build()  # Higher high
        ]
        
        patterns = self.pattern_matcher.find_downtrend_patterns(swings)
        
        # Should find no valid downtrend patterns
        self.assertEqual(len(patterns), 0)
    
    def test_find_sideways_patterns(self):
        """Test sideways pattern detection"""
        candles = TrendTestDataFactory.create_simple_sideways_scenario()
        swings = SwingDetector(BasicSwingDetectionStrategy()).detect_swings(candles)
        
        patterns = self.pattern_matcher.find_sideways_patterns(candles, swings, 0, len(candles)-1)
        
        if patterns:  # May or may not find sideways depending on exact data
            for pattern in patterns:
                self.assertEqual(pattern.pattern_type, TrendDirection.SIDEWAYS)
                TrendTestAssertions.assert_valid_trend_pattern(pattern)


class TestTrendValidation(unittest.TestCase):
    """Test trend breakout validation"""
    
    def setUp(self):
        self.config = TrendAnalysisConfig()
        self.validator = TrendBreakoutValidator(self.config)
        self.candle_builder = CandleBuilder()
    
    def test_validate_uptrend_breakout(self):
        """Test uptrend breakout validation"""
        # Create L-H-L pattern
        swings = [
            SwingPointBuilder().at_candle(1).with_price(95).swing_low().build(),
            SwingPointBuilder().at_candle(3).with_price(105).swing_high().build(),
            SwingPointBuilder().at_candle(5).with_price(100).swing_low().build()
        ]
        
        pattern = (TrendPatternBuilder()
                  .uptrend_pattern()
                  .with_swings(*swings)
                  .build())
        
        # Test breakout above swing high
        breakout_candle = self.candle_builder.with_ohlc(106, 110, 105, 108).build()
        is_valid = self.validator.validate_uptrend_breakout(pattern, breakout_candle)
        self.assertTrue(is_valid)
        
        # Test no breakout
        no_breakout_candle = self.candle_builder.with_ohlc(102, 104, 101, 103).build()
        is_valid = self.validator.validate_uptrend_breakout(pattern, no_breakout_candle)
        self.assertFalse(is_valid)
    
    def test_validate_downtrend_breakout(self):
        """Test downtrend breakout validation"""
        # Create H-L-H pattern
        swings = [
            SwingPointBuilder().at_candle(1).with_price(105).swing_high().build(),
            SwingPointBuilder().at_candle(3).with_price(95).swing_low().build(),
            SwingPointBuilder().at_candle(5).with_price(100).swing_high().build()
        ]
        
        pattern = (TrendPatternBuilder()
                  .downtrend_pattern()
                  .with_swings(*swings)
                  .build())
        
        # Test breakdown below swing low
        breakdown_candle = self.candle_builder.with_ohlc(96, 98, 90, 92).build()
        is_valid = self.validator.validate_downtrend_breakout(pattern, breakdown_candle)
        self.assertTrue(is_valid)
        
        # Test no breakdown
        no_breakdown_candle = self.candle_builder.with_ohlc(97, 99, 96, 98).build()
        is_valid = self.validator.validate_downtrend_breakout(pattern, no_breakdown_candle)
        self.assertFalse(is_valid)
    
    def test_validate_moveout_strength(self):
        """Test moveout strength validation using body-to-wick ratio"""
        # Strong moveout (large body, small wicks)
        strong_candle = self.candle_builder.large_body(8, 0.5).build()
        is_strong = self.validator.validate_moveout_strength(strong_candle)
        self.assertTrue(is_strong)
        
        # Weak moveout (small body, large wicks)
        weak_candle = self.candle_builder.large_wicks(1, 5).build()
        is_strong = self.validator.validate_moveout_strength(weak_candle)
        self.assertFalse(is_strong)


class TestTrendFactory(unittest.TestCase):
    """Test trend creation from patterns"""
    
    def setUp(self):
        self.config = TrendAnalysisConfig()
        self.factory = TrendFactory(self.config)
    
    def test_create_uptrend_from_pattern(self):
        """Test creating uptrend from L-H-L pattern"""
        swings = [
            SwingPointBuilder().at_candle(1).with_price(95).swing_low().build(),
            SwingPointBuilder().at_candle(3).with_price(105).swing_high().build(),
            SwingPointBuilder().at_candle(5).with_price(100).swing_low().build()
        ]
        
        pattern = (TrendPatternBuilder()
                  .uptrend_pattern()
                  .with_swings(*swings)
                  .build())
        
        trend = self.factory.create_trend_from_pattern(pattern, 7, True, True)
        
        self.assertEqual(trend.direction, TrendDirection.UP)
        self.assertTrue(trend.breakout_confirmed)
        self.assertTrue(trend.moveout_confirmed)
        self.assertTrue(trend.is_active)
        TrendTestAssertions.assert_valid_trend(trend)
        TrendTestAssertions.assert_uptrend_logic(trend)
    
    def test_create_downtrend_from_pattern(self):
        """Test creating downtrend from H-L-H pattern"""
        swings = [
            SwingPointBuilder().at_candle(1).with_price(105).swing_high().build(),
            SwingPointBuilder().at_candle(3).with_price(95).swing_low().build(),
            SwingPointBuilder().at_candle(5).with_price(100).swing_high().build()
        ]
        
        pattern = (TrendPatternBuilder()
                  .downtrend_pattern()
                  .with_swings(*swings)
                  .build())
        
        trend = self.factory.create_trend_from_pattern(pattern, 7, True, False)
        
        self.assertEqual(trend.direction, TrendDirection.DOWN)
        self.assertTrue(trend.breakout_confirmed)
        self.assertFalse(trend.moveout_confirmed)
        TrendTestAssertions.assert_valid_trend(trend)
        TrendTestAssertions.assert_downtrend_logic(trend)
    
    def test_trend_id_increment(self):
        """Test that trend IDs are incremented correctly"""
        pattern = TrendPatternBuilder().uptrend_pattern().build()
        
        trend1 = self.factory.create_trend_from_pattern(pattern, 5)
        trend2 = self.factory.create_trend_from_pattern(pattern, 7)
        
        self.assertEqual(trend2.trend_id, trend1.trend_id + 1)


class TestTrendManager(unittest.TestCase):
    """Test trend management functionality"""
    
    def setUp(self):
        self.config = TrendAnalysisConfig()
        self.manager = TrendManager(self.config)
        self.factory = TrendFactory(self.config)
    
    def test_add_and_get_trends(self):
        """Test adding trends and retrieving them"""
        pattern = TrendPatternBuilder().uptrend_pattern().build()
        trend = self.factory.create_trend_from_pattern(pattern, 5)
        
        self.manager.add_trend(trend)
        
        self.assertIn(trend, self.manager.active_trends)
        self.assertIn(trend, self.manager.get_all_trends())
    
    def test_terminate_trend(self):
        """Test trend termination"""
        pattern = TrendPatternBuilder().uptrend_pattern().build()
        trend = self.factory.create_trend_from_pattern(pattern, 5)
        
        self.manager.add_trend(trend)
        
        # Terminate trend
        genesis_swing = SwingPointBuilder().at_candle(10).swing_high().build()
        genesis_point = self.manager.terminate_trend(trend, 10, genesis_swing)
        
        self.assertNotIn(trend, self.manager.active_trends)
        self.assertEqual(len(self.manager.terminated_trends), 1)
        self.assertIsNotNone(genesis_point)
        self.assertEqual(genesis_point.swing, genesis_swing)
    
    def test_resolve_temporal_conflicts(self):
        """Test resolution of overlapping trends"""
        # Create two overlapping trends
        pattern1 = TrendPatternBuilder().uptrend_pattern().with_indices(0, 10).build()
        pattern2 = TrendPatternBuilder().downtrend_pattern().with_indices(5, 15).build()
        
        trend1 = self.factory.create_trend_from_pattern(pattern1, 10)
        trend2 = self.factory.create_trend_from_pattern(pattern2, 15)
        
        # Make trend1 more dominant
        trend1 = Trend(
            trend_id=trend1.trend_id,
            direction=trend1.direction,
            start_index=trend1.start_index,
            end_index=trend1.end_index,
            controlling_swing=trend1.controlling_swing,
            formation_pattern=trend1.formation_pattern,
            significance=trend1.significance,
            is_active=trend1.is_active,
            price_range=100.0,  # Larger range
            duration=15,        # Longer duration
            dominance_score=1500.0  # Higher dominance
        )
        
        self.manager.add_trend(trend1)
        self.manager.add_trend(trend2)
        
        # Resolve conflicts
        self.manager.resolve_temporal_conflicts(20)
        
        # Should keep the more dominant trend active
        active_ids = [t.trend_id for t in self.manager.active_trends]
        self.assertIn(trend1.trend_id, active_ids)


class TestIntegratedTrendAnalysis(unittest.TestCase):
    """Integration tests for the complete trend analysis engine"""
    
    def setUp(self):
        self.config = TrendAnalysisConfig()
        self.engine = TrendAnalysisEngine(self.config)
    
    def test_analyze_simple_uptrend(self):
        """Test complete analysis of simple uptrend"""
        candles = TrendTestDataFactory.create_simple_uptrend_scenario()
        
        result = self.engine.analyze_trends(candles)
        
        TrendTestAssertions.assert_analysis_result_consistency(result)
        self.assertGreater(len(result.swings), 0)
        
        # Should detect at least one trend
        if result.trends:
            TrendTestAssertions.assert_contains_trend_direction(
                list(result.trends), TrendDirection.UP
            )
    
    def test_analyze_simple_downtrend(self):
        """Test complete analysis of simple downtrend"""
        candles = TrendTestDataFactory.create_simple_downtrend_scenario()
        
        result = self.engine.analyze_trends(candles)
        
        TrendTestAssertions.assert_analysis_result_consistency(result)
        
        # Should detect at least one trend
        if result.trends:
            TrendTestAssertions.assert_contains_trend_direction(
                list(result.trends), TrendDirection.DOWN
            )
    
    def test_analyze_multi_trend_scenario(self):
        """Test analysis of complex multi-trend scenario"""
        candles = TrendTestDataFactory.create_multi_trend_scenario()
        
        result = self.engine.analyze_trends(candles)
        
        TrendTestAssertions.assert_analysis_result_consistency(result)
        self.assertGreater(len(result.trends), 1)
        
        # Should detect multiple trend types
        directions = [t.direction for t in result.trends]
        self.assertGreater(len(set(directions)), 1, "Should detect multiple trend directions")
    
    def test_analyze_with_custom_window(self):
        """Test analysis with custom analysis window"""
        candles = TrendTestDataFactory.create_multi_trend_scenario()
        
        # Analyze only middle portion
        start_idx = len(candles) // 4
        end_idx = 3 * len(candles) // 4
        
        result = self.engine.analyze_trends(candles, start_idx, end_idx)
        
        TrendTestAssertions.assert_analysis_result_consistency(result)
        self.assertEqual(result.analysis_window, (start_idx, end_idx))
        self.assertEqual(result.total_candles_analyzed, end_idx - start_idx + 1)
    
    def test_trend_classification(self):
        """Test that trends are properly classified by significance"""
        candles = TrendTestDataFactory.create_multi_trend_scenario()
        
        result = self.engine.analyze_trends(candles)
        
        # Check that some trends were classified as major
        major_trends = [t for t in result.trends if t.significance == TrendSignificance.MAJOR]
        minor_trends = [t for t in result.trends if t.significance == TrendSignificance.MINOR]
        
        # At least some classification should have occurred
        self.assertGreater(len(major_trends) + len(minor_trends), 0)
        
        # Major trends should have longer duration/larger range
        if major_trends and minor_trends:
            avg_major_duration = np.mean([t.duration for t in major_trends])
            avg_minor_duration = np.mean([t.duration for t in minor_trends])
            
            # Major trends should generally be longer (allow some tolerance)
            self.assertGreaterEqual(avg_major_duration * 0.8, avg_minor_duration * 0.8)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        self.engine = TrendAnalysisEngine()
    
    def test_empty_candle_list(self):
        """Test analysis with empty candle list"""
        with self.assertRaises(ValueError):
            self.engine.analyze_trends([])
    
    def test_single_candle(self):
        """Test analysis with single candle"""
        edge_cases = TrendTestDataFactory.create_edge_case_scenarios()
        single_candle = edge_cases['single_candle']
        
        result = self.engine.analyze_trends(single_candle)
        
        # Should complete without error but find no trends
        self.assertEqual(len(result.swings), 0)
        self.assertEqual(len(result.trends), 0)
    
    def test_identical_candles(self):
        """Test analysis with identical candles (no swings possible)"""
        edge_cases = TrendTestDataFactory.create_edge_case_scenarios()
        identical_candles = edge_cases['identical_candles']
        
        result = self.engine.analyze_trends(identical_candles)
        
        # Should find no swings and no trends
        self.assertEqual(len(result.swings), 0)
        self.assertEqual(len(result.trends), 0)
    
    def test_extreme_volatility(self):
        """Test analysis with extreme price volatility"""
        edge_cases = TrendTestDataFactory.create_edge_case_scenarios()
        volatile_candles = edge_cases['extreme_volatility']
        
        result = self.engine.analyze_trends(volatile_candles)
        
        # Should handle without crashing
        TrendTestAssertions.assert_analysis_result_consistency(result)
    
    def test_invalid_analysis_window(self):
        """Test error handling for invalid analysis windows"""
        candles = TrendTestDataFactory.create_simple_uptrend_scenario()
        
        # Start after end
        with self.assertRaises(ValueError):
            self.engine.analyze_trends(candles, 10, 5)
        
        # Start beyond data
        with self.assertRaises(ValueError):
            self.engine.analyze_trends(candles, len(candles) + 1, len(candles) + 5)


class TestPropertyBased(unittest.TestCase):
    """Property-based tests using randomly generated data"""
    
    def setUp(self):
        self.engine = TrendAnalysisEngine()
    
    def test_random_scenarios_produce_valid_results(self):
        """Test that random valid scenarios always produce valid results"""
        for _ in range(5):  # Run multiple iterations
            # Generate random trending market
            direction = np.random.choice(list(TrendDirection))
            candles = MarketConditionGenerator.create_trending_market(
                direction, duration=30, strength=0.7
            )
            
            result = self.engine.analyze_trends(candles)
            
            # Properties that should always hold
            TrendTestAssertions.assert_analysis_result_consistency(result)
            self.assertEqual(len(result.candles), len(candles))
            
            # All detected trends should be valid
            for trend in result.trends:
                TrendTestAssertions.assert_valid_trend(trend)
    
    def test_choppy_market_robustness(self):
        """Test algorithm robustness with choppy market conditions"""
        for _ in range(3):
            candles = MarketConditionGenerator.create_choppy_market(
                duration=40, volatility=0.8
            )
            
            result = self.engine.analyze_trends(candles)
            
            # Should handle choppy conditions without error
            TrendTestAssertions.assert_analysis_result_consistency(result)
            
            # In choppy conditions, might find few or no clear trends
            self.assertGreaterEqual(len(result.trends), 0)
    
    def test_invariant_trend_sequence_logic(self):
        """Test invariant: trend sequences should follow logical progression"""
        for _ in range(5):
            candles = MarketConditionGenerator.create_multi_timeframe_scenario(duration=60)
            
            result = self.engine.analyze_trends(candles)
            
            if len(result.trends) > 1:
                TrendTestAssertions.assert_trend_sequence_logical(list(result.trends))


class TestPerformance(unittest.TestCase):
    """Performance tests for the algorithm"""
    
    def test_large_dataset_performance(self):
        """Test performance with large dataset"""
        import time
        
        # Generate large dataset
        candles = MarketConditionGenerator.create_multi_timeframe_scenario(duration=500)
        engine = TrendAnalysisEngine()
        
        start_time = time.time()
        result = engine.analyze_trends(candles)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(execution_time, 10.0, "Algorithm should complete within 10 seconds")
        TrendTestAssertions.assert_analysis_result_consistency(result)
    
    def test_memory_usage_reasonable(self):
        """Test that memory usage is reasonable for large datasets"""
        import sys
        
        candles = MarketConditionGenerator.create_trending_market(
            TrendDirection.UP, duration=200, strength=0.8
        )
        engine = TrendAnalysisEngine()
        
        initial_size = sys.getsizeof(candles)
        result = engine.analyze_trends(candles)
        result_size = sys.getsizeof(result)
        
        # Result should not be disproportionately larger than input
        self.assertLess(result_size, initial_size * 15)


class TestUtilities(unittest.TestCase):
    """Test utility functions"""
    
    def test_trend_scenario_builder(self):
        """Test trend scenario builder creates valid scenarios"""
        candles = (TrendScenarioBuilder()
                  .with_base_price(100)
                  .add_uptrend_sequence(8, 0.7)
                  .add_sideways_sequence(5, 2.0)
                  .add_downtrend_sequence(6, 0.8)
                  .build())
        
        self.assertEqual(len(candles), 19)  # 8 + 5 + 6
        
        # All candles should be valid
        for candle in candles:
            self.assertGreater(candle.high, 0)
            self.assertGreater(candle.low, 0)
            self.assertGreaterEqual(candle.high, max(candle.open, candle.close))
            self.assertLessEqual(candle.low, min(candle.open, candle.close))
    
    def test_trend_data_processor(self):
        """Test trend data processing utilities"""
        # Create some test trends
        pattern = TrendPatternBuilder().uptrend_pattern().build()
        factory = TrendFactory(TrendAnalysisConfig())
        
        trends = [
            factory.create_trend_from_pattern(pattern, 5),
            factory.create_trend_from_pattern(pattern, 10)
        ]
        
        summary = TrendDataProcessor.extract_trend_summary(trends)
        
        self.assertEqual(summary['total_trends'], 2)
        self.assertGreater(summary['avg_duration'], 0)
        self.assertIn('UP', summary['direction_counts'])
    
    def test_trend_validation_utils(self):
        """Test trend validation utilities"""
        # Create valid trend
        swings = [
            SwingPointBuilder().at_candle(1).with_price(95).swing_low().build(),
            SwingPointBuilder().at_candle(3).with_price(105).swing_high().build(),
            SwingPointBuilder().at_candle(5).with_price(100).swing_low().build()
        ]
        
        pattern = (TrendPatternBuilder()
                  .uptrend_pattern()
                  .with_swings(*swings)
                  .build())
        
        factory = TrendFactory(TrendAnalysisConfig())
        trend = factory.create_trend_from_pattern(pattern, 7)
        
        candles = TrendTestDataFactory.create_simple_uptrend_scenario()
        issues = TrendValidationUtils.validate_trend_logic(trend, candles)
        
        # Should find no issues with valid trend
        self.assertEqual(len(issues), 0)


class TestMockIntegration(unittest.TestCase):
    """Test integration with mock objects"""
    
    def test_engine_with_mocked_components(self):
        """Test that engine works with mocked components"""
        # Create mocks
        mock_detector = TrendMockObjects.create_mock_swing_detector()
        mock_matcher = TrendMockObjects.create_mock_pattern_matcher()
        
        # Test that mocks have expected behavior
        candles = TrendTestDataFactory.create_simple_uptrend_scenario()
        
        swings = mock_detector.detect_swings(candles)
        self.assertGreater(len(swings), 0)
        
        patterns = mock_matcher.find_uptrend_patterns(swings)
        self.assertGreater(len(patterns), 0)

class TestBodyToWickRatio(unittest.TestCase):
    """Test the body_to_wick_ratio property on Candle objects"""
    
    def setUp(self):
        self.candle_builder = CandleBuilder()
    
    def test_body_to_wick_ratio_bullish_candle(self):
        """Test body-to-wick ratio calculation for bullish candle"""
        # Create bullish candle: open=100, high=108, low=98, close=105
        # Body size = |105 - 100| = 5
        # Upper wick = 108 - 105 = 3
        # Lower wick = 100 - 98 = 2
        # Total wick = 3 + 2 = 5
        # Ratio = 5 / 5 = 1.0
        candle = self.candle_builder.with_ohlc(100, 108, 98, 105).build()
        
        self.assertEqual(candle.body_to_wick_ratio, 1.0)
    
    def test_body_to_wick_ratio_bearish_candle(self):
        """Test body-to-wick ratio calculation for bearish candle"""
        # Create bearish candle: open=105, high=108, low=98, close=100
        # Body size = |100 - 105| = 5
        # Upper wick = 108 - 105 = 3
        # Lower wick = 100 - 98 = 2
        # Total wick = 3 + 2 = 5
        # Ratio = 5 / 5 = 1.0
        candle = self.candle_builder.with_ohlc(105, 108, 98, 100).build()
        
        self.assertEqual(candle.body_to_wick_ratio, 1.0)
    
    def test_body_to_wick_ratio_doji(self):
        """Test body-to-wick ratio for doji candle (open = close)"""
        # Create doji: open=100, high=103, low=97, close=100
        # Body size = |100 - 100| = 0
        # Upper wick = 103 - 100 = 3
        # Lower wick = 100 - 97 = 3
        # Total wick = 3 + 3 = 6
        # Ratio = 0 / 6 = 0.0
        candle = self.candle_builder.doji(wick_size=3.0).build()
        
        self.assertEqual(candle.body_to_wick_ratio, 0.0)
    
    def test_body_to_wick_ratio_no_wicks(self):
        """Test body-to-wick ratio when there are no wicks"""
        # Create candle with no wicks: open=100, high=105, low=100, close=105
        # Body size = |105 - 100| = 5
        # Upper wick = 105 - 105 = 0
        # Lower wick = 100 - 100 = 0
        # Total wick = 0 + 0 = 0
        # Ratio = 5 / 0 = infinity
        candle = self.candle_builder.with_ohlc(100, 105, 100, 105).build()
        
        self.assertEqual(candle.body_to_wick_ratio, float('inf'))
    
    def test_body_to_wick_ratio_large_body_small_wicks(self):
        """Test ratio for large body with small wicks (LEG candle)"""
        # Create LEG candle with large body
        candle = self.candle_builder.large_body(body_size=10.0, small_wicks=0.5).build()
        
        # Should have high body-to-wick ratio (>1.0)
        self.assertGreater(candle.body_to_wick_ratio, 1.0)
        self.assertLess(candle.body_to_wick_ratio, float('inf'))
    
    def test_body_to_wick_ratio_small_body_large_wicks(self):
        """Test ratio for small body with large wicks (BASE candle)"""
        # Create BASE candle with small body
        candle = self.candle_builder.large_wicks(body_size=1.0, wick_size=5.0).build()
        
        # Should have low body-to-wick ratio (<1.0)
        self.assertLess(candle.body_to_wick_ratio, 1.0)
        self.assertGreater(candle.body_to_wick_ratio, 0.0)
    
    def test_body_to_wick_ratio_trend_detection_threshold(self):
        """Test that body-to-wick ratio works for trend detection thresholds"""
        # Test candle above typical trend detection threshold (2.0)
        strong_candle = self.candle_builder.reset().with_ohlc(100, 112, 99, 110).build()
        # Body = 10, Upper wick = 2, Lower wick = 1, Total wick = 3
        # Ratio = 10/3 ≈ 3.33
        
        self.assertGreater(strong_candle.body_to_wick_ratio, 2.0)
        
        # Test candle below typical trend detection threshold (2.0)
        weak_candle = self.candle_builder.reset().with_ohlc(100, 108, 95, 102).build()
        # Body = 2, Upper wick = 6, Lower wick = 5, Total wick = 11
        # Ratio = 2/11 ≈ 0.18
        
        self.assertLess(weak_candle.body_to_wick_ratio, 2.0)
    
    def test_body_to_wick_ratio_various_configurations(self):
        """Test various candle configurations for consistency"""
        test_cases = [
            # (open, high, low, close, expected_comparison)
            (100, 110, 95, 108, "high_ratio"),    # Large bullish body, small wicks
            (108, 110, 95, 100, "high_ratio"),    # Large bearish body, small wicks
            (100, 105, 98, 101, "low_ratio"),     # Small body, moderate wicks
            (100, 102, 98, 100, "zero_ratio"),    # Doji
            (100, 100, 100, 100, "undefined"),    # Flat line (should not occur in real data)
        ]
        
        for open_price, high, low, close, expected in test_cases:
            with self.subTest(case=f"{open_price}-{high}-{low}-{close}"):
                if expected == "undefined":
                    # Skip flat line case as it's invalid
                    continue
                
                candle = self.candle_builder.reset().with_ohlc(open_price, high, low, close).build()
                ratio = candle.body_to_wick_ratio
                
                if expected == "high_ratio":
                    self.assertGreater(ratio, 1.0, f"Expected high ratio for {open_price}-{high}-{low}-{close}")
                elif expected == "low_ratio":
                    self.assertLess(ratio, 1.0, f"Expected low ratio for {open_price}-{high}-{low}-{close}")
                    self.assertGreater(ratio, 0.0, f"Expected positive ratio for {open_price}-{high}-{low}-{close}")
                elif expected == "zero_ratio":
                    self.assertEqual(ratio, 0.0, f"Expected zero ratio for doji {open_price}-{high}-{low}-{close}")
    
    def test_body_to_wick_ratio_precision(self):
        """Test precision of body-to-wick ratio calculations"""
        # Test with precise decimal values
        candle = self.candle_builder.reset().with_ohlc(100.0, 105.5, 99.25, 103.75).build()
        
        # Body = |103.75 - 100.0| = 3.75
        # Upper wick = 105.5 - 103.75 = 1.75
        # Lower wick = 100.0 - 99.25 = 0.75
        # Total wick = 1.75 + 0.75 = 2.5
        # Ratio = 3.75 / 2.5 = 1.5
        
        self.assertAlmostEqual(candle.body_to_wick_ratio, 1.5, places=10)
    
    def test_body_to_wick_ratio_integration_with_builders(self):
        """Test that body-to-wick ratio works correctly with builder methods"""
        # Test bullish() method creates candles with correct ratios
        bullish_candle = self.candle_builder.reset().bullish(8.0).build()
        self.assertGreater(bullish_candle.body_to_wick_ratio, 0.0)
        
        # Test bearish() method creates candles with correct ratios
        bearish_candle = self.candle_builder.reset().bearish(6.0).build()
        self.assertGreater(bearish_candle.body_to_wick_ratio, 0.0)
        
        # Test large_body() creates high ratios
        leg_candle = self.candle_builder.reset().large_body(10.0, 0.5).build()
        self.assertGreater(leg_candle.body_to_wick_ratio, 5.0)
        
        # Test large_wicks() creates low ratios
        base_candle = self.candle_builder.reset().large_wicks(1.0, 5.0).build()
        self.assertLess(base_candle.body_to_wick_ratio, 0.5)
    
    def test_body_to_wick_ratio_edge_cases(self):
        """Test edge cases for body-to-wick ratio"""
        # Very small body with very large wicks
        tiny_body = self.candle_builder.reset().with_ohlc(100.0, 120.0, 80.0, 100.01).build()
        self.assertLess(tiny_body.body_to_wick_ratio, 0.001)
        
        # Very large body with very small wicks
        huge_body = self.candle_builder.reset().with_ohlc(100.0, 150.01, 99.99, 150.0).build()
        self.assertGreater(huge_body.body_to_wick_ratio, 100.0)
    
    def test_body_to_wick_ratio_trend_classification_compatibility(self):
        """Test that body-to-wick ratios work for trend classification"""
        # Create candles that should classify as LEG candles
        leg_candles = [
            self.candle_builder.reset().large_body(10, 0.5).build(),
            self.candle_builder.reset().bullish(8).large_body(8, 0.3).build(),
            self.candle_builder.reset().bearish(12).large_body(12, 0.2).build(),
        ]
        
        for candle in leg_candles:
            self.assertGreater(candle.body_to_wick_ratio, 1.0, 
                              "LEG candles should have body-to-wick ratio > 1.0")
        
        # Create candles that should classify as BASE candles
        base_candles = [
            self.candle_builder.reset().large_wicks(1, 5).build(),
            self.candle_builder.reset().doji(3).build(),
            self.candle_builder.reset().bullish(0.5).large_wicks(0.5, 4).build(),
        ]
        
        for candle in base_candles:
            self.assertLessEqual(candle.body_to_wick_ratio, 1.0, 
                                "BASE candles should have body-to-wick ratio <= 1.0")
    
    def test_body_to_wick_ratio_property_immutability(self):
        """Test that body-to-wick ratio is calculated fresh each time (no caching issues)"""
        candle = self.candle_builder.reset().with_ohlc(100, 105, 95, 103).build()
        
        # Get ratio multiple times and ensure consistency
        ratio1 = candle.body_to_wick_ratio
        ratio2 = candle.body_to_wick_ratio
        ratio3 = candle.body_to_wick_ratio
        
        self.assertEqual(ratio1, ratio2)
        self.assertEqual(ratio2, ratio3)
        self.assertEqual(ratio1, ratio3)

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)