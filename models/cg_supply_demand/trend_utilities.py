"""
Complete production utilities for the Trend Detection Algorithm
Includes all original utilities PLUS TrendTestDataFactory moved from test utilities
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from dataclasses import replace
import numpy as np

from trend_core_models import (
    SwingPoint, SwingType, Trend, TrendDirection, TrendSignificance,
    TrendPattern, TrendAnalysisConfig, VisualizationData
)

# Use existing Candle from supply/demand refactoring
from utilities import Candle


class TrendScenarioBuilder:
    """
    Builder for creating realistic trend scenarios.
    Useful for backtesting, simulation, and testing specific market conditions.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> 'TrendScenarioBuilder':
        """Reset builder to default state"""
        self._candles = []
        self._base_time = datetime(2024, 1, 1, 9, 0)
        self._current_price = 100.0
        self._candle_interval_minutes = 5
        return self
    
    def with_base_price(self, price: float) -> 'TrendScenarioBuilder':
        """Set starting price for the scenario"""
        self._current_price = price
        return self
    
    def with_base_time(self, timestamp: datetime) -> 'TrendScenarioBuilder':
        """Set starting timestamp"""
        self._base_time = timestamp
        return self
    
    def with_interval(self, minutes: int) -> 'TrendScenarioBuilder':
        """Set candle interval in minutes"""
        self._candle_interval_minutes = minutes
        return self
    
    def add_uptrend_sequence(self, candle_count: int = 8, strength: float = 0.7) -> 'TrendScenarioBuilder':
        """
        Add uptrend sequence that will form L-H-L pattern.
        
        Args:
            candle_count: Number of candles in the sequence
            strength: Trend strength (0.0 = weak, 1.0 = very strong)
        """
        sequence_candles = self._generate_uptrend_candles(candle_count, strength)
        self._candles.extend(sequence_candles)
        return self
    
    def add_downtrend_sequence(self, candle_count: int = 8, strength: float = 0.7) -> 'TrendScenarioBuilder':
        """
        Add downtrend sequence that will form H-L-H pattern.
        
        Args:
            candle_count: Number of candles in the sequence
            strength: Trend strength (0.0 = weak, 1.0 = very strong)
        """
        sequence_candles = self._generate_downtrend_candles(candle_count, strength)
        self._candles.extend(sequence_candles)
        return self
    
    def add_sideways_sequence(self, candle_count: int = 10, range_size: float = 3.0) -> 'TrendScenarioBuilder':
        """
        Add sideways/consolidation sequence.
        
        Args:
            candle_count: Number of candles in the consolidation
            range_size: Price range of the consolidation
        """
        sequence_candles = self._generate_sideways_candles(candle_count, range_size)
        self._candles.extend(sequence_candles)
        return self
    
    def add_noise_sequence(self, candle_count: int = 5, volatility: float = 0.5) -> 'TrendScenarioBuilder':
        """
        Add noise/random movement sequence.
        
        Args:
            candle_count: Number of noise candles
            volatility: How volatile the noise is
        """
        sequence_candles = self._generate_noise_candles(candle_count, volatility)
        self._candles.extend(sequence_candles)
        return self
    
    def build(self) -> List[Candle]:
        """Build the complete scenario using existing CandleBuilder"""
        # Use your existing CandleBuilder from supply/demand refactoring
        from utilities import CandleBuilder
        
        result_candles = []
        for i, spec in enumerate(self._candles):
            timestamp = self._base_time + timedelta(minutes=i * self._candle_interval_minutes)
            
            candle = CandleBuilder().with_timestamp(timestamp).with_ohlc(
                spec['open'], spec['high'], spec['low'], spec['close']
            ).with_volume(spec.get('volume', 1000 + i * 50)).build()
            
            result_candles.append(candle)
        
        return result_candles
    
    def _generate_uptrend_candles(self, count: int, strength: float) -> List[Dict]:
        """Generate candles that form an uptrend pattern"""
        candles = []
        
        # Phase 1: Initial low (swing low)
        low_price = self._current_price * 0.97  # 3% below current
        candles.append(self._create_candle_spec(
            close=low_price, range_size=2.0, bullish=False
        ))
        
        # Phase 2: Recovery and higher high (swing high)
        for i in range(3):
            move_up = (self._current_price * 1.08 - low_price) * (i + 1) / 3
            target_price = low_price + move_up
            candles.append(self._create_candle_spec(
                close=target_price, range_size=2.5, bullish=True
            ))
        
        # Phase 3: Pullback to higher low (swing low)
        higher_low = self._current_price * 1.02  # Higher than initial low
        for i in range(2):
            pullback_price = candles[-1]['close'] - (candles[-1]['close'] - higher_low) * (i + 1) / 2
            candles.append(self._create_candle_spec(
                close=pullback_price, range_size=2.0, bullish=False
            ))
        
        # Phase 4: Breakout (confirms uptrend)
        breakout_target = candles[3]['close'] * 1.03  # Above swing high
        for i in range(count - 6):
            progress = (i + 1) / max(1, count - 6)
            target_price = higher_low + (breakout_target - higher_low) * progress
            
            # Add some volatility but maintain upward bias
            is_bullish = np.random.random() < (0.5 + strength * 0.3)
            range_size = 2.0 + np.random.random() * 2.0
            
            candles.append(self._create_candle_spec(
                close=target_price, range_size=range_size, bullish=is_bullish
            ))
        
        self._current_price = candles[-1]['close']
        return candles
    
    def _generate_downtrend_candles(self, count: int, strength: float) -> List[Dict]:
        """Generate candles that form a downtrend pattern"""
        candles = []
        
        # Phase 1: Initial high (swing high)
        high_price = self._current_price * 1.03  # 3% above current
        candles.append(self._create_candle_spec(
            close=high_price, range_size=2.0, bullish=True
        ))
        
        # Phase 2: Decline to lower low (swing low)
        for i in range(3):
            move_down = (high_price - self._current_price * 0.92) * (i + 1) / 3
            target_price = high_price - move_down
            candles.append(self._create_candle_spec(
                close=target_price, range_size=2.5, bullish=False
            ))
        
        # Phase 3: Rally to lower high (swing high)
        lower_high = self._current_price * 0.98  # Lower than initial high
        for i in range(2):
            rally_price = candles[-1]['close'] + (lower_high - candles[-1]['close']) * (i + 1) / 2
            candles.append(self._create_candle_spec(
                close=rally_price, range_size=2.0, bullish=True
            ))
        
        # Phase 4: Breakdown (confirms downtrend)
        breakdown_target = candles[3]['close'] * 0.97  # Below swing low
        for i in range(count - 6):
            progress = (i + 1) / max(1, count - 6)
            target_price = lower_high - (lower_high - breakdown_target) * progress
            
            # Add some volatility but maintain downward bias
            is_bullish = np.random.random() < (0.5 - strength * 0.3)
            range_size = 2.0 + np.random.random() * 2.0
            
            candles.append(self._create_candle_spec(
                close=target_price, range_size=range_size, bullish=is_bullish
            ))
        
        self._current_price = candles[-1]['close']
        return candles
    
    def _generate_sideways_candles(self, count: int, range_size: float) -> List[Dict]:
        """Generate candles that form sideways consolidation"""
        candles = []
        
        # Define range boundaries
        range_center = self._current_price
        range_high = range_center + range_size / 2
        range_low = range_center - range_size / 2
        
        for i in range(count):
            # Oscillate within the range
            position_in_cycle = (i / max(1, count - 1)) * 2 * np.pi
            target_ratio = (np.sin(position_in_cycle) + 1) / 2  # 0 to 1
            
            # Add noise
            noise = (np.random.random() - 0.5) * 0.3
            target_ratio = max(0, min(1, target_ratio + noise))
            
            target_price = range_low + (range_high - range_low) * target_ratio
            
            # Create small-bodied candles with larger wicks (typical for consolidation)
            candles.append(self._create_candle_spec(
                close=target_price, range_size=1.5, bullish=np.random.random() > 0.5
            ))
        
        self._current_price = candles[-1]['close']
        return candles
    
    def _generate_noise_candles(self, count: int, volatility: float) -> List[Dict]:
        """Generate random noise candles"""
        candles = []
        
        for i in range(count):
            # Random walk with specified volatility
            price_change_pct = (np.random.random() - 0.5) * volatility * 0.02
            target_price = self._current_price * (1 + price_change_pct)
            
            range_size = 1.0 + np.random.random() * 2.0
            is_bullish = np.random.random() > 0.5
            
            candles.append(self._create_candle_spec(
                close=target_price, range_size=range_size, bullish=is_bullish
            ))
            
            self._current_price = target_price
        
        return candles
    
    def _create_candle_spec(self, close: float, range_size: float, bullish: bool) -> Dict:
        """Create OHLC specification for a candle"""
        if bullish:
            open_price = close - (range_size * 0.6)  # Body is 60% of range
            high = close + (range_size * 0.2)        # Upper wick is 20%
            low = open_price - (range_size * 0.2)    # Lower wick is 20%
        else:
            open_price = close + (range_size * 0.6)  # Body is 60% of range
            high = open_price + (range_size * 0.2)   # Upper wick is 20%
            low = close - (range_size * 0.2)         # Lower wick is 20%
        
        return {
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': 1000 + np.random.randint(0, 1000)
        }


class SwingPointBuilder:
    """
    Builder for creating swing points programmatically.
    Useful for testing and scenario creation.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> 'SwingPointBuilder':
        """Reset to default values"""
        self._candle_index = 0
        self._price = 100.0
        self._swing_type = SwingType.HIGH
        self._timestamp = datetime(2024, 1, 1, 9, 0)
        return self
    
    def at_candle(self, index: int) -> 'SwingPointBuilder':
        """Set candle index"""
        self._candle_index = index
        return self
    
    def with_price(self, price: float) -> 'SwingPointBuilder':
        """Set swing price"""
        self._price = price
        return self
    
    def swing_high(self) -> 'SwingPointBuilder':
        """Set as swing high"""
        self._swing_type = SwingType.HIGH
        return self
    
    def swing_low(self) -> 'SwingPointBuilder':
        """Set as swing low"""
        self._swing_type = SwingType.LOW
        return self
    
    def with_timestamp(self, timestamp: datetime) -> 'SwingPointBuilder':
        """Set timestamp"""
        self._timestamp = timestamp
        return self
    
    def build(self) -> SwingPoint:
        """Build the swing point"""
        return SwingPoint(
            candle_index=self._candle_index,
            price=self._price,
            swing_type=self._swing_type,
            timestamp=self._timestamp
        )


class TrendPatternBuilder:
    """
    Builder for creating trend patterns programmatically.
    Useful for testing pattern recognition logic.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> 'TrendPatternBuilder':
        """Reset to default values"""
        self._formation_swings = []
        self._pattern_type = TrendDirection.UP
        self._start_index = 0
        self._end_index = 2
        return self
    
    def uptrend_pattern(self) -> 'TrendPatternBuilder':
        """Set as uptrend pattern"""
        self._pattern_type = TrendDirection.UP
        return self
    
    def downtrend_pattern(self) -> 'TrendPatternBuilder':
        """Set as downtrend pattern"""
        self._pattern_type = TrendDirection.DOWN
        return self
    
    def sideways_pattern(self) -> 'TrendPatternBuilder':
        """Set as sideways pattern"""
        self._pattern_type = TrendDirection.SIDEWAYS
        return self
    
    def with_swings(self, *swings: SwingPoint) -> 'TrendPatternBuilder':
        """Set formation swings"""
        self._formation_swings = list(swings)
        if swings:
            self._start_index = min(s.candle_index for s in swings)
            self._end_index = max(s.candle_index for s in swings)
        return self
    
    def with_indices(self, start: int, end: int) -> 'TrendPatternBuilder':
        """Set start and end indices"""
        self._start_index = start
        self._end_index = end
        return self
    
    def build(self) -> TrendPattern:
        """Build the trend pattern"""
        # If no swings are provided, create default swings based on pattern type
        if not self._formation_swings:
            self._formation_swings = self._create_default_swings()
        
        return TrendPattern(
            formation_swings=tuple(self._formation_swings),
            pattern_type=self._pattern_type,
            start_index=self._start_index,
            end_index=self._end_index
        )
    
    def _create_default_swings(self) -> List[SwingPoint]:
        """Create default swings for the pattern if none are provided"""
        base_time = datetime(2024, 1, 1, 9, 0)
        
        if self._pattern_type == TrendDirection.UP:
            # Default L-H-L uptrend pattern
            return [
                SwingPoint(0, 95.0, SwingType.LOW, base_time),
                SwingPoint(1, 105.0, SwingType.HIGH, base_time + timedelta(minutes=5)),
                SwingPoint(2, 100.0, SwingType.LOW, base_time + timedelta(minutes=10))
            ]
        elif self._pattern_type == TrendDirection.DOWN:
            # Default H-L-H downtrend pattern
            return [
                SwingPoint(0, 105.0, SwingType.HIGH, base_time),
                SwingPoint(1, 95.0, SwingType.LOW, base_time + timedelta(minutes=5)),
                SwingPoint(2, 100.0, SwingType.HIGH, base_time + timedelta(minutes=10))
            ]
        else:  # SIDEWAYS
            # Default sideways pattern with alternating highs and lows
            return [
                SwingPoint(0, 98.0, SwingType.LOW, base_time),
                SwingPoint(1, 102.0, SwingType.HIGH, base_time + timedelta(minutes=5)),
                SwingPoint(2, 98.5, SwingType.LOW, base_time + timedelta(minutes=10))
            ]


class TrendTestDataFactory:
    """
    Factory for creating predefined test scenarios and edge cases.
    Responsibility: Generate known test patterns for algorithm validation.
    
    MOVED FROM TEST UTILITIES - useful for production backtesting and simulation.
    """
    
    @staticmethod
    def create_simple_uptrend_scenario() -> List[Candle]:
        """Create a simple uptrend scenario for testing L-H-L detection"""
        return (TrendScenarioBuilder()
                .with_base_price(100)
                .add_uptrend_sequence(candle_count=12, strength=0.9)  # Stronger and longer
                .build())
    
    @staticmethod
    def create_simple_downtrend_scenario() -> List[Candle]:
        """Create a simple downtrend scenario for testing H-L-H detection"""
        return (TrendScenarioBuilder()
                .with_base_price(100)
                .add_downtrend_sequence(candle_count=12, strength=0.9)  # Stronger and longer
                .build())
    
    @staticmethod
    def create_simple_sideways_scenario() -> List[Candle]:
        """Create a simple sideways scenario for testing consolidation detection"""
        return (TrendScenarioBuilder()
                .with_base_price(100)
                .add_sideways_sequence(candle_count=15, range_size=2.0)  # Tighter range
                .build())
    
    @staticmethod
    def create_multi_trend_scenario() -> List[Candle]:
        """Create scenario with multiple trend types for integration testing"""
        return (TrendScenarioBuilder()
                .with_base_price(100)
                .add_uptrend_sequence(12, 0.8)
                .add_noise_sequence(2, 0.2)  # Reduced noise
                .add_downtrend_sequence(12, 0.8)
                .add_noise_sequence(2, 0.2)
                .add_sideways_sequence(8, 1.5)  # Tighter sideways
                .build())
    
    @staticmethod
    def create_ambiguous_pattern_scenario() -> List[Candle]:
        """Create scenario with ambiguous patterns that should be rejected"""
        return (TrendScenarioBuilder()
                .with_base_price(100)
                .add_noise_sequence(20, 1.0)  # Pure noise - no clear patterns
                .build())
    
    @staticmethod
    def create_overlapping_trends_scenario() -> List[Candle]:
        """Create scenario that tests trend conflict resolution"""
        builder = TrendScenarioBuilder().with_base_price(100)
        
        # Create overlapping trend signals by mixing sequences
        candles = []
        
        # Start with partial uptrend
        uptrend_candles = builder.add_uptrend_sequence(8, 0.8).build()
        candles.extend(uptrend_candles[:5])  # Only first part
        
        # Insert sideways that overlaps
        builder.reset().with_base_price(uptrend_candles[4].close)
        sideways_candles = builder.add_sideways_sequence(6, 1.5).build()
        candles.extend(sideways_candles)
        
        # Continue with conflicting downtrend
        builder.reset().with_base_price(sideways_candles[-1].close)
        downtrend_candles = builder.add_downtrend_sequence(8, 0.8).build()
        candles.extend(downtrend_candles)
        
        return candles
    
    @staticmethod
    def create_edge_case_scenarios() -> Dict[str, List[Candle]]:
        """Create various edge case scenarios for boundary testing"""
        # Use your existing CandleBuilder
        from utilities import CandleBuilder
        
        # Single candle
        single_candle = [CandleBuilder().build()]
        
        # Two candles
        two_candles = [
            CandleBuilder().build(),
            CandleBuilder().bullish(5).build()
        ]
        
        # Three candles (minimum for swing detection)
        three_candles = [
            CandleBuilder().with_ohlc(100, 102, 98, 99).build(),
            CandleBuilder().with_ohlc(99, 105, 97, 103).build(),  # Swing high
            CandleBuilder().with_ohlc(103, 104, 100, 101).build()
        ]
        
        # All identical candles (no swings possible)
        identical_candles = [CandleBuilder().with_ohlc(100, 100, 100, 100).build() for _ in range(10)]
        
        # Extreme volatility (large gaps)
        extreme_volatility = [
            CandleBuilder().with_ohlc(100, 105, 95, 102).build(),
            CandleBuilder().with_ohlc(150, 155, 148, 152).build(),  # Big gap up
            CandleBuilder().with_ohlc(80, 85, 75, 82).build()       # Big gap down  
        ]
        
        return {
            'empty_list': [],
            'single_candle': single_candle,
            'two_candles': two_candles,
            'three_candles': three_candles,
            'identical_candles': identical_candles,
            'extreme_volatility': extreme_volatility
        }
    
    @staticmethod
    def create_genesis_point_test_scenario() -> List[Candle]:
        """Create scenario specifically for testing genesis point logic"""
        return (TrendScenarioBuilder()
                .with_base_price(100)
                .add_uptrend_sequence(10, 0.8)      # Creates first trend
                .add_downtrend_sequence(8, 0.7)     # Should terminate first and create genesis
                .add_sideways_sequence(6, 2.0)      # Uses genesis point
                .add_uptrend_sequence(12, 0.9)      # Another trend from new genesis
                .build())
    
    @staticmethod
    def create_weak_trend_scenario() -> List[Candle]:
        """Create scenario with weak trends that should be classified as minor/consolidation"""
        return (TrendScenarioBuilder()
                .with_base_price(100)
                .add_uptrend_sequence(4, 0.3)      # Short, weak uptrend
                .add_downtrend_sequence(3, 0.2)    # Very weak downtrend
                .add_sideways_sequence(5, 1.0)     # Very tight range
                .build())


class MarketConditionGenerator:
    """
    Generate specific market conditions for backtesting and simulation.
    Useful for testing algorithm behavior under different market regimes.
    
    This is a production utility for creating realistic market scenarios.
    """
    
    @staticmethod
    def create_trending_market(direction: TrendDirection, duration: int = 50,
                              strength: float = 0.8) -> List[Candle]:
        """Create a strongly trending market"""
        builder = TrendScenarioBuilder()
        
        if direction == TrendDirection.UP:
            # Create multiple uptrend sequences
            segments = max(1, duration // 15)
            for _ in range(segments):
                builder.add_uptrend_sequence(15, strength)
                if np.random.random() > 0.7:  # Occasional pullback
                    builder.add_noise_sequence(3, 0.3)
        
        elif direction == TrendDirection.DOWN:
            # Create multiple downtrend sequences
            segments = max(1, duration // 15)
            for _ in range(segments):
                builder.add_downtrend_sequence(15, strength)
                if np.random.random() > 0.7:  # Occasional bounce
                    builder.add_noise_sequence(3, 0.3)
        
        else:  # SIDEWAYS
            # Create extended consolidation
            builder.add_sideways_sequence(duration, 5.0)
        
        return builder.build()
    
    @staticmethod
    def create_choppy_market(duration: int = 50, volatility: float = 0.8) -> List[Candle]:
        """Create a choppy, directionless market"""
        builder = TrendScenarioBuilder()
        
        # Alternate between small trends and consolidations
        remaining = duration
        while remaining > 0:
            segment_size = min(np.random.randint(5, 12), remaining)
            
            choice = np.random.choice(['up', 'down', 'sideways', 'noise'], p=[0.25, 0.25, 0.3, 0.2])
            
            if choice == 'up':
                builder.add_uptrend_sequence(segment_size, 0.4)  # Weak uptrend
            elif choice == 'down':
                builder.add_downtrend_sequence(segment_size, 0.4)  # Weak downtrend
            elif choice == 'sideways':
                builder.add_sideways_sequence(segment_size, 2.0)  # Tight range
            else:
                builder.add_noise_sequence(segment_size, volatility)
            
            remaining -= segment_size
        
        return builder.build()
    
    @staticmethod
    def create_multi_timeframe_scenario(duration: int = 100) -> List[Candle]:
        """Create scenario with trends on multiple timeframes"""
        builder = TrendScenarioBuilder()
        
        # Major uptrend with intermediate corrections
        builder.add_uptrend_sequence(20, 0.8)      # Strong up move
        builder.add_sideways_sequence(15, 3.0)     # Consolidation
        builder.add_downtrend_sequence(10, 0.6)    # Correction
        builder.add_uptrend_sequence(25, 0.9)      # Continuation up
        builder.add_sideways_sequence(20, 4.0)     # Distribution
        builder.add_downtrend_sequence(30, 0.8)    # Major reversal
        
        return builder.build()


class TrendDataProcessor:
    """
    Utility functions for processing trend analysis data.
    Useful for converting between formats and preparing data for different uses.
    """
    
    @staticmethod
    def extract_trend_summary(trends: List[Trend]) -> Dict:
        """Extract summary statistics from trends"""
        if not trends:
            return {
                'total_trends': 0,
                'avg_duration': 0,
                'avg_price_range': 0,
                'direction_counts': {'UP': 0, 'DOWN': 0, 'SIDEWAYS': 0},
                'significance_counts': {'MAJOR': 0, 'MINOR': 0, 'CONSOLIDATION': 0}
            }
        
        direction_counts = {direction.value.upper(): 0 for direction in TrendDirection}
        significance_counts = {sig.value.upper(): 0 for sig in TrendSignificance}
        
        total_duration = 0
        total_range = 0
        
        for trend in trends:
            direction_counts[trend.direction.value.upper()] += 1
            significance_counts[trend.significance.value.upper()] += 1
            total_duration += trend.duration
            total_range += trend.price_range
        
        return {
            'total_trends': len(trends),
            'avg_duration': total_duration / len(trends),
            'avg_price_range': total_range / len(trends),
            'direction_counts': direction_counts,
            'significance_counts': significance_counts
        }
    
    @staticmethod
    def convert_to_visualization_format(analysis_result: 'TrendAnalysisResult', 
                                       candles: List['Candle']) -> Dict:
        """Convert analysis result to format suitable for visualization"""
        # Prepare swing data
        swing_data = []
        for swing in analysis_result.swings:
            swing_data.append({
                'index': swing.candle_index,
                'price': swing.price,
                'type': swing.swing_type.value.upper(),
                'timestamp': swing.timestamp
            })
        
        # Prepare trend data
        trend_data = []
        for trend in analysis_result.trends:
            trend_info = {
                'id': trend.trend_id,
                'direction': trend.direction.value.upper(),
                'start': trend.start_index,
                'end': trend.end_index,
                'significance': trend.significance.value.upper(),
                'is_active': trend.is_active,
                'duration': trend.duration,
                'price_range': trend.price_range
            }
            
            # Add controlling swing info
            if trend.controlling_swing:
                trend_info['controlling_swing'] = {
                    'index': trend.controlling_swing.candle_index,
                    'price': trend.controlling_swing.price,
                    'type': trend.controlling_swing.swing_type.value.upper()
                }
            
            # Add formation swings
            trend_info['formation_swings'] = []
            for swing in trend.formation_pattern.formation_swings:
                trend_info['formation_swings'].append({
                    'index': swing.candle_index,
                    'price': swing.price,
                    'type': swing.swing_type.value.upper()
                })
            
            # Add sideways-specific data
            if trend.direction == TrendDirection.SIDEWAYS:
                trend_info['range_high'] = trend.range_high
                trend_info['range_low'] = trend.range_low
            
            trend_data.append(trend_info)
        
        return {
            'candles': candles,
            'swings': swing_data,
            'trends': trend_data,
            'active_trends': [t for t in trend_data if t['is_active']],
            'major_trends': [t for t in trend_data if t['significance'] == 'MAJOR'],
            'analysis_window': analysis_result.analysis_window,
            'summary': TrendDataProcessor.extract_trend_summary(list(analysis_result.trends))
        }


class TrendValidationUtils:
    """
    Utility functions for validating trend analysis results.
    """
    
    @staticmethod
    def validate_trend_logic(trend: Trend, candles: List[Candle]) -> List[str]:
        """
        Validate trend follows expected logic patterns.
        Returns list of validation issues (empty if valid).
        """
        issues = []
        
        # Basic validation
        if trend.start_index < 0:
            issues.append("Start index cannot be negative")
        
        if trend.end_index is not None and trend.start_index >= trend.end_index:
            issues.append("Start index must be less than end index")
        
        # Validate formation pattern consistency
        pattern = trend.formation_pattern
        if len(pattern.formation_swings) < 2:
            issues.append("Formation pattern must have at least 2 swings")
        
        # Direction-specific validation
        if trend.direction == TrendDirection.UP:
            issues.extend(TrendValidationUtils._validate_uptrend_logic(trend, candles))
        elif trend.direction == TrendDirection.DOWN:
            issues.extend(TrendValidationUtils._validate_downtrend_logic(trend, candles))
        elif trend.direction == TrendDirection.SIDEWAYS:
            issues.extend(TrendValidationUtils._validate_sideways_logic(trend, candles))
        
        return issues
    
    @staticmethod
    def _validate_uptrend_logic(trend: Trend, candles: List[Candle]) -> List[str]:
        """Validate uptrend-specific logic"""
        issues = []
        
        # Check for L-H-L pattern in a 3-swing formation
        swings = trend.formation_pattern.formation_swings
        if len(swings) >= 3:
            if not (swings[0].swing_type == SwingType.LOW and
                   swings[1].swing_type == SwingType.HIGH and
                   swings[2].swing_type == SwingType.LOW):
                issues.append("Uptrend should have L-H-L formation pattern")
            
            # Check for higher low
            if swings[2].price <= swings[0].price:
                issues.append("Uptrend should have higher low in formation")
        
        return issues
    
    @staticmethod
    def _validate_downtrend_logic(trend: Trend, candles: List[Candle]) -> List[str]:
        """Validate downtrend-specific logic"""
        issues = []
        
        # Check for H-L-H pattern in a 3-swing formation
        swings = trend.formation_pattern.formation_swings
        if len(swings) >= 3:
            if not (swings[0].swing_type == SwingType.HIGH and
                   swings[1].swing_type == SwingType.LOW and
                   swings[2].swing_type == SwingType.HIGH):
                issues.append("Downtrend should have H-L-H formation pattern")
            
            # Check for lower high
            if swings[2].price >= swings[0].price:
                issues.append("Downtrend should have lower high in formation")
        
        return issues
    
    @staticmethod
    def _validate_sideways_logic(trend: Trend, candles: List[Candle]) -> List[str]:
        """Validate sideways trend logic"""
        issues = []
        
        if trend.range_high is None or trend.range_low is None:
            issues.append("Sideways trend must have range_high and range_low defined")
        elif trend.range_high <= trend.range_low:
            issues.append("Sideways trend range_high must be greater than range_low")
        
        return issues
    
    @staticmethod
    def validate_analysis_result(result: 'TrendAnalysisResult', candles: List[Candle]) -> List[str]:
        """Validate complete analysis result for consistency"""
        issues = []
        
        # Validate each trend individually
        for trend in result.trends:
            trend_issues = TrendValidationUtils.validate_trend_logic(trend, candles)
            for issue in trend_issues:
                issues.append(f"Trend {trend.trend_id}: {issue}")
        
        # Validate result consistency
        active_count = sum(1 for t in result.trends if t.is_active)
        if active_count != len(result.active_trends):
            issues.append("Active trends count mismatch")
        
        major_count = sum(1 for t in result.trends if t.significance == TrendSignificance.MAJOR)
        if major_count != len(result.major_trends):
            issues.append("Major trends count mismatch")
        
        return issues