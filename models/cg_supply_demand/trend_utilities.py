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
    IMPROVED IMPLEMENTATION of TrendScenarioBuilder
    
    Creates realistic candle sequences with proper:
    - Candle continuity (open = previous close)  
    - Realistic OHLC relationships
    - Strong enough moves to be detectable
    - Clear swing structures for L-H-L and H-L-H patterns
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
    
    def add_strong_uptrend_sequence(self, candle_count: int = 10, strength: float = 0.8) -> 'TrendScenarioBuilder':
        """
        Create a STRONG uptrend with clear L-H-L pattern that will be detected.
        
        Key improvements over old system:
        1. Proper candle continuity 
        2. Strong enough moves (5-15% swings)
        3. Clear geometric L-H-L structure
        4. Realistic OHLC relationships
        """
        print(f"Building strong uptrend: {candle_count} candles, strength {strength}")
        
        # Phase 1: Create the initial LOW (genesis point)
        initial_low_candles = self._create_swing_low_sequence(
            target_price=self._current_price * 0.92,  # 8% drop to create strong low
            candle_count=2,
            strength=0.8
        )
        
        # Phase 2: Rally to create the HIGH  
        swing_high_target = self._current_price * 1.12  # 12% above starting price
        rally_candles = self._create_swing_high_sequence(
            target_price=swing_high_target,
            candle_count=3,
            strength=strength
        )
        
        # Phase 3: Pullback to create HIGHER LOW
        higher_low_target = self._current_price * 0.96  # 4% drop, but higher than initial low
        pullback_candles = self._create_swing_low_sequence(
            target_price=higher_low_target,
            candle_count=2,
            strength=0.6
        )
        
        # Phase 4: Breakout above the swing high (this triggers trend detection)
        breakout_target = swing_high_target * 1.05  # 5% above swing high
        remaining_candles = candle_count - len(initial_low_candles) - len(rally_candles) - len(pullback_candles)
        breakout_candles = self._create_strong_move_sequence(
            target_price=breakout_target,
            candle_count=max(3, remaining_candles),
            direction='up',
            strength=strength
        )
        
        # Combine all phases
        all_candles = initial_low_candles + rally_candles + pullback_candles + breakout_candles
        self._candles.extend(all_candles)
        
        print(f"Created uptrend phases:")
        print(f"  - Initial low: {len(initial_low_candles)} candles to {initial_low_candles[-1]['close']:.2f}")
        print(f"  - Rally high: {len(rally_candles)} candles to {rally_candles[-1]['close']:.2f}")
        print(f"  - Higher low: {len(pullback_candles)} candles to {pullback_candles[-1]['close']:.2f}")
        print(f"  - Breakout: {len(breakout_candles)} candles to {breakout_candles[-1]['close']:.2f}")
        
        return self
    
    def add_strong_downtrend_sequence(self, candle_count: int = 10, strength: float = 0.8) -> 'TrendScenarioBuilder':
        """
        Create a STRONG downtrend with clear H-L-H pattern that will be detected.
        """
        print(f"Building strong downtrend: {candle_count} candles, strength {strength}")
        
        # Phase 1: Create the initial HIGH (genesis point)
        initial_high_candles = self._create_swing_high_sequence(
            target_price=self._current_price * 1.08,  # 8% rally to create strong high
            candle_count=2,
            strength=0.8
        )
        
        # Phase 2: Decline to create the LOW
        swing_low_target = self._current_price * 0.88  # 12% below starting price
        decline_candles = self._create_swing_low_sequence(
            target_price=swing_low_target,
            candle_count=3,
            strength=strength
        )
        
        # Phase 3: Rally to create LOWER HIGH
        lower_high_target = self._current_price * 1.04  # 4% rally, but lower than initial high
        rally_candles = self._create_swing_high_sequence(
            target_price=lower_high_target,
            candle_count=2,
            strength=0.6
        )
        
        # Phase 4: Breakdown below the swing low (this triggers trend detection)
        breakdown_target = swing_low_target * 0.95  # 5% below swing low
        remaining_candles = candle_count - len(initial_high_candles) - len(decline_candles) - len(rally_candles)
        breakdown_candles = self._create_strong_move_sequence(
            target_price=breakdown_target,
            candle_count=max(3, remaining_candles),
            direction='down',
            strength=strength
        )
        
        # Combine all phases
        all_candles = initial_high_candles + decline_candles + rally_candles + breakdown_candles
        self._candles.extend(all_candles)
        
        print(f"Created downtrend phases:")
        print(f"  - Initial high: {len(initial_high_candles)} candles to {initial_high_candles[-1]['close']:.2f}")
        print(f"  - Swing low: {len(decline_candles)} candles to {decline_candles[-1]['close']:.2f}")
        print(f"  - Lower high: {len(rally_candles)} candles to {rally_candles[-1]['close']:.2f}")
        print(f"  - Breakdown: {len(breakdown_candles)} candles to {breakdown_candles[-1]['close']:.2f}")
        
        return self
    
    def add_tight_sideways_sequence(self, candle_count: int = 12, range_percent: float = 3.0) -> 'TrendScenarioBuilder':
        """
        Create a tight sideways range with low momentum candles.
        """
        print(f"Building sideways: {candle_count} candles, {range_percent}% range")
        
        range_center = self._current_price
        range_high = range_center * (1 + range_percent / 200)  # Half range above
        range_low = range_center * (1 - range_percent / 200)   # Half range below
        
        print(f"Sideways range: {range_low:.2f} to {range_high:.2f}")
        
        sideways_candles = []
        
        for i in range(candle_count):
            # Oscillate within the range with noise
            cycle_position = (i / max(1, candle_count - 1)) * 2 * np.pi
            base_position = (np.sin(cycle_position) + 1) / 2  # 0 to 1
            
            # Add some randomness
            noise = (np.random.random() - 0.5) * 0.4
            target_position = max(0, min(1, base_position + noise))
            
            target_price = range_low + (range_high - range_low) * target_position
            
            # Create small-bodied candles with larger wicks (typical for consolidation)
            candle_spec = self._create_realistic_candle(
                target_close=target_price,
                body_size_percent=1.0,  # Small 1% bodies
                wick_size_percent=1.5,  # Larger 1.5% wicks
                bullish_bias=0.5  # No directional bias
            )
            
            sideways_candles.append(candle_spec)
        
        self._candles.extend(sideways_candles)
        print(f"Created {len(sideways_candles)} sideways candles")
        
        return self
    
    def _create_swing_low_sequence(self, target_price: float, candle_count: int, strength: float) -> List[Dict]:
        """Create a sequence of candles that form a swing low"""
        candles = []
        
        start_price = self._current_price
        price_drop = target_price - start_price
        
        for i in range(candle_count):
            progress = (i + 1) / candle_count
            
            # Non-linear progression (more drop early, then stabilize)
            adjusted_progress = progress ** (2 - strength)  # strength affects curvature
            current_target = start_price + (price_drop * adjusted_progress)
            
            # Make the last candle the actual swing low with a spike down
            if i == candle_count - 1:
                # Final candle: spike low but close higher (typical swing low behavior)
                candle_spec = self._create_realistic_candle(
                    target_close=current_target,
                    body_size_percent=2.0,
                    wick_size_percent=3.0,  # Large lower wick for swing low
                    bullish_bias=0.7  # Close higher in the candle range
                )
            else:
                # Declining candles
                candle_spec = self._create_realistic_candle(
                    target_close=current_target,
                    body_size_percent=2.5,
                    wick_size_percent=1.5,
                    bullish_bias=0.2  # Bearish bias
                )
            
            candles.append(candle_spec)
        
        return candles
    
    def _create_swing_high_sequence(self, target_price: float, candle_count: int, strength: float) -> List[Dict]:
        """Create a sequence of candles that form a swing high"""
        candles = []
        
        start_price = self._current_price
        price_gain = target_price - start_price
        
        for i in range(candle_count):
            progress = (i + 1) / candle_count
            
            # Non-linear progression
            adjusted_progress = progress ** (2 - strength)
            current_target = start_price + (price_gain * adjusted_progress)
            
            # Make the last candle the actual swing high with a spike up
            if i == candle_count - 1:
                # Final candle: spike high but close lower (typical swing high behavior)
                candle_spec = self._create_realistic_candle(
                    target_close=current_target,
                    body_size_percent=2.0,
                    wick_size_percent=3.0,  # Large upper wick for swing high
                    bullish_bias=0.3  # Close lower in the candle range
                )
            else:
                # Rising candles
                candle_spec = self._create_realistic_candle(
                    target_close=current_target,
                    body_size_percent=2.5,
                    wick_size_percent=1.5,
                    bullish_bias=0.8  # Bullish bias
                )
            
            candles.append(candle_spec)
        
        return candles
    
    def _create_strong_move_sequence(self, target_price: float, candle_count: int, 
                                   direction: str, strength: float) -> List[Dict]:
        """Create a strong directional move (breakout/breakdown)"""
        candles = []
        
        start_price = self._current_price
        price_change = target_price - start_price
        
        for i in range(candle_count):
            progress = (i + 1) / candle_count
            current_target = start_price + (price_change * progress)
            
            # Strong moves have larger bodies and smaller wicks
            bullish_bias = 0.8 if direction == 'up' else 0.2
            
            candle_spec = self._create_realistic_candle(
                target_close=current_target,
                body_size_percent=3.0 * strength,  # Larger bodies for strong moves
                wick_size_percent=1.0,  # Smaller wicks
                bullish_bias=bullish_bias
            )
            
            candles.append(candle_spec)
        
        return candles
    
    def _create_realistic_candle(self, target_close: float, body_size_percent: float, 
                               wick_size_percent: float, bullish_bias: float) -> Dict:
        """
        Create a single realistic candle with proper OHLC relationships.
        
        Key improvements:
        1. Open = previous candle's close (proper continuity)
        2. Realistic OHLC construction 
        3. Configurable body/wick sizes
        4. Proper bullish/bearish bias
        """
        
        # CRITICAL FIX: Open equals previous candle's close (except first candle)
        if self._candles:
            open_price = self._candles[-1]['close']  # Continuity!
        else:
            open_price = self._current_price
        
        close_price = target_close
        
        # Calculate body and wick sizes as absolute values
        body_size = abs(close_price - open_price)
        if body_size < open_price * (body_size_percent / 100):
            # Ensure minimum body size
            if close_price > open_price:
                close_price = open_price + (open_price * body_size_percent / 100)
            else:
                close_price = open_price - (open_price * body_size_percent / 100)
        
        wick_size = open_price * (wick_size_percent / 100)
        
        # Determine high and low based on bullish bias
        if bullish_bias > 0.5:  # Bullish candle
            if close_price < open_price:  # Force bullish if bias says so
                close_price = open_price + abs(close_price - open_price)
            
            # High extends above the top
            high = max(open_price, close_price) + wick_size * (1 - bullish_bias + 0.5)
            # Low extends below the bottom  
            low = min(open_price, close_price) - wick_size * bullish_bias
            
        else:  # Bearish candle
            if close_price > open_price:  # Force bearish if bias says so
                close_price = open_price - abs(close_price - open_price)
            
            # High extends above the top
            high = max(open_price, close_price) + wick_size * (1 - bullish_bias)
            # Low extends below the bottom
            low = min(open_price, close_price) - wick_size * (bullish_bias + 0.5)
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        # Update current price for next candle
        self._current_price = close_price
        
        return {
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': 1000 + np.random.randint(0, 1000)
        }
    
    def build(self) -> List[Candle]:
        """Build the complete scenario using existing CandleBuilder"""
        from utilities import CandleBuilder
        
        result_candles = []
        for i, spec in enumerate(self._candles):
            timestamp = self._base_time + timedelta(minutes=i * self._candle_interval_minutes)
            
            candle = CandleBuilder().with_timestamp(timestamp).with_ohlc(
                spec['open'], spec['high'], spec['low'], spec['close']
            ).with_volume(spec.get('volume', 1000 + i * 50)).build()
            
            result_candles.append(candle)
        
        print(f"Built {len(result_candles)} realistic candles")
        return result_candles


class TrendTestDataFactory:
    """
    IMPROVED IMPLEMENTATION of TrendTestDataFactory
    
    Creates realistic test scenarios that will actually trigger trend detection.
    """
    
    @staticmethod
    def create_strong_uptrend_scenario() -> List[Candle]:
        """Create a strong uptrend that WILL be detected"""
        print("\n=== Creating Strong Uptrend Scenario ===")
        candles = (TrendScenarioBuilder()
                  .with_base_price(100)
                  .add_strong_uptrend_sequence(candle_count=12, strength=0.9)
                  .build())
        
        print(f"Strong uptrend: {candles[0].close:.2f} → {candles[-1].close:.2f} ({((candles[-1].close/candles[0].close-1)*100):+.1f}%)")
        return candles
    
    @staticmethod
    def create_strong_downtrend_scenario() -> List[Candle]:
        """Create a strong downtrend that WILL be detected"""
        print("\n=== Creating Strong Downtrend Scenario ===")
        candles = (TrendScenarioBuilder()
                  .with_base_price(100)
                  .add_strong_downtrend_sequence(candle_count=12, strength=0.9)
                  .build())
        
        print(f"Strong downtrend: {candles[0].close:.2f} → {candles[-1].close:.2f} ({((candles[-1].close/candles[0].close-1)*100):+.1f}%)")
        return candles
    
    @staticmethod
    def create_tight_sideways_scenario() -> List[Candle]:
        """Create a tight sideways range that should be detected as consolidation"""
        print("\n=== Creating Tight Sideways Scenario ===")
        candles = (TrendScenarioBuilder()
                  .with_base_price(100)
                  .add_tight_sideways_sequence(candle_count=15, range_percent=2.5)
                  .build())
        
        high_price = max(c.high for c in candles)
        low_price = min(c.low for c in candles)
        range_pct = ((high_price - low_price) / low_price) * 100
        print(f"Tight sideways: {low_price:.2f} to {high_price:.2f} ({range_pct:.1f}% range)")
        return candles
    
    @staticmethod
    def create_multi_trend_scenario() -> List[Candle]:
        """Create scenario with multiple trend phases"""
        print("\n=== Creating Multi-Trend Scenario ===")
        builder = TrendScenarioBuilder().with_base_price(100)
        
        # Phase 1: Strong uptrend
        builder.add_strong_uptrend_sequence(10, 0.8)
        
        # Phase 2: Sideways consolidation
        builder.add_tight_sideways_sequence(8, 3.0)
        
        # Phase 3: Strong downtrend
        builder.add_strong_downtrend_sequence(10, 0.8)
        
        candles = builder.build()
        print(f"Multi-trend: {len(candles)} candles with 3 phases")
        return candles
    
    @staticmethod
    def create_noisy_scenario() -> List[Candle]:
        """Create noisy market with no clear trends"""
        print("\n=== Creating Noisy Scenario ===")
        builder = TrendScenarioBuilder().with_base_price(100)
        
        # Multiple small, conflicting moves
        for _ in range(4):
            # Small random moves in different directions
            if np.random.random() > 0.5:
                builder.add_strong_uptrend_sequence(5, 0.3)  # Weak moves
            else:
                builder.add_strong_downtrend_sequence(5, 0.3)
        
        candles = builder.build()
        print(f"Noisy market: {len(candles)} candles with conflicting signals")
        return candles


class MarketConditionGenerator:
    """
    IMPROVED IMPLEMENTATION of MarketConditionGenerator
    
    Generate realistic market conditions for backtesting.
    """
    
    @staticmethod
    def create_trending_market(direction: TrendDirection, duration: int = 50,
                              strength: float = 0.8) -> List[Candle]:
        """Create a strongly trending market with realistic structure"""
        builder = TrendScenarioBuilder()
        
        remaining_duration = duration
        while remaining_duration > 0:
            # Create trend segments with occasional consolidations
            segment_size = min(np.random.randint(8, 15), remaining_duration)
            
            if direction == TrendDirection.UP:
                builder.add_strong_uptrend_sequence(segment_size, strength)
            elif direction == TrendDirection.DOWN:
                builder.add_strong_downtrend_sequence(segment_size, strength)
            else:  # SIDEWAYS
                builder.add_tight_sideways_sequence(segment_size, 3.0)
            
            remaining_duration -= segment_size
            
            # Occasional small consolidation
            if remaining_duration > 5 and np.random.random() > 0.7:
                consolidation_size = min(5, remaining_duration)
                builder.add_tight_sideways_sequence(consolidation_size, 2.0)
                remaining_duration -= consolidation_size
        
        return builder.build()
    
    @staticmethod
    def create_choppy_market(duration: int = 50) -> List[Candle]:
        """Create a choppy, directionless market"""
        builder = TrendScenarioBuilder()
        
        remaining_duration = duration
        while remaining_duration > 0:
            segment_size = min(np.random.randint(5, 10), remaining_duration)
            
            # Random direction with weak strength
            choice = np.random.choice(['up', 'down', 'sideways'])
            
            if choice == 'up':
                builder.add_strong_uptrend_sequence(segment_size, 0.4)
            elif choice == 'down':
                builder.add_strong_downtrend_sequence(segment_size, 0.4)
            else:
                builder.add_tight_sideways_sequence(segment_size, 4.0)
            
            remaining_duration -= segment_size
        
        return builder.build()


# Validation function to test the new system
def validate_realistic_data_generation():
    """
    Test function to validate the new realistic data generation.
    Run this to verify the new system works correctly.
    """
    print("=" * 60)
    print("VALIDATING REALISTIC DATA GENERATION SYSTEM")
    print("=" * 60)
    
    # Test each scenario type
    scenarios = {
        "Strong Uptrend": TrendTestDataFactory.create_strong_uptrend_scenario(),
        "Strong Downtrend": TrendTestDataFactory.create_strong_downtrend_scenario(),
        "Tight Sideways": TrendTestDataFactory.create_tight_sideways_scenario(),
    }
    
    for name, candles in scenarios.items():
        print(f"\n--- Validating {name} ---")
        
        # Check candle continuity
        continuity_breaks = 0
        for i in range(1, len(candles)):
            if abs(candles[i].open - candles[i-1].close) > 0.01:  # Allow tiny rounding errors
                continuity_breaks += 1
        
        print(f"Candle continuity: {continuity_breaks} breaks out of {len(candles)-1} transitions")
        
        # Check OHLC validity
        invalid_ohlc = 0
        for candle in candles:
            if not (candle.low <= min(candle.open, candle.close) and
                   candle.high >= max(candle.open, candle.close)):
                invalid_ohlc += 1
        
        print(f"OHLC validity: {invalid_ohlc} invalid out of {len(candles)} candles")
        
        # Check price movement magnitude
        total_range = max(c.high for c in candles) - min(c.low for c in candles)
        range_percent = (total_range / candles[0].close) * 100
        net_change_percent = ((candles[-1].close / candles[0].close) - 1) * 100
        
        print(f"Price movement: {range_percent:.1f}% total range, {net_change_percent:+.1f}% net change")
        
        # Basic realism check
        if continuity_breaks == 0 and invalid_ohlc == 0 and range_percent > 5:
            print(f"✅ {name}: REALISTIC DATA GENERATED")
        else:
            print(f"❌ {name}: DATA ISSUES DETECTED")
    
    print(f"\n{'='*60}")
    print("VALIDATION COMPLETE")
    print("If all scenarios show ✅, the new system is working correctly!")
    print("=" * 60)

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