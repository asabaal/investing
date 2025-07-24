"""
Fixed business logic components for the Trend Detection Algorithm
UPDATED: Swing detection, controlling swing management, genesis point logic
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Protocol, Tuple, Dict, Set
import numpy as np
from datetime import datetime

from trend_core_models import (
    SwingPoint, SwingType, Trend, TrendDirection, TrendSignificance,
    TrendPattern, TrendMetrics, TrendAnalysisConfig, TrendAnalysisResult,
    GenesisPoint
)

# Use your existing Candle class from the supply/demand refactoring
from utilities import Candle


class SwingDetectionStrategy(Protocol):
    """Protocol for swing detection strategies"""
    
    def detect_swings(self, candles: List[Candle]) -> List[SwingPoint]:
        """Detect swing points in candle data"""
        ...


class BasicSwingDetectionStrategy:
    """
    CORRECTED: Simplified swing detection that actually works correctly.
    Responsibility: Detect swing points using reliable local extrema detection.
    """
    
    def __init__(self, lookback_period: int = 1):  # REVERTED: Back to 1 for reliability
        self.lookback_period = lookback_period
    
    def detect_swings(self, candles: List[Candle]) -> List[SwingPoint]:
        """CORRECTED: Reliable swing detection with proper alternation"""
        if len(candles) < 3:
            return []
        
        all_extrema = []
        
        # Step 1: Find ALL local extrema (don't worry about alternation yet)
        for i in range(1, len(candles) - 1):  # Check interior candles only
            swing = self._check_swing_at_index(candles, i)
            if swing:
                all_extrema.append(swing)
        
        # Step 2: Enforce alternation while preserving important swings
        return self._enforce_alternation_with_preservation(all_extrema)
    
    def _check_swing_at_index(self, candles: List[Candle], index: int) -> Optional[SwingPoint]:
        """
        MODIFIED: Use close prices for swing detection - much better for trend analysis.
        This catches candle 12 type scenarios naturally.
        """
        current = candles[index]
        prev_candle = candles[index - 1]
        next_candle = candles[index + 1]
        
        # Swing high: close higher than both neighbors
        if current.close > prev_candle.close and current.close > next_candle.close:
            return SwingPoint(
                candle_index=index,
                price=current.close,  # Use close price
                swing_type=SwingType.HIGH,
                timestamp=current.timestamp
            )
        
        # Swing low: close lower than both neighbors  
        if current.close < prev_candle.close and current.close < next_candle.close:
            return SwingPoint(
                candle_index=index,
                price=current.close,  # Use close price
                swing_type=SwingType.LOW,
                timestamp=current.timestamp
            )
        
        return None
    
    def _enforce_alternation_with_preservation(self, extrema: List[SwingPoint]) -> List[SwingPoint]:
        """
        CORRECTED: Enforce alternation while preserving significant intermediate swings.
        This is the key fix - we need to be smarter about which swings to keep.
        """
        if not extrema:
            return []
        
        # Sort by candle index to ensure chronological order
        extrema.sort(key=lambda s: s.candle_index)
        
        result = [extrema[0]]  # Always keep first swing
        
        for current_swing in extrema[1:]:
            last_swing = result[-1]
            
            if current_swing.swing_type != last_swing.swing_type:
                # Different types - this maintains alternation, always add
                result.append(current_swing)
            else:
                # Same type - need to decide whether to keep both or replace
                if current_swing.swing_type == SwingType.HIGH:
                    if current_swing.price > last_swing.price * 1.01:  # 1% higher
                        # Significantly higher high - keep both (important intermediate swing)
                        result.append(current_swing)
                    elif current_swing.price > last_swing.price:
                        # Higher but not significant - replace with the higher one
                        result[-1] = current_swing
                    # If lower, ignore it
                else:  # SwingType.LOW
                    if current_swing.price < last_swing.price * 0.99:  # 1% lower
                        # Significantly lower low - keep both (important intermediate swing)
                        result.append(current_swing)
                    elif current_swing.price < last_swing.price:
                        # Lower but not significant - replace with the lower one
                        result[-1] = current_swing
                    # If higher, ignore it
        
        return result


class SwingDetector:
    """
    Responsibility: Orchestrate swing detection using pluggable strategies.
    NO CHANGES - interface remains the same
    """
    
    def __init__(self, strategy: SwingDetectionStrategy):
        self._strategy = strategy
    
    def detect_swings(self, candles: List[Candle]) -> List[SwingPoint]:
        """Detect swings using the configured strategy"""
        return self._strategy.detect_swings(candles)


class PatternMatcher:
    """Clean pattern detection with single responsibility methods"""
    
    def __init__(self, config):
        self.config = config
    
    def find_uptrend_patterns(self, swings: List[SwingPoint]) -> List[TrendPattern]:
        """
        SIMPLE ORCHESTRATOR: Find L-H-L uptrend patterns
        Responsibility: Coordinate the pattern detection process only
        """
        print(f"Finding uptrend patterns from {len(swings)} swings")
        
        # 1. Find all potential L-H-L combinations
        lhl_combinations = self._find_lhl_combinations(swings)
        print(f"Found {len(lhl_combinations)} L-H-L combinations")
        
        # 2. Validate each combination
        valid_patterns = []
        for combination in lhl_combinations:
            if self._is_valid_lhl_pattern(combination, swings):
                pattern = self._create_uptrend_pattern(combination)
                valid_patterns.append(pattern)
                print(f"✅ Valid pattern: {[s.candle_index for s in combination]}")
            else:
                print(f"❌ Invalid pattern: {[s.candle_index for s in combination]}")
        
        return valid_patterns
    
    def find_downtrend_patterns(self, swings: List[SwingPoint]) -> List[TrendPattern]:
        """
        SIMPLE ORCHESTRATOR: Find H-L-H downtrend patterns
        Responsibility: Coordinate the pattern detection process only
        """
        print(f"Finding downtrend patterns from {len(swings)} swings")
        
        # 1. Find all potential H-L-H combinations
        hlh_combinations = self._find_hlh_combinations(swings)
        print(f"Found {len(hlh_combinations)} H-L-H combinations")
        
        # 2. Validate each combination
        valid_patterns = []
        for combination in hlh_combinations:
            if self._is_valid_hlh_pattern(combination, swings):
                pattern = self._create_downtrend_pattern(combination)
                valid_patterns.append(pattern)
                print(f"✅ Valid pattern: {[s.candle_index for s in combination]}")
            else:
                print(f"❌ Invalid pattern: {[s.candle_index for s in combination]}")
        
        return valid_patterns
    
    # =================================================================
    # SINGLE RESPONSIBILITY METHODS - Each easily testable
    # =================================================================
    
    def _find_lhl_combinations(self, swings: List[SwingPoint]) -> List[Tuple[SwingPoint, SwingPoint, SwingPoint]]:
        """
        SINGLE RESPONSIBILITY: Find all possible L-H-L combinations
        Input: List of swings
        Output: List of (low, high, low) tuples
        Testable: Easy to verify with known swing input
        """
        combinations = []
        
        for i, low1 in enumerate(swings):
            if low1.swing_type != SwingType.LOW:
                continue
                
            for j, high in enumerate(swings[i+1:], i+1):
                if high.swing_type != SwingType.HIGH:
                    continue
                    
                for k, low2 in enumerate(swings[j+1:], j+1):
                    if low2.swing_type != SwingType.LOW:
                        continue
                        
                    combinations.append((low1, high, low2))
        
        return combinations
    
    def _find_hlh_combinations(self, swings: List[SwingPoint]) -> List[Tuple[SwingPoint, SwingPoint, SwingPoint]]:
        """
        SINGLE RESPONSIBILITY: Find all possible H-L-H combinations
        Input: List of swings
        Output: List of (high, low, high) tuples
        Testable: Easy to verify with known swing input
        """
        combinations = []
        
        for i, high1 in enumerate(swings):
            if high1.swing_type != SwingType.HIGH:
                continue
                
            for j, low in enumerate(swings[i+1:], i+1):
                if low.swing_type != SwingType.LOW:
                    continue
                    
                for k, high2 in enumerate(swings[j+1:], j+1):
                    if high2.swing_type != SwingType.HIGH:
                        continue
                        
                    combinations.append((high1, low, high2))
        
        return combinations
    
    def _is_valid_lhl_pattern(self, combination: Tuple[SwingPoint, SwingPoint, SwingPoint], 
                             all_swings: List[SwingPoint]) -> bool:
        """
        SINGLE RESPONSIBILITY: Validate if L-H-L combination is a valid pattern
        Input: (low, high, low) tuple and all swings
        Output: True if valid pattern
        Testable: Easy to test with specific swing combinations
        """
        low1, high, low2 = combination
        
        # Check 1: Price relationships
        if not self._has_valid_lhl_price_structure(low1, high, low2):
            return False
        
        # Check 2: Genesis validation (the key fix!)
        if not self._is_valid_genesis_low(low1, combination, all_swings):
            return False
        
        # Check 3: Reasonable time spacing
        if not self._has_reasonable_timing(combination):
            return False
        
        return True
    
    def _is_valid_hlh_pattern(self, combination: Tuple[SwingPoint, SwingPoint, SwingPoint], 
                             all_swings: List[SwingPoint]) -> bool:
        """
        SINGLE RESPONSIBILITY: Validate if H-L-H combination is a valid pattern
        Input: (high, low, high) tuple and all swings
        Output: True if valid pattern
        Testable: Easy to test with specific swing combinations
        """
        high1, low, high2 = combination
        
        # Check 1: Price relationships
        if not self._has_valid_hlh_price_structure(high1, low, high2):
            return False
        
        # Check 2: Genesis validation (the key fix!)
        if not self._is_valid_genesis_high(high1, combination, all_swings):
            return False
        
        # Check 3: Reasonable time spacing
        if not self._has_reasonable_timing(combination):
            return False
        
        return True
    
    def _has_valid_lhl_price_structure(self, low1: SwingPoint, high: SwingPoint, low2: SwingPoint) -> bool:
        """
        SINGLE RESPONSIBILITY: Check if L-H-L has valid price relationships
        Input: Three swing points
        Output: True if prices form valid L-H-L structure
        Testable: Trivial to test with different price combinations
        """
        # High must be significantly higher than both lows
        if high.price <= low1.price * 1.005:  # 0.5% minimum
            return False
        if high.price <= low2.price * 1.005:
            return False
        
        # Low2 must be higher than low1 (higher low)
        if low2.price <= low1.price * 1.002:  # 0.2% minimum higher
            return False
        
        return True
    
    def _has_valid_hlh_price_structure(self, high1: SwingPoint, low: SwingPoint, high2: SwingPoint) -> bool:
        """
        SINGLE RESPONSIBILITY: Check if H-L-H has valid price relationships
        Input: Three swing points
        Output: True if prices form valid H-L-H structure
        Testable: Trivial to test with different price combinations
        """
        # Low must be significantly lower than both highs
        if low.price >= high1.price * 0.995:  # 0.5% minimum
            return False
        if low.price >= high2.price * 0.995:
            return False
        
        # High2 must be lower than high1 (lower high)
        if high2.price >= high1.price * 0.998:  # 0.2% minimum lower
            return False
        
        return True
    
    def _is_valid_genesis_low(self, potential_genesis: SwingPoint, 
                             combination: Tuple[SwingPoint, SwingPoint, SwingPoint],
                             all_swings: List[SwingPoint]) -> bool:
        """
        SINGLE RESPONSIBILITY: Validate genesis low (THE KEY FIX!)
        Input: Potential genesis swing and pattern combination
        Output: True if this is actually the lowest low in the pattern timeframe
        Testable: Easy to test with different swing sequences
        """
        low1, high, low2 = combination
        
        # Find all swings between genesis and pattern end
        pattern_swings = self._get_swings_in_timeframe(
            all_swings, 
            potential_genesis.candle_index, 
            low2.candle_index
        )
        
        # Check if any low in this timeframe is lower than potential genesis
        for swing in pattern_swings:
            if swing.swing_type == SwingType.LOW and swing.price < potential_genesis.price:
                print(f"    Genesis validation failed: Found lower low at candle {swing.candle_index} ({swing.price}) vs genesis {potential_genesis.candle_index} ({potential_genesis.price})")
                return False
        
        return True
    
    def _is_valid_genesis_high(self, potential_genesis: SwingPoint,
                              combination: Tuple[SwingPoint, SwingPoint, SwingPoint],
                              all_swings: List[SwingPoint]) -> bool:
        """
        SINGLE RESPONSIBILITY: Validate genesis high
        Input: Potential genesis swing and pattern combination
        Output: True if this is actually the highest high in the pattern timeframe
        Testable: Easy to test with different swing sequences
        """
        high1, low, high2 = combination
        
        # Find all swings between genesis and pattern end
        pattern_swings = self._get_swings_in_timeframe(
            all_swings,
            potential_genesis.candle_index,
            high2.candle_index
        )
        
        # Check if any high in this timeframe is higher than potential genesis
        for swing in pattern_swings:
            if swing.swing_type == SwingType.HIGH and swing.price > potential_genesis.price:
                print(f"    Genesis validation failed: Found higher high at candle {swing.candle_index} ({swing.price}) vs genesis {potential_genesis.candle_index} ({potential_genesis.price})")
                return False
        
        return True
    
    def _get_swings_in_timeframe(self, all_swings: List[SwingPoint], 
                                start_candle: int, end_candle: int) -> List[SwingPoint]:
        """
        SINGLE RESPONSIBILITY: Get swings within a time range
        Input: All swings and time boundaries
        Output: Swings within the timeframe
        Testable: Easy to verify with known swing list and boundaries
        """
        return [swing for swing in all_swings 
                if start_candle <= swing.candle_index <= end_candle]
    
    def _has_reasonable_timing(self, combination: Tuple[SwingPoint, SwingPoint, SwingPoint]) -> bool:
        """
        SINGLE RESPONSIBILITY: Check if pattern has reasonable time spacing
        Input: Pattern combination
        Output: True if timing is reasonable
        Testable: Easy to test with different time spacings
        """
        swing1, swing2, swing3 = combination
        
        # Pattern shouldn't be too compressed or too extended
        total_span = swing3.candle_index - swing1.candle_index
        
        if total_span < 2:  # Too compressed
            return False
        if total_span > 20:  # Too extended
            return False
        
        return True
    
    def _create_uptrend_pattern(self, combination: Tuple[SwingPoint, SwingPoint, SwingPoint]) -> TrendPattern:
        """
        SINGLE RESPONSIBILITY: Create TrendPattern object from L-H-L combination
        Input: Valid L-H-L combination
        Output: TrendPattern object
        Testable: Easy to verify pattern properties
        """
        low1, high, low2 = combination
        
        return TrendPattern(
            formation_swings=tuple(combination),
            pattern_type=TrendDirection.UP,
            start_index=low1.candle_index,
            end_index=low2.candle_index
        )
    
    def _create_downtrend_pattern(self, combination: Tuple[SwingPoint, SwingPoint, SwingPoint]) -> TrendPattern:
        """
        SINGLE RESPONSIBILITY: Create TrendPattern object from H-L-H combination
        Input: Valid H-L-H combination
        Output: TrendPattern object
        Testable: Easy to verify pattern properties
        """
        high1, low, high2 = combination
        
        return TrendPattern(
            formation_swings=tuple(combination),
            pattern_type=TrendDirection.DOWN,
            start_index=high1.candle_index,
            end_index=high2.candle_index
        )

    def find_sideways_patterns(self, candles: List[Candle], swings: List[SwingPoint],
                            start_idx: int, end_idx: int) -> List[TrendPattern]:
        """
        CLEAN SIDEWAYS DETECTION: Find sideways/consolidation patterns
        Responsibility: Detect low-momentum, range-bound patterns
        """
        print(f"Finding sideways patterns from candles {start_idx}-{end_idx}")
        
        patterns = []
        
        if end_idx - start_idx < 5:  # Need minimum data
            print("  Not enough data for sideways pattern")
            return patterns
        
        # Get candles in the period
        period_candles = candles[start_idx:end_idx + 1]
        if not period_candles:
            print("  No candles in period")
            return patterns
        
        # Check if this period qualifies as sideways
        if self._is_sideways_period(period_candles, swings, start_idx, end_idx):
            pattern = self._create_sideways_pattern(swings, start_idx, end_idx)
            if pattern:
                patterns.append(pattern)
                print(f"  ✅ Found sideways pattern: candles {start_idx}-{end_idx}")
        else:
            print(f"  ❌ Period {start_idx}-{end_idx} does not qualify as sideways")
        
        return patterns

    def _is_sideways_period(self, period_candles: List[Candle], swings: List[SwingPoint],
                        start_idx: int, end_idx: int) -> bool:
        """
        SINGLE RESPONSIBILITY: Check if period qualifies as sideways
        Input: Period candles and swing points
        Output: True if period is sideways/consolidation
        Testable: Easy to test with different market conditions
        """
        # Check 1: Narrow price range
        if not self._has_narrow_price_range(period_candles):
            print("    Failed: Price range too wide")
            return False
        
        # Check 2: Low momentum (small body-to-wick ratios)
        if not self._has_low_momentum(period_candles):
            print("    Failed: Momentum too high")
            return False
        
        # Check 3: Sufficient swing activity
        period_swings = self._get_swings_in_period(swings, start_idx, end_idx)
        if len(period_swings) < 3:
            print("    Failed: Not enough swings in period")
            return False
        
        print("    ✅ Period qualifies as sideways")
        return True

    def _has_narrow_price_range(self, period_candles: List[Candle]) -> bool:
        """
        SINGLE RESPONSIBILITY: Check if price range is narrow enough for sideways
        Input: Candles in the period
        Output: True if range is narrow
        Testable: Easy to test with different price ranges
        """
        range_high = max(c.high for c in period_candles)
        range_low = min(c.low for c in period_candles)
        range_size = range_high - range_low
        
        avg_price = (range_high + range_low) / 2
        range_pct = range_size / avg_price if avg_price > 0 else 0
        
        # More strict threshold for sideways detection
        threshold = self.config.sideways_range_threshold * 0.7  # 30% stricter
        
        print(f"    Range check: {range_pct:.3f} <= {threshold:.3f}")
        return range_pct <= threshold

    def _has_low_momentum(self, period_candles: List[Candle]) -> bool:
        """
        SINGLE RESPONSIBILITY: Check if momentum is low enough for sideways
        Input: Candles in the period  
        Output: True if momentum is low
        Testable: Easy to test with different momentum levels
        """
        # Check momentum of recent candles
        momentum_candles = period_candles[-3:] if len(period_candles) >= 3 else period_candles
        avg_momentum = sum(c.body_to_wick_ratio for c in momentum_candles) / len(momentum_candles)
        
        # More strict momentum requirement
        threshold = self.config.moveout_threshold * 0.6  # Stricter than before
        
        print(f"    Momentum check: {avg_momentum:.3f} < {threshold:.3f}")
        return avg_momentum < threshold

    def _get_swings_in_period(self, swings: List[SwingPoint], start_idx: int, end_idx: int) -> List[SwingPoint]:
        """
        SINGLE RESPONSIBILITY: Get swings within a specific period
        Input: All swings and period boundaries
        Output: Swings within the period
        Testable: Easy to verify with known swing list
        """
        return [swing for swing in swings 
                if start_idx <= swing.candle_index <= end_idx]

    def _create_sideways_pattern(self, swings: List[SwingPoint], start_idx: int, end_idx: int) -> Optional[TrendPattern]:
        """
        SINGLE RESPONSIBILITY: Create sideways pattern from period
        Input: Swings and period boundaries
        Output: TrendPattern object for sideways trend
        Testable: Easy to verify pattern properties
        """
        period_swings = self._get_swings_in_period(swings, start_idx, end_idx)
        
        if len(period_swings) < 3:
            return None
        
        # Take first few swings as formation swings
        formation_swings = period_swings[:4] if len(period_swings) >= 4 else period_swings
        
        return TrendPattern(
            formation_swings=tuple(formation_swings),
            pattern_type=TrendDirection.SIDEWAYS,
            start_index=start_idx,
            end_index=end_idx
        )        

class TrendBreakoutValidator:
    """
    Responsibility: Validate trend breakouts and confirmations.
    NO CHANGES - works correctly
    """
    
    def __init__(self, config: TrendAnalysisConfig):
        self.config = config
    
    def validate_uptrend_breakout(self, pattern: TrendPattern, current_candle: Candle) -> bool:
        """Validate uptrend breakout above swing high"""
        if pattern.pattern_type != TrendDirection.UP:
            return False
        
        # Find the swing high in the pattern
        swing_high = max(s for s in pattern.formation_swings if s.swing_type == SwingType.HIGH)
        
        return current_candle.high > swing_high.price
    
    def validate_downtrend_breakout(self, pattern: TrendPattern, current_candle: Candle) -> bool:
        """Validate downtrend breakdown below swing low"""
        if pattern.pattern_type != TrendDirection.DOWN:
            return False
        
        # Find the swing low in the pattern
        swing_low = min(s for s in pattern.formation_swings if s.swing_type == SwingType.LOW)
        
        return current_candle.low < swing_low.price
    
    def validate_moveout_strength(self, current_candle: Candle) -> bool:
        """Validate strength of moveout using body-to-wick ratio"""
        return current_candle.body_to_wick_ratio > self.config.moveout_threshold


class TrendFactory:
    """
    UPDATED: Correct genesis point and controlling swing initialization.
    Responsibility: Create Trend objects from patterns and validation results.
    """
    
    def __init__(self, config: TrendAnalysisConfig):
        self.config = config
        self._trend_counter = 0
    
    def create_trend_from_pattern(self, pattern: TrendPattern, current_candle_index: int,
                                 breakout_confirmed: bool = False, moveout_confirmed: bool = False,
                                 genesis_point: Optional[SwingPoint] = None) -> Trend:
        """UPDATED: Create trend with correct genesis point and controlling swing logic"""
        self._trend_counter += 1
        
        # UPDATED: Determine controlling swing and genesis point correctly
        controlling_swing = None
        genesis_swing = None
        
        if pattern.pattern_type == TrendDirection.UP:
            # For uptrend: controlling swing is the most recent swing LOW
            low_swings = [s for s in pattern.formation_swings if s.swing_type == SwingType.LOW]
            if low_swings:
                controlling_swing = max(low_swings, key=lambda s: s.candle_index)  # Most recent low
            
            # Genesis point is the original swing low that was broken out from
            if low_swings:
                genesis_swing = min(low_swings, key=lambda s: s.price)  # Lowest low in pattern
                
        elif pattern.pattern_type == TrendDirection.DOWN:
            # For downtrend: controlling swing is the most recent swing HIGH
            high_swings = [s for s in pattern.formation_swings if s.swing_type == SwingType.HIGH]
            if high_swings:
                controlling_swing = max(high_swings, key=lambda s: s.candle_index)  # Most recent high
            
            # Genesis point is the original swing high that was broken out from
            if high_swings:
                genesis_swing = max(high_swings, key=lambda s: s.price)  # Highest high in pattern
        
        # Calculate initial metrics
        formation_swings = pattern.formation_swings
        price_range = max(s.price for s in formation_swings) - min(s.price for s in formation_swings)
        duration = pattern.end_index - pattern.start_index + 1
        
        # Handle sideways-specific fields
        range_high = None
        range_low = None
        if pattern.pattern_type == TrendDirection.SIDEWAYS:
            range_high = max(s.price for s in formation_swings)
            range_low = min(s.price for s in formation_swings)
        
        return Trend(
            trend_id=self._trend_counter,
            direction=pattern.pattern_type,
            start_index=pattern.start_index,
            end_index=None,  # Active trend
            controlling_swing=controlling_swing,
            formation_pattern=pattern,
            is_active=True,
            price_range=price_range,
            duration=duration,
            moveout_confirmed=moveout_confirmed,
            breakout_confirmed=breakout_confirmed,
            range_high=range_high,
            range_low=range_low,
            genesis_point=genesis_swing  # UPDATED: Genesis point at trend start
        )


class TrendTerminationDetector:
    """
    Responsibility: Detect when trends should be terminated.
    NO CHANGES - logic is correct
    """
    
    def __init__(self, config: TrendAnalysisConfig):
        self.config = config
    
    def check_trend_termination(self, trend: Trend, current_candle: Candle,
                               candle_index: int) -> Tuple[bool, Optional[SwingPoint]]:
        """
        Check if trend should be terminated.
        Returns (should_terminate, genesis_swing)
        """
        if not trend.is_active or not trend.controlling_swing:
            return False, None
        
        terminated = False
        genesis_swing = None
        
        if trend.direction == TrendDirection.UP:
            if current_candle.low < trend.controlling_swing.price:
                terminated = True
                genesis_swing = self._find_termination_swing(
                    current_candle, candle_index, SwingType.HIGH
                )
        
        elif trend.direction == TrendDirection.DOWN:
            if current_candle.high > trend.controlling_swing.price:
                terminated = True
                genesis_swing = self._find_termination_swing(
                    current_candle, candle_index, SwingType.LOW
                )
        
        elif trend.direction == TrendDirection.SIDEWAYS:
            if trend.range_high and trend.range_low:
                range_size = trend.range_high - trend.range_low
                moveout_threshold = range_size * 0.3
                
                if current_candle.high > trend.range_high + moveout_threshold:
                    terminated = True
                    genesis_swing = self._find_termination_swing(
                        current_candle, candle_index, SwingType.HIGH
                    )
                elif current_candle.low < trend.range_low - moveout_threshold:
                    terminated = True
                    genesis_swing = self._find_termination_swing(
                        current_candle, candle_index, SwingType.LOW
                    )
        
        return terminated, genesis_swing
    
    def _find_termination_swing(self, current_candle: Candle, candle_index: int,
                               swing_type: SwingType) -> SwingPoint:
        """Create a swing point for trend termination"""
        if swing_type == SwingType.HIGH:
            price = current_candle.high
        else:
            price = current_candle.low
        
        return SwingPoint(
            candle_index=candle_index,
            price=price,
            swing_type=swing_type,
            timestamp=current_candle.timestamp
        )


class TrendClassifier:
    """
    Responsibility: Classify trend significance and calculate metrics.
    NO CHANGES - works correctly
    """
    
    def __init__(self, config: TrendAnalysisConfig):
        self.config = config
    
    def classify_trend_significance(self, trend: Trend, candles: List[Candle]) -> TrendSignificance:
        """Classify trend as MAJOR, MINOR, or CONSOLIDATION"""
        # Calculate actual duration and price range
        end_idx = trend.end_index if trend.end_index else len(candles) - 1
        duration = end_idx - trend.start_index + 1
        
        # Calculate price range from actual candles
        trend_candles = candles[trend.start_index:end_idx + 1]
        if trend_candles:
            actual_range = max(c.high for c in trend_candles) - min(c.low for c in trend_candles)
        else:
            actual_range = trend.price_range
        
        # Classify based on duration and range
        if (duration >= self.config.major_trend_min_duration and 
            actual_range >= self.config.major_trend_min_range):
            return TrendSignificance.MAJOR
        elif duration >= 3:
            return TrendSignificance.MINOR
        else:
            return TrendSignificance.CONSOLIDATION
    
    def calculate_trend_metrics(self, trend: Trend, candles: List[Candle]) -> TrendMetrics:
        """Calculate comprehensive trend metrics"""
        end_idx = trend.end_index if trend.end_index else len(candles) - 1
        duration = end_idx - trend.start_index + 1
        
        trend_candles = candles[trend.start_index:end_idx + 1]
        if not trend_candles:
            return TrendMetrics(
                duration=duration,
                price_range=trend.price_range,
                volatility=0.0,
                strength=0.0,
                dominance_score=0.0
            )
        
        # Calculate price range
        price_range = max(c.high for c in trend_candles) - min(c.low for c in trend_candles)
        
        # Calculate volatility (standard deviation of closes)
        closes = [c.close for c in trend_candles]
        volatility = float(np.std(closes)) if len(closes) > 1 else 0.0
        
        # Calculate trend strength (directional consistency)
        if trend.direction == TrendDirection.UP:
            ups = sum(1 for c in trend_candles if c.close > c.open)
            strength = ups / len(trend_candles)
        elif trend.direction == TrendDirection.DOWN:
            downs = sum(1 for c in trend_candles if c.close < c.open)
            strength = downs / len(trend_candles)
        else:  # SIDEWAYS
            # For sideways, strength is inverse of volatility
            strength = 1.0 / (1.0 + volatility) if volatility > 0 else 1.0
        
        # Calculate dominance score
        dominance_score = duration * price_range * strength
        
        return TrendMetrics(
            duration=duration,
            price_range=price_range,
            volatility=volatility,
            strength=strength,
            dominance_score=dominance_score
        )


class TrendManager:
    """
    UPDATED: Add controlling swing update functionality.
    Responsibility: Manage active trends and resolve conflicts.
    """
    
    def __init__(self, config: TrendAnalysisConfig):
        self.config = config
        self.active_trends: List[Trend] = []
        self.terminated_trends: List[Trend] = []
        self.genesis_points: List[GenesisPoint] = []
    
    def add_trend(self, trend: Trend) -> None:
        """Add a new active trend"""
        self.active_trends.append(trend)
    
    def update_controlling_swings(self, new_swing: SwingPoint) -> None:
        """
        UPDATED: Update controlling swings for active trends when new swing is confirmed.
        This is KEY for proper trend termination logic.
        """
        for i, trend in enumerate(self.active_trends):
            if trend.direction == TrendDirection.UP:
                # For uptrend, controlling swing is swing LOW
                if (new_swing.swing_type == SwingType.LOW and 
                    trend.controlling_swing and
                    new_swing.price > trend.controlling_swing.price and
                    new_swing.candle_index > trend.controlling_swing.candle_index):
                    
                    # Create updated trend with new controlling swing
                    updated_trend = Trend(
                        trend_id=trend.trend_id,
                        direction=trend.direction,
                        start_index=trend.start_index,
                        end_index=trend.end_index,
                        controlling_swing=new_swing,  # UPDATED controlling swing
                        formation_pattern=trend.formation_pattern,
                        significance=trend.significance,
                        is_active=trend.is_active,
                        price_range=trend.price_range,
                        duration=trend.duration,
                        moveout_confirmed=trend.moveout_confirmed,
                        breakout_confirmed=trend.breakout_confirmed,
                        range_high=trend.range_high,
                        range_low=trend.range_low,
                        genesis_point=trend.genesis_point,
                        dominance_score=trend.dominance_score
                    )
                    self.active_trends[i] = updated_trend
            
            elif trend.direction == TrendDirection.DOWN:
                # For downtrend, controlling swing is swing HIGH
                if (new_swing.swing_type == SwingType.HIGH and 
                    trend.controlling_swing and
                    new_swing.price < trend.controlling_swing.price and
                    new_swing.candle_index > trend.controlling_swing.candle_index):
                    
                    # Create updated trend with new controlling swing
                    updated_trend = Trend(
                        trend_id=trend.trend_id,
                        direction=trend.direction,
                        start_index=trend.start_index,
                        end_index=trend.end_index,
                        controlling_swing=new_swing,  # UPDATED controlling swing
                        formation_pattern=trend.formation_pattern,
                        significance=trend.significance,
                        is_active=trend.is_active,
                        price_range=trend.price_range,
                        duration=trend.duration,
                        moveout_confirmed=trend.moveout_confirmed,
                        breakout_confirmed=trend.breakout_confirmed,
                        range_high=trend.range_high,
                        range_low=trend.range_low,
                        genesis_point=trend.genesis_point,
                        dominance_score=trend.dominance_score
                    )
                    self.active_trends[i] = updated_trend
    
    def terminate_trend(self, trend: Trend, candle_index: int, 
                       genesis_swing: Optional[SwingPoint] = None) -> Optional[GenesisPoint]:
        """Terminate a trend and optionally create genesis point"""
        if trend in self.active_trends:
            self.active_trends.remove(trend)
        
        # Create terminated trend with end index
        terminated_trend = Trend(
            trend_id=trend.trend_id,
            direction=trend.direction,
            start_index=trend.start_index,
            end_index=candle_index,
            controlling_swing=trend.controlling_swing,
            formation_pattern=trend.formation_pattern,
            significance=trend.significance,
            is_active=False,
            price_range=trend.price_range,
            duration=trend.duration,
            moveout_confirmed=trend.moveout_confirmed,
            breakout_confirmed=trend.breakout_confirmed,
            range_high=trend.range_high,
            range_low=trend.range_low,
            genesis_point=trend.genesis_point,
            dominance_score=trend.dominance_score
        )
        
        self.terminated_trends.append(terminated_trend)
        
        # Create genesis point if swing provided
        genesis_point = None
        if genesis_swing:
            genesis_point = GenesisPoint(
                swing=genesis_swing,
                candle_index=candle_index,
                terminated_trend_id=trend.trend_id,
                genesis_type=genesis_swing.swing_type
            )
            self.genesis_points.append(genesis_point)
        
        return genesis_point
    
    def resolve_temporal_conflicts(self, candle_index: int) -> None:
        """Resolve overlapping active trends with more lenient criteria"""
        if len(self.active_trends) <= 1:
            return
        
        # Find overlapping trends
        conflicts = []
        for i, trend1 in enumerate(self.active_trends):
            for j, trend2 in enumerate(self.active_trends[i+1:], i+1):
                if self._trends_overlap(trend1, trend2, candle_index):
                    conflicts.append((trend1, trend2))
        
        # Group conflicts and resolve
        if conflicts:
            self._resolve_conflict_groups(conflicts, candle_index)
    
    def _trends_overlap(self, trend1: Trend, trend2: Trend, current_candle_index: int) -> bool:
        """Check if two trends overlap temporally with more lenient criteria"""
        end1 = trend1.end_index if trend1.end_index else current_candle_index
        end2 = trend2.end_index if trend2.end_index else current_candle_index
        
        # Calculate overlap
        overlap_start = max(trend1.start_index, trend2.start_index)
        overlap_end = min(end1, end2)
        
        if overlap_end <= overlap_start:
            return False  # No overlap
        
        overlap_duration = overlap_end - overlap_start
        min_duration = min(end1 - trend1.start_index, end2 - trend2.start_index)
        
        overlap_ratio = overlap_duration / min_duration if min_duration > 0 else 0
        
        # More lenient overlap threshold
        return overlap_ratio > self.config.max_trend_overlap * 1.5  # 50% more lenient
    
    def _resolve_conflict_groups(self, conflicts: List[Tuple[Trend, Trend]], candle_index: int) -> None:
        """Resolve conflict groups by keeping most dominant trends"""
        # Build conflict groups
        conflict_groups = []
        for trend1, trend2 in conflicts:
            added_to_group = False
            for group in conflict_groups:
                if trend1 in group or trend2 in group:
                    if trend1 not in group:
                        group.append(trend1)
                    if trend2 not in group:
                        group.append(trend2)
                    added_to_group = True
                    break
            
            if not added_to_group:
                conflict_groups.append([trend1, trend2])
        
        # Resolve each group - keep most dominant
        for group in conflict_groups:
            group.sort(key=lambda t: (t.dominance_score, t.duration, t.price_range), reverse=True)
            winner = group[0]
            losers = group[1:]
            
            for loser in losers:
                self.terminate_trend(loser, candle_index)
    
    def get_all_trends(self) -> List[Trend]:
        """Get all trends (active and terminated)"""
        return self.active_trends + self.terminated_trends


class TrendAnalysisEngine:
    """
    UPDATED: Add controlling swing updates to main analysis loop.
    Responsibility: Orchestrate the complete trend analysis process.
    """
    
    def __init__(self, config: TrendAnalysisConfig | None = None):
        self.config = config or TrendAnalysisConfig()
        
        # Initialize components with corrected strategies
        self.swing_detector = SwingDetector(BasicSwingDetectionStrategy(lookback_period=1))  # CORRECTED: Back to 1
        self.pattern_matcher = PatternMatcher(self.config)
        self.breakout_validator = TrendBreakoutValidator(self.config)
        self.trend_factory = TrendFactory(self.config)
        self.termination_detector = TrendTerminationDetector(self.config)
        self.trend_classifier = TrendClassifier(self.config)
        self.trend_manager = TrendManager(self.config)
    
    def analyze_trends(self, candles: List[Candle], analysis_start: int = 0,
                      analysis_end: Optional[int] = None) -> TrendAnalysisResult:
        """
        UPDATED: Add controlling swing updates to main analysis loop.
        Perform complete trend analysis.
        """
        if analysis_end is None:
            analysis_end = len(candles) - 1
        
        if not candles or analysis_start >= len(candles):
            raise ValueError("Invalid analysis parameters")
        
        # Get analysis window candles
        analysis_candles = candles[analysis_start:analysis_end + 1]
        
        # Handle single candle case
        if len(candles) == 1:
            return TrendAnalysisResult(
                candles=tuple(analysis_candles),
                swings=tuple(),
                detected_patterns=tuple(),
                trends=tuple(),
                active_trends=tuple(),
                major_trends=tuple(),
                current_trend=None,
                analysis_window=(0, 0),
                total_candles_analyzed=1
            )
        
        # Reset state
        self.trend_manager = TrendManager(self.config)
        
        # Step 1: Detect all swing points
        all_swings = self.swing_detector.detect_swings(candles)
        analysis_swings = [s for s in all_swings 
                          if analysis_start <= s.candle_index <= analysis_end]
        
        # Step 2: Process each candle in the analysis window
        detected_patterns = []
        for candle_index in range(analysis_start, analysis_end + 1):
            current_candle = candles[candle_index]
            current_swings = [s for s in all_swings if s.candle_index <= candle_index]
            
            # UPDATED: Check for new swing confirmations and update controlling swings
            current_swing = next((s for s in all_swings if s.candle_index == candle_index), None)
            if current_swing:
                self.trend_manager.update_controlling_swings(current_swing)
            
            # Check trend terminations
            self._process_trend_terminations(current_candle, candle_index)
            
            # Detect new patterns (prioritize up/down over sideways)
            new_patterns = self._detect_new_patterns(candles, current_swings, candle_index)
            detected_patterns.extend(new_patterns)
            
            # Resolve conflicts
            self.trend_manager.resolve_temporal_conflicts(candle_index)
        
        # Step 3: Classify all trends (unchanged)
        all_trends = self.trend_manager.get_all_trends()
        for trend in all_trends:
            significance = self.trend_classifier.classify_trend_significance(trend, candles)
            metrics = self.trend_classifier.calculate_trend_metrics(trend, candles)
            
            # Update trend with classification and metrics
            updated_trend = Trend(
                trend_id=trend.trend_id,
                direction=trend.direction,
                start_index=trend.start_index,
                end_index=trend.end_index,
                controlling_swing=trend.controlling_swing,
                formation_pattern=trend.formation_pattern,
                significance=significance,
                is_active=trend.is_active,
                price_range=metrics.price_range,
                duration=metrics.duration,
                moveout_confirmed=trend.moveout_confirmed,
                breakout_confirmed=trend.breakout_confirmed,
                range_high=trend.range_high,
                range_low=trend.range_low,
                genesis_point=trend.genesis_point,
                dominance_score=metrics.dominance_score
            )
            
            # Replace in manager
            if trend in self.trend_manager.active_trends:
                index = self.trend_manager.active_trends.index(trend)
                self.trend_manager.active_trends[index] = updated_trend
            if trend in self.trend_manager.terminated_trends:
                index = self.trend_manager.terminated_trends.index(trend)
                self.trend_manager.terminated_trends[index] = updated_trend
        
        # Step 4: Prepare results
        final_trends = self.trend_manager.get_all_trends()
        active_trends = [t for t in final_trends if t.is_active]
        major_trends = [t for t in final_trends if t.significance == TrendSignificance.MAJOR]
        current_trend = active_trends[-1] if active_trends else None
        
        return TrendAnalysisResult(
            candles=tuple(analysis_candles),
            swings=tuple(analysis_swings),
            detected_patterns=tuple(detected_patterns),
            trends=tuple(final_trends),
            active_trends=tuple(active_trends),
            major_trends=tuple(major_trends),
            current_trend=current_trend,
            analysis_window=(analysis_start, analysis_end),
            total_candles_analyzed=analysis_end - analysis_start + 1
        )
    
    def _process_trend_terminations(self, current_candle: Candle, candle_index: int) -> None:
        """Process trend terminations for current candle"""
        for trend in self.trend_manager.active_trends[:]:  # Copy list to avoid modification during iteration
            should_terminate, genesis_swing = self.termination_detector.check_trend_termination(
                trend, current_candle, candle_index
            )
            
            if should_terminate:
                self.trend_manager.terminate_trend(trend, candle_index, genesis_swing)
    
    def _detect_new_patterns(self, candles: List[Candle], swings: List[SwingPoint],
                           candle_index: int) -> List[TrendPattern]:
        """Detect new trend patterns at current candle (prioritize up/down trends)"""
        current_candle = candles[candle_index]
        new_patterns = []
        
        # FIRST: Find uptrend patterns (higher priority)
        up_patterns = self.pattern_matcher.find_uptrend_patterns(swings)
        for pattern in up_patterns:
            if self.breakout_validator.validate_uptrend_breakout(pattern, current_candle):
                moveout_confirmed = self.breakout_validator.validate_moveout_strength(current_candle)
                trend = self.trend_factory.create_trend_from_pattern(
                    pattern, candle_index, breakout_confirmed=True, 
                    moveout_confirmed=moveout_confirmed
                )
                self.trend_manager.add_trend(trend)
                new_patterns.append(pattern)
        
        # SECOND: Find downtrend patterns (higher priority)
        down_patterns = self.pattern_matcher.find_downtrend_patterns(swings)
        for pattern in down_patterns:
            if self.breakout_validator.validate_downtrend_breakout(pattern, current_candle):
                moveout_confirmed = self.breakout_validator.validate_moveout_strength(current_candle)
                trend = self.trend_factory.create_trend_from_pattern(
                    pattern, candle_index, breakout_confirmed=True,
                    moveout_confirmed=moveout_confirmed
                )
                self.trend_manager.add_trend(trend)
                new_patterns.append(pattern)
        
        # THIRD: Find sideways patterns only if no up/down trends found (lower priority)
        if not new_patterns and candle_index >= 8:  # Need more history and no directional trends
            sideways_patterns = self.pattern_matcher.find_sideways_patterns(
                candles, swings, max(0, candle_index - 15), candle_index
            )
            for pattern in sideways_patterns:
                trend = self.trend_factory.create_trend_from_pattern(
                    pattern, candle_index, breakout_confirmed=False, moveout_confirmed=False
                )
                self.trend_manager.add_trend(trend)
                new_patterns.append(pattern)
        
        return new_patterns