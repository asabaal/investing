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
    """
    UPDATED: Enhanced pattern matching for better L-H-L detection.
    Responsibility: Identify trend patterns from swing points.
    """
    
    def __init__(self, config: TrendAnalysisConfig):
        self.config = config
    
    def find_uptrend_patterns(self, swings: List[SwingPoint]) -> List[TrendPattern]:
        """UPDATED: Find L-H-L uptrend patterns with flexible intermediate swings"""
        patterns = []
        
        # Look for significant L-H-L patterns, not just consecutive triplets
        for i in range(len(swings)):
            if swings[i].swing_type == SwingType.LOW:
                # Found potential trend start low - look for higher high
                for j in range(i + 1, min(i + 8, len(swings))):  # Limit search window
                    if (swings[j].swing_type == SwingType.HIGH and 
                        swings[j].price > swings[i].price * 1.01):  # Must be significantly higher
                        
                        # Found higher high - look for higher low
                        for k in range(j + 1, min(j + 6, len(swings))):  # Limit search window
                            if (swings[k].swing_type == SwingType.LOW and 
                                swings[k].price > swings[i].price * 1.005):  # Must be higher low
                                
                                # Found valid L-H-L pattern
                                pattern = TrendPattern(
                                    formation_swings=tuple([swings[i], swings[j], swings[k]]),
                                    pattern_type=TrendDirection.UP,
                                    start_index=swings[i].candle_index,
                                    end_index=swings[k].candle_index
                                )
                                patterns.append(pattern)
                                break  # Found pattern for this L-H combination
                        break  # Move to next low after finding valid high
        
        return self._filter_overlapping_patterns(patterns)
    
    def find_downtrend_patterns(self, swings: List[SwingPoint]) -> List[TrendPattern]:
        """UPDATED: Find H-L-H downtrend patterns with flexible intermediate swings"""
        patterns = []
        
        # Look for significant H-L-H patterns
        for i in range(len(swings)):
            if swings[i].swing_type == SwingType.HIGH:
                # Found potential trend start high - look for lower low
                for j in range(i + 1, min(i + 8, len(swings))):  # Limit search window
                    if (swings[j].swing_type == SwingType.LOW and 
                        swings[j].price < swings[i].price * 0.99):  # Must be significantly lower
                        
                        # Found lower low - look for lower high
                        for k in range(j + 1, min(j + 6, len(swings))):  # Limit search window
                            if (swings[k].swing_type == SwingType.HIGH and 
                                swings[k].price < swings[i].price * 0.995):  # Must be lower high
                                
                                # Found valid H-L-H pattern
                                pattern = TrendPattern(
                                    formation_swings=tuple([swings[i], swings[j], swings[k]]),
                                    pattern_type=TrendDirection.DOWN,
                                    start_index=swings[i].candle_index,
                                    end_index=swings[k].candle_index
                                )
                                patterns.append(pattern)
                                break  # Found pattern for this H-L combination
                        break  # Move to next high after finding valid low
        
        return self._filter_overlapping_patterns(patterns)
    
    def _filter_overlapping_patterns(self, patterns: List[TrendPattern]) -> List[TrendPattern]:
        """Remove overlapping patterns, keeping the most significant ones"""
        if len(patterns) <= 1:
            return patterns
        
        # Sort by pattern strength (price range)
        def pattern_strength(pattern):
            prices = [s.price for s in pattern.formation_swings]
            return max(prices) - min(prices)
        
        patterns.sort(key=pattern_strength, reverse=True)
        
        filtered = []
        for pattern in patterns:
            # Check if this pattern overlaps significantly with existing ones
            overlaps = False
            for existing in filtered:
                if (pattern.start_index < existing.end_index and 
                    pattern.end_index > existing.start_index):
                    # Overlapping - skip this one
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(pattern)
        
        return filtered
    
    def find_sideways_patterns(self, candles: List[Candle], swings: List[SwingPoint],
                              start_idx: int, end_idx: int) -> List[TrendPattern]:
        """Find sideways/consolidation patterns (unchanged for now)"""
        patterns = []
        
        if end_idx - start_idx < 5:  # Require more data for sideways
            return patterns
        
        # Analyze price range in the period
        period_candles = candles[start_idx:end_idx + 1]
        if not period_candles:
            return patterns
        
        range_high = max(c.high for c in period_candles)
        range_low = min(c.low for c in period_candles)
        range_size = range_high - range_low
        
        # Check if range is narrow enough for sideways (more strict)
        avg_price = (range_high + range_low) / 2
        range_pct = range_size / avg_price if avg_price > 0 else 0
        
        # More strict threshold for sideways detection
        if range_pct <= self.config.sideways_range_threshold * 0.7:  # 30% stricter
            # Check for low momentum (small body-to-wick ratios)
            momentum_candles = period_candles[-3:] if len(period_candles) >= 3 else period_candles
            avg_momentum = np.mean([c.body_to_wick_ratio for c in momentum_candles])
            
            # More strict momentum requirement
            if avg_momentum < self.config.moveout_threshold * 0.6:  # Stricter than before
                # Find representative swings in the period
                period_swings = [s for s in swings if start_idx <= s.candle_index <= end_idx]
                
                if len(period_swings) >= 3:  # Require more swings
                    pattern = TrendPattern(
                        formation_swings=tuple(period_swings[:4]),  # Take first few swings
                        pattern_type=TrendDirection.SIDEWAYS,
                        start_index=start_idx,
                        end_index=end_idx
                    )
                    patterns.append(pattern)
        
        return patterns


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