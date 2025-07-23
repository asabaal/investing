"""
Final fixed business logic components for the Trend Detection Algorithm
Improved swing detection, pattern recognition, and analysis result handling
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
    Responsibility: Detect swing points using basic high/low comparison.
    Ensures proper alternation of swing types.
    """
    
    def __init__(self, lookback_period: int = 1):
        self.lookback_period = lookback_period
    
    def detect_swings(self, candles: List[Candle]) -> List[SwingPoint]:
        """Detect swing points using basic comparison method with alternation enforcement"""
        if len(candles) < 3:
            return []
        
        candidate_swings = []
        
        # Find all potential swing points
        for i in range(self.lookback_period, len(candles) - self.lookback_period):
            swing = self._check_swing_at_index(candles, i)
            if swing:
                candidate_swings.append(swing)
        
        # Check potential swing at last candle
        if len(candles) >= 2:
            last_swing = self._check_last_candle_swing(candles)
            if last_swing:
                candidate_swings.append(last_swing)
        
        # Enforce alternation pattern
        return self._enforce_alternation(candidate_swings)
    
    def _check_swing_at_index(self, candles: List[Candle], index: int) -> Optional[SwingPoint]:
        """Check if candle at index forms a swing point"""
        current = candles[index]
        
        # Check for swing high
        is_swing_high = True
        for offset in range(-self.lookback_period, self.lookback_period + 1):
            if offset == 0:
                continue
            compare_idx = index + offset
            if 0 <= compare_idx < len(candles):
                if current.high <= candles[compare_idx].high:
                    is_swing_high = False
                    break
        
        if is_swing_high:
            return SwingPoint(
                candle_index=index,
                price=current.high,
                swing_type=SwingType.HIGH,
                timestamp=current.timestamp
            )
        
        # Check for swing low
        is_swing_low = True
        for offset in range(-self.lookback_period, self.lookback_period + 1):
            if offset == 0:
                continue
            compare_idx = index + offset
            if 0 <= compare_idx < len(candles):
                if current.low >= candles[compare_idx].low:
                    is_swing_low = False
                    break
        
        if is_swing_low:
            return SwingPoint(
                candle_index=index,
                price=current.low,
                swing_type=SwingType.LOW,
                timestamp=current.timestamp
            )
        
        return None
    
    def _check_last_candle_swing(self, candles: List[Candle]) -> Optional[SwingPoint]:
        """Check if last candle could be a swing point"""
        if len(candles) < 3:
            return None
        
        last_idx = len(candles) - 1
        current = candles[last_idx]
        
        # Simple check: higher than previous two candles
        if (current.high > candles[last_idx - 1].high and 
            current.high > candles[last_idx - 2].high):
            return SwingPoint(
                candle_index=last_idx,
                price=current.high,
                swing_type=SwingType.HIGH,
                timestamp=current.timestamp
            )
        
        # Simple check: lower than previous two candles
        if (current.low < candles[last_idx - 1].low and 
            current.low < candles[last_idx - 2].low):
            return SwingPoint(
                candle_index=last_idx,
                price=current.low,
                swing_type=SwingType.LOW,
                timestamp=current.timestamp
            )
        
        return None
    
    def _enforce_alternation(self, candidate_swings: List[SwingPoint]) -> List[SwingPoint]:
        """Enforce alternating HIGH-LOW pattern by filtering out consecutive same types"""
        if not candidate_swings:
            return []
        
        filtered_swings = [candidate_swings[0]]  # Always keep first swing
        
        for swing in candidate_swings[1:]:
            last_swing = filtered_swings[-1]
            
            # Only add if it's different type from last swing
            if swing.swing_type != last_swing.swing_type:
                filtered_swings.append(swing)
            else:
                # Keep the more extreme swing of the same type
                if swing.swing_type == SwingType.HIGH:
                    if swing.price > last_swing.price:
                        filtered_swings[-1] = swing  # Replace with higher high
                elif swing.swing_type == SwingType.LOW:
                    if swing.price < last_swing.price:
                        filtered_swings[-1] = swing  # Replace with lower low
        
        return filtered_swings


class SwingDetector:
    """
    Responsibility: Orchestrate swing detection using pluggable strategies.
    """
    
    def __init__(self, strategy: SwingDetectionStrategy):
        self._strategy = strategy
    
    def detect_swings(self, candles: List[Candle]) -> List[SwingPoint]:
        """Detect swings using the configured strategy"""
        return self._strategy.detect_swings(candles)


class PatternMatcher:
    """
    Responsibility: Identify trend patterns from swing points.
    Prioritizes up/down trends over sideways patterns.
    """
    
    def __init__(self, config: TrendAnalysisConfig):
        self.config = config
    
    def find_uptrend_patterns(self, swings: List[SwingPoint]) -> List[TrendPattern]:
        """Find L-H-L uptrend patterns"""
        patterns = []
        
        for i in range(len(swings) - 2):
            pattern_swings = swings[i:i+3]
            
            if (pattern_swings[0].swing_type == SwingType.LOW and
                pattern_swings[1].swing_type == SwingType.HIGH and
                pattern_swings[2].swing_type == SwingType.LOW):
                
                # Check for higher low (more strict requirement)
                if pattern_swings[2].price > pattern_swings[0].price * 1.005:  # At least 0.5% higher
                    pattern = TrendPattern(
                        formation_swings=tuple(pattern_swings),
                        pattern_type=TrendDirection.UP,
                        start_index=pattern_swings[0].candle_index,
                        end_index=pattern_swings[2].candle_index
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def find_downtrend_patterns(self, swings: List[SwingPoint]) -> List[TrendPattern]:
        """Find H-L-H downtrend patterns"""
        patterns = []
        
        for i in range(len(swings) - 2):
            pattern_swings = swings[i:i+3]
            
            if (pattern_swings[0].swing_type == SwingType.HIGH and
                pattern_swings[1].swing_type == SwingType.LOW and
                pattern_swings[2].swing_type == SwingType.HIGH):
                
                # Check for lower high (more strict requirement)
                if pattern_swings[2].price < pattern_swings[0].price * 0.995:  # At least 0.5% lower
                    pattern = TrendPattern(
                        formation_swings=tuple(pattern_swings),
                        pattern_type=TrendDirection.DOWN,
                        start_index=pattern_swings[0].candle_index,
                        end_index=pattern_swings[2].candle_index
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def find_sideways_patterns(self, candles: List[Candle], swings: List[SwingPoint],
                              start_idx: int, end_idx: int) -> List[TrendPattern]:
        """Find sideways/consolidation patterns (less aggressive detection)"""
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
    Responsibility: Create Trend objects from patterns and validation results.
    """
    
    def __init__(self, config: TrendAnalysisConfig):
        self.config = config
        self._trend_counter = 0
    
    def create_trend_from_pattern(self, pattern: TrendPattern, current_candle_index: int,
                                 breakout_confirmed: bool = False, moveout_confirmed: bool = False,
                                 genesis_point: Optional[SwingPoint] = None) -> Trend:
        """Create a trend from a validated pattern"""
        self._trend_counter += 1
        
        # Determine controlling swing
        controlling_swing = None
        if pattern.pattern_type == TrendDirection.UP:
            # Find the lowest swing point among LOWs
            low_swings = [s for s in pattern.formation_swings if s.swing_type == SwingType.LOW]
            if low_swings:
                controlling_swing = min(low_swings, key=lambda s: s.price)
        elif pattern.pattern_type == TrendDirection.DOWN:
            # Find the highest swing point among HIGHs
            high_swings = [s for s in pattern.formation_swings if s.swing_type == SwingType.HIGH]
            if high_swings:
                controlling_swing = max(high_swings, key=lambda s: s.price)
        
        # Calculate initial metrics
        formation_swings = pattern.formation_swings
        if pattern.pattern_type == TrendDirection.UP:
            price_range = max(s.price for s in formation_swings) - min(s.price for s in formation_swings)
        elif pattern.pattern_type == TrendDirection.DOWN:
            price_range = max(s.price for s in formation_swings) - min(s.price for s in formation_swings)
        else:  # SIDEWAYS
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
            genesis_point=genesis_point
        )


class TrendTerminationDetector:
    """
    Responsibility: Detect when trends should be terminated.
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
    Responsibility: Orchestrate the complete trend analysis process.
    """
    
    def __init__(self, config: TrendAnalysisConfig | None = None):
        self.config = config or TrendAnalysisConfig()
        
        # Initialize components
        self.swing_detector = SwingDetector(BasicSwingDetectionStrategy())
        self.pattern_matcher = PatternMatcher(self.config)
        self.breakout_validator = TrendBreakoutValidator(self.config)
        self.trend_factory = TrendFactory(self.config)
        self.termination_detector = TrendTerminationDetector(self.config)
        self.trend_classifier = TrendClassifier(self.config)
        self.trend_manager = TrendManager(self.config)
    
    def analyze_trends(self, candles: List[Candle], analysis_start: int = 0,
                      analysis_end: Optional[int] = None) -> TrendAnalysisResult:
        """
        Perform complete trend analysis.
        
        Args:
            candles: List of market candles to analyze (using existing Candle from supply/demand)
            analysis_start: Start index for analysis window
            analysis_end: End index for analysis window (None = end of data)
            
        Returns:
            Complete trend analysis results
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
        
        # Step 2: Process each candle in the analysis window (prioritize up/down patterns)
        detected_patterns = []
        for candle_index in range(analysis_start, analysis_end + 1):
            current_candle = candles[candle_index]
            current_swings = [s for s in all_swings if s.candle_index <= candle_index]
            
            # Check trend terminations
            self._process_trend_terminations(current_candle, candle_index)
            
            # Detect new patterns (prioritize up/down over sideways)
            new_patterns = self._detect_new_patterns(candles, current_swings, candle_index)
            detected_patterns.extend(new_patterns)
            
            # Resolve conflicts
            self.trend_manager.resolve_temporal_conflicts(candle_index)
        
        # Step 3: Classify all trends
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