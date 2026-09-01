"""
Trade Scoring System

A decision tree model for scoring trading opportunities based on supply/demand zones.
Implements a 0-10 point scoring system across 6 key metrics:

1. Zone Strength (0-2 pts): Leg out distance and opposing zone breakout
2. Time/Base (0-1 pts): Number of candles in base segment  
3. Freshness (0-2 pts): Zone testing/violation status
4. Trend Alignment (0-2 pts): Zone direction vs current trend
5. Price Position (0-1 pts): Position in long-term price range
6. Profit Potential (0-2 pts): Risk/reward ratio
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd


class ZoneType(Enum):
    DEMAND = "demand"
    SUPPLY = "supply"


class TrendDirection(Enum):
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


class FreshnessStatus(Enum):
    UNTESTED = "untested"
    PARTIAL_PENETRATION = "partial_penetration"  # <50%
    DEEP_PENETRATION = "deep_penetration"       # >50% but not violated
    VIOLATED = "violated"


@dataclass
class Zone:
    """Represents a supply/demand zone"""
    zone_type: ZoneType
    high: float
    low: float
    base_candles: int
    leg_out_start: float
    leg_out_end: float
    opposing_zones: List[Tuple[float, float]] = None
    
    @property
    def range_size(self) -> float:
        return self.high - self.low
    
    @property
    def leg_out_distance(self) -> float:
        return abs(self.leg_out_end - self.leg_out_start)


@dataclass
class MarketContext:
    """Market context for scoring"""
    current_trend: TrendDirection
    current_price: float
    long_term_high: float
    long_term_low: float
    freshness_status: FreshnessStatus
    penetration_percentage: float = 0.0
    target_zone_distance: Optional[float] = None


class TradeScorer:
    """Main trade scoring system"""
    
    def __init__(self):
        self.max_score = 10.0
    
    def score_zone_strength(self, zone: Zone) -> float:
        """
        Score zone strength (0-2 points)
        - 2:1 leg out distance vs zone range: 1 pt
        - Broke opposing zone: 1 pt
        - Both components are independent - can get 0, 1, or 2 points total
        """
        score = 0.0
        
        # Component 1: Check if leg out moved at least 2:1 compared to zone range
        if zone.leg_out_distance >= (2 * zone.range_size):
            score += 1.0
        
        # Component 2: Check if leg out broke opposing zones (independent of ratio)
        if zone.opposing_zones:
            leg_out_broke_zone = self._check_opposing_zone_breakout(zone)
            if leg_out_broke_zone:
                score += 1.0
        
        return score
    
    def score_time_base(self, zone: Zone) -> float:
        """
        Score base segment time (0-1 points)
        - 1-3 candles: 1 pt
        - 4-6 candles: 0.5 pt
        - >6 candles: 0 pt
        """
        if 1 <= zone.base_candles <= 3:
            return 1.0
        elif 4 <= zone.base_candles <= 6:
            return 0.5
        else:
            return 0.0
    
    def score_freshness(self, context: MarketContext) -> float:
        """
        Score zone freshness (0-2 points)
        - Untested: 2 pts
        - <50% penetration: 1 pt
        - >50% penetration but not violated: 0 pts
        - Violated: Zone invalid (negative score to indicate invalid)
        """
        if context.freshness_status == FreshnessStatus.VIOLATED:
            return -1.0  # Invalid zone
        elif context.freshness_status == FreshnessStatus.UNTESTED:
            return 2.0
        elif (context.freshness_status == FreshnessStatus.PARTIAL_PENETRATION and 
              context.penetration_percentage < 50):
            return 1.0
        elif context.freshness_status == FreshnessStatus.DEEP_PENETRATION:
            return 0.0
        else:
            return 0.0
    
    def score_trend_alignment(self, zone: Zone, context: MarketContext) -> float:
        """
        Score trend alignment (0-2 points)
        - Zone/trend align: 2 pts
        - Sideways trend: 1 pt
        - Opposing trend: 0 pts
        """
        if context.current_trend == TrendDirection.SIDEWAYS:
            return 1.0
        
        # Check directional alignment
        zone_bullish = (zone.zone_type == ZoneType.DEMAND)
        trend_bullish = (context.current_trend == TrendDirection.UP)
        
        if zone_bullish == trend_bullish:
            return 2.0  # Aligned
        else:
            return 0.0  # Opposing
    
    def score_price_position(self, zone: Zone, context: MarketContext) -> float:
        """
        Score price position in long-term range (0-1 points)
        - Most extreme third (favorable direction): 1 pt
        - Middle third: 0.5 pt
        - Other extreme third: 0 pt
        """
        long_term_range = context.long_term_high - context.long_term_low
        third_size = long_term_range / 3
        
        # Determine which third the zone is in
        zone_mid = (zone.high + zone.low) / 2
        position_from_low = zone_mid - context.long_term_low
        
        if position_from_low <= third_size:
            # Bottom third
            third_position = "bottom"
        elif position_from_low <= 2 * third_size:
            # Middle third
            third_position = "middle"
        else:
            # Top third
            third_position = "top"
        
        # Score based on zone type and position
        if zone.zone_type == ZoneType.DEMAND:
            # For demand zones, bottom third is most favorable
            if third_position == "bottom":
                return 1.0
            elif third_position == "middle":
                return 0.5
            else:
                return 0.0
        else:  # Supply zone
            # For supply zones, top third is most favorable
            if third_position == "top":
                return 1.0
            elif third_position == "middle":
                return 0.5
            else:
                return 0.0
    
    def score_profit_potential(self, zone: Zone, context: MarketContext) -> float:
        """
        Score profit potential ratio (0-2 points)
        - >=5:1 leg out distance vs zone size: 2 pts
        - >=3:1 leg out distance vs zone size: 1 pt
        - <3:1: 0 pts
        
        This measures actual profit potential based on how far the leg out moved
        compared to the zone range (profit potential ratio).
        """
        if zone.leg_out_distance == 0 or zone.range_size == 0:
            return 0.0
        
        ratio = zone.leg_out_distance / zone.range_size
        
        if ratio >= 5.0:
            return 2.0
        elif ratio >= 3.0:
            return 1.0
        else:
            return 0.0
    
    def calculate_total_score(self, zone: Zone, context: MarketContext) -> Dict[str, float]:
        """
        Calculate total trade score and return breakdown
        """
        scores = {
            'zone_strength': self.score_zone_strength(zone),
            'time_base': self.score_time_base(zone),
            'freshness': self.score_freshness(context),
            'trend_alignment': self.score_trend_alignment(zone, context),
            'price_position': self.score_price_position(zone, context),
            'profit_potential': self.score_profit_potential(zone, context)
        }
        
        # Check if zone is invalid (violated)
        if scores['freshness'] < 0:
            scores['total'] = -1.0
            scores['status'] = 'INVALID'
            return scores
        
        scores['total'] = sum(scores.values())
        scores['status'] = 'VALID'
        
        return scores
    
    def _check_opposing_zone_breakout(self, zone: Zone) -> bool:
        """
        Check if leg out broke through any opposing zones
        """
        if not zone.opposing_zones:
            return False
        
        leg_out_range = (min(zone.leg_out_start, zone.leg_out_end), 
                        max(zone.leg_out_start, zone.leg_out_end))
        
        for opp_low, opp_high in zone.opposing_zones:
            # Check if leg out range overlaps with opposing zone
            if (leg_out_range[0] <= opp_high and leg_out_range[1] >= opp_low):
                return True
        
        return False


def create_example_zone() -> Tuple[Zone, MarketContext]:
    """Create example zone and context for testing"""
    zone = Zone(
        zone_type=ZoneType.DEMAND,
        high=100.0,
        low=98.0,
        base_candles=2,
        leg_out_start=99.0,
        leg_out_end=105.0,
        opposing_zones=[(102.0, 103.0)]
    )
    
    context = MarketContext(
        current_trend=TrendDirection.UP,
        current_price=99.5,
        long_term_high=120.0,
        long_term_low=80.0,
        freshness_status=FreshnessStatus.UNTESTED,
        target_zone_distance=10.0
    )
    
    return zone, context


if __name__ == "__main__":
    # Example usage
    scorer = TradeScorer()
    zone, context = create_example_zone()
    
    scores = scorer.calculate_total_score(zone, context)
    
    print("Trade Scoring Results:")
    print("=" * 30)
    for metric, score in scores.items():
        if metric not in ['total', 'status']:
            print(f"{metric.replace('_', ' ').title()}: {score:.1f}")
    print("-" * 30)
    print(f"Total Score: {scores['total']:.1f}/10.0")
    print(f"Status: {scores['status']}")