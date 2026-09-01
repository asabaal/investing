# Trade Setup Examples - Workshop Materials

This directory contains comprehensive examples for the trade scoring system, designed to help workshop and refine the scoring definitions.

## Overview

The trade scoring system evaluates setups on a 0-10 point scale across 6 key metrics:

1. **Zone Strength** (0-2 pts): Leg out distance + opposing zone breakout  
2. **Time/Base** (0-1 pts): Number of candles in base segment
3. **Freshness** (0-2 pts): Zone testing/violation status
4. **Trend Alignment** (0-2 pts): Zone direction vs current trend  
5. **Price Position** (0-1 pts): Position in long-term price range
6. **Profit Potential** (0-2 pts): Risk/reward ratio

## File Structure

### Complete Setup Examples
- `perfect_setup_analysis.html` - 9.5/10 score - Nearly perfect trade setup
- `strong_setup_analysis.html` - 6.0/10 score - Strong but not perfect  
- `moderate_setup_analysis.html` - 3.5/10 score - Mixed signals
- `weak_setup_analysis.html` - 4.0/10 score - Weak opportunity
- `poor_setup_analysis.html` - 2.5/10 score - Poor setup
- `invalid_setup_analysis.html` - Invalid - Violated zone

### Metric Deep Dives
- `metric_zone_strength_deep_dive.html` - Zone strength isolated examples
- `metric_time_base_deep_dive.html` - Time/base scoring examples
- `metric_freshness_deep_dive.html` - Freshness status examples
- `metric_trend_alignment_deep_dive.html` - Trend alignment examples
- `metric_price_position_deep_dive.html` - Price position examples  
- `metric_profit_potential_deep_dive.html` - R:R ratio examples

### Analysis Reports
- `scoring_summary_report.html` - Comprehensive analysis across all examples

## Key Insights for Workshop Discussions

### 1. Zone Strength (0-2 points)
- **Critical Rule**: Leg out must move ≥2:1 compared to zone range
- **Bonus Point**: Breaking through opposing zones adds +1 point
- **Workshop Question**: Should the 2:1 ratio be adjustable based on timeframe?

### 2. Time/Base (0-1 points)
- **1-3 candles**: 1.0 point (strongest - decisive move)
- **4-6 candles**: 0.5 points (acceptable)
- **>6 candles**: 0.0 points (too much consolidation)
- **Workshop Question**: Should this scale with timeframe?

### 3. Freshness (0-2 points)
- **Untested**: 2.0 points (highest probability)
- **<50% penetration**: 1.0 point (still valid)
- **>50% penetration**: 0.0 points (weakened)
- **Violated**: Invalid (-1.0 - no trade)
- **Workshop Question**: Are the penetration thresholds correct?

### 4. Trend Alignment (0-2 points)
- **Perfect alignment**: 2.0 points (demand in uptrend, supply in downtrend)
- **Sideways trend**: 1.0 point (neutral)
- **Counter-trend**: 0.0 points (fighting the trend)
- **Workshop Question**: How do we define trend strength requirements?

### 5. Price Position (0-1 points)
- **Favorable third**: 1.0 point (demand at bottom, supply at top)
- **Middle third**: 0.5 points (neutral)
- **Unfavorable third**: 0.0 points
- **Workshop Question**: What timeframe should define "long-term range"?

### 6. Profit Potential (0-2 points)
- **≥5:1 leg out ratio**: 2.0 points (excellent profit potential)
- **≥3:1 leg out ratio**: 1.0 point (good profit potential)
- **<3:1 leg out ratio**: 0.0 points (avoid - poor potential)
- **Workshop Question**: Should minimum ratio vary by setup quality?

## Workshop Discussion Points

### Score Distribution Analysis
From the examples generated:
- **Perfect Setup (9.5/10)**: Untested demand zone in strong uptrend with excellent R:R
- **Strong Setup (6.0/10)**: Minor zone test in uptrend with good R:R  
- **Moderate Setup (3.5/10)**: Deeply tested supply zone in sideways market
- **Weak Setup (4.0/10)**: Counter-trend demand zone with poor R:R
- **Poor Setup (2.5/10)**: Large base, counter-trend, minimal leg out
- **Invalid Setup (-1.0)**: Zone completely violated

### Key Patterns Observed
1. **Trend alignment** has major impact on total score
2. **Freshness** is critical - violated zones invalidate setups
3. **Time/base** acts as a quality filter  
4. **Profit potential ratio** separates good from great setups
5. **Price position** provides fine-tuning

### Refinement Questions for Workshop

1. **Weighting**: Should metrics have different weights based on importance?
2. **Thresholds**: Are current scoring thresholds optimal?
3. **Edge Cases**: How to handle boundary conditions?
4. **Timeframe**: Should scoring adjust for different timeframes?
5. **Market Conditions**: Should scoring adapt to volatility regimes?

## Using These Examples

1. **Open the HTML files** in a web browser to see interactive visualizations
2. **Study the scoring breakdowns** to understand how each metric contributes
3. **Compare examples** across the scoring spectrum
4. **Focus on metric deep dives** to understand individual components
5. **Use for training** - these represent the "ground truth" for the scoring system

## Next Steps

1. Review all examples and validate scoring logic
2. Adjust thresholds based on workshop feedback
3. Test with real market data
4. Integrate with existing formation detection system
5. Build automated setup scanner using these rules

---

*Generated by trade_setup_visualizer.py and metric_deep_dive_analyzer.py*
*Use these examples to workshop and refine your scoring system definitions*