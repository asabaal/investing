# USO Swing Point Algorithm Rules

## Overview
This document defines the algorithmic rules for identifying swing points, trends, and trading signals in USO price data. We're building this incrementally based on weekly chart analysis.

## Core Definitions

### 1. Trend Detection
- **5-Candle Monotonic Rule**: A trend is confirmed when 5 consecutive candles move in the same direction
- **Genesis Candle**: The candle that becomes the start of a confirmed trend (e.g., candle 22 becomes genesis of downtrend when candle 27 confirms it)
- **Trend Types**: 
  - UPTREND: 5 consecutive candles with higher lows
  - DOWNTREND: 5 consecutive candles with lower highs

### 2. Swing Point Classification

#### Local Extrema vs Significant Swing Points
**Problem Identified**: Not all local extrema should be considered swing points.

**Example from Weekly Analysis**:
- Candle 25 contains a local extrema BUT candle 26 is completely contained within candle 25's price range
- Therefore, candle 25's extrema should NOT be considered a significant swing point
- Candle 29 is the true local minimum in this sequence

#### Proposed Rules:
1. **Local Extrema**: Any candle that is a mathematical high/low relative to neighbors
2. **Significant Swing Point**: A local extrema that EXCEEDS the range of the previous H/L extrema

### 3. Initial Weekly Chart Swing Points Analysis
Based on manual review of weekly USO chart:

**Candidate Swing Points** (Index, Type):
- 4L (Low)
- 8H (High) 
- 11L (Low)
- 22H (High) - Genesis of confirmed downtrend at candle 27
- 29L (Low) - True minimum (not 25L due to containment rule)
- 33H (High)
- 34L (Low)
- 45H (High)

**Note**: This list needs refinement using precise algorithmic rules.

## Algorithmic Implementation Strategy

### Phase 1: Trend Detection
1. Implement 5-candle monotonic detection
2. Identify genesis candles when trends confirm
3. Track trend state changes

### Phase 2: Swing Point Validation
1. Identify all local extrema mathematically
2. Apply containment filter (eliminate extrema contained within previous candle ranges)
3. Apply range-breaking filter (only keep extrema that exceed previous H/L ranges)

### Phase 3: Comparison Analysis
1. Compare new algorithm results to existing supply/demand model
2. Identify discrepancies and improvements
3. Document edge cases and refinements needed

## Questions to Resolve

1. **Range-Breaking Threshold**: What constitutes "exceeding" the range of previous H/L extrema?
   - Exact price level breakthrough?
   - Percentage threshold?
   - Body vs wick considerations?

2. **Containment Rule Precision**: How exactly do we define "completely contained"?
   - Full OHLC within previous candle range?
   - Just body contained?
   - Wick tolerance allowed?

3. **Trend Continuation vs Reversal**: How do we handle swing points that occur during confirmed trends?

## Implementation Status

- [ ] 5-candle monotonic trend detection
- [ ] Genesis candle identification
- [ ] Local extrema mathematical detection
- [ ] Containment rule implementation
- [ ] Range-breaking rule implementation
- [ ] Swing point labeling on charts
- [ ] Comparison with existing algorithm

## Next Steps

1. Fix datetime display on chart (in progress)
2. Implement basic trend detection algorithm
3. Code swing point detection with containment rules
4. Add visual swing point labels to chart
5. Compare results with existing supply/demand model

---
*Document created: 2025-08-06*
*Last updated: 2025-08-06*