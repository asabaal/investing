"""
SLIDING WINDOW INVESTIGATION: Test if the algorithm is designed for multi-timeframe analysis
==========================================================================================
This investigates whether 53 trends is actually CORRECT behavior for sliding window visualization.
"""

# Required imports
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple

from trend_business_logic import (
    TrendAnalysisEngine, SwingDetector, BasicSwingDetectionStrategy,
    PatternMatcher, TrendFactory, TrendBreakoutValidator,
    TrendTerminationDetector, TrendClassifier, TrendManager
)

from trend_core_models import (
    SwingPoint, SwingType, Trend, TrendDirection, TrendSignificance,
    TrendPattern, TrendAnalysisConfig, TrendAnalysisResult
)

from utilities import Candle, CandleBuilder


def create_31_candle_test_scenario() -> List[Candle]:
    """Create the test scenario"""
    test_specs = [
        {'close': 100, 'body': 1.0, 'bullish': True},   # 0
        {'close': 99, 'body': 1.5, 'bullish': False},   # 1
        {'close': 101, 'body': 1.2, 'bullish': True},   # 2
        {'close': 97, 'body': 3.5, 'bullish': False},   # 3
        {'close': 100, 'body': 2.0, 'bullish': True},   # 4
        {'close': 108, 'body': 4.0, 'bullish': True},   # 5
        {'close': 105, 'body': 1.5, 'bullish': False},  # 6
        {'close': 103, 'body': 2.5, 'bullish': False},  # 7
        {'close': 106, 'body': 2.8, 'bullish': True},   # 8
        {'close': 112, 'body': 4.5, 'bullish': True},   # 9
        {'close': 116, 'body': 3.8, 'bullish': True},   # 10
        {'close': 120, 'body': 3.5, 'bullish': True},   # 11
        {'close': 118, 'body': 1.8, 'bullish': False},  # 12
        {'close': 122, 'body': 3.0, 'bullish': True},   # 13
        {'close': 124, 'body': 2.5, 'bullish': True},   # 14
        {'close': 122, 'body': 0.6, 'bullish': False},  # 15
        {'close': 125, 'body': 0.7, 'bullish': True},   # 16
        {'close': 121, 'body': 0.5, 'bullish': False},  # 17
        {'close': 124, 'body': 0.8, 'bullish': True},   # 18
        {'close': 116.5, 'body': 2.0, 'bullish': False}, # 19
        {'close': 125, 'body': 0.7, 'bullish': True},   # 20
        {'close': 121, 'body': 0.5, 'bullish': False},  # 21
        {'close': 123, 'body': 0.6, 'bullish': True},   # 22
        {'close': 124, 'body': 0.7, 'bullish': True},   # 23
        {'close': 122, 'body': 0.5, 'bullish': False},  # 24
        {'close': 118, 'body': 4.0, 'bullish': False},  # 25
        {'close': 121, 'body': 2.8, 'bullish': True},   # 26
        {'close': 115, 'body': 4.5, 'bullish': False},  # 27
        {'close': 111, 'body': 4.0, 'bullish': False},  # 28
        {'close': 108, 'body': 3.8, 'bullish': False},  # 29
        {'close': 105, 'body': 3.5, 'bullish': False},  # 30
    ]
    
    builder = CandleBuilder()
    base_time = datetime(2024, 1, 1, 9, 0)
    candles = []
    
    for i, spec in enumerate(test_specs):
        timestamp = base_time + timedelta(minutes=i * 5)
        close = spec['close']
        body = spec['body']
        is_bullish = spec['bullish']
        
        if is_bullish:
            open_price = close - body
            high = close + 1.0
            low = open_price - 0.5
        else:
            open_price = close + body  
            high = open_price + 0.5
            low = close - 1.0
        
        candle = builder.reset().with_timestamp(timestamp).with_ohlc(
            open_price, high, low, close
        ).build()
        candles.append(candle)
    
    return candles


def analyze_sliding_window_behavior():
    """Test if the algorithm is designed for sliding window analysis"""
    print("🔍 SLIDING WINDOW TREND ANALYSIS INVESTIGATION")
    print("=" * 60)
    
    candles = create_31_candle_test_scenario()
    config = TrendAnalysisConfig()
    engine = TrendAnalysisEngine(config)
    
    # Full analysis
    full_result = engine.analyze_trends(candles)
    
    print(f"📊 FULL WINDOW ANALYSIS (candles 0-30):")
    print(f"  Total trends: {len(full_result.trends)}")
    print(f"  Active trends: {len(full_result.active_trends)}")
    print()
    
    # Test different time windows to see if trends are contextually relevant
    windows = [
        (0, 15),   # Early period - should show early uptrend
        (10, 25),  # Middle period - should show different trends  
        (15, 30),  # Late period - should show downtrend
        (5, 20),   # Custom window - should show overlapping trends
    ]
    
    print("🪟 SLIDING WINDOW TEST:")
    print("=" * 30)
    
    for start, end in windows:
        print(f"\n📋 WINDOW: Candles {start}-{end} ({end-start+1} candles)")
        print("-" * 40)
        
        window_candles = candles[start:end+1]
        window_result = engine.analyze_trends(window_candles, analysis_start=0, analysis_end=len(window_candles)-1)
        
        print(f"  Trends in this window: {len(window_result.trends)}")
        print(f"  Active trends: {len(window_result.active_trends)}")
        
        # Show trend breakdown
        uptrends = [t for t in window_result.trends if t.direction == TrendDirection.UP]
        downtrends = [t for t in window_result.trends if t.direction == TrendDirection.DOWN] 
        sideways = [t for t in window_result.trends if t.direction == TrendDirection.SIDEWAYS]
        
        print(f"    📈 Uptrends: {len(uptrends)}")
        print(f"    📉 Downtrends: {len(downtrends)}")
        print(f"    ➡️  Sideways: {len(sideways)}")
        
        # Show first few trends with their spans
        if window_result.trends:
            print(f"  Sample trends:")
            for i, trend in enumerate(window_result.trends[:3]):
                adjusted_start = start + (trend.start_index or 0)  # Adjust back to original indices
                adjusted_end = start + (trend.end_index or len(window_candles)-1) 
                print(f"    {i+1}. {trend.direction.value} trend: original candles {adjusted_start}-{adjusted_end}")


def analyze_trend_overlaps():
    """Analyze how trends overlap in the full dataset"""
    print(f"\n🔄 TREND OVERLAP ANALYSIS:")
    print("=" * 30)
    
    candles = create_31_candle_test_scenario()
    config = TrendAnalysisConfig()
    engine = TrendAnalysisEngine(config)
    
    result = engine.analyze_trends(candles)
    
    # Group trends by time periods
    time_periods = {
        "Early (0-10)": [],
        "Middle (10-20)": [], 
        "Late (20-30)": [],
        "Spanning multiple": []
    }
    
    for trend in result.trends:
        start = trend.start_index or 0
        end = trend.end_index or 30
        
        if end <= 10:
            time_periods["Early (0-10)"].append(trend)
        elif start >= 20:
            time_periods["Late (20-30)"].append(trend)
        elif start >= 10 and end <= 20:
            time_periods["Middle (10-20)"].append(trend)
        else:
            time_periods["Spanning multiple"].append(trend)
    
    for period, trends in time_periods.items():
        print(f"\n{period}: {len(trends)} trends")
        if trends:
            for trend in trends[:3]:  # Show first 3
                span = f"{trend.start_index or 0}-{trend.end_index or 30}"
                print(f"  - {trend.direction.value} trend (candles {span})")


def test_visualization_relevance():
    """Test if trends are relevant for different visualization windows"""
    print(f"\n🖼️  VISUALIZATION RELEVANCE TEST:")
    print("=" * 40)
    
    candles = create_31_candle_test_scenario()
    config = TrendAnalysisConfig()
    engine = TrendAnalysisEngine(config)
    
    # Simulate different chart zoom levels
    zoom_scenarios = [
        ("Full Chart", 0, 30),
        ("Uptrend Focus", 3, 19),     # Focus on main uptrend period
        ("Consolidation Focus", 15, 25), # Focus on sideways period
        ("Downtrend Focus", 20, 30),     # Focus on downtrend period
    ]
    
    print("Testing chart zoom scenarios:")
    
    for scenario_name, start, end in zoom_scenarios:
        print(f"\n📊 {scenario_name} (candles {start}-{end}):")
        
        # Analyze just this window
        window_candles = candles[start:end+1]
        window_result = engine.analyze_trends(window_candles)
        
        relevant_trends = []
        for trend in window_result.trends:
            # Check if trend is actually active/relevant in this window
            trend_start = trend.start_index or 0
            trend_end = trend.end_index or len(window_candles)-1
            
            # Consider a trend relevant if it spans a meaningful portion of the window
            window_size = end - start + 1
            trend_span = trend_end - trend_start + 1
            relevance_ratio = trend_span / window_size
            
            if relevance_ratio >= 0.2:  # At least 20% of the window
                relevant_trends.append((trend, relevance_ratio))
        
        print(f"  Total trends detected: {len(window_result.trends)}")
        print(f"  Relevant trends (>20% window): {len(relevant_trends)}")
        
        for trend, ratio in relevant_trends[:3]:  # Show top 3
            print(f"    - {trend.direction.value} ({ratio:.1%} of window)")


def investigate_algorithm_design():
    """Investigate if the algorithm is designed for this sliding window approach"""
    print(f"\n🔬 ALGORITHM DESIGN INVESTIGATION:")
    print("=" * 40)
    
    print("Key Questions:")
    print("1. Does the algorithm detect trends for sub-windows within the data?")
    print("2. Are trends contextually relevant to their time periods?")
    print("3. Is the high trend count (53) actually correct behavior?")
    print("4. Does this support flexible chart visualization?")
    
    candles = create_31_candle_test_scenario()
    config = TrendAnalysisConfig()
    engine = TrendAnalysisEngine(config)
    
    # Test: Does the algorithm find different trends when given different starting points?
    print(f"\n🧪 SUB-WINDOW TREND DETECTION TEST:")
    
    # Test overlapping windows
    windows = [(0, 20), (5, 25), (10, 30)]
    
    for i, (start, end) in enumerate(windows):
        window_candles = candles[start:end+1]
        result = engine.analyze_trends(window_candles)
        
        print(f"  Window {i+1} (candles {start}-{end}): {len(result.trends)} trends")
        
        # Check for unique patterns in each window
        uptrends = len([t for t in result.trends if t.direction == TrendDirection.UP])
        downtrends = len([t for t in result.trends if t.direction == TrendDirection.DOWN])
        
        print(f"    Up: {uptrends}, Down: {downtrends}")
    
    print(f"\n💡 HYPOTHESIS TEST RESULTS:")
    if len(result.trends) > 10:
        print("✅ HIGH TREND COUNT suggests algorithm IS designed for multi-timeframe analysis")
        print("✅ This supports flexible visualization with sliding/scaling")
        print("✅ 53 trends might be CORRECT behavior for this approach")
    else:
        print("❌ Low trend count suggests single-timeframe approach")
        print("❌ May not support flexible visualization")


if __name__ == "__main__":
    analyze_sliding_window_behavior()
    analyze_trend_overlaps()
    test_visualization_relevance()
    investigate_algorithm_design()
    
    print(f"\n🎯 CONCLUSION:")
    print("=" * 20)
    print("If the algorithm IS designed for sliding window analysis:")
    print("  ✅ 53 trends could be correct")
    print("  ✅ Multiple overlapping trends make sense")
    print("  ✅ Supports flexible chart visualization")
    print()
    print("If it's NOT designed for this:")
    print("  ❌ 53 trends indicates a bug")
    print("  ❌ Should have 3-4 main trends")
    print("  ❌ Need to fix trend management logic")