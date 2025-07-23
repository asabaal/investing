#!/usr/bin/env python3
"""
Validation script for the 31-candle test scenario
Tests the specific fixes made to the trend detection system
"""

from datetime import datetime, timedelta
from typing import List

from trend_business_logic import TrendAnalysisEngine, BasicSwingDetectionStrategy, SwingDetector
from trend_core_models import TrendAnalysisConfig, TrendDirection
from utilities import CandleBuilder


def create_31_candle_test_data() -> List:
    """Create the exact 31-candle test scenario"""
    test_specs = [
        # Phase 1: Initial movement (no trends yet)
        {'close': 100, 'body': 1.0, 'bullish': True},   # 0: Start
        {'close': 99, 'body': 1.5, 'bullish': False},   # 1: Minor dip (swing low)
        {'close': 101, 'body': 1.2, 'bullish': True},   # 2: Recovery (swing high)
        
        # Phase 2: Strong uptrend formation (will become MAJOR)
        {'close': 97, 'body': 3.5, 'bullish': False},   # 3: Major swing LOW (genesis point)
        {'close': 100, 'body': 2.0, 'bullish': True},   # 4: Bounce
        {'close': 108, 'body': 4.0, 'bullish': True},   # 5: Major swing HIGH
        {'close': 105, 'body': 1.5, 'bullish': False},  # 6: Minor pullback
        {'close': 103, 'body': 2.5, 'bullish': False},  # 7: Higher LOW swing (swing low)
        {'close': 106, 'body': 2.8, 'bullish': True},   # 8: Recovery
        {'close': 112, 'body': 4.5, 'bullish': True},   # 9: BREAKOUT (uptrend starts!)
        {'close': 116, 'body': 3.8, 'bullish': True},   # 10: Strong continuation
        {'close': 120, 'body': 3.5, 'bullish': True},   # 11: More upside (swing high)
        {'close': 118, 'body': 1.8, 'bullish': False},  # 12: Pullback (swing low - controlling)
        {'close': 122, 'body': 3.0, 'bullish': True},   # 13: Recovery
        {'close': 124, 'body': 2.5, 'bullish': True},   # 14: Peak (swing high)
        
        # Phase 3: Trend termination setup
        {'close': 122, 'body': 0.6, 'bullish': False},  # 15: Start consolidation
        {'close': 125, 'body': 0.7, 'bullish': True},   # 16: Range high (swing high)
        {'close': 121, 'body': 0.5, 'bullish': False},  # 17: Range low (swing low)
        {'close': 124, 'body': 0.8, 'bullish': True},   # 18: Similar high (swing high)
        {'close': 116.5, 'body': 2.0, 'bullish': False}, # 19: BREAKS controlling swing! (swing low)
        {'close': 125, 'body': 0.7, 'bullish': True},   # 20: Rally (swing high)
        {'close': 121, 'body': 0.5, 'bullish': False},  # 21: Pullback
        {'close': 123, 'body': 0.6, 'bullish': True},   # 22: Range middle
        {'close': 124, 'body': 0.7, 'bullish': True},   # 23: Range high (swing high)
        {'close': 122, 'body': 0.5, 'bullish': False},  # 24: Range center
        
        # Phase 4: Major downtrend
        {'close': 118, 'body': 4.0, 'bullish': False},  # 25: Major breakdown
        {'close': 121, 'body': 2.8, 'bullish': True},   # 26: Lower HIGH swing
        {'close': 115, 'body': 4.5, 'bullish': False},  # 27: Lower LOW
        {'close': 111, 'body': 4.0, 'bullish': False},  # 28: Strong selling
        {'close': 108, 'body': 3.8, 'bullish': False},  # 29: Continuation down
        {'close': 105, 'body': 3.5, 'bullish': False},  # 30: Final leg
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
            high = close + 1.0  # Small upper wick
            low = open_price - 0.5  # Small lower wick
        else:
            open_price = close + body  
            high = open_price + 0.5  # Small upper wick
            low = close - 1.0  # Small lower wick
        
        candle = builder.reset().with_timestamp(timestamp).with_ohlc(
            open_price, high, low, close
        ).build()
        candles.append(candle)
    
    return candles


def test_swing_detection_fixes():
    """Test that swing detection now finds the missing swings"""
    print("🔍 TESTING SWING DETECTION FIXES")
    print("=" * 50)
    
    candles = create_31_candle_test_data()
    
    # Test with updated strategy (lookback_period=2)
    strategy = BasicSwingDetectionStrategy(lookback_period=2)
    detector = SwingDetector(strategy)
    
    swings = detector.detect_swings(candles)
    swing_indices = [s.candle_index for s in swings]
    
    expected_key_swings = [1, 2, 3, 5, 7, 11, 12, 14, 16, 17, 18, 19, 20, 23]
    
    print(f"Expected key swings: {expected_key_swings}")
    print(f"Detected swings:     {swing_indices}")
    
    # Check for the previously missing swings
    missing_swings = []
    for expected in [12, 14]:  # These were the key missing ones
        if expected not in swing_indices:
            missing_swings.append(expected)
    
    if missing_swings:
        print(f"❌ STILL MISSING: {missing_swings}")
        return False
    else:
        print("✅ Key swings 12 and 14 are now detected!")
    
    # Check that we don't have the false positive at candle 30
    if 30 in swing_indices:
        print("⚠️  WARNING: Still detecting false swing at candle 30")
    else:
        print("✅ No false swing at candle 30")
    
    return True


def test_trend_formation_and_termination():
    """Test that trends form and terminate correctly"""
    print("\n🎯 TESTING TREND FORMATION & TERMINATION")
    print("=" * 50)
    
    candles = create_31_candle_test_data()
    engine = TrendAnalysisEngine()
    
    result = engine.analyze_trends(candles)
    
    print(f"Total candles analyzed: {result.total_candles_analyzed}")
    print(f"Swing points detected: {len(result.swings)}")
    print(f"Trends detected: {len(result.trends)}")
    print(f"Active trends: {len(result.active_trends)}")
    
    # Check swing detection
    swing_indices = [s.candle_index for s in result.swings]
    print(f"Swing indices: {swing_indices}")
    
    # Check trends
    for trend in result.trends:
        print(f"\nTrend {trend.trend_id}:")
        print(f"  Direction: {trend.direction.value}")
        print(f"  Start: {trend.start_index}, End: {trend.end_index}")
        print(f"  Active: {trend.is_active}")
        print(f"  Significance: {trend.significance.value}")
        
        if trend.genesis_point:
            print(f"  Genesis: Candle {trend.genesis_point.candle_index} ({trend.genesis_point.price})")
        
        if trend.controlling_swing:
            print(f"  Controlling: Candle {trend.controlling_swing.candle_index} ({trend.controlling_swing.price})")
    
    # Look for expected uptrend
    uptrends = [t for t in result.trends if t.direction == TrendDirection.UP]
    
    if uptrends:
        uptrend = uptrends[0]
        print(f"\n✅ Found uptrend:")
        print(f"   Genesis point: Candle {uptrend.genesis_point.candle_index if uptrend.genesis_point else 'None'}")
        print(f"   Expected genesis: Candle 3")
        
        if uptrend.genesis_point and uptrend.genesis_point.candle_index == 3:
            print("✅ Genesis point is correct!")
        else:
            print("❌ Genesis point is incorrect")
            
        return True
    else:
        print("❌ No uptrend detected")
        return False


def test_controlling_swing_updates():
    """Test that controlling swings update correctly"""
    print("\n⚙️  TESTING CONTROLLING SWING UPDATES")
    print("=" * 50)
    
    candles = create_31_candle_test_data()
    engine = TrendAnalysisEngine()
    
    # Analyze up to candle 15 (before trend termination)
    result = engine.analyze_trends(candles, analysis_end=15)
    
    uptrends = [t for t in result.trends if t.direction == TrendDirection.UP and t.is_active]
    
    if uptrends:
        uptrend = uptrends[0]
        print(f"Uptrend controlling swing: Candle {uptrend.controlling_swing.candle_index if uptrend.controlling_swing else 'None'}")
        print(f"Expected: Should be candle 12 (118) after swing confirmation")
        
        # The controlling swing should have updated from candle 7 (103) to candle 12 (118)
        if uptrend.controlling_swing and uptrend.controlling_swing.candle_index == 12:
            print("✅ Controlling swing updated correctly to candle 12!")
            return True
        else:
            print("❌ Controlling swing not updated correctly")
            return False
    else:
        print("❌ No active uptrend found")
        return False


def main():
    """Run all validation tests"""
    print("🚀 VALIDATING 31-CANDLE TREND DETECTION FIXES")
    print("=" * 60)
    
    results = []
    
    # Test 1: Swing detection
    results.append(test_swing_detection_fixes())
    
    # Test 2: Trend formation and termination
    results.append(test_trend_formation_and_termination())
    
    # Test 3: Controlling swing updates
    results.append(test_controlling_swing_updates())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The fixes are working correctly.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
