#!/usr/bin/env python3
"""
Simple debug script to understand swing detection behavior
Run this independently to see what's happening with the algorithm
"""

import sys
from datetime import datetime, timedelta
from typing import List

try:
    from trend_business_logic import SwingDetector, BasicSwingDetectionStrategy
    from utilities import CandleBuilder
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the correct directory with all modules available.")
    sys.exit(1)


def create_simple_test_data() -> List:
    """Create simple test data to debug swing detection"""
    # Simple case that should definitely work
    test_specs = [
        {'close': 100, 'body': 1.0, 'bullish': True},   # 0: Neutral
        {'close': 95, 'body': 2.0, 'bullish': False},   # 1: Should be swing LOW
        {'close': 105, 'body': 3.0, 'bullish': True},   # 2: Should be swing HIGH
        {'close': 98, 'body': 2.0, 'bullish': False},   # 3: Should be swing LOW
        {'close': 102, 'body': 1.0, 'bullish': True},   # 4: Neutral
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


def debug_simple_case():
    """Debug swing detection on simple test case"""
    print("🔍 DEBUGGING SIMPLE SWING DETECTION")
    print("=" * 50)
    
    candles = create_simple_test_data()
    
    print("Test candles:")
    print("Index | Open   | High   | Low    | Close  | Expected")
    print("-" * 55)
    
    for i, candle in enumerate(candles):
        expected = ""
        if i == 1:
            expected = "LOW"
        elif i == 2:
            expected = "HIGH"
        elif i == 3:
            expected = "LOW"
        
        print(f"{i:5} | {candle.open:6.1f} | {candle.high:6.1f} | {candle.low:6.1f} | {candle.close:6.1f} | {expected}")
    
    # Test swing detection
    strategy = BasicSwingDetectionStrategy(lookback_period=1)
    detector = SwingDetector(strategy)
    
    swings = detector.detect_swings(candles)
    
    print(f"\nDetected {len(swings)} swings:")
    for swing in swings:
        print(f"  Candle {swing.candle_index}: {swing.swing_type.value.upper()} at {swing.price:.1f}")
    
    swing_indices = [s.candle_index for s in swings]
    print(f"\nSwing indices: {swing_indices}")
    
    # Manual check of each candle
    print("\nManual extrema check:")
    for i in range(1, len(candles) - 1):  # Only interior candles
        swing = strategy._check_swing_at_index(candles, i)
        current = candles[i]
        prev = candles[i-1]
        next_candle = candles[i+1]
        
        if swing:
            print(f"  Candle {i}: ✅ {swing.swing_type.value.upper()} at {swing.price:.1f}")
        else:
            high_vs_prev = current.high > prev.high
            high_vs_next = current.high > next_candle.high
            low_vs_prev = current.low < prev.low
            low_vs_next = current.low < next_candle.low
            
            print(f"  Candle {i}: ❌ Not swing")
            print(f"    High {current.high:.1f}: > prev({prev.high:.1f})? {high_vs_prev}, > next({next_candle.high:.1f})? {high_vs_next}")
            print(f"    Low {current.low:.1f}: < prev({prev.low:.1f})? {low_vs_prev}, < next({next_candle.low:.1f})? {low_vs_next}")


def create_31_candle_debug_data() -> List:
    """Create the 31-candle test scenario for debugging"""
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


def debug_31_candle_case():
    """Debug swing detection on first 15 candles of the 31-candle scenario"""
    print("\n🔍 DEBUGGING 31-CANDLE SCENARIO (First 15 candles)")
    print("=" * 60)
    
    candles = create_31_candle_debug_data()
    
    print("First 15 candles OHLC:")
    print("Index | Open   | High   | Low    | Close  | High>Neighbors? | Low<Neighbors?")
    print("-" * 75)
    
    for i, candle in enumerate(candles):
        # Check if this could be a swing
        high_check = ""
        low_check = ""
        
        if 0 < i < len(candles) - 1:
            prev = candles[i-1]
            next_candle = candles[i+1]
            
            high_vs_prev = candle.high > prev.high
            high_vs_next = candle.high > next_candle.high
            high_check = f"{high_vs_prev} & {high_vs_next} = {high_vs_prev and high_vs_next}"
            
            low_vs_prev = candle.low < prev.low
            low_vs_next = candle.low < next_candle.low
            low_check = f"{low_vs_prev} & {low_vs_next} = {low_vs_prev and low_vs_next}"
        
        print(f"{i:5} | {candle.open:6.1f} | {candle.high:6.1f} | {candle.low:6.1f} | {candle.close:6.1f} | {high_check:15} | {low_check}")
    
    # Test swing detection
    strategy = BasicSwingDetectionStrategy(lookback_period=1)
    detector = SwingDetector(strategy)
    
    swings = detector.detect_swings(candles)
    
    print(f"\nDetected {len(swings)} swings:")
    for swing in swings:
        print(f"  Candle {swing.candle_index}: {swing.swing_type.value.upper()} at {swing.price:.1f}")
    
    swing_indices = [s.candle_index for s in swings]
    print(f"\nSwing indices: {swing_indices}")
    print(f"Expected some of: [1, 2, 3, 5, 7, 11, 12, 14]")


def main():
    """Run debug analysis"""
    print("🚀 SWING DETECTION DEBUG ANALYSIS")
    print("=" * 50)
    
    try:
        # Test 1: Simple case
        debug_simple_case()
        
        # Test 2: First 15 candles of 31-candle scenario
        debug_31_candle_case()
        
        print("\n✅ Debug analysis complete!")
        print("Use this information to understand why certain swings are/aren't detected.")
        
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
