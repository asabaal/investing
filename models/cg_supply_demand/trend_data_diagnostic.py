#!/usr/bin/env python3
"""
Trend Data Generation Diagnostic Script

This script analyzes the current data generation system to understand why
it's not producing data that triggers trend detection.

Run this script and feed the output to Claude for analysis.
"""

import sys
from datetime import datetime, timedelta
import numpy as np

# Import your trend detection components
from trend_core_models import *
from trend_business_logic import *
from trend_utilities import *
from utilities import Candle, CandleBuilder

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---")

def analyze_candle_data(candles, scenario_name):
    """Analyze the raw candle data for trend characteristics"""
    print_subsection(f"Raw Data Analysis: {scenario_name}")
    
    if not candles:
        print("ERROR: No candles provided!")
        return
    
    print(f"Number of candles: {len(candles)}")
    
    # Price statistics
    opens = [c.open for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    closes = [c.close for c in candles]
    
    print(f"Price range: {min(lows):.2f} to {max(highs):.2f}")
    print(f"Total range: {max(highs) - min(lows):.2f} ({((max(highs) - min(lows)) / min(lows) * 100):.1f}%)")
    print(f"Start close: {closes[0]:.2f}")
    print(f"End close: {closes[-1]:.2f}")
    print(f"Net change: {closes[-1] - closes[0]:.2f} ({((closes[-1] - closes[0]) / closes[0] * 100):.1f}%)")
    
    # Directional movement analysis
    up_candles = sum(1 for c in candles if c.close > c.open)
    down_candles = sum(1 for c in candles if c.close < c.open)
    doji_candles = len(candles) - up_candles - down_candles
    
    print(f"Candle direction: {up_candles} up, {down_candles} down, {doji_candles} doji")
    print(f"Directional bias: {(up_candles / len(candles) * 100):.1f}% bullish")
    
    # Movement strength analysis
    avg_body_size = np.mean([abs(c.close - c.open) for c in candles])
    avg_wick_total = np.mean([(c.high - max(c.open, c.close)) + (min(c.open, c.close) - c.low) for c in candles])
    
    print(f"Average body size: {avg_body_size:.2f}")
    print(f"Average total wick: {avg_wick_total:.2f}")
    print(f"Body-to-wick ratio: {avg_body_size / avg_wick_total:.2f}" if avg_wick_total > 0 else "Body-to-wick ratio: N/A")
    
    # Print first and last few candles for inspection
    print(f"\nFirst 3 candles:")
    for i, c in enumerate(candles[:3]):
        print(f"  {i}: O={c.open:.2f} H={c.high:.2f} L={c.low:.2f} C={c.close:.2f}")
    
    print(f"\nLast 3 candles:")
    for i, c in enumerate(candles[-3:], len(candles)-3):
        print(f"  {i}: O={c.open:.2f} H={c.high:.2f} L={c.low:.2f} C={c.close:.2f}")

def analyze_swing_detection(candles, swings, scenario_name):
    """Analyze swing point detection results"""
    print_subsection(f"Swing Detection Analysis: {scenario_name}")
    
    print(f"Number of swings detected: {len(swings)}")
    
    if not swings:
        print("ERROR: No swings detected!")
        print("This means swing detection failed completely.")
        print("Checking swing detection requirements...")
        
        # Check if we have minimum 3 candles
        if len(candles) < 3:
            print(f"  - ISSUE: Only {len(candles)} candles, need at least 3 for swing detection")
        else:
            print(f"  - OK: Have {len(candles)} candles (>= 3)")
        
        # Check for price movement between candles
        if len(candles) >= 3:
            close_prices = [c.close for c in candles]
            print(f"  - Close price sequence: {[f'{p:.2f}' for p in close_prices[:5]]}")
            
            # Check for local extrema manually
            potential_swings = []
            for i in range(1, len(candles) - 1):
                prev_close = candles[i-1].close
                curr_close = candles[i].close
                next_close = candles[i+1].close
                
                if curr_close > prev_close and curr_close > next_close:
                    potential_swings.append(f"High at candle {i} ({curr_close:.2f})")
                elif curr_close < prev_close and curr_close < next_close:
                    potential_swings.append(f"Low at candle {i} ({curr_close:.2f})")
            
            print(f"  - Manual swing check found: {len(potential_swings)} potential swings")
            for swing in potential_swings[:3]:  # Show first 3
                print(f"    {swing}")
        
        return
    
    # Analyze detected swings
    swing_highs = [s for s in swings if s.swing_type == SwingType.HIGH]
    swing_lows = [s for s in swings if s.swing_type == SwingType.LOW]
    
    print(f"Swing breakdown: {len(swing_highs)} highs, {len(swing_lows)} lows")
    
    print(f"\nSwing sequence:")
    for i, swing in enumerate(swings):
        print(f"  {i+1}. Candle {swing.candle_index}: {swing.swing_type.value.upper()} at {swing.price:.2f}")
    
    # Check swing alternation
    if len(swings) > 1:
        alternation_ok = True
        for i in range(1, len(swings)):
            if swings[i].swing_type == swings[i-1].swing_type:
                alternation_ok = False
                break
        print(f"Swing alternation: {'OK' if alternation_ok else 'BROKEN (consecutive same types)'}")
    
    # Analyze spacing between swings
    if len(swings) > 1:
        spacings = [swings[i+1].candle_index - swings[i].candle_index for i in range(len(swings)-1)]
        print(f"Swing spacing (candles): {spacings}")
        print(f"Average spacing: {np.mean(spacings):.1f} candles")

def analyze_pattern_recognition(swings, patterns, scenario_name):
    """Analyze pattern recognition results"""
    print_subsection(f"Pattern Recognition Analysis: {scenario_name}")
    
    print(f"Patterns detected: {len(patterns)}")
    
    if not patterns:
        print("ERROR: No patterns detected!")
        
        # Check if we have enough swings for patterns
        if len(swings) < 3:
            print(f"  - ISSUE: Only {len(swings)} swings, need at least 3 for L-H-L or H-L-H patterns")
        else:
            print(f"  - OK: Have {len(swings)} swings (>= 3)")
            
            # Manual pattern check
            print(f"  - Manual pattern analysis:")
            
            # Look for L-H-L patterns
            lhl_candidates = []
            for i in range(len(swings) - 2):
                s1, s2, s3 = swings[i], swings[i+1], swings[i+2]
                if (s1.swing_type == SwingType.LOW and 
                    s2.swing_type == SwingType.HIGH and 
                    s3.swing_type == SwingType.LOW):
                    
                    higher_low = s3.price > s1.price
                    price_diff = ((s3.price - s1.price) / s1.price * 100)
                    lhl_candidates.append(f"L-H-L at indices {s1.candle_index}-{s2.candle_index}-{s3.candle_index}, higher_low={higher_low} ({price_diff:+.1f}%)")
            
            # Look for H-L-H patterns  
            hlh_candidates = []
            for i in range(len(swings) - 2):
                s1, s2, s3 = swings[i], swings[i+1], swings[i+2]
                if (s1.swing_type == SwingType.HIGH and 
                    s2.swing_type == SwingType.LOW and 
                    s3.swing_type == SwingType.HIGH):
                    
                    lower_high = s3.price < s1.price
                    price_diff = ((s3.price - s1.price) / s1.price * 100)
                    hlh_candidates.append(f"H-L-H at indices {s1.candle_index}-{s2.candle_index}-{s3.candle_index}, lower_high={lower_high} ({price_diff:+.1f}%)")
            
            print(f"    L-H-L candidates: {len(lhl_candidates)}")
            for candidate in lhl_candidates:
                print(f"      {candidate}")
                
            print(f"    H-L-H candidates: {len(hlh_candidates)}")
            for candidate in hlh_candidates:
                print(f"      {candidate}")
        
        return
    
    # Analyze detected patterns
    for i, pattern in enumerate(patterns):
        print(f"\nPattern {i+1}:")
        print(f"  Type: {pattern.pattern_type.value.upper()}")
        print(f"  Candle range: {pattern.start_index} to {pattern.end_index}")
        print(f"  Formation swings: {len(pattern.formation_swings)}")
        
        swing_desc = []
        for swing in pattern.formation_swings:
            swing_desc.append(f"{swing.swing_type.value.upper()}@{swing.candle_index}({swing.price:.2f})")
        print(f"  Swing sequence: {' → '.join(swing_desc)}")

def analyze_trend_detection(trends, scenario_name):
    """Analyze trend detection results"""
    print_subsection(f"Trend Detection Analysis: {scenario_name}")
    
    print(f"Trends detected: {len(trends)}")
    
    if not trends:
        print("ERROR: No trends detected!")
        print("This means pattern validation or breakout confirmation failed.")
        return
    
    for i, trend in enumerate(trends):
        print(f"\nTrend {i+1}:")
        print(f"  ID: {trend.trend_id}")
        print(f"  Direction: {trend.direction.value.upper()}")
        print(f"  Significance: {trend.significance.value.upper()}")
        print(f"  Status: {'ACTIVE' if trend.is_active else 'TERMINATED'}")
        print(f"  Duration: {trend.duration} candles")
        print(f"  Price range: {trend.price_range:.2f}")
        print(f"  Start: candle {trend.start_index}")
        print(f"  End: candle {trend.end_index if trend.end_index else 'ACTIVE'}")
        
        if trend.controlling_swing:
            print(f"  Controlling swing: {trend.controlling_swing.swing_type.value.upper()} at {trend.controlling_swing.price:.2f} (candle {trend.controlling_swing.candle_index})")
        
        print(f"  Breakout confirmed: {trend.breakout_confirmed}")
        print(f"  Moveout confirmed: {trend.moveout_confirmed}")

def run_algorithm_diagnostics():
    """Run comprehensive diagnostics on the data generation system"""
    
    print_section("TREND DATA GENERATION DIAGNOSTIC REPORT")
    print(f"Generated at: {datetime.now()}")
    print(f"Purpose: Analyze why data generation isn't producing detectable trends")
    
    # Initialize the analysis engine
    config = TrendAnalysisConfig()
    engine = TrendAnalysisEngine(config)
    
    print_section("CONFIGURATION")
    print(f"Sideways range threshold: {config.sideways_range_threshold}")
    print(f"Moveout threshold: {config.moveout_threshold}")
    print(f"Major trend min duration: {config.major_trend_min_duration}")
    print(f"Major trend min range: {config.major_trend_min_range}")
    print(f"Swing lookback period: {config.swing_lookback_period}")
    
    # Test scenarios from the current data generation system
    test_scenarios = {
        "Simple Uptrend": TrendTestDataFactory.create_simple_uptrend_scenario(),
        "Simple Downtrend": TrendTestDataFactory.create_simple_downtrend_scenario(),
        "Simple Sideways": TrendTestDataFactory.create_simple_sideways_scenario(),
    }
    
    for scenario_name, candles in test_scenarios.items():
        print_section(f"SCENARIO: {scenario_name}")
        
        # Analyze raw data
        analyze_candle_data(candles, scenario_name)
        
        try:
            # Run trend analysis
            result = engine.analyze_trends(candles)
            
            # Analyze each stage
            analyze_swing_detection(candles, result.swings, scenario_name)
            analyze_pattern_recognition(result.swings, result.detected_patterns, scenario_name)
            analyze_trend_detection(result.trends, scenario_name)
            
            # Summary for this scenario
            print_subsection("SCENARIO SUMMARY")
            print(f"SUCCESS METRICS:")
            print(f"  - Candles: {len(candles)}")
            print(f"  - Swings: {len(result.swings)}")
            print(f"  - Patterns: {len(result.detected_patterns)}")
            print(f"  - Trends: {len(result.trends)}")
            print(f"  - Active trends: {len(result.active_trends)}")
            print(f"  - Major trends: {len(result.major_trends)}")
            
            if len(result.trends) == 0:
                print(f"  ❌ FAILED: No trends detected for '{scenario_name}' scenario")
            elif len(result.trends) > 0:
                directions = [t.direction.value for t in result.trends]
                print(f"  ✅ SUCCESS: Detected {len(result.trends)} trends: {directions}")
            
        except Exception as e:
            print(f"ERROR analyzing {scenario_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print_section("DIAGNOSTIC RECOMMENDATIONS")
    print("Based on the analysis above, here are the likely issues:")
    print("1. If no swings detected: Price movements too small or no local extrema")
    print("2. If swings but no patterns: L-H-L/H-L-H formations not valid")
    print("3. If patterns but no trends: Breakout validation failing")
    print("4. Check price ratios, movement sizes, and breakout strengths")
    print("\nRecommended next steps:")
    print("- Increase price movement magnitudes in data generation")
    print("- Ensure clear L-H-L and H-L-H geometric patterns")
    print("- Add stronger breakout moves above/below swing levels")
    print("- Test with manually constructed ideal scenarios")

if __name__ == "__main__":
    run_algorithm_diagnostics()
