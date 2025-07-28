"""
Debug Script: Analyze what happens at Candle 20
=====================================================
This script will step through the trend analysis at candle 20 
to identify why the expected transitions aren't happening.
"""

def debug_candle_20_transition(test_candles, engine):
    """Comprehensive debug analysis of candle 20 transition"""
    
    print("🔍 DEBUGGING CANDLE 20 TRANSITION")
    print("=" * 60)
    
    # Step 1: Analyze up to candle 19 (before the issue)
    print("\n📊 STEP 1: Analysis up to Candle 19")
    result_19 = engine.analyze_trends(test_candles, analysis_start=0, analysis_end=19)
    
    print(f"Active trends at candle 19: {len(result_19.active_trends)}")
    for trend in result_19.active_trends:
        print(f"  - {trend.direction.value.upper()} trend T{trend.trend_id} "
              f"(start: {trend.start_index}, controlling: {trend.controlling_swing.candle_index if trend.controlling_swing else 'None'})")
    
    # Step 2: Analyze up to candle 20 (the problematic candle)
    print("\n📊 STEP 2: Analysis up to Candle 20")
    result_20 = engine.analyze_trends(test_candles, analysis_start=0, analysis_end=20)
    
    print(f"Active trends at candle 20: {len(result_20.active_trends)}")
    for trend in result_20.active_trends:
        print(f"  - {trend.direction.value.upper()} trend T{trend.trend_id} "
              f"(start: {trend.start_index}, controlling: {trend.controlling_swing.candle_index if trend.controlling_swing else 'None'})")
    
    print(f"Terminated trends: {len([t for t in result_20.trends if not t.is_active])}")
    for trend in [t for t in result_20.trends if not t.is_active]:
        print(f"  - {trend.direction.value.upper()} trend T{trend.trend_id} "
              f"(ended at: {trend.end_index})")
    
    # Step 3: Check swing points around candle 20
    print("\n🎯 STEP 3: Swing Analysis around Candle 20")
    relevant_swings = [s for s in result_20.swings if 15 <= s.candle_index <= 20]
    print(f"Swings in range 15-20:")
    for swing in relevant_swings:
        print(f"  - Candle {swing.candle_index}: {swing.swing_type.value.upper()} at {swing.price:.1f}")
    
    # Step 4: Test termination logic manually
    print("\n🔧 STEP 4: Manual Termination Check")
    candle_20 = test_candles[20]
    print(f"Candle 20: high={candle_20.high:.1f}, low={candle_20.low:.1f}, close={candle_20.close:.1f}")
    
    # Check each active trend from candle 19 for termination at candle 20
    from trend_business_logic import TrendTerminationDetector
    termination_detector = TrendTerminationDetector(engine.config)
    
    for trend in result_19.active_trends:
        should_terminate, genesis_swing = termination_detector.check_trend_termination(
            trend, candle_20, 20
        )
        print(f"Trend T{trend.trend_id} ({trend.direction.value.upper()}): "
              f"should_terminate={should_terminate}")
        if trend.controlling_swing:
            print(f"  - Controlling swing: candle {trend.controlling_swing.candle_index} at {trend.controlling_swing.price:.1f}")
            if trend.direction.name == 'DOWN':
                print(f"  - Termination check: candle_20.high ({candle_20.high:.1f}) > controlling ({trend.controlling_swing.price:.1f})? {candle_20.high > trend.controlling_swing.price}")
    
    # Step 5: Test sideways detection manually
    print("\n🔄 STEP 5: Manual Sideways Detection")
    swings_20 = [s for s in result_20.swings if s.candle_index <= 20]
    
    # Test sideways detection for period 16-20
    pattern_matcher = engine.pattern_matcher
    sideways_patterns = pattern_matcher.find_sideways_patterns(
        test_candles, swings_20, 16, 20
    )
    print(f"Sideways patterns detected in range 16-20: {len(sideways_patterns)}")
    
    if sideways_patterns:
        for i, pattern in enumerate(sideways_patterns):
            print(f"  Pattern {i+1}: start={pattern.start_index}, end={pattern.end_index}")
    else:
        # Debug why no sideways patterns
        print("  ❌ No sideways patterns detected. Checking requirements...")
        
        # Check period length
        period_length = 20 - 16 + 1
        print(f"  - Period length: {period_length} (need >= 5)")
        
        # Check if period qualifies as sideways
        period_candles = test_candles[16:21]  # candles 16-20
        if period_candles:
            # Test narrow price range
            range_high = max(c.high for c in period_candles)
            range_low = min(c.low for c in period_candles)
            range_size = range_high - range_low
            avg_price = (range_high + range_low) / 2
            range_pct = range_size / avg_price if avg_price > 0 else 0
            threshold = engine.config.sideways_range_threshold * 0.7
            
            print(f"  - Price range check: {range_pct:.3f} <= {threshold:.3f}? {range_pct <= threshold}")
            
            # Test low momentum
            momentum_candles = period_candles[-3:] if len(period_candles) >= 3 else period_candles
            avg_momentum = sum(c.body_to_wick_ratio for c in momentum_candles) / len(momentum_candles)
            momentum_threshold = engine.config.moveout_threshold * 0.6
            
            print(f"  - Momentum check: {avg_momentum:.3f} < {momentum_threshold:.3f}? {avg_momentum < momentum_threshold}")
            
            # Check swing count
            period_swings = [s for s in swings_20 if 16 <= s.candle_index <= 20]
            print(f"  - Swing count: {len(period_swings)} (need >= 3)")
    
    # Step 6: Check pattern detection priority logic
    print("\n⚖️ STEP 6: Pattern Detection Priority Analysis")
    
    # Simulate what happens in _detect_new_patterns at candle 20
    current_swings = [s for s in result_20.swings if s.candle_index <= 20]
    
    up_patterns = pattern_matcher.find_uptrend_patterns(current_swings)
    down_patterns = pattern_matcher.find_downtrend_patterns(current_swings)
    
    print(f"Uptrend patterns at candle 20: {len(up_patterns)}")
    print(f"Downtrend patterns at candle 20: {len(down_patterns)}")
    
    directional_patterns_found = len(up_patterns) + len(down_patterns)
    print(f"Total directional patterns: {directional_patterns_found}")
    
    if directional_patterns_found > 0:
        print("  ⚠️ ISSUE: Directional patterns found - this prevents sideways detection!")
        print("  The current logic: if not new_patterns and candle_index >= 8:")
        print("  This means sideways is only checked if NO directional patterns exist.")
    else:
        print("  ✅ No directional patterns - sideways detection would proceed")
    
    # Step 7: Summary and Recommendations
    print("\n📋 STEP 7: Summary and Recommendations")
    print("-" * 40)
    
    # Compare trends between candle 19 and 20
    trends_19_active = set(t.trend_id for t in result_19.active_trends)
    trends_20_active = set(t.trend_id for t in result_20.active_trends)
    
    terminated_trend_ids = trends_19_active - trends_20_active
    new_trend_ids = trends_20_active - trends_19_active
    
    print(f"Trends terminated at candle 20: {terminated_trend_ids}")
    print(f"New trends created at candle 20: {new_trend_ids}")
    
    # Check for expected sideways trend
    sideways_trends_20 = [t for t in result_20.active_trends if t.direction.name == 'SIDEWAYS']
    print(f"Sideways trends at candle 20: {len(sideways_trends_20)}")
    
    if len(sideways_trends_20) == 0:
        print("❌ ISSUE CONFIRMED: No sideways trend detected at candle 20")
        print("🔧 LIKELY CAUSES:")
        print("   1. Sideways detection is blocked by directional pattern priority")
        print("   2. Downtrend not properly terminated")
        print("   3. Sideways period doesn't meet algorithm requirements")
    else:
        print("✅ Sideways trend found - issue may be elsewhere")

# Run the debug analysis
debug_candle_20_transition(test_candles, engine)