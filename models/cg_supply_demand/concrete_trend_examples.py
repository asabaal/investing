"""
CONCRETE EXAMPLES: What trends should be detected in the 31-candle scenario
=========================================================================
Let's enumerate the specific trends that should be detected when processing each candle as a window start.
"""

def enumerate_expected_trends():
    """
    Write down the specific trends that should be detected in our 31-candle scenario.
    Using swings: [1, 2, 3, 5, 7, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26]
    """
    
    print("📋 CONCRETE TREND EXAMPLES FOR 31-CANDLE SCENARIO")
    print("=" * 60)
    
    # Our swing points (from actual detection)
    swings = {
        1: ('LOW', 99),      2: ('HIGH', 101),    3: ('LOW', 97),
        5: ('HIGH', 108),    7: ('LOW', 103),     11: ('HIGH', 120),
        12: ('LOW', 118),    14: ('HIGH', 124),   15: ('LOW', 122),
        16: ('HIGH', 125),   17: ('LOW', 121),    18: ('HIGH', 124),
        19: ('LOW', 116.5),  20: ('HIGH', 125),   21: ('LOW', 121),
        23: ('HIGH', 124),   25: ('LOW', 118),    26: ('HIGH', 121)
    }
    
    # Candle close prices (for breakout validation)
    closes = {
        0: 100, 1: 99, 2: 101, 3: 97, 4: 100, 5: 108, 6: 105, 7: 103, 8: 106, 9: 112,
        10: 116, 11: 120, 12: 118, 13: 122, 14: 124, 15: 122, 16: 125, 17: 121, 18: 124,
        19: 116.5, 20: 125, 21: 121, 22: 123, 23: 124, 24: 122, 25: 118, 26: 121,
        27: 115, 28: 111, 29: 108, 30: 105
    }
    
    expected_trends = []
    
    print("🔍 STEP-BY-STEP TREND DETECTION:")
    print("-" * 40)
    
    # Process each candle as window start
    for window_start in range(31):
        print(f"\n📊 PROCESSING CANDLE {window_start} AS WINDOW START:")
        
        # Get available swings from this window start
        available_swings = [(idx, swing_type, price) for idx, (swing_type, price) in swings.items() if idx >= window_start]
        
        if len(available_swings) < 3:
            print(f"   ⏭️  Not enough swings from candle {window_start} ({len(available_swings)} swings)")
            continue
            
        print(f"   Available swings: {[(idx, swing_type) for idx, swing_type, price in available_swings[:6]]}")  # Show first 6
        
        # Look for L-H-L patterns (UPTRENDS)
        uptrend_patterns = find_lhl_patterns(available_swings)
        for pattern in uptrend_patterns[:2]:  # Show first 2
            l1_idx, h_idx, l2_idx = pattern
            print(f"   📈 L-H-L pattern found: {l1_idx}-{h_idx}-{l2_idx}")
            
            # Check if there's a breakout candle
            h_price = swings[h_idx][1]
            breakout_candle = find_breakout_candle(closes, h_idx + 1, h_price, 'up')
            
            if breakout_candle:
                trend_id = f"UP_{window_start}_{l1_idx}_{h_idx}_{l2_idx}_{breakout_candle}"
                expected_trends.append({
                    'id': trend_id,
                    'type': 'UPTREND',
                    'window_start': window_start,
                    'pattern': f"{l1_idx}-{h_idx}-{l2_idx}",
                    'breakout': breakout_candle,
                    'genesis': l1_idx
                })
                print(f"   ✅ UPTREND: Pattern {l1_idx}-{h_idx}-{l2_idx}, breakout at candle {breakout_candle}")
        
        # Look for H-L-H patterns (DOWNTRENDS)  
        downtrend_patterns = find_hlh_patterns(available_swings)
        for pattern in downtrend_patterns[:2]:  # Show first 2
            h1_idx, l_idx, h2_idx = pattern
            print(f"   📉 H-L-H pattern found: {h1_idx}-{l_idx}-{h2_idx}")
            
            # Check if there's a breakout candle
            l_price = swings[l_idx][1] 
            breakout_candle = find_breakout_candle(closes, l_idx + 1, l_price, 'down')
            
            if breakout_candle:
                trend_id = f"DOWN_{window_start}_{h1_idx}_{l_idx}_{h2_idx}_{breakout_candle}"
                expected_trends.append({
                    'id': trend_id,
                    'type': 'DOWNTREND', 
                    'window_start': window_start,
                    'pattern': f"{h1_idx}-{l_idx}-{h2_idx}",
                    'breakout': breakout_candle,
                    'genesis': h1_idx
                })
                print(f"   ✅ DOWNTREND: Pattern {h1_idx}-{l_idx}-{h2_idx}, breakout at candle {breakout_candle}")
        
        # Limit output for readability
        if window_start >= 5:  # Show first 6 windows in detail
            print(f"   ... (continuing for remaining candles)")
            break
    
    print(f"\n📊 SUMMARY OF EXPECTED TRENDS:")
    print("-" * 30)
    print(f"Total expected trends: {len(expected_trends)}")
    
    uptrends = [t for t in expected_trends if t['type'] == 'UPTREND']
    downtrends = [t for t in expected_trends if t['type'] == 'DOWNTREND']
    
    print(f"Uptrends: {len(uptrends)}")
    print(f"Downtrends: {len(downtrends)}")
    
    print(f"\n🎯 KEY UPTREND EXAMPLES:")
    for trend in uptrends[:5]:  # Show first 5
        print(f"  - {trend['pattern']} → breakout at {trend['breakout']} (window {trend['window_start']})")
    
    print(f"\n🎯 KEY DOWNTREND EXAMPLES:")  
    for trend in downtrends[:5]:  # Show first 5
        print(f"  - {trend['pattern']} → breakout at {trend['breakout']} (window {trend['window_start']})")
    
    return expected_trends


def find_lhl_patterns(swings):
    """Find L-H-L patterns from available swings"""
    patterns = []
    
    for i in range(len(swings)):
        if swings[i][1] == 'LOW':  # Found potential first low
            for j in range(i + 1, len(swings)):
                if swings[j][1] == 'HIGH' and swings[j][2] > swings[i][2]:  # Higher high
                    for k in range(j + 1, len(swings)):
                        if (swings[k][1] == 'LOW' and 
                            swings[k][2] > swings[i][2] and  # Higher low
                            swings[k][2] < swings[j][2]):    # Lower than high
                            patterns.append((swings[i][0], swings[j][0], swings[k][0]))
    
    return patterns


def find_hlh_patterns(swings):
    """Find H-L-H patterns from available swings"""
    patterns = []
    
    for i in range(len(swings)):
        if swings[i][1] == 'HIGH':  # Found potential first high
            for j in range(i + 1, len(swings)):
                if swings[j][1] == 'LOW' and swings[j][2] < swings[i][2]:  # Lower low
                    for k in range(j + 1, len(swings)):
                        if (swings[k][1] == 'HIGH' and 
                            swings[k][2] < swings[i][2] and  # Lower high
                            swings[k][2] > swings[j][2]):    # Higher than low
                            patterns.append((swings[i][0], swings[j][0], swings[k][0]))
    
    return patterns


def find_breakout_candle(closes, start_candle, trigger_price, direction):
    """Find the first candle that breaks out above/below trigger price"""
    for candle_idx in range(start_candle, min(start_candle + 10, len(closes))):  # Look ahead max 10 candles
        if candle_idx in closes:
            if direction == 'up' and closes[candle_idx] > trigger_price:
                return candle_idx
            elif direction == 'down' and closes[candle_idx] < trigger_price:
                return candle_idx
    return None


if __name__ == "__main__":
    expected_trends = enumerate_expected_trends()
    
    print(f"\n💡 WHAT THIS TELLS US:")
    print("=" * 30)
    print("1. Each window start can produce multiple valid trends")
    print("2. Early windows will have more uptrend opportunities") 
    print("3. Later windows will have more downtrend opportunities")
    print("4. Total trends depends on pattern + breakout combinations")
    print("5. This is why we might see many trends - each is valid for its context")
    
    if len(expected_trends) > 0:
        reasonable_range = f"{len(expected_trends)//2}-{len(expected_trends)}"
        print(f"\n🎯 REASONABLE TREND COUNT: {reasonable_range}")
        print(f"If we're getting 53 trends, that suggests the algorithm is working")
        print(f"but maybe detecting more patterns than this manual analysis shows.")
