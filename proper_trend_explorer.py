#!/usr/bin/env python3
"""
Proper Trend Explorer - Browse Individual Trends
Uses the WORKING swing point system and shows each trend with all details:
- Relevant swing points
- All candles between 
- Breakout candle
- Termination violation candle
"""

import plotly.graph_objects as go
import plotly.io as pio
from swing_point_detector import SwingPointDetector
from simple_trend_progression import find_first_breakout_candle
from trend_termination_progression import find_first_violation_candle
import json

# Dark theme
pio.templates.default = "plotly_dark"

def find_all_proper_trends(df, swing_points):
    """
    Find ALL trend formations using the working swing point system
    WITH PROPER VALIDATION - swing points must remain valid until breakout
    INCLUDES: Uptrends, Downtrends, and Sideways trends
    """
    
    formations = []
    
    # Convert swing points to easier format
    swing_list = [(sp['index'], sp['type'], sp['price']) for sp in swing_points]
    
    print(f"🔍 Scanning {len(swing_list)} swing points for trend formations...")
    print(f"   ✅ Adding validation: swing points must remain valid until breakout")
    print(f"   ✅ Adding sideways trend detection: balanced expansion within 5%")
    
    valid_count = 0
    invalid_count = 0
    sideways_count = 0
    
    # Look for SL → SH → SL patterns (uptrend formations)
    for i in range(len(swing_list) - 2):
        if (swing_list[i][1] == 'LOW' and 
            swing_list[i+1][1] == 'HIGH' and 
            swing_list[i+2][1] == 'LOW'):
            
            sl1_idx, sl1_type, sl1_price = swing_list[i]
            sh1_idx, sh1_type, sh1_price = swing_list[i+1]
            sl2_idx, sl2_type, sl2_price = swing_list[i+2]
            
            # Check for higher low (uptrend condition)
            if sl2_price > sl1_price:
                # Find breakout origin
                breakout_idx = find_first_breakout_candle(df, sl2_idx, sh1_price, 'above')
                
                if breakout_idx:
                    # VALIDATE: Check if swing points remain valid until breakout
                    is_valid, reason = validate_uptrend_formation(swing_list, i, breakout_idx)
                    
                    if is_valid:
                        formation = {
                            'type': 'UPTREND',
                            'sl1': {'idx': sl1_idx, 'price': sl1_price},
                            'sh1': {'idx': sh1_idx, 'price': sh1_price}, 
                            'sl2': {'idx': sl2_idx, 'price': sl2_price},
                            'breakout': {'idx': breakout_idx, 'price': df.iloc[breakout_idx]['high']},
                            'controlling_swing': {'idx': sl1_idx, 'price': sl1_price},
                            'formation_date': df.iloc[breakout_idx]['datetime'],
                            'setup_quality': 'VALID'
                        }
                        formations.append(formation)
                        valid_count += 1
                    else:
                        invalid_count += 1
                        # Optionally print first few invalid reasons for debugging
                        if invalid_count <= 5:
                            print(f"   ❌ Invalid uptrend SL1=${sl1_price:.2f}→SH1=${sh1_price:.2f}→SL2=${sl2_price:.2f}: {reason}")
    
    # Look for SH → SL → SH patterns (downtrend formations)
    for i in range(len(swing_list) - 2):
        if (swing_list[i][1] == 'HIGH' and 
            swing_list[i+1][1] == 'LOW' and 
            swing_list[i+2][1] == 'HIGH'):
            
            sh1_idx, sh1_type, sh1_price = swing_list[i]
            sl1_idx, sl1_type, sl1_price = swing_list[i+1]
            sh2_idx, sh2_type, sh2_price = swing_list[i+2]
            
            # Check for lower high (downtrend condition)
            if sh2_price < sh1_price:
                # Find breakdown origin
                breakout_idx = find_first_breakout_candle(df, sh2_idx, sl1_price, 'below')
                
                if breakout_idx:
                    # VALIDATE: Check if swing points remain valid until breakout
                    is_valid, reason = validate_downtrend_formation(swing_list, i, breakout_idx)
                    
                    if is_valid:
                        formation = {
                            'type': 'DOWNTREND',
                            'sh1': {'idx': sh1_idx, 'price': sh1_price},
                            'sl1': {'idx': sl1_idx, 'price': sl1_price},
                            'sh2': {'idx': sh2_idx, 'price': sh2_price}, 
                            'breakout': {'idx': breakout_idx, 'price': df.iloc[breakout_idx]['low']},
                            'controlling_swing': {'idx': sh1_idx, 'price': sh1_price},
                            'formation_date': df.iloc[breakout_idx]['datetime'],
                            'setup_quality': 'VALID'
                        }
                        formations.append(formation)
                        valid_count += 1
                    else:
                        invalid_count += 1
                        # Optionally print first few invalid reasons for debugging
                        if invalid_count <= 5:
                            print(f"   ❌ Invalid downtrend SH1=${sh1_price:.2f}→SL1=${sl1_price:.2f}→SH2=${sh2_price:.2f}: {reason}")
    
    # Look for SIDEWAYS trends (4+ swing points with balanced expansion)
    sideways_formations = find_sideways_trends(df, swing_list)
    formations.extend(sideways_formations)
    sideways_count = len(sideways_formations)
    
    # FILTER: Remove sideways trends that overlap with directional trends
    original_count = len(formations)
    formations = filter_overlapping_sideways_trends(formations)
    filtered_count = original_count - len(formations)
    final_sideways = len([f for f in formations if f['type'] == 'SIDEWAYS'])
    
    print(f"✅ Validation complete: {valid_count} uptrends/downtrends, {invalid_count} invalidated, {sideways_count} sideways trends")
    if filtered_count > 0:
        print(f"✅ Filtered out {filtered_count} sideways trends that overlapped with directional trends")
        print(f"   Final: {final_sideways} sideways trends remaining")
    return formations

def validate_uptrend_formation(swing_list, formation_start_idx, breakout_idx):
    """
    Validate uptrend formation: SL1 must remain the controlling low until breakout
    
    Args:
        swing_list: List of (idx, type, price) tuples
        formation_start_idx: Index in swing_list where SL1 is located
        breakout_idx: Candle index where breakout occurs
        
    Returns:
        tuple: (is_valid, reason)
    """
    
    sl1_idx, sl1_type, sl1_price = swing_list[formation_start_idx]
    sh1_idx, sh1_type, sh1_price = swing_list[formation_start_idx + 1]
    sl2_idx, sl2_type, sl2_price = swing_list[formation_start_idx + 2]
    
    # Check all swing points that occur after SL2 and before/at breakout
    for i in range(formation_start_idx + 3, len(swing_list)):
        swing_idx, swing_type, swing_price = swing_list[i]
        
        # Stop checking once we're past the breakout
        if swing_idx > breakout_idx:
            break
            
        # Check for invalidating swing lows
        if swing_type == 'LOW':
            if swing_price <= sl2_price:
                return False, f"Swing low at {swing_idx} (${swing_price:.2f}) ≤ SL2 (${sl2_price:.2f}) - SL1 no longer controlling"
        
        # Check for invalidating swing highs 
        if swing_type == 'HIGH':
            if swing_price >= sh1_price:
                return False, f"Swing high at {swing_idx} (${swing_price:.2f}) ≥ SH1 (${sh1_price:.2f}) - breakout level compromised"
    
    return True, "Valid formation"

def validate_downtrend_formation(swing_list, formation_start_idx, breakout_idx):
    """
    Validate downtrend formation: SH1 must remain the controlling high until breakout
    
    Args:
        swing_list: List of (idx, type, price) tuples
        formation_start_idx: Index in swing_list where SH1 is located  
        breakout_idx: Candle index where breakout occurs
        
    Returns:
        tuple: (is_valid, reason)
    """
    
    sh1_idx, sh1_type, sh1_price = swing_list[formation_start_idx]
    sl1_idx, sl1_type, sl1_price = swing_list[formation_start_idx + 1] 
    sh2_idx, sh2_type, sh2_price = swing_list[formation_start_idx + 2]
    
    # Check all swing points that occur after SH2 and before/at breakout
    for i in range(formation_start_idx + 3, len(swing_list)):
        swing_idx, swing_type, swing_price = swing_list[i]
        
        # Stop checking once we're past the breakout
        if swing_idx > breakout_idx:
            break
            
        # Check for invalidating swing highs
        if swing_type == 'HIGH':
            if swing_price >= sh2_price:
                return False, f"Swing high at {swing_idx} (${swing_price:.2f}) ≥ SH2 (${sh2_price:.2f}) - SH1 no longer controlling"
        
        # Check for invalidating swing lows
        if swing_type == 'LOW':
            if swing_price <= sl1_price:
                return False, f"Swing low at {swing_idx} (${swing_price:.2f}) ≤ SL1 (${sl1_price:.2f}) - breakdown level compromised"
    
    return True, "Valid formation"

def find_sideways_trends(df, swing_list):
    """
    Find SIDEWAYS trends using CORRECTED logic:
    - Trend ORIGINATES at 1st swing point (not 4th)
    - Uses 2 MOST RECENT swing points to define dynamic range
    - Supports contraction and balanced expansion within 5%
    - Need minimum 4 swing points for confirmation
    """
    
    formations = []
    
    print(f"🔍 Scanning for sideways trends (corrected logic: origin at 1st swing, dynamic range)...")
    
    # Look for potential sideways starting points (need at least 4 swings)
    for start_idx in range(len(swing_list) - 3):
        sideways_formation = attempt_sideways_formation(df, swing_list, start_idx)
        if sideways_formation:
            formations.append(sideways_formation)
    
    return formations

def attempt_sideways_formation(df, swing_list, start_idx):
    """
    Attempt to form a sideways trend starting at start_idx
    CORRECTED LOGIC:
    - Origin at 1st swing point
    - Dynamic range from 2 most recent swings
    - Supports contraction and balanced expansion
    """
    
    if start_idx + 3 >= len(swing_list):
        return None
    
    # CORRECTED: Origin at the FIRST swing point
    origin_swing = swing_list[start_idx]
    origin_idx, origin_type, origin_price = origin_swing
    
    # Start with first 4 swings for initial confirmation
    initial_swings = swing_list[start_idx:start_idx + 4]
    
    # Get initial range from first 4 swings
    initial_highs = [s[2] for s in initial_swings if s[1] == 'HIGH']
    initial_lows = [s[2] for s in initial_swings if s[1] == 'LOW']
    
    if len(initial_highs) < 2 or len(initial_lows) < 2:
        return None  # Need at least 2 highs and 2 lows
    
    # Continue adding swings while maintaining sideways behavior
    sideways_swings = initial_swings[:]
    
    for next_idx in range(start_idx + 4, len(swing_list)):
        next_swing = swing_list[next_idx]
        swing_idx, swing_type, swing_price = next_swing
        
        # CORRECTED: Use 2 MOST RECENT swings to define current range
        recent_swings = sideways_swings[-4:]  # Get last 4 swings to find most recent pair
        recent_highs = [s[2] for s in recent_swings if s[1] == 'HIGH']
        recent_lows = [s[2] for s in recent_swings if s[1] == 'LOW']
        
        if not recent_highs or not recent_lows:
            sideways_swings.append(next_swing)
            continue
            
        current_high = max(recent_highs)
        current_low = min(recent_lows)
        current_range = current_high - current_low
        
        # Check if adding this swing maintains balanced behavior
        if swing_type == 'HIGH':
            if swing_price > current_high:
                # Range expansion upward
                upward_expansion = swing_price - current_high
                # Check if we have corresponding downward expansion (within 5% tolerance)
                initial_range = max(initial_highs) - min(initial_lows)
                max_allowed_expansion = initial_range * 0.05  # 5% of initial range
                
                # If expansion is too large compared to other direction, break sideways
                recent_low_expansion = min(initial_lows) - current_low if current_low < min(initial_lows) else 0
                if upward_expansion > recent_low_expansion + max_allowed_expansion:
                    break  # Imbalanced expansion - end sideways trend
                
        else:  # LOW
            if swing_price < current_low:
                # Range expansion downward
                downward_expansion = current_low - swing_price
                # Check if we have corresponding upward expansion (within 5% tolerance)
                initial_range = max(initial_highs) - min(initial_lows)
                max_allowed_expansion = initial_range * 0.05  # 5% of initial range
                
                # If expansion is too large compared to other direction, break sideways
                recent_high_expansion = current_high - max(initial_highs) if current_high > max(initial_highs) else 0
                if downward_expansion > recent_high_expansion + max_allowed_expansion:
                    break  # Imbalanced expansion - end sideways trend
        
        sideways_swings.append(next_swing)
    
    # Only create formation if we have meaningful sideways action (4+ swings minimum)
    if len(sideways_swings) >= 4:
        # Get final range from 2 most recent swings
        final_swings = sideways_swings[-4:]
        final_highs = [s[2] for s in final_swings if s[1] == 'HIGH']
        final_lows = [s[2] for s in final_swings if s[1] == 'LOW']
        
        final_high = max(final_highs) if final_highs else max(initial_highs)
        final_low = min(final_lows) if final_lows else min(initial_lows)
        
        # Find the actual termination by checking candle-by-candle for violations
        last_swing_idx = sideways_swings[-1][0]
        actual_end_idx = last_swing_idx
        
        # Scan forward from the last swing to find first violation
        for candle_idx in range(last_swing_idx + 1, min(len(df), last_swing_idx + 20)):
            candle = df.iloc[candle_idx]
            
            # Check if this candle violates the current range (from most recent 2 swings)
            if candle['high'] > final_high or candle['low'] < final_low:
                actual_end_idx = candle_idx - 1  # End at the candle before violation
                break
            actual_end_idx = candle_idx  # Keep extending if no violation
        
        formation = {
            'type': 'SIDEWAYS',
            'start_swing': {'idx': origin_idx, 'price': origin_price},  # CORRECTED: Origin at 1st swing
            'end_swing': {'idx': actual_end_idx, 'price': df.iloc[actual_end_idx]['close']},
            'high_level': final_high,  # From most recent swings
            'low_level': final_low,    # From most recent swings
            'range_size': final_high - final_low,
            'swing_count': len(sideways_swings),
            'formation_date': df.iloc[origin_idx]['datetime'],  # CORRECTED: Use origin date
            'termination_date': df.iloc[actual_end_idx]['datetime'],
            'setup_quality': 'VALID',
            'last_swing_idx': last_swing_idx
        }
        return formation
    
    return None

def filter_overlapping_sideways_trends(formations):
    """
    Filter out sideways trends that overlap with ACTIVE directional trends
    Allow sideways trends AFTER directional trends have terminated
    """
    
    # We need termination data to do this properly, so for now just return a basic filter
    # This function will be called AFTER terminations are found
    return formations

def filter_overlapping_sideways_trends_with_terminations(formations, terminations):
    """
    Filter out sideways trends that overlap with ACTIVE directional trends
    Allow sideways trends AFTER directional trends have terminated
    """
    
    # Create termination lookup
    termination_by_formation = {}
    for term in terminations:
        formation_key = id(term['formation'])
        termination_by_formation[formation_key] = term
    
    # Separate directional and sideways trends
    directional_trends = [f for f in formations if f['type'] in ['UPTREND', 'DOWNTREND']]
    sideways_trends = [f for f in formations if f['type'] == 'SIDEWAYS']
    
    # Keep all directional trends
    filtered_formations = directional_trends[:]
    
    # Filter sideways trends that don't overlap with ACTIVE directional trends
    for sideways in sideways_trends:
        sideways_start = sideways['start_swing']['idx']
        sideways_end = sideways['end_swing']['idx']
        
        # Check if this sideways trend overlaps with any ACTIVE directional trend
        overlaps = False
        for directional in directional_trends:
            if 'breakout' in directional:
                dir_start = directional['breakout']['idx']
                
                # Get actual directional trend end from termination data
                formation_key = id(directional)
                termination = termination_by_formation.get(formation_key)
                if termination:
                    dir_end = termination['violation_idx']
                else:
                    # If no termination, assume trend is still active
                    dir_end = dir_start + 100  # Large range for active trends
                
                # Check for overlap with ACTIVE directional trend
                if not (sideways_end < dir_start or sideways_start > dir_end):
                    # But allow sideways if it starts AFTER the directional trend terminates
                    if termination and sideways_start >= termination['violation_idx']:
                        continue  # No overlap - sideways starts after termination
                    overlaps = True
                    break
        
        # Only keep sideways trend if it doesn't overlap with active directional trends
        if not overlaps:
            filtered_formations.append(sideways)
    
    return filtered_formations

def calculate_trend_duration(formation, origin_idx, df_length):
    """
    Calculate trend duration properly for different trend types
    """
    if formation['type'] == 'SIDEWAYS':
        # For sideways trends, use the start and end swings
        start_idx = formation['start_swing']['idx']
        end_idx = formation['end_swing']['idx']
        return end_idx - start_idx
    else:
        # For directional trends without termination, use remaining data
        return df_length - origin_idx

def find_all_proper_terminations(df, formations, swing_points):
    """
    Find ALL trend terminations with DYNAMIC CONTROLLING SWING UPDATES
    Updates controlling swings as new extremes form during the trend
    """
    
    terminations = []
    
    print(f"🔍 Scanning {len(formations)} formations for terminations with dynamic controlling updates...")
    
    for formation in formations:
        termination = find_trend_termination_with_updates(df, formation, swing_points)
        if termination:
            terminations.append(termination)
    
    return terminations

def find_trend_termination_with_updates(df, formation, all_swing_points):
    """
    Find trend termination using HYBRID approach:
    - Swing points update controlling levels (trailing stops)
    - ANY candle can violate and terminate the trend
    """
    
    # Initial controlling swing should be SL2 for uptrends, SH2 for downtrends, N/A for sideways
    if formation['type'] == 'UPTREND':
        original_controlling_price = formation['sl2']['price']  # SL2 is the setup swing for uptrends
        current_controlling_idx = formation['sl2']['idx']
    elif formation['type'] == 'DOWNTREND':
        original_controlling_price = formation['sh2']['price']  # SH2 is the setup swing for downtrends  
        current_controlling_idx = formation['sh2']['idx']
    else:  # SIDEWAYS
        # Sideways trends have their own termination logic - they end when range is violated
        return find_sideways_termination(df, formation)
    
    current_controlling_price = original_controlling_price
    formation_idx = formation['breakout']['idx']  # Directional trends always have breakout
    
    # Track updates for debugging
    updates_count = 0
    
    # Get swing points that occur AFTER the breakout for controlling updates
    relevant_swings = [sp for sp in all_swing_points if sp['index'] > formation_idx]
    relevant_swings.sort(key=lambda x: x['index'])
    
    # Scan each candle after breakout
    for candle_idx in range(formation_idx + 1, len(df)):
        candle = df.iloc[candle_idx]
        
        # First: Check if this candle corresponds to a swing point that could update controlling level
        swing_at_this_candle = next((sp for sp in relevant_swings if sp['index'] == candle_idx), None)
        
        if swing_at_this_candle and formation['type'] == 'UPTREND':
            # For uptrend: swing lows can update controlling swing
            if swing_at_this_candle['type'] == 'LOW':
                if swing_at_this_candle['price'] > current_controlling_price:
                    current_controlling_price = swing_at_this_candle['price']
                    current_controlling_idx = candle_idx
                    updates_count += 1
                    
        elif swing_at_this_candle and formation['type'] == 'DOWNTREND':
            # For downtrend: swing highs can update controlling swing
            if swing_at_this_candle['type'] == 'HIGH':
                if swing_at_this_candle['price'] < current_controlling_price:
                    current_controlling_price = swing_at_this_candle['price']
                    current_controlling_idx = candle_idx
                    updates_count += 1
        
        # Second: Check if ANY candle violates the current controlling swing
        violation_occurred = False
        violation_price = None
        
        if formation['type'] == 'UPTREND':
            # Uptrend violation: any candle low breaks below controlling swing
            if candle['low'] < current_controlling_price:
                violation_occurred = True
                violation_price = candle['low']
        else:  # DOWNTREND
            # Downtrend violation: any candle high breaks above controlling swing
            if candle['high'] > current_controlling_price:
                violation_occurred = True
                violation_price = candle['high']
        
        if violation_occurred:
            # Found violation!
            termination = {
                'formation': formation,
                'violation_idx': candle_idx,
                'violation_price': violation_price,
                'violation_date': candle['datetime'],
                'trend_duration': candle_idx - formation_idx,
                'violation_size': abs(current_controlling_price - violation_price),
                'original_controlling_price': original_controlling_price,
                'final_controlling_price': current_controlling_price,
                'final_controlling_idx': current_controlling_idx,
                'controlling_updates': updates_count
            }
            return termination
    
    # No violation found - trend remains active
    return None

def find_sideways_termination(df, formation):
    """
    Find termination for sideways trends using the range violation logic
    """
    
    # For sideways trends, termination is already built into the formation
    # The end_swing represents where the trend terminated
    start_idx = formation['start_swing']['idx']
    end_idx = formation['end_swing']['idx']
    
    # Check if this sideways trend actually terminated (has violation)
    if end_idx < len(df) - 1:
        # Find the actual violation candle (first candle that broke the range)
        violation_idx = None
        violation_price = None
        
        for candle_idx in range(end_idx, min(len(df), end_idx + 10)):
            candle = df.iloc[candle_idx]
            
            # Check if this candle violates the sideways range
            if candle['high'] > formation['high_level'] or candle['low'] < formation['low_level']:
                violation_idx = candle_idx
                if candle['high'] > formation['high_level']:
                    violation_price = candle['high']
                else:
                    violation_price = candle['low']
                break
        
        if violation_idx is not None:
            termination = {
                'formation': formation,
                'violation_idx': violation_idx,
                'violation_price': violation_price,
                'violation_date': df.iloc[violation_idx]['datetime'],
                'trend_duration': end_idx - start_idx,
                'violation_size': abs(violation_price - (formation['high_level'] + formation['low_level']) / 2),
                'range_high': formation['high_level'],
                'range_low': formation['low_level']
            }
            return termination
    
    # No violation found - sideways trend is still active or data ended
    return None

def get_controlling_swing_for_trend(formation):
    """
    Get appropriate controlling swing data for different trend types
    """
    if formation['type'] == 'SIDEWAYS':
        # For sideways trends, use the range center as "controlling" level
        center_price = (formation['high_level'] + formation['low_level']) / 2
        return {'idx': formation['start_swing']['idx'], 'price': center_price}
    elif formation['type'] == 'UPTREND':
        # For uptrends, use SL2 as controlling swing
        return formation.get('sl2', {'idx': 0, 'price': 0})
    else:  # DOWNTREND
        # For downtrends, use SH2 as controlling swing  
        return formation.get('sh2', {'idx': 0, 'price': 0})

def create_proper_trend_explorer(df, formations, terminations, all_swing_points):
    """
    Create a proper trend explorer showing each trend with all details
    """
    
    print(f"🎯 Creating proper trend explorer for {len(formations)} formations...")
    
    # Create termination lookup
    termination_by_formation = {}
    for term in terminations:
        formation_key = id(term['formation'])
        termination_by_formation[formation_key] = term
    
    # Prepare data for each trend
    trend_data = []
    
    for i, formation in enumerate(formations):
        formation_key = id(formation)
        termination = termination_by_formation.get(formation_key)
        
        # Determine time window around the trend - show MORE context
        if formation['type'] == 'SIDEWAYS':
            origin_idx = formation['start_swing']['idx']  # Sideways trends have origins, not breakouts
        else:
            origin_idx = formation['breakout']['idx']  # Directional trends have breakouts
        
        # Get swing point indices
        if formation['type'] == 'UPTREND':
            swing_indices = [
                formation['sl1']['idx'],
                formation['sh1']['idx'], 
                formation['sl2']['idx']
            ]
        elif formation['type'] == 'DOWNTREND':
            swing_indices = [
                formation['sh1']['idx'],
                formation['sl1']['idx'],
                formation['sh2']['idx']
            ]
        else:  # SIDEWAYS
            swing_indices = [
                formation['start_swing']['idx'],
                formation['end_swing']['idx']
            ]
        
        # Window should show: context before first swing + formation + context after
        first_swing_idx = min(swing_indices)
        
        if formation['type'] == 'SIDEWAYS':
            # For sideways trends, use start and end swings with context
            start_idx = formation['start_swing']['idx']
            end_idx = formation['end_swing']['idx']
            window_start = max(0, start_idx - 20)
            window_end = min(len(df) - 1, end_idx + 20)
        elif termination:
            end_idx = termination['violation_idx']
            # Show context: 20 candles before first swing, formation, 10 candles after termination
            window_start = max(0, first_swing_idx - 20)
            window_end = min(len(df) - 1, end_idx + 10)
        else:
            # Active trend - show to end with more context
            window_start = max(0, first_swing_idx - 20)
            window_end = min(len(df) - 1, breakout_idx + 50)  # Show 50 candles after breakout
        
        # Get swing point details with proper structure
        swing_points = []
        
        if formation['type'] == 'UPTREND':
            # Add formation swings
            swing_points = [
                {'idx': formation['sl1']['idx'], 'price': formation['sl1']['price'], 'type': 'SL1', 'role': 'First Low'},
                {'idx': formation['sh1']['idx'], 'price': formation['sh1']['price'], 'type': 'SH1', 'role': 'High (Resistance)'},
                {'idx': formation['sl2']['idx'], 'price': formation['sl2']['price'], 'type': 'SL2', 'role': 'Setup Low'}
            ]
            
            # Add ALL swing points that occur during the trend
            if termination:
                formation_idx = formation['breakout']['idx']
                violation_idx = termination['violation_idx']
                
                # Get all swings during the trend (both highs and lows)
                trend_swings = [sp for sp in all_swing_points if 
                              sp['index'] > formation_idx and 
                              sp['index'] < violation_idx]
                trend_swings.sort(key=lambda x: x['index'])
                
                sl_count = 3  # Start numbering after SL1, SL2
                sh_count = 2  # Start numbering after SH1
                
                for swing in trend_swings:
                    if swing['type'] == 'LOW':
                        # Check if this low updated the controlling swing
                        is_controlling_update = swing['price'] > formation['sl2']['price']
                        role = f'Controlling Update #{sl_count-2}' if is_controlling_update else 'Trend Low'
                        
                        swing_points.append({
                            'idx': swing['index'], 
                            'price': swing['price'], 
                            'type': f'SL{sl_count}', 
                            'role': role
                        })
                        sl_count += 1
                        
                    else:  # HIGH
                        swing_points.append({
                            'idx': swing['index'], 
                            'price': swing['price'], 
                            'type': f'SH{sh_count}', 
                            'role': 'Trend High'
                        })
                        sh_count += 1
        elif formation['type'] == 'DOWNTREND':
            # Add formation swings  
            swing_points = [
                {'idx': formation['sh1']['idx'], 'price': formation['sh1']['price'], 'type': 'SH1', 'role': 'First High'},
                {'idx': formation['sl1']['idx'], 'price': formation['sl1']['price'], 'type': 'SL1', 'role': 'Low (Support)'},
                {'idx': formation['sh2']['idx'], 'price': formation['sh2']['price'], 'type': 'SH2', 'role': 'Setup High'}
            ]
            
            # Add ALL swing points that occur during the trend
            if termination:
                formation_idx = formation['breakout']['idx']
                violation_idx = termination['violation_idx']
                
                # Get all swings during the trend (both highs and lows)
                trend_swings = [sp for sp in all_swing_points if 
                              sp['index'] > formation_idx and 
                              sp['index'] < violation_idx]
                trend_swings.sort(key=lambda x: x['index'])
                
                sh_count = 3  # Start numbering after SH1, SH2
                sl_count = 2  # Start numbering after SL1
                
                for swing in trend_swings:
                    if swing['type'] == 'HIGH':
                        # Check if this high updated the controlling swing
                        is_controlling_update = swing['price'] < formation['sh2']['price']
                        role = f'Controlling Update #{sh_count-2}' if is_controlling_update else 'Trend High'
                        
                        swing_points.append({
                            'idx': swing['index'], 
                            'price': swing['price'], 
                            'type': f'SH{sh_count}', 
                            'role': role
                        })
                        sh_count += 1
                        
                    else:  # LOW
                        swing_points.append({
                            'idx': swing['index'], 
                            'price': swing['price'], 
                            'type': f'SL{sl_count}', 
                            'role': 'Trend Low'
                        })
                        sl_count += 1
        else:  # SIDEWAYS
            # For sideways trends, show all swing points in the range
            swing_points = []
            start_idx = formation['start_swing']['idx']
            end_idx = formation['end_swing']['idx']
            
            # Find all swings in the sideways range
            range_swings = [sp for sp in all_swing_points if 
                          sp['index'] >= start_idx and sp['index'] <= end_idx]
            range_swings.sort(key=lambda x: x['index'])
            
            sh_count = 1
            sl_count = 1
            
            for swing in range_swings:
                if swing['type'] == 'HIGH':
                    swing_points.append({
                        'idx': swing['index'], 
                        'price': swing['price'], 
                        'type': f'SH{sh_count}', 
                        'role': f'Range High #{sh_count}'
                    })
                    sh_count += 1
                else:  # LOW
                    swing_points.append({
                        'idx': swing['index'], 
                        'price': swing['price'], 
                        'type': f'SL{sl_count}', 
                        'role': f'Range Low #{sl_count}'
                    })
                    sl_count += 1
        
        trend_info = {
            'id': i + 1,
            'type': formation['type'],
            'formation': formation,
            'termination': termination,
            'window_start': window_start,
            'window_end': window_end,
            'origin_idx': origin_idx,  # Use origin_idx for both directional and sideways
            'swing_points': swing_points,
            'controlling_swing': get_controlling_swing_for_trend(formation),
            'duration': termination['trend_duration'] if termination else calculate_trend_duration(formation, origin_idx, len(df)),
            'formation_date': formation['formation_date'].strftime('%Y-%m-%d') if hasattr(formation['formation_date'], 'strftime') else str(formation['formation_date']),
            'status': 'Terminated' if termination else 'Active',
            # Add controlling swing update info
            'original_controlling': get_controlling_swing_for_trend(formation).get('price', 0),
            'final_controlling': termination.get('final_controlling_price') if termination else get_controlling_swing_for_trend(formation).get('price', 0),
            'controlling_updates': termination.get('controlling_updates', 0) if termination else 0
        }
        
        if termination:
            trend_info['termination_idx'] = termination['violation_idx']
            trend_info['violation_price'] = termination['violation_price']
            trend_info['violation_size'] = termination['violation_size']
            trend_info['termination_date'] = termination['violation_date'].strftime('%Y-%m-%d') if hasattr(termination['violation_date'], 'strftime') else str(termination['violation_date'])
        
        trend_data.append(trend_info)
    
    # Create the enhanced HTML
    create_proper_explorer_html(df, trend_data)

def create_proper_explorer_html(df, trend_data):
    """Create the proper trend explorer HTML"""
    
    # Convert dataframe to JSON for JavaScript
    df_json = []
    for i, row in df.iterrows():
        df_json.append({
            'idx': i,
            'datetime': row['datetime'].isoformat() if hasattr(row['datetime'], 'isoformat') else str(row['datetime']),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        })
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Proper Trend Explorer - Detailed Analysis</title>
    <script src="https://cdn.plot.ly/plotly-2.25.2.min.js"></script>
    <style>
        body {{ 
            background-color: #1e1e1e; 
            color: white; 
            font-family: Arial, sans-serif; 
            margin: 0;
            padding: 15px;
            font-size: 14px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .header h1 {{
            color: #00ff88;
            margin-bottom: 5px;
            font-size: 24px;
        }}
        .control-panel {{
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            border: 2px solid #333;
            display: grid;
            grid-template-columns: 350px 1fr 200px;
            gap: 20px;
            align-items: center;
        }}
        .trend-selector {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .trend-info {{
            background: rgba(0,50,0,0.3);
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #00ff88;
            min-height: 120px;
        }}
        .trend-info h3 {{
            margin: 0 0 8px 0;
            color: #00ff88;
            font-size: 16px;
        }}
        .info-item {{
            margin: 3px 0;
            font-size: 12px;
            line-height: 1.3;
        }}
        .navigation {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .nav-btn {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
            font-size: 13px;
        }}
        .nav-btn:hover {{ 
            background: #0056b3; 
            transform: translateY(-1px);
        }}
        .nav-btn:disabled {{
            background: #555;
            cursor: not-allowed;
            transform: none;
        }}
        .trend-counter {{
            text-align: center;
            font-size: 16px;
            font-weight: bold;
            color: #ffaa00;
            margin-bottom: 8px;
        }}
        
        .uptrend {{ border-left: 4px solid #00ff00; }}
        .downtrend {{ border-left: 4px solid #ff0000; }}
        .terminated {{ background: rgba(100,0,0,0.1); }}
        .active {{ background: rgba(0,100,0,0.1); }}
        
        select {{
            background: #333;
            color: white;
            border: 1px solid #555;
            padding: 6px;
            border-radius: 4px;
            width: 100%;
            font-size: 12px;
        }}
        
        label {{
            color: #ffaa00;
            font-weight: bold;
            font-size: 12px;
        }}
        
        #chart {{ 
            margin-top: 15px;
            min-height: 600px;
        }}
        
        .swing-details {{
            background: rgba(0,0,100,0.1);
            border: 1px solid #007bff;
            padding: 8px;
            margin: 5px 0;
            border-radius: 4px;
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Proper Trend Explorer</h1>
        <p>Browse through {len(trend_data)} trend formations with complete swing point analysis</p>
    </div>
    
    <div class="control-panel">
        <div class="trend-selector">
            <label>Filter by Trend Type:</label>
            <select id="typeFilter" onchange="filterTrendsByType()">
                <option value="ALL">All Trends</option>
                <option value="UPTREND">Uptrends Only</option>
                <option value="DOWNTREND">Downtrends Only</option>
                <option value="SIDEWAYS">Sideways Only</option>
            </select>
            
            <label>Select Trend:</label>
            <select id="trendSelect" onchange="loadTrend()">
                {chr(10).join([f'<option value="{i}">{trend["type"]} #{trend["id"]} ({trend["formation_date"]}) - {trend["status"]}</option>' for i, trend in enumerate(trend_data)])}
            </select>
            
            <div class="trend-counter" id="trendCounter">
                Trend 1 of {len(trend_data)}
            </div>
        </div>
        
        <div class="trend-info" id="trendInfo">
            <h3 id="trendTitle">Loading...</h3>
            <div id="trendDetails"></div>
        </div>
        
        <div class="navigation">
            <button class="nav-btn" onclick="previousTrend()" id="prevBtn">← Previous</button>
            <button class="nav-btn" onclick="nextTrend()" id="nextBtn">Next →</button>
            <button class="nav-btn" onclick="jumpToRandom()" style="background: #ff8c00;">🎲 Random</button>
        </div>
    </div>
    
    <div id="chart"></div>
    
    <script>
        // Data from Python
        const dfData = {json.dumps(df_json)};
        const allTrendData = {json.dumps(trend_data, default=str)};
        let filteredTrendData = [...allTrendData];
        let currentTrendIndex = 0;
        
        function filterTrendsByType() {{
            const typeFilter = document.getElementById('typeFilter');
            const selectedType = typeFilter.value;
            
            if (selectedType === 'ALL') {{
                filteredTrendData = [...allTrendData];
            }} else {{
                filteredTrendData = allTrendData.filter(trend => trend.type === selectedType);
            }}
            
            // Update the trend selector dropdown
            updateTrendSelector();
            
            // Reset to first trend in filtered list
            currentTrendIndex = 0;
            if (filteredTrendData.length > 0) {{
                displayTrend(currentTrendIndex);
            }}
        }}
        
        function updateTrendSelector() {{
            const trendSelect = document.getElementById('trendSelect');
            trendSelect.innerHTML = '';
            
            filteredTrendData.forEach((trend, index) => {{
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `${{trend.type}} #${{trend.id}} (${{trend.formation_date}}) - ${{trend.status}}`;
                trendSelect.appendChild(option);
            }});
            
            // Update counter
            const counter = document.getElementById('trendCounter');
            counter.textContent = filteredTrendData.length > 0 ? 
                `Trend 1 of ${{filteredTrendData.length}}` : 
                'No trends found';
        }}
        
        function loadTrend() {{
            const select = document.getElementById('trendSelect');
            currentTrendIndex = parseInt(select.value);
            displayTrend(currentTrendIndex);
        }}
        
        function displayTrend(index) {{
            if (index >= filteredTrendData.length) return;
            const trend = filteredTrendData[index];
            
            console.log('displayTrend called for:', trend.type, 'trend #' + trend.id);
            console.log('Window:', trend.window_start, '-', trend.window_end);
            console.log('Formation:', trend.formation);
            
            const windowData = dfData.slice(trend.window_start, trend.window_end + 1);
            console.log('WindowData length:', windowData.length);
            
            // Update trend info panel
            updateTrendInfo(trend);
            
            // Create candlestick chart
            const traces = [];
            
            // Add candlestick data
            traces.push({{
                type: 'candlestick',
                x: windowData.map(d => d.datetime),
                open: windowData.map(d => d.open),
                high: windowData.map(d => d.high),
                low: windowData.map(d => d.low),
                close: windowData.map(d => d.close),
                name: 'USO',
                increasing: {{ line: {{ color: '#00ff88' }} }},
                decreasing: {{ line: {{ color: '#ff4444' }} }}
            }});
            
            // Add swing points with detailed labels
            const swingColors = {{
                'SL1': '#00aaff', 'SL2': '#00aaff', 'SH1': '#ffaa00', 'SH2': '#ffaa00'
            }};
            
            trend.swing_points.forEach(sp => {{
                if (sp.idx >= trend.window_start && sp.idx <= trend.window_end) {{
                    const candle = dfData[sp.idx];
                    const color = swingColors[sp.type] || '#ffffff';
                    
                    traces.push({{
                        type: 'scatter',
                        mode: 'markers+text',
                        x: [candle.datetime],
                        y: [sp.price],
                        marker: {{
                            color: color,
                            size: 14,
                            symbol: sp.type.includes('H') ? 'triangle-up' : 'triangle-down',
                            line: {{ color: 'white', width: 2 }}
                        }},
                        text: [sp.type],
                        textposition: sp.type.includes('H') ? 'top center' : 'bottom center',
                        textfont: {{ size: 11, color: 'white', family: 'Arial Black' }},
                        name: `${{sp.type}} (${{sp.role}})`,
                        hovertemplate: `<b>${{sp.type}}</b><br>${{sp.role}}<br>Price: $${{sp.price.toFixed(2)}}<br>Candle: ${{sp.idx}}<extra></extra>`
                    }});
                }}
            }});
            
            // Add breakout origin marker
            if (trend.breakout_idx >= trend.window_start && trend.breakout_idx <= trend.window_end) {{
                const breakoutCandle = dfData[trend.breakout_idx];
                traces.push({{
                    type: 'scatter',
                    mode: 'markers+text',
                    x: [breakoutCandle.datetime],
                    y: [trend.type === 'UPTREND' ? breakoutCandle.high : breakoutCandle.low],
                    marker: {{
                        color: '#ffff00',
                        size: 18,
                        symbol: 'star',
                        line: {{ color: 'black', width: 2 }}
                    }},
                    text: ['BREAKOUT'],
                    textposition: 'middle right',
                    textfont: {{ size: 12, color: '#ffff00', family: 'Arial Black' }},
                    name: 'Breakout Origin',
                    hovertemplate: `<b>Trend Formation Origin</b><br>Breakout Price: $${{(trend.type === 'UPTREND' ? breakoutCandle.high : breakoutCandle.low).toFixed(2)}}<br>Candle: ${{trend.breakout_idx}}<extra></extra>`
                }});
            }}
            
            // Add termination marker if exists
            if (trend.termination && trend.termination_idx >= trend.window_start && trend.termination_idx <= trend.window_end) {{
                const termCandle = dfData[trend.termination_idx];
                traces.push({{
                    type: 'scatter',
                    mode: 'markers+text',
                    x: [termCandle.datetime],
                    y: [trend.violation_price],
                    marker: {{
                        color: '#ff0000',
                        size: 16,
                        symbol: 'x',
                        line: {{ color: 'white', width: 2 }}
                    }},
                    text: ['VIOLATION'],
                    textposition: 'middle left',
                    textfont: {{ size: 11, color: '#ff0000', family: 'Arial Black' }},
                    name: 'Trend Termination',
                    hovertemplate: `<b>Trend Violation</b><br>Violation: $${{trend.violation_price.toFixed(2)}}<br>Size: $${{trend.violation_size.toFixed(2)}}<br>Candle: ${{trend.termination_idx}}<extra></extra>`
                }});
            }}
            
            
            // Add controlling swing horizontal line
            if (trend.controlling_swing && trend.controlling_swing.price !== undefined) {{
                const controlPrice = trend.controlling_swing.price;
                
                // For sideways trends, use actual trend start/end dates
                let startDate, endDate;
                if (trend.type === 'SIDEWAYS') {{
                    const startIdx = Math.max(0, trend.formation.start_swing.idx - trend.window_start);
                    const endIdx = Math.min(windowData.length - 1, trend.formation.end_swing.idx - trend.window_start);
                    startDate = windowData[startIdx].datetime;
                    endDate = windowData[endIdx].datetime;
                }} else {{
                    // For directional trends, use full window
                    startDate = windowData[0].datetime;
                    endDate = windowData[windowData.length - 1].datetime;
                }}
                
                let lineColor = '#ff0000';  // Default red
                if (trend.type === 'UPTREND') lineColor = '#00ff00';  // Green for uptrends
                else if (trend.type === 'SIDEWAYS') lineColor = '#ffaa00';  // Orange for sideways
                
                traces.push({{
                    type: 'scatter',
                    mode: 'lines',
                    x: [startDate, endDate],
                    y: [controlPrice, controlPrice],
                    line: {{
                        color: lineColor,
                        width: 2,
                        dash: trend.status === 'Terminated' ? 'solid' : 'dash'
                    }},
                    name: trend.type === 'SIDEWAYS' ? 'Range Center' : 'Control Level',
                    showlegend: true,
                    hovertemplate: `<b>${{trend.type === 'SIDEWAYS' ? 'Range Center' : 'Controlling Level'}}</b><br>Price: $${{controlPrice.toFixed(2)}}<br>Status: ${{trend.status}}<extra></extra>`
                }});
            }}
            
            // Calculate dynamic Y-axis range based on trend type
            let minPrice = Infinity;
            let maxPrice = -Infinity;
            
            try {{
                if (trend.type === 'SIDEWAYS') {{
                    console.log('Calculating range for SIDEWAYS trend');
                    console.log('Formation data:', trend.formation);
                    
                    // For sideways trends, use ALL candles in the trend duration PLUS violation candle
                    if (trend.formation && trend.formation.start_swing && trend.formation.end_swing) {{
                        const sidewaysStartIdx = Math.max(0, trend.formation.start_swing.idx - trend.window_start);
                        let sidewaysEndIdx = Math.min(windowData.length - 1, trend.formation.end_swing.idx - trend.window_start);
                        
                        // IMPORTANT: Include violating candle if it exists
                        if (trend.termination && trend.termination_idx) {{
                            const violationIdx = Math.min(windowData.length - 1, trend.termination_idx - trend.window_start);
                            sidewaysEndIdx = Math.max(sidewaysEndIdx, violationIdx);
                            console.log('Including violation candle at idx:', trend.termination_idx);
                        }}
                        
                        for (let i = sidewaysStartIdx; i <= sidewaysEndIdx; i++) {{
                            const candle = windowData[i];
                            if (candle) {{
                                minPrice = Math.min(minPrice, candle.low);
                                maxPrice = Math.max(maxPrice, candle.high);
                            }}
                        }}
                        
                        console.log('Range from candle', sidewaysStartIdx, 'to', sidewaysEndIdx, ':', minPrice, '-', maxPrice);
                    }} else {{
                        // Fallback: use window data if formation data is missing
                        minPrice = Math.min(...windowData.map(d => d.low));
                        maxPrice = Math.max(...windowData.map(d => d.high));
                        console.log('Using window data fallback:', minPrice, '-', maxPrice);
                    }}
                }} else {{
                    // For directional trends, find min/max from ALL candles in the trend duration
                    if (trend.origin_idx !== undefined && trend.termination && trend.termination_idx) {{
                        // Use actual trend duration: from origin to termination (INCLUDING violating candle)
                        const trendStartIdx = Math.max(0, trend.origin_idx - trend.window_start);
                        const trendEndIdx = Math.min(windowData.length - 1, trend.termination_idx - trend.window_start);
                        
                        console.log('Directional trend range from candle', trendStartIdx, 'to', trendEndIdx, '(including violation)');
                        
                        for (let i = trendStartIdx; i <= trendEndIdx; i++) {{
                            const candle = windowData[i];
                            if (candle) {{
                                minPrice = Math.min(minPrice, candle.low);
                                maxPrice = Math.max(maxPrice, candle.high);
                            }}
                        }}
                    }} else {{
                        // Fallback: use all swing points if trend duration can't be determined
                        trend.swing_points.forEach(swing => {{
                            minPrice = Math.min(minPrice, swing.price);
                            maxPrice = Math.max(maxPrice, swing.price);
                        }});
                        
                        // Also check termination price if exists
                        if (trend.termination) {{
                            minPrice = Math.min(minPrice, trend.termination.violation_price);
                            maxPrice = Math.max(maxPrice, trend.termination.violation_price);
                        }}
                    }}
                }}
            }} catch (rangeError) {{
                console.error('Error calculating price range:', rangeError);
                // Ultimate fallback
                minPrice = Math.min(...windowData.map(d => d.low));
                maxPrice = Math.max(...windowData.map(d => d.high));
            }}
            
            console.log('Final price range:', minPrice, '-', maxPrice);
            
            // Add 10% padding above and below
            const range = maxPrice - minPrice;
            const padding = Math.max(range * 0.1, 0.5); // At least $0.50 padding
            let yMin = minPrice - padding;
            let yMax = maxPrice + padding;
            
            // Ensure valid range (fallback if something went wrong)
            if (!isFinite(yMin) || !isFinite(yMax) || yMin >= yMax) {{
                const allPrices = windowData.flatMap(d => [d.open, d.high, d.low, d.close]);
                yMin = Math.min(...allPrices) * 0.95;
                yMax = Math.max(...allPrices) * 1.05;
            }}
            
            const layout = {{
                title: {{
                    text: `${{trend.type}} #${{trend.id}} - ${{trend.formation_date}} (${{trend.status}}) - ${{trend.duration}} candles`,
                    font: {{ size: 16, color: 'white' }},
                    x: 0.5
                }},
                xaxis: {{
                    title: 'Date',
                    gridcolor: 'rgba(100,100,100,0.2)',
                    type: 'date'
                }},
                yaxis: {{
                    title: 'Price ($)',
                    gridcolor: 'rgba(100,100,100,0.2)',
                    range: [yMin, yMax]
                }},
                template: 'plotly_dark',
                height: 600,
                showlegend: true,
                legend: {{
                    x: 0.02, y: 0.98,
                    bgcolor: 'rgba(0,0,0,0.8)',
                    bordercolor: 'white',
                    borderwidth: 1
                }},
                plot_bgcolor: '#1e1e1e',
                paper_bgcolor: '#1e1e1e'
            }};
            
            console.log('About to render chart for', trend.type, 'trend');
            console.log('Traces count:', traces.length);
            console.log('Y-axis range:', layout.yaxis.range);
            
            try {{
                Plotly.newPlot('chart', traces, layout);
                console.log('Chart rendered successfully');
            }} catch (error) {{
                console.error('Error rendering chart:', error);
                console.error('Trend data:', trend);
                console.error('Layout:', layout);
                console.error('Traces:', traces);
            }}
            
            // Update navigation
            document.getElementById('prevBtn').disabled = index === 0;
            document.getElementById('nextBtn').disabled = index === filteredTrendData.length - 1;
            document.getElementById('trendCounter').textContent = `Trend ${{index + 1}} of ${{filteredTrendData.length}}`;
            document.getElementById('trendSelect').value = index;
        }}
        
        function updateTrendInfo(trend) {{
            const infoDiv = document.getElementById('trendInfo');
            const titleDiv = document.getElementById('trendTitle');
            const detailsDiv = document.getElementById('trendDetails');
            
            // Set class for styling
            infoDiv.className = `trend-info ${{trend.type.toLowerCase()}} ${{trend.status.toLowerCase()}}`;
            
            titleDiv.textContent = `${{trend.type}} #${{trend.id}}`;
            
            let detailsHTML = `
                <div class="info-item"><strong>Formation:</strong> ${{trend.formation_date}}</div>
                <div class="info-item"><strong>Duration:</strong> ${{trend.duration}} candles</div>
                <div class="info-item"><strong>Status:</strong> ${{trend.status}}</div>
                <div class="info-item"><strong>Original Control:</strong> $${{(trend.original_controlling || 0).toFixed(2)}}</div>
                <div class="info-item"><strong>Final Control:</strong> $${{(trend.final_controlling || 0).toFixed(2)}} (${{trend.controlling_updates || 0}} updates)</div>
            `;
            
            // Add swing point details
            detailsHTML += '<div class="swing-details"><strong>Swing Points:</strong><br>';
            trend.swing_points.forEach(sp => {{
                detailsHTML += `${{sp.type}}: $${{sp.price.toFixed(2)}} (${{sp.role}})<br>`;
            }});
            detailsHTML += `${{trend.type === 'SIDEWAYS' ? 'Origin' : 'Breakout'}}: Candle ${{trend.origin_idx || trend.breakout_idx}}</div>`;
            
            if (trend.termination) {{
                detailsHTML += `
                    <div class="info-item"><strong>Termination:</strong> ${{trend.termination_date}}</div>
                    <div class="info-item"><strong>Violation:</strong> $${{(trend.violation_price || 0).toFixed(2)}} ($${{(trend.violation_size || 0).toFixed(2)}} break)</div>
                `;
            }}
            
            detailsDiv.innerHTML = detailsHTML;
        }}
        
        function previousTrend() {{
            if (currentTrendIndex > 0) {{
                currentTrendIndex--;
                displayTrend(currentTrendIndex);
                updateNavigationButtons();
            }}
        }}
        
        function nextTrend() {{
            if (currentTrendIndex < filteredTrendData.length - 1) {{
                currentTrendIndex++;
                displayTrend(currentTrendIndex);
                updateNavigationButtons();
            }}
        }}
        
        function jumpToRandom() {{
            if (filteredTrendData.length > 0) {{
                currentTrendIndex = Math.floor(Math.random() * filteredTrendData.length);
                displayTrend(currentTrendIndex);
                updateNavigationButtons();
            }}
        }}
        
        function updateNavigationButtons() {{
            // Update navigation buttons
            document.getElementById('prevBtn').disabled = currentTrendIndex === 0;
            document.getElementById('nextBtn').disabled = currentTrendIndex === filteredTrendData.length - 1;
            document.getElementById('trendCounter').textContent = `Trend ${{currentTrendIndex + 1}} of ${{filteredTrendData.length}}`;
            document.getElementById('trendSelect').value = currentTrendIndex;
        }}
        
        // Initialize with first trend
        displayTrend(0);
    </script>
</body>
</html>'''
    
    # Save the file
    filename = 'proper_trend_explorer.html'
    with open(filename, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Proper trend explorer saved as: {filename}")
    print(f"🎯 Browse through {len(trend_data)} trends with complete swing point details")

def main():
    """Create proper trend explorer with working swing point system"""
    
    print("🎯 PROPER TREND EXPLORER")
    print("Using WORKING swing point system with detailed trend analysis")
    print("=" * 70)
    
    # Get data using the working system
    from uso_supply_demand_visualizer import SupplyDemandVisualizer
    
    visualizer = SupplyDemandVisualizer()
    df_full = visualizer.fetch_uso_data(timeframe='daily')
    
    df = df_full.reset_index()
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})
    
    # Use the WORKING swing point system
    detector = SwingPointDetector(lookback_periods=1, equal_levels_count=False)
    swing_points = detector.detect_swing_points(df)
    
    print(f"📊 Dataset: {len(df)} candles")
    print(f"🎯 Swing points: {len(swing_points)} (using working system)")
    
    # Find ALL formations (not sequential, just all detected)
    formations = find_all_proper_trends(df, swing_points)
    terminations = find_all_proper_terminations(df, formations, swing_points)
    
    # Apply improved filtering AFTER we have termination data
    original_count = len(formations)
    formations = filter_overlapping_sideways_trends_with_terminations(formations, terminations)
    filtered_count = original_count - len(formations)
    if filtered_count > 0:
        print(f"🔄 Re-filtered {filtered_count} sideways trends using termination data")
    
    print(f"✅ Found {len(formations)} trend formations")
    print(f"✅ Found {len(terminations)} trend terminations")
    
    # Create the explorer
    create_proper_trend_explorer(df, formations, terminations, swing_points)
    
    print(f"\\n🎯 PROPER TREND EXPLORER COMPLETE!")
    print(f"   📁 File: proper_trend_explorer.html")
    print(f"   📊 Features: All {len(formations)} trends with swing points, breakouts, and terminations")
    print(f"   🔍 Navigate: Previous/Next buttons or dropdown selector")
    
    return formations, terminations

if __name__ == "__main__":
    main()