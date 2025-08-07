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
    """
    
    formations = []
    
    # Convert swing points to easier format
    swing_list = [(sp['index'], sp['type'], sp['price']) for sp in swing_points]
    
    print(f"🔍 Scanning {len(swing_list)} swing points for trend formations...")
    print(f"   ✅ Adding validation: swing points must remain valid until breakout")
    
    valid_count = 0
    invalid_count = 0
    
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
    
    print(f"✅ Validation complete: {valid_count} valid formations, {invalid_count} invalidated")
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
    
    # Initial controlling swing should be SL2 for uptrends, SH2 for downtrends
    if formation['type'] == 'UPTREND':
        original_controlling_price = formation['sl2']['price']  # SL2 is the setup swing for uptrends
        current_controlling_idx = formation['sl2']['idx']
    else:
        original_controlling_price = formation['sh2']['price']  # SH2 is the setup swing for downtrends  
        current_controlling_idx = formation['sh2']['idx']
    
    current_controlling_price = original_controlling_price
    formation_idx = formation['breakout']['idx']
    
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
        breakout_idx = formation['breakout']['idx']
        
        # Get swing point indices
        if formation['type'] == 'UPTREND':
            swing_indices = [
                formation['sl1']['idx'],
                formation['sh1']['idx'], 
                formation['sl2']['idx']
            ]
        else:
            swing_indices = [
                formation['sh1']['idx'],
                formation['sl1']['idx'],
                formation['sh2']['idx']
            ]
        
        # Window should show: context before first swing + formation + context after
        first_swing_idx = min(swing_indices)
        
        if termination:
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
        else:
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
        
        trend_info = {
            'id': i + 1,
            'type': formation['type'],
            'formation': formation,
            'termination': termination,
            'window_start': window_start,
            'window_end': window_end,
            'breakout_idx': breakout_idx,
            'swing_points': swing_points,
            'controlling_swing': formation['controlling_swing'],
            'duration': termination['trend_duration'] if termination else (len(df) - breakout_idx),
            'formation_date': formation['formation_date'].strftime('%Y-%m-%d') if hasattr(formation['formation_date'], 'strftime') else str(formation['formation_date']),
            'status': 'Terminated' if termination else 'Active',
            # Add controlling swing update info
            'original_controlling': formation['controlling_swing']['price'] if termination else formation['controlling_swing']['price'],
            'final_controlling': termination.get('final_controlling_price') if termination else formation['controlling_swing']['price'],
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
        const trendData = {json.dumps(trend_data, default=str)};
        let currentTrendIndex = 0;
        
        function loadTrend() {{
            const select = document.getElementById('trendSelect');
            currentTrendIndex = parseInt(select.value);
            displayTrend(currentTrendIndex);
        }}
        
        function displayTrend(index) {{
            const trend = trendData[index];
            const windowData = dfData.slice(trend.window_start, trend.window_end + 1);
            
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
            const controlPrice = trend.controlling_swing.price;
            const startDate = windowData[0].datetime;
            const endDate = windowData[windowData.length - 1].datetime;
            
            traces.push({{
                type: 'scatter',
                mode: 'lines',
                x: [startDate, endDate],
                y: [controlPrice, controlPrice],
                line: {{
                    color: trend.type === 'UPTREND' ? '#00ff00' : '#ff0000',
                    width: 2,
                    dash: trend.status === 'Terminated' ? 'solid' : 'dash'
                }},
                name: 'Control Level',
                showlegend: true,
                hovertemplate: `<b>Controlling Level</b><br>Price: $${{controlPrice.toFixed(2)}}<br>Status: ${{trend.status}}<extra></extra>`
            }});
            
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
                    gridcolor: 'rgba(100,100,100,0.2)'
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
            
            Plotly.newPlot('chart', traces, layout);
            
            // Update navigation
            document.getElementById('prevBtn').disabled = index === 0;
            document.getElementById('nextBtn').disabled = index === trendData.length - 1;
            document.getElementById('trendCounter').textContent = `Trend ${{index + 1}} of ${{trendData.length}}`;
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
                <div class="info-item"><strong>Original Control:</strong> $${{trend.original_controlling.toFixed(2)}}</div>
                <div class="info-item"><strong>Final Control:</strong> $${{trend.final_controlling.toFixed(2)}} (${{trend.controlling_updates}} updates)</div>
            `;
            
            // Add swing point details
            detailsHTML += '<div class="swing-details"><strong>Swing Points:</strong><br>';
            trend.swing_points.forEach(sp => {{
                detailsHTML += `${{sp.type}}: $${{sp.price.toFixed(2)}} (${{sp.role}})<br>`;
            }});
            detailsHTML += `Breakout: Candle ${{trend.breakout_idx}}</div>`;
            
            if (trend.termination) {{
                detailsHTML += `
                    <div class="info-item"><strong>Termination:</strong> ${{trend.termination_date}}</div>
                    <div class="info-item"><strong>Violation:</strong> $${{trend.violation_price.toFixed(2)}} ($${{trend.violation_size.toFixed(2)}} break)</div>
                `;
            }}
            
            detailsDiv.innerHTML = detailsHTML;
        }}
        
        function previousTrend() {{
            if (currentTrendIndex > 0) {{
                currentTrendIndex--;
                displayTrend(currentTrendIndex);
            }}
        }}
        
        function nextTrend() {{
            if (currentTrendIndex < trendData.length - 1) {{
                currentTrendIndex++;
                displayTrend(currentTrendIndex);
            }}
        }}
        
        function jumpToRandom() {{
            currentTrendIndex = Math.floor(Math.random() * trendData.length);
            displayTrend(currentTrendIndex);
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