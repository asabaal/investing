#!/usr/bin/env python3
"""
Sequential Proper Trend Analysis 
Uses the CORRECT trend detection system with SEQUENTIAL processing
Combines proper trend logic with mathematical function property
"""

import plotly.graph_objects as go
import plotly.io as pio
from simple_trend_progression import find_first_breakout_candle
from trend_termination_progression import find_first_violation_candle
import json

# Dark theme
pio.templates.default = "plotly_dark"

def find_sequential_proper_trends(df, swing_points):
    """
    Find trends SEQUENTIALLY using the PROPER trend system
    Each candle index has at most one trend state
    
    Uses the correct 3-swing + breakout detection system
    """
    
    print(f"🔄 SEQUENTIAL PROPER TREND DETECTION - Processing {len(swing_points)} swing points...")
    
    sequential_formations = []
    sequential_terminations = []
    
    # Convert swing points to chronological list
    swing_list = [(sp['index'], sp['type'], sp['price']) for sp in swing_points]
    swing_list.sort(key=lambda x: x[0])  # Sort by index
    
    print(f"📊 Swing points sorted chronologically: {len(swing_list)}")
    
    # Track processing state
    current_swing_idx = 0
    active_trend = None
    last_processed_candle = -1
    
    while current_swing_idx < len(swing_list) - 2:
        
        # Skip swing points that are before our last processed candle
        if swing_list[current_swing_idx][0] <= last_processed_candle:
            current_swing_idx += 1
            continue
        
        print(f"\\n🔍 Processing swing {current_swing_idx}: Candle {swing_list[current_swing_idx][0]} ({swing_list[current_swing_idx][1]})")
        
        # If we have an active trend, monitor for termination first
        if active_trend:
            termination = check_proper_trend_termination(df, active_trend, swing_list[current_swing_idx][0])
            if termination:
                print(f"🛑 TREND TERMINATED: {active_trend['type']} at candle {termination['violation_idx']}")
                sequential_terminations.append(termination)
                last_processed_candle = termination['violation_idx']
                active_trend = None
                # Continue from this swing point to look for new trends
                continue
        
        # Look for new trend formation if no active trend
        if not active_trend:
            new_trend = attempt_proper_trend_formation(df, swing_list, current_swing_idx)
            if new_trend:
                print(f"✅ NEW TREND FORMED: {new_trend['type']} at candle {new_trend['breakout']['idx']}")
                sequential_formations.append(new_trend)
                active_trend = new_trend
                last_processed_candle = new_trend['breakout']['idx']
                # Skip ahead to avoid processing swings used in this formation
                current_swing_idx += 2  # We used 3 swings, skip to avoid reprocessing
            else:
                current_swing_idx += 1
        else:
            current_swing_idx += 1
    
    # Check if final trend needs termination
    if active_trend:
        print(f"📈 Final trend ({active_trend['type']}) remains active to end of data")
    
    print(f"\\n✅ SEQUENTIAL PROPER PROCESSING COMPLETE:")
    print(f"   📈 Formations: {len(sequential_formations)}")
    print(f"   🛑 Terminations: {len(sequential_terminations)}")
    
    return sequential_formations, sequential_terminations

def attempt_proper_trend_formation(df, swing_list, start_idx):
    """
    Attempt to form a trend using PROPER 3-swing + breakout system
    """
    
    if start_idx + 2 >= len(swing_list):
        return None
    
    swing1 = swing_list[start_idx]      # (idx, type, price)
    swing2 = swing_list[start_idx + 1]  # (idx, type, price)
    swing3 = swing_list[start_idx + 2]  # (idx, type, price)
    
    # Check for UPTREND pattern: SL → SH → SL with higher low
    if (swing1[1] == 'LOW' and swing2[1] == 'HIGH' and swing3[1] == 'LOW' and
        swing3[2] > swing1[2]):  # Higher low condition
        
        print(f"  🔍 Checking UPTREND: SL1={swing1[2]:.2f} → SH1={swing2[2]:.2f} → SL2={swing3[2]:.2f}")
        
        # Use PROPER breakout detection - find first candle that breaks above SH1
        breakout_idx = find_first_breakout_candle(df, swing3[0], swing2[2], 'above')
        
        if breakout_idx:
            formation = {
                'type': 'UPTREND',
                'sl1': {'idx': swing1[0], 'price': swing1[2]},
                'sh1': {'idx': swing2[0], 'price': swing2[2]},
                'sl2': {'idx': swing3[0], 'price': swing3[2]},
                'breakout': {'idx': breakout_idx, 'price': df.iloc[breakout_idx]['high']},
                'controlling_swing': {'idx': swing1[0], 'price': swing1[2]},
                'formation_date': df.iloc[breakout_idx]['datetime'],
                'setup_quality': 'VALID'
            }
            return formation
    
    # Check for DOWNTREND pattern: SH → SL → SH with lower high
    elif (swing1[1] == 'HIGH' and swing2[1] == 'LOW' and swing3[1] == 'HIGH' and
          swing3[2] < swing1[2]):  # Lower high condition
        
        print(f"  🔍 Checking DOWNTREND: SH1={swing1[2]:.2f} → SL1={swing2[2]:.2f} → SH2={swing3[2]:.2f}")
        
        # Use PROPER breakout detection - find first candle that breaks below SL1
        breakout_idx = find_first_breakout_candle(df, swing3[0], swing2[2], 'below')
        
        if breakout_idx:
            formation = {
                'type': 'DOWNTREND',
                'sh1': {'idx': swing1[0], 'price': swing1[2]},
                'sl1': {'idx': swing2[0], 'price': swing2[2]},
                'sh2': {'idx': swing3[0], 'price': swing3[2]},
                'breakout': {'idx': breakout_idx, 'price': df.iloc[breakout_idx]['low']},
                'controlling_swing': {'idx': swing1[0], 'price': swing1[2]},
                'formation_date': df.iloc[breakout_idx]['datetime'],
                'setup_quality': 'VALID'
            }
            return formation
    
    return None

def check_proper_trend_termination(df, active_trend, current_candle_idx):
    """
    Check if the active trend should be terminated using PROPER violation detection
    """
    
    controlling_price = active_trend['controlling_swing']['price']
    formation_idx = active_trend['breakout']['idx']
    
    # Only check from current position forward (don't re-check old candles)
    search_start = max(current_candle_idx, formation_idx + 1)  # Check immediately after breakout
    
    if active_trend['type'] == 'UPTREND':
        # Look for violation below controlling swing (SL1) using PROPER function
        violation_idx = find_first_violation_candle(df, search_start, controlling_price, 'below')
    else:
        # Look for violation above controlling swing (SH1) using PROPER function 
        violation_idx = find_first_violation_candle(df, search_start, controlling_price, 'above')
    
    if violation_idx:
        termination = {
            'formation': active_trend,
            'violation_idx': violation_idx,
            'violation_price': df.iloc[violation_idx]['low'] if active_trend['type'] == 'UPTREND' else df.iloc[violation_idx]['high'],
            'violation_date': df.iloc[violation_idx]['datetime'],
            'trend_duration': violation_idx - formation_idx,
            'violation_size': abs(controlling_price - (df.iloc[violation_idx]['low'] if active_trend['type'] == 'UPTREND' else df.iloc[violation_idx]['high']))
        }
        return termination
    
    return None

def verify_sequential_function_property(formations, terminations, df_length):
    """
    Verify that the sequential trends form a proper mathematical function
    """
    
    print(f"\\n🔍 VERIFYING MATHEMATICAL FUNCTION PROPERTY...")
    
    # Create trend state mapping for each candle
    trend_states = [None] * df_length  # None = no trend
    
    # Map terminated trends
    termination_by_formation = {id(term['formation']): term for term in terminations}
    
    overlaps_found = 0
    
    for formation in formations:
        start_idx = formation['breakout']['idx']
        formation_id = id(formation)
        
        # Determine end index
        if formation_id in termination_by_formation:
            end_idx = termination_by_formation[formation_id]['violation_idx']
        else:
            end_idx = df_length - 1  # Active trend goes to end
        
        # Check for overlaps and assign states
        for candle_idx in range(start_idx, min(end_idx + 1, df_length)):
            if trend_states[candle_idx] is not None:
                print(f"❌ OVERLAP FOUND: Candle {candle_idx} already has trend {trend_states[candle_idx]}, trying to assign {formation['type']}")
                overlaps_found += 1
            else:
                trend_states[candle_idx] = formation['type']
    
    # Report results
    if overlaps_found == 0:
        print(f"✅ FUNCTION PROPERTY VERIFIED: No overlapping trends found!")
    else:
        print(f"❌ FUNCTION PROPERTY VIOLATED: {overlaps_found} overlapping candle states found")
    
    # Statistics
    active_candles = sum(1 for state in trend_states if state is not None)
    uptrend_candles = sum(1 for state in trend_states if state == 'UPTREND')
    downtrend_candles = sum(1 for state in trend_states if state == 'DOWNTREND')
    coverage = (active_candles / df_length) * 100
    
    print(f"\\n📊 SEQUENTIAL TREND STATISTICS:")
    print(f"   Total candles: {df_length}")
    print(f"   Trend coverage: {active_candles} candles ({coverage:.1f}%)")
    print(f"   Uptrend candles: {uptrend_candles} ({uptrend_candles/df_length*100:.1f}%)")
    print(f"   Downtrend candles: {downtrend_candles} ({downtrend_candles/df_length*100:.1f}%)")
    print(f"   No-trend candles: {df_length - active_candles} ({(df_length-active_candles)/df_length*100:.1f}%)")
    
    return overlaps_found == 0

def create_sequential_proper_chart_with_controls(df, formations, terminations):
    """Create chart with all the visual controls that were lost"""
    
    print(f"\\n📊 Creating sequential proper trend chart with full controls...")
    
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
    
    # Prepare formation data for JavaScript
    formation_data = []
    termination_by_formation = {id(term['formation']): term for term in terminations}
    
    for i, formation in enumerate(formations):
        formation_key = id(formation)
        termination = termination_by_formation.get(formation_key)
        
        formation_info = {
            'id': i + 1,
            'type': formation['type'],
            'formation_idx': formation['breakout']['idx'],
            'formation_price': formation['breakout']['price'],
            'formation_date': formation['formation_date'].isoformat() if hasattr(formation['formation_date'], 'isoformat') else str(formation['formation_date']),
            'controlling_price': formation['controlling_swing']['price'],
            'sl1_idx': formation.get('sl1', {}).get('idx'),
            'sh1_idx': formation.get('sh1', {}).get('idx') if formation['type'] == 'UPTREND' else formation.get('sl1', {}).get('idx'),
            'sl2_idx': formation.get('sl2', {}).get('idx') if formation['type'] == 'UPTREND' else formation.get('sh2', {}).get('idx'),
            'status': 'Terminated' if termination else 'Active'
        }
        
        if termination:
            formation_info['termination_idx'] = termination['violation_idx']
            formation_info['violation_price'] = termination['violation_price']
            formation_info['termination_date'] = termination['violation_date'].isoformat() if hasattr(termination['violation_date'], 'isoformat') else str(termination['violation_date'])
            formation_info['duration'] = termination['trend_duration']
        else:
            formation_info['termination_idx'] = None
            formation_info['violation_price'] = None
            formation_info['termination_date'] = None
            formation_info['duration'] = len(df) - formation['breakout']['idx']
        
        formation_data.append(formation_info)
    
    # Create the HTML with ALL controls
    create_enhanced_html_with_controls(df_json, formation_data)

def create_enhanced_html_with_controls(df_data, formation_data):
    """Create the enhanced HTML with full visual controls"""
    
    min_price = min(d['low'] for d in df_data)
    max_price = max(d['high'] for d in df_data)
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Sequential Proper Trend Analysis - Full Controls</title>
    <script src="https://cdn.plot.ly/plotly-2.25.2.min.js"></script>
    <style>
        body {{ 
            background-color: #1e1e1e; 
            color: white; 
            font-family: Arial, sans-serif; 
            margin: 0;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #00ff88;
            margin-bottom: 10px;
        }}
        .controls-panel {{
            background: rgba(0,0,0,0.6);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 2px solid #333;
        }}
        
        /* Price Range Controls */
        .price-controls {{
            margin-bottom: 30px;
            padding: 15px;
            background: rgba(0,100,0,0.1);
            border-radius: 8px;
            border: 1px solid #00ff88;
        }}
        .price-controls h3 {{
            margin: 0 0 15px 0;
            color: #00ff88;
        }}
        .price-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 200px;
            gap: 20px;
            align-items: end;
        }}
        
        /* Date Range Controls */
        .date-controls {{
            margin-bottom: 30px;
            padding: 15px;
            background: rgba(100,50,0,0.1);
            border-radius: 8px;
            border: 1px solid #ffaa00;
        }}
        .date-controls h3 {{
            margin: 0 0 15px 0;
            color: #ffaa00;
        }}
        .date-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr 200px;
            gap: 20px;
            align-items: end;
        }}
        
        /* Statistics Panel */
        .stats-panel {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-box {{
            background: rgba(0,50,100,0.1);
            border: 2px solid #007bff;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-number {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .stat-label {{ font-size: 14px; color: #ccc; }}
        
        /* Buttons */
        .control-btn {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }}
        .control-btn:hover {{ 
            background: #0056b3; 
            transform: translateY(-1px);
        }}
        
        /* Inputs */
        input[type="range"] {{
            width: 100%;
            margin: 5px 0;
        }}
        label {{
            color: white;
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }}
        
        #chart {{ 
            margin-top: 20px;
            min-height: 700px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Sequential Proper Trend Analysis</h1>
        <p>Mathematical Function Property - Each Candle Has At Most One Trend State</p>
    </div>
    
    <div class="controls-panel">
        
        <!-- Statistics Panel -->
        <div class="stats-panel">
            <div class="stat-box">
                <div class="stat-number" id="total-trends">{len(formation_data)}</div>
                <div class="stat-label">Total Trends</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" id="uptrends">-</div>
                <div class="stat-label">Uptrends</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" id="downtrends">-</div>
                <div class="stat-label">Downtrends</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" id="coverage">-</div>
                <div class="stat-label">Coverage %</div>
            </div>
        </div>
        
        <!-- Price Range Controls -->
        <div class="price-controls">
            <h3>💰 Price Range Filter</h3>
            <div class="price-grid">
                <div>
                    <label>Min Price: $<span id="minPriceValue">{min_price:.2f}</span></label>
                    <input type="range" id="minPriceSlider" 
                           min="{min_price:.2f}" max="{max_price:.2f}" 
                           value="{min_price:.2f}" step="0.1">
                </div>
                <div>
                    <label>Max Price: $<span id="maxPriceValue">{max_price:.2f}</span></label>
                    <input type="range" id="maxPriceSlider" 
                           min="{min_price:.2f}" max="{max_price:.2f}" 
                           value="{max_price:.2f}" step="0.1">
                </div>
                <div>
                    <button class="control-btn" onclick="resetPriceRange()">Reset Price Range</button>
                </div>
            </div>
        </div>
        
        <!-- Date Range Controls -->
        <div class="date-controls">
            <h3>📅 Date Range Controls</h3>
            <div class="date-grid">
                <div>
                    <label>Timeline Zoom: <span id="zoomLevelValue">100%</span></label>
                    <input type="range" id="dateZoomSlider" 
                           min="5" max="100" value="100" step="5">
                    <div style="font-size: 12px; color: #ccc; margin-top: 5px;">
                        ← More Focused | Full Timeline →
                    </div>
                </div>
                <div>
                    <label>Position: <span id="timelinePosition">50%</span></label>
                    <input type="range" id="timelinePositionSlider" 
                           min="0" max="100" value="50" step="1">
                </div>
                <div>
                    <button class="control-btn" onclick="resetDateZoom()">Reset Date Range</button>
                </div>
            </div>
        </div>
        
    </div>
    
    <div id="chart"></div>
    
    <script>
        // Data from Python
        const candleData = {json.dumps(df_data)};
        const trendData = {json.dumps(formation_data, default=str)};
        
        // Current view state
        let currentPriceMin = {min_price};
        let currentPriceMax = {max_price};
        let currentZoom = 100;
        let currentPosition = 50;
        
        function updateChart() {{
            // Calculate visible candle range
            const totalCandles = candleData.length;
            const visibleCandles = Math.max(10, Math.floor(totalCandles * currentZoom / 100));
            const startIdx = Math.floor((totalCandles - visibleCandles) * currentPosition / 100);
            const endIdx = Math.min(startIdx + visibleCandles, totalCandles);
            
            // Get visible data
            const visibleData = candleData.slice(startIdx, endIdx);
            
            // Filter trends that are visible in this range
            const visibleTrends = trendData.filter(trend => {{
                const trendStart = trend.formation_idx;
                const trendEnd = trend.termination_idx || totalCandles - 1;
                return (trendStart < endIdx && trendEnd >= startIdx);
            }});
            
            // Update statistics
            updateStatistics(visibleTrends);
            
            // Create chart traces
            const traces = [];
            
            // Candlestick trace
            traces.push({{
                type: 'candlestick',
                x: visibleData.map(d => d.datetime),
                open: visibleData.map(d => d.open),
                high: visibleData.map(d => d.high),
                low: visibleData.map(d => d.low),
                close: visibleData.map(d => d.close),
                name: 'USO',
                increasing: {{ line: {{ color: '#00ff88' }} }},
                decreasing: {{ line: {{ color: '#ff4444' }} }}
            }});
            
            // Add trend lines
            visibleTrends.forEach((trend, i) => {{
                const trendStart = Math.max(0, trend.formation_idx - startIdx);
                const trendEnd = trend.termination_idx ? 
                    Math.min(visibleData.length - 1, trend.termination_idx - startIdx) : 
                    visibleData.length - 1;
                
                if (trendStart < visibleData.length && trendEnd >= 0) {{
                    const color = trend.type === 'UPTREND' ? '#00ff00' : '#ff0000';
                    const startDate = visibleData[Math.max(0, trendStart)].datetime;
                    const endDate = visibleData[Math.min(visibleData.length - 1, trendEnd)].datetime;
                    
                    traces.push({{
                        type: 'scatter',
                        mode: 'lines',
                        x: [startDate, endDate],
                        y: [trend.controlling_price, trend.violation_price || trend.controlling_price],
                        line: {{
                            color: color,
                            width: 3,
                            dash: trend.status === 'Active' ? 'dash' : 'solid'
                        }},
                        name: `${{trend.type}} #${{trend.id}}`,
                        hovertemplate: `<b>${{trend.type}} #${{trend.id}}</b><br>` +
                                     `Status: ${{trend.status}}<br>` +
                                     `Duration: ${{trend.duration}} candles<br>` +
                                     `Control: $${{trend.controlling_price.toFixed(2)}}<extra></extra>`
                    }});
                    
                    // Add formation marker
                    const formationIdx = trend.formation_idx - startIdx;
                    if (formationIdx >= 0 && formationIdx < visibleData.length) {{
                        traces.push({{
                            type: 'scatter',
                            mode: 'markers',
                            x: [visibleData[formationIdx].datetime],
                            y: [trend.formation_price],
                            marker: {{
                                color: color,
                                size: 12,
                                symbol: 'diamond',
                                line: {{ color: 'white', width: 2 }}
                            }},
                            name: `${{trend.type}} Origin`,
                            showlegend: false,
                            hovertemplate: `<b>${{trend.type}} Formation</b><br>` +
                                         `Origin: $${{trend.formation_price.toFixed(2)}}<br>` +
                                         `Date: ${{trend.formation_date}}<extra></extra>`
                        }});
                    }}
                    
                    // Add termination marker if exists and visible
                    if (trend.termination_idx && trend.termination_idx >= startIdx && trend.termination_idx < endIdx) {{
                        const termIdx = trend.termination_idx - startIdx;
                        traces.push({{
                            type: 'scatter',
                            mode: 'markers',
                            x: [visibleData[termIdx].datetime],
                            y: [trend.violation_price],
                            marker: {{
                                color: '#ff6600',
                                size: 15,
                                symbol: 'x',
                                line: {{ color: 'white', width: 2 }}
                            }},
                            name: 'Termination',
                            showlegend: false,
                            hovertemplate: `<b>Trend Termination</b><br>` +
                                         `Violation: $${{trend.violation_price.toFixed(2)}}<br>` +
                                         `Date: ${{trend.termination_date}}<extra></extra>`
                        }});
                    }}
                }}
            }});
            
            const layout = {{
                title: {{
                    text: `Sequential Proper Trends: ${{visibleTrends.length}} Visible (Mathematical Function)`,
                    font: {{ size: 18, color: 'white' }},
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
                    range: [currentPriceMin, currentPriceMax]
                }},
                template: 'plotly_dark',
                height: 700,
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
        }}
        
        function updateStatistics(visibleTrends) {{
            const uptrends = visibleTrends.filter(t => t.type === 'UPTREND').length;
            const downtrends = visibleTrends.filter(t => t.type === 'DOWNTREND').length;
            
            // Calculate coverage (simplified)
            let totalCoverage = 0;
            visibleTrends.forEach(trend => {{
                totalCoverage += trend.duration;
            }});
            const avgCoverage = visibleTrends.length > 0 ? Math.round(totalCoverage / visibleTrends.length) : 0;
            
            document.getElementById('total-trends').textContent = visibleTrends.length;
            document.getElementById('uptrends').textContent = uptrends;
            document.getElementById('downtrends').textContent = downtrends;
            document.getElementById('coverage').textContent = avgCoverage + ' avg';
        }}
        
        function updatePriceRange() {{
            const minSlider = document.getElementById('minPriceSlider');
            const maxSlider = document.getElementById('maxPriceSlider');
            
            currentPriceMin = parseFloat(minSlider.value);
            currentPriceMax = parseFloat(maxSlider.value);
            
            // Ensure min <= max
            if (currentPriceMin > currentPriceMax) {{
                currentPriceMax = currentPriceMin + 0.1;
                maxSlider.value = currentPriceMax;
            }}
            
            document.getElementById('minPriceValue').textContent = currentPriceMin.toFixed(2);
            document.getElementById('maxPriceValue').textContent = currentPriceMax.toFixed(2);
            
            updateChart();
        }}
        
        function updateDateRange() {{
            const zoomSlider = document.getElementById('dateZoomSlider');
            const positionSlider = document.getElementById('timelinePositionSlider');
            
            currentZoom = parseInt(zoomSlider.value);
            currentPosition = parseInt(positionSlider.value);
            
            document.getElementById('zoomLevelValue').textContent = currentZoom + '%';
            document.getElementById('timelinePosition').textContent = currentPosition + '%';
            
            updateChart();
        }}
        
        function resetPriceRange() {{
            currentPriceMin = {min_price};
            currentPriceMax = {max_price};
            
            document.getElementById('minPriceSlider').value = currentPriceMin;
            document.getElementById('maxPriceSlider').value = currentPriceMax;
            document.getElementById('minPriceValue').textContent = currentPriceMin.toFixed(2);
            document.getElementById('maxPriceValue').textContent = currentPriceMax.toFixed(2);
            
            updateChart();
        }}
        
        function resetDateZoom() {{
            currentZoom = 100;
            currentPosition = 50;
            
            document.getElementById('dateZoomSlider').value = 100;
            document.getElementById('timelinePositionSlider').value = 50;
            document.getElementById('zoomLevelValue').textContent = '100%';
            document.getElementById('timelinePosition').textContent = '50%';
            
            updateChart();
        }}
        
        // Event listeners
        document.getElementById('minPriceSlider').addEventListener('input', updatePriceRange);
        document.getElementById('maxPriceSlider').addEventListener('input', updatePriceRange);
        document.getElementById('dateZoomSlider').addEventListener('input', updateDateRange);
        document.getElementById('timelinePositionSlider').addEventListener('input', updateDateRange);
        
        // Initialize chart
        updateChart();
    </script>
</body>
</html>'''
    
    # Save the HTML file
    filename = 'sequential_proper_trends.html'
    with open(filename, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Sequential proper trend chart with full controls saved as: {filename}")

def main():
    """Run the sequential proper trend analysis"""
    
    print("🎯 SEQUENTIAL PROPER TREND ANALYSIS")
    print("Using CORRECT trend system with SEQUENTIAL processing")
    print("=" * 70)
    
    # Get data using the proper system
    from uso_supply_demand_visualizer import SupplyDemandVisualizer
    from swing_point_detector import SwingPointDetector
    
    visualizer = SupplyDemandVisualizer()
    df_full = visualizer.fetch_uso_data(timeframe='daily')
    
    df = df_full.reset_index()
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})
    
    # Get swing points using the WORKING swing point system
    detector = SwingPointDetector(lookback_periods=1, equal_levels_count=False)
    swing_points = detector.detect_swing_points(df)
    
    print(f"🎯 Using WORKING swing point system:")
    print(f"   Swing points found: {len(swing_points)}")
    if swing_points:
        swing_highs = [s for s in swing_points if s['type'] == 'HIGH']
        swing_lows = [s for s in swing_points if s['type'] == 'LOW']
        print(f"   • Swing highs: {len(swing_highs)}")
        print(f"   • Swing lows: {len(swing_lows)}")
        
        # Show first few to verify
        print(f"   First 5 swing points:")
        for i, swing in enumerate(swing_points[:5]):
            print(f"     {i+1}. Candle {swing['index']}: {swing['type']} at ${swing['price']:.2f}")
    else:
        print("   ❌ No swing points found!")
    
    print(f"📊 Dataset: {len(df)} candles, {len(swing_points)} swing points")
    
    # Run sequential PROPER detection
    formations, terminations = find_sequential_proper_trends(df, swing_points)
    
    # Verify function property
    is_function = verify_sequential_function_property(formations, terminations, len(df))
    
    # Create visualization with ALL controls
    create_sequential_proper_chart_with_controls(df, formations, terminations)
    
    print(f"\\n🎯 SEQUENTIAL PROPER ANALYSIS RESULTS:")
    print(f"   Mathematical function: {'✅ YES' if is_function else '❌ NO'}")
    print(f"   Formations: {len(formations)}")
    print(f"   Terminations: {len(terminations)}")
    print(f"   Active trends: {len(formations) - len(terminations)}")
    print(f"   📁 Full controls available in: sequential_proper_trends.html")
    
    return formations, terminations

if __name__ == "__main__":
    main()