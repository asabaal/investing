#!/usr/bin/env python3
"""
Intuitive Trend Visualizer
Clean, user-friendly trend visualization with:
- Intuitive slider to control time window
- Shows only trends with origins in the visible window
- Smooth, easy-to-use interface
"""

import plotly.graph_objects as go
import plotly.io as pio
from proper_trend_explorer import find_all_proper_trends, find_all_proper_terminations, filter_overlapping_sideways_trends_with_terminations
from swing_point_detector import SwingPointDetector
from uso_supply_demand_visualizer import SupplyDemandVisualizer
import json
from datetime import datetime, timedelta

# Dark theme
pio.templates.default = "plotly_dark"

def create_intuitive_trend_visualizer():
    """
    Create an intuitive trend visualizer with proper slider controls
    """
    
    print("🎯 INTUITIVE TREND VISUALIZER")
    print("Creating user-friendly interface with slider controls")
    print("=" * 60)
    
    # Get the data
    visualizer = SupplyDemandVisualizer()
    df_full = visualizer.fetch_uso_data(timeframe='daily')
    df = df_full.reset_index()
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})
    
    # Get swing points using the working system
    detector = SwingPointDetector(lookback_periods=1, equal_levels_count=False)
    swing_points = detector.detect_swing_points(df)
    
    print(f"📊 Dataset: {len(df)} candles, {len(swing_points)} swing points")
    
    # Find all trend formations
    formations = find_all_proper_trends(df, swing_points)
    terminations = find_all_proper_terminations(df, formations, swing_points)
    
    # Apply improved filtering AFTER we have termination data (same as trend explorer and CSV analytics)
    original_count = len(formations)
    formations = filter_overlapping_sideways_trends_with_terminations(formations, terminations)
    filtered_count = original_count - len(formations)
    if filtered_count > 0:
        print(f"🔄 Filtered out {filtered_count} overlapping sideways trends")
    
    print(f"✅ Found {len(formations)} trend formations")
    print(f"   • {len([f for f in formations if f['type'] == 'UPTREND'])} uptrends")
    print(f"   • {len([f for f in formations if f['type'] == 'DOWNTREND'])} downtrends") 
    print(f"   • {len([f for f in formations if f['type'] == 'SIDEWAYS'])} sideways trends")
    
    # Prepare data for JavaScript
    df_json = df.to_dict('records')
    for i, record in enumerate(df_json):
        record['index'] = i
        if 'datetime' in record:
            record['datetime'] = record['datetime'].isoformat() if hasattr(record['datetime'], 'isoformat') else str(record['datetime'])
    
    # Prepare trend data
    trend_data = []
    for i, formation in enumerate(formations):
        # Get breakout index based on trend type
        if formation['type'] == 'SIDEWAYS':
            breakout_idx = formation['start_swing']['idx']
        else:
            breakout_idx = get_breakout_index(formation)
        
        # Find termination if exists
        termination = None
        for term in terminations:
            if term['formation'] == formation:
                termination = term
                break
        
        # Clean formation data for JSON serialization
        clean_formation = {}
        for key, value in formation.items():
            if hasattr(value, 'strftime'):  # datetime objects
                clean_formation[key] = value.strftime('%Y-%m-%d')
            elif isinstance(value, dict):
                # Clean nested dicts
                clean_dict = {}
                for k, v in value.items():
                    if hasattr(v, 'strftime'):
                        clean_dict[k] = v.strftime('%Y-%m-%d') 
                    else:
                        clean_dict[k] = v
                clean_formation[key] = clean_dict
            else:
                clean_formation[key] = value

        trend_info = {
            'id': i + 1,
            'type': formation['type'],
            'breakout_idx': breakout_idx,
            'formation_date': formation['formation_date'].strftime('%Y-%m-%d') if hasattr(formation['formation_date'], 'strftime') else str(formation['formation_date']),
            'termination_idx': termination['violation_idx'] if termination else None,
            'violation_price': termination['violation_price'] if termination else None,
            'formation': clean_formation
        }
        
        trend_data.append(trend_info)
    
    # Create the HTML interface
    html_content = create_html_interface(df_json, trend_data, swing_points)
    
    # Save the visualizer
    with open('intuitive_trend_visualizer.html', 'w') as f:
        f.write(html_content)
    
    print(f"✅ Intuitive trend visualizer saved as: intuitive_trend_visualizer.html")
    print(f"🎯 Features:")
    print(f"   • Intuitive slider to control time window")
    print(f"   • Shows trends with origins in visible window")
    print(f"   • Clean, user-friendly interface")
    print(f"   • Smooth drag-to-explore interaction")
    
def get_breakout_index(formation):
    """Get the breakout index for directional trends"""
    if formation['type'] == 'UPTREND':
        return formation.get('sl1', {}).get('idx', 0)  # Uptrends start from swing LOW
    elif formation['type'] == 'DOWNTREND':
        return formation.get('sh1', {}).get('idx', 0)  # Downtrends start from swing HIGH
    return 0

def create_html_interface(df_data, trend_data, swing_points):
    """Create the HTML interface with intuitive controls"""
    
    # Convert swing points to JSON-serializable format
    swing_data = []
    for sp in swing_points:
        swing_data.append({
            'index': sp['index'],
            'type': sp['type'], 
            'price': sp['price']
        })
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Intuitive Trend Visualizer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            background: #1a1a1a; 
            color: white; 
            margin: 0; 
            padding: 20px; 
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        
        .controls {{
            background: #2a2a2a;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }}
        
        .slider-container {{
            margin: 20px 0;
        }}
        
        .slider-label {{
            font-weight: bold;
            margin-bottom: 10px;
            display: block;
        }}
        
        .time-slider {{
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: #444;
            outline: none;
            -webkit-appearance: none;
            margin: 10px 0;
        }}
        
        .time-slider::-webkit-slider-thumb {{
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #00ff88;
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }}
        
        .time-slider::-moz-range-thumb {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #00ff88;
            cursor: pointer;
            border: none;
        }}
        
        .window-info {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 10px 0;
            font-size: 14px;
        }}
        
        .trend-counter {{
            background: #333;
            padding: 10px 15px;
            border-radius: 5px;
            display: inline-block;
        }}
        
        .chart-container {{
            background: #2a2a2a;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }}
        
        .trend-type-filter {{
            margin: 10px 0;
        }}
        
        .filter-checkbox {{
            margin: 0 15px 0 5px;
        }}
        
        .filter-label {{
            margin-right: 20px;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Intuitive Trend Visualizer</h1>
            <p>Drag the slider to explore different time periods • Only trends starting in the visible window are shown</p>
        </div>
        
        <div class="controls">
            <div class="slider-container">
                <label class="slider-label">Time Window (Drag to explore different periods)</label>
                <input type="range" id="windowSlider" class="time-slider" min="0" max="800" value="0" step="1">
                
                <div class="window-info">
                    <span id="windowStart">Start: 2006-05-02</span>
                    <span id="windowEnd">End: 2008-05-02</span>
                </div>
            </div>
            
            <div class="slider-container">
                <label class="slider-label">Price Window Position (Drag to move up/down through price levels)</label>
                <input type="range" id="pricePositionSlider" class="time-slider" min="0" max="100" value="50" step="1">
                
                <label class="slider-label" style="margin-top: 15px;">Price Window Size (Drag to zoom in/out)</label>
                <input type="range" id="priceZoomSlider" class="time-slider" min="10" max="100" value="100" step="1">
                
                <div class="window-info">
                    <span id="priceRangeInfo">Price range: Auto-fit</span>
                    <button id="resetPriceZoom" style="background: #444; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer; margin-left: 10px;">Reset</button>
                </div>
            </div>
            
            <div class="trend-type-filter">
                <strong>Show Trend Types:</strong>
                <label class="filter-label">
                    <input type="checkbox" id="showUptrends" class="filter-checkbox" checked> 
                    ↗️ Uptrends
                </label>
                <label class="filter-label">
                    <input type="checkbox" id="showDowntrends" class="filter-checkbox" checked> 
                    ↘️ Downtrends  
                </label>
                <label class="filter-label">
                    <input type="checkbox" id="showSideways" class="filter-checkbox" checked> 
                    ↔️ Sideways
                </label>
            </div>
            
            <div class="trend-counter" id="trendCounter">
                Loading trends...
            </div>
        </div>
        
        <div class="chart-container">
            <div id="chart" style="height: 700px;"></div>
        </div>
    </div>

    <script>
        // Data
        const dfData = {json.dumps(df_data)};
        const trendData = {json.dumps(trend_data)};
        const swingData = {json.dumps(swing_data)};
        
        // Chart state
        let currentWindowStart = 0;
        let currentWindowSize = 500; // Show 500 candles at a time
        let pricePosition = 50; // 0-100, position within full price range
        let priceZoomLevel = 100; // 10-100, percentage of full range to show
        let manualPriceRange = null;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            setupSlider();
            setupPriceZoom();
            updateChart();
            setupFilters();
        }});
        
        function setupSlider() {{
            const slider = document.getElementById('windowSlider');
            const maxStart = Math.max(0, dfData.length - currentWindowSize);
            slider.max = maxStart;
            
            slider.addEventListener('input', function() {{
                currentWindowStart = parseInt(this.value);
                updateChart();
                updateWindowInfo();
            }});
            
            updateWindowInfo();
        }}
        
        function setupPriceZoom() {{
            const pricePositionSlider = document.getElementById('pricePositionSlider');
            const priceZoomSlider = document.getElementById('priceZoomSlider');
            const resetButton = document.getElementById('resetPriceZoom');
            
            pricePositionSlider.addEventListener('input', function() {{
                pricePosition = parseInt(this.value);
                updateChart();
                updatePriceRangeInfo();
            }});
            
            priceZoomSlider.addEventListener('input', function() {{
                priceZoomLevel = parseInt(this.value);
                updateChart();
                updatePriceRangeInfo();
            }});
            
            resetButton.addEventListener('click', function() {{
                pricePosition = 50;
                priceZoomLevel = 100;
                pricePositionSlider.value = 50;
                priceZoomSlider.value = 100;
                manualPriceRange = null;
                updateChart();
                updatePriceRangeInfo();
            }});
        }}
        
        function setupFilters() {{
            ['showUptrends', 'showDowntrends', 'showSideways'].forEach(id => {{
                document.getElementById(id).addEventListener('change', updateChart);
            }});
        }}
        
        function updateWindowInfo() {{
            const startCandle = dfData[currentWindowStart];
            const endCandle = dfData[Math.min(currentWindowStart + currentWindowSize - 1, dfData.length - 1)];
            
            document.getElementById('windowStart').textContent = 
                `Start: ${{new Date(startCandle.datetime).toLocaleDateString()}}`;
            document.getElementById('windowEnd').textContent = 
                `End: ${{new Date(endCandle.datetime).toLocaleDateString()}}`;
        }}
        
        function updatePriceRangeInfo() {{
            if (priceZoomLevel === 100) {{
                document.getElementById('priceRangeInfo').textContent = 'Price range: Auto-fit';
            }} else {{
                const windowEnd = Math.min(currentWindowStart + currentWindowSize, dfData.length);
                const windowData = dfData.slice(currentWindowStart, windowEnd);
                const fullMinPrice = Math.min(...windowData.map(d => d.low));
                const fullMaxPrice = Math.max(...windowData.map(d => d.high));
                const fullRange = fullMaxPrice - fullMinPrice;
                
                // Calculate zoom window size as percentage of full range
                const zoomWindowSize = fullRange * (priceZoomLevel / 100);
                
                // Calculate position within the available range
                const availableRange = fullRange - zoomWindowSize;
                const positionOffset = availableRange * (pricePosition / 100);
                
                const zoomedMin = fullMinPrice + positionOffset;
                const zoomedMax = zoomedMin + zoomWindowSize;
                
                document.getElementById('priceRangeInfo').textContent = 
                    `Price range: $${{zoomedMin.toFixed(2)}} - $${{zoomedMax.toFixed(2)}}`;
            }}
        }}
        
        function getVisibleTrends() {{
            const windowEnd = currentWindowStart + currentWindowSize;
            
            // Filter trends that have breakout (origin) within the visible window
            const visibleTrends = trendData.filter(trend => {{
                return trend.breakout_idx >= currentWindowStart && trend.breakout_idx < windowEnd;
            }});
            
            // Apply type filters
            const showUptrends = document.getElementById('showUptrends').checked;
            const showDowntrends = document.getElementById('showDowntrends').checked;
            const showSideways = document.getElementById('showSideways').checked;
            
            const typeFilteredTrends = visibleTrends.filter(trend => {{
                if (trend.type === 'UPTREND') return showUptrends;
                if (trend.type === 'DOWNTREND') return showDowntrends;
                if (trend.type === 'SIDEWAYS') return showSideways;
                return true;
            }});
            
            // ADDITIONAL FILTERING: Remove overlapping trends, keep only the longest one
            return filterOverlappingTrendsInWindow(typeFilteredTrends);
        }}
        
        function filterOverlappingTrendsInWindow(trends) {{
            // For each candle position in the window, find which trends cover it
            // Then keep only the longest trend for overlapping ranges
            
            const filteredTrends = [];
            const processedTrends = new Set();
            
            // Group trends by type to handle separately
            const trendsByType = {{
                'UPTREND': trends.filter(t => t.type === 'UPTREND'),
                'DOWNTREND': trends.filter(t => t.type === 'DOWNTREND'),  
                'SIDEWAYS': trends.filter(t => t.type === 'SIDEWAYS')
            }};
            
            // Process each type separately
            ['UPTREND', 'DOWNTREND', 'SIDEWAYS'].forEach(trendType => {{
                const typeTrends = trendsByType[trendType];
                
                typeTrends.forEach(trend => {{
                    if (processedTrends.has(trend.id)) return;
                    
                    // Find all trends that overlap with this one
                    const overlappingTrends = typeTrends.filter(other => {{
                        if (other.id === trend.id) return true;
                        
                        // Calculate overlap ranges
                        const trend1Start = trend.breakout_idx;
                        const trend1End = trend.termination_idx || (currentWindowStart + currentWindowSize - 1);
                        
                        const trend2Start = other.breakout_idx;  
                        const trend2End = other.termination_idx || (currentWindowStart + currentWindowSize - 1);
                        
                        // Check if ranges overlap
                        return !(trend1End < trend2Start || trend2End < trend1Start);
                    }});
                    
                    if (overlappingTrends.length === 1) {{
                        // No overlap, add this trend
                        filteredTrends.push(trend);
                        processedTrends.add(trend.id);
                    }} else {{
                        // Multiple overlapping trends - keep the longest one
                        const longestTrend = overlappingTrends.reduce((longest, current) => {{
                            const currentLength = (current.termination_idx || (currentWindowStart + currentWindowSize - 1)) - current.breakout_idx;
                            const longestLength = (longest.termination_idx || (currentWindowStart + currentWindowSize - 1)) - longest.breakout_idx;
                            return currentLength > longestLength ? current : longest;
                        }});
                        
                        if (!processedTrends.has(longestTrend.id)) {{
                            filteredTrends.push(longestTrend);
                            processedTrends.add(longestTrend.id);
                        }}
                        
                        // Mark all overlapping trends as processed
                        overlappingTrends.forEach(t => processedTrends.add(t.id));
                    }}
                }});
            }});
            
            return filteredTrends;
        }}
        
        function updateChart() {{
            const windowEnd = Math.min(currentWindowStart + currentWindowSize, dfData.length);
            const windowData = dfData.slice(currentWindowStart, windowEnd);
            const visibleTrends = getVisibleTrends();
            
            // Update trend counter with debugging info
            const rawVisibleTrends = trendData.filter(trend => {{
                return trend.breakout_idx >= currentWindowStart && trend.breakout_idx < (currentWindowStart + currentWindowSize);
            }});
            
            document.getElementById('trendCounter').textContent = 
                `${{visibleTrends.length}} trends displayed (${{rawVisibleTrends.length}} originated in window)`;
            
            // Create candlestick trace
            const traces = [];
            
            traces.push({{
                type: 'candlestick',
                x: windowData.map(d => d.datetime),
                open: windowData.map(d => d.open),
                high: windowData.map(d => d.high),
                low: windowData.map(d => d.low),
                close: windowData.map(d => d.close),
                name: 'USO Price',
                increasing: {{ line: {{ color: '#00ff88' }} }},
                decreasing: {{ line: {{ color: '#ff4444' }} }}
            }});
            
            // Add trend lines and markers
            const trendColors = {{
                'UPTREND': '#00ff88',
                'DOWNTREND': '#ff4444', 
                'SIDEWAYS': '#ffaa00'
            }};
            
            visibleTrends.forEach(trend => {{
                const color = trendColors[trend.type];
                
                if (trend.type === 'SIDEWAYS') {{
                    // For sideways trends, draw horizontal line at center of range
                    const startIdx = trend.formation.start_swing.idx;
                    const endIdx = trend.formation.end_swing.idx;
                    const highLevel = trend.formation.high_level;
                    const lowLevel = trend.formation.low_level;
                    const centerLevel = (highLevel + lowLevel) / 2;
                    
                    if (startIdx >= currentWindowStart || endIdx >= currentWindowStart) {{
                        const startCandle = dfData[Math.max(startIdx, currentWindowStart)];
                        const endCandle = dfData[Math.min(endIdx, windowEnd - 1)];
                        
                        // Horizontal trend line
                        traces.push({{
                            type: 'scatter',
                            mode: 'lines',
                            x: [startCandle.datetime, endCandle.datetime],
                            y: [centerLevel, centerLevel],
                            line: {{ color: color, width: 3, dash: 'solid' }},
                            name: `Sideways #${{trend.id}}`,
                            hovertemplate: `<b>Sideways Trend #${{trend.id}}</b><br>Range: $${{lowLevel.toFixed(2)}} - $${{highLevel.toFixed(2)}}<br>Center: $${{centerLevel.toFixed(2)}}<extra></extra>`
                        }});
                        
                        // Start marker
                        if (startIdx >= currentWindowStart && startIdx < windowEnd) {{
                            traces.push({{
                                type: 'scatter',
                                mode: 'markers+text',
                                x: [dfData[startIdx].datetime],
                                y: [centerLevel],
                                marker: {{ color: color, size: 8, symbol: 'diamond' }},
                                text: ['S'],
                                textposition: 'middle center',
                                textfont: {{ size: 8, color: 'white', family: 'Arial Black' }},
                                showlegend: false,
                                hovertemplate: `<b>Sideways Start #${{trend.id}}</b><br>Date: ${{trend.formation_date}}<extra></extra>`
                            }});
                        }}
                    }}
                }} else {{
                    // For directional trends, find most extreme point and draw line
                    const breakoutCandle = dfData[trend.breakout_idx];
                    let extremePrice, extremeDate;
                    
                    // Find the most extreme point in the trend direction
                    const trendEndIdx = trend.termination_idx || Math.min(windowEnd - 1, dfData.length - 1);
                    let mostExtreme = trend.type === 'UPTREND' ? -Infinity : Infinity;
                    let extremeIdx = trend.breakout_idx;
                    
                    for (let i = trend.breakout_idx; i <= trendEndIdx && i < dfData.length; i++) {{
                        const candle = dfData[i];
                        const price = trend.type === 'UPTREND' ? candle.high : candle.low;
                        
                        if ((trend.type === 'UPTREND' && price > mostExtreme) ||
                            (trend.type === 'DOWNTREND' && price < mostExtreme)) {{
                            mostExtreme = price;
                            extremeIdx = i;
                        }}
                    }}
                    
                    extremePrice = mostExtreme;
                    extremeDate = dfData[extremeIdx].datetime;
                    
                    // Draw trend line from origin to extreme point
                    if ((trend.breakout_idx >= currentWindowStart && trend.breakout_idx < windowEnd) ||
                        (extremeIdx >= currentWindowStart && extremeIdx < windowEnd)) {{
                        traces.push({{
                            type: 'scatter',
                            mode: 'lines',
                            x: [breakoutCandle.datetime, extremeDate],
                            y: [trend.type === 'UPTREND' ? breakoutCandle.low : breakoutCandle.high, extremePrice],
                            line: {{ color: color, width: 3, dash: 'solid' }},
                            name: `${{trend.type}} #${{trend.id}}`,
                            hovertemplate: `<b>${{trend.type}} #${{trend.id}}</b><br>Origin: $${{(trend.type === 'UPTREND' ? breakoutCandle.low : breakoutCandle.high).toFixed(2)}}<br>Extreme: $${{extremePrice.toFixed(2)}}<extra></extra>`
                        }});
                        
                        // Origin marker
                        if (trend.breakout_idx >= currentWindowStart && trend.breakout_idx < windowEnd) {{
                            traces.push({{
                                type: 'scatter',
                                mode: 'markers+text',
                                x: [breakoutCandle.datetime],
                                y: [trend.type === 'UPTREND' ? breakoutCandle.low : breakoutCandle.high],
                                marker: {{ color: color, size: 8, symbol: 'diamond' }},
                                text: [trend.type.charAt(0)],
                                textposition: 'middle center',
                                textfont: {{ size: 8, color: 'white', family: 'Arial Black' }},
                                showlegend: false,
                                hovertemplate: `<b>${{trend.type}} Origin #${{trend.id}}</b><br>Date: ${{trend.formation_date}}<extra></extra>`
                            }});
                        }}
                        
                        // Extreme point marker
                        if (extremeIdx >= currentWindowStart && extremeIdx < windowEnd) {{
                            traces.push({{
                                type: 'scatter',
                                mode: 'markers',
                                x: [extremeDate],
                                y: [extremePrice],
                                marker: {{ color: color, size: 6, symbol: 'star' }},
                                showlegend: false,
                                hovertemplate: `<b>${{trend.type}} Peak #${{trend.id}}</b><br>Price: $${{extremePrice.toFixed(2)}}<extra></extra>`
                            }});
                        }}
                    }}
                }}
                
                // Add violation marker if exists and in window
                if (trend.termination_idx && trend.termination_idx >= currentWindowStart && trend.termination_idx < windowEnd) {{
                    const violationCandle = dfData[trend.termination_idx];
                    traces.push({{
                        type: 'scatter',
                        mode: 'markers+text',
                        x: [violationCandle.datetime],
                        y: [trend.violation_price],
                        marker: {{
                            color: '#ff0000',
                            size: 10,
                            symbol: 'x',
                            line: {{ color: 'white', width: 1 }}
                        }},
                        text: ['✕'],
                        textposition: 'middle center',
                        textfont: {{ size: 8, color: '#ff0000', family: 'Arial Black' }},
                        showlegend: false,
                        hovertemplate: `<b>Trend Violation</b><br>Price: $${{trend.violation_price.toFixed(2)}}<br>Trend #${{trend.id}}<extra></extra>`
                    }});
                }}
            }});
            
            // Add smaller swing points
            const visibleSwings = swingData.filter(swing => 
                swing.index >= currentWindowStart && swing.index < windowEnd
            );
            
            const swingHighs = visibleSwings.filter(s => s.type === 'HIGH');
            const swingLows = visibleSwings.filter(s => s.type === 'LOW');
            
            if (swingHighs.length > 0) {{
                traces.push({{
                    type: 'scatter',
                    mode: 'markers',
                    x: swingHighs.map(s => dfData[s.index].datetime),
                    y: swingHighs.map(s => s.price),
                    marker: {{ color: '#ffaa00', size: 3, symbol: 'triangle-down', opacity: 0.7 }},
                    name: 'Swing Highs',
                    hovertemplate: '<b>Swing High</b><br>Price: $%{{y:.2f}}<extra></extra>'
                }});
            }}
            
            if (swingLows.length > 0) {{
                traces.push({{
                    type: 'scatter',
                    mode: 'markers', 
                    x: swingLows.map(s => dfData[s.index].datetime),
                    y: swingLows.map(s => s.price),
                    marker: {{ color: '#00aaff', size: 3, symbol: 'triangle-up', opacity: 0.7 }},
                    name: 'Swing Lows',
                    hovertemplate: '<b>Swing Low</b><br>Price: $%{{y:.2f}}<extra></extra>'
                }});
            }}
            
            // Calculate price range for Y-axis
            let yAxisRange = null;
            if (priceZoomLevel < 100) {{
                const windowEnd = Math.min(currentWindowStart + currentWindowSize, dfData.length);
                const windowData = dfData.slice(currentWindowStart, windowEnd);
                const fullMinPrice = Math.min(...windowData.map(d => d.low));
                const fullMaxPrice = Math.max(...windowData.map(d => d.high));
                const fullRange = fullMaxPrice - fullMinPrice;
                
                // Calculate zoom window size as percentage of full range
                const zoomWindowSize = fullRange * (priceZoomLevel / 100);
                
                // Calculate position within the available range
                const availableRange = fullRange - zoomWindowSize;
                const positionOffset = availableRange * (pricePosition / 100);
                
                const zoomedMin = fullMinPrice + positionOffset;
                const zoomedMax = zoomedMin + zoomWindowSize;
                
                // Add small padding for visual comfort
                const padding = zoomWindowSize * 0.02;
                yAxisRange = [zoomedMin - padding, zoomedMax + padding];
            }}
            
            // Layout
            const layout = {{
                title: {{
                    text: `USO Trend Analysis - ${{visibleTrends.length}} trends in window`,
                    font: {{ size: 18, color: 'white' }}
                }},
                xaxis: {{
                    title: 'Date',
                    gridcolor: '#444',
                    color: 'white'
                }},
                yaxis: {{
                    title: 'Price ($)',
                    gridcolor: '#444', 
                    color: 'white',
                    range: yAxisRange
                }},
                plot_bgcolor: '#1a1a1a',
                paper_bgcolor: '#2a2a2a',
                font: {{ color: 'white' }},
                legend: {{ 
                    bgcolor: 'rgba(42,42,42,0.8)',
                    bordercolor: '#666',
                    borderwidth: 1
                }},
                margin: {{ t: 80, b: 50, l: 60, r: 60 }}
            }};
            
            Plotly.newPlot('chart', traces, layout, {{
                displayModeBar: true,
                modeBarButtonsToRemove: ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
                displaylogo: false
            }});
        }}
    </script>
</body>
</html>
    """
    
    return html_content

if __name__ == "__main__":
    create_intuitive_trend_visualizer()