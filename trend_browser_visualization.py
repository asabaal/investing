#!/usr/bin/env python3
"""
Trend Browser Visualization
Shows each individual trend formation/termination in separate time windows
for detailed analysis of what's happening
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import json

# Dark theme
pio.templates.default = "plotly_dark"

def create_trend_browser(df, formations, terminations):
    """
    Create a browseable HTML file showing each trend in its own time window
    """
    
    print(f"🎯 Creating trend browser for {len(formations)} formations...")
    
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
        
        # Determine time window around the trend
        breakout_idx = formation['breakout']['idx']
        
        if termination:
            end_idx = termination['violation_idx']
            trend_duration = end_idx - breakout_idx
            # Show some context before and after
            window_start = max(0, breakout_idx - max(10, trend_duration // 2))
            window_end = min(len(df) - 1, end_idx + max(10, trend_duration // 2))
        else:
            # Active trend - show to end of data
            trend_duration = len(df) - breakout_idx
            window_start = max(0, breakout_idx - max(10, trend_duration // 4))
            window_end = len(df) - 1
        
        # Get swing point details
        if formation['type'] == 'UPTREND':
            swing_points = [
                {'idx': formation['sl1']['idx'], 'price': formation['sl1']['price'], 'type': 'SL1'},
                {'idx': formation['sh1']['idx'], 'price': formation['sh1']['price'], 'type': 'SH1'},
                {'idx': formation['sl2']['idx'], 'price': formation['sl2']['price'], 'type': 'SL2'}
            ]
            controlling_swing = {'idx': formation['sl1']['idx'], 'price': formation['sl1']['price'], 'type': 'SL1 (Control)'}
        else:
            swing_points = [
                {'idx': formation['sh1']['idx'], 'price': formation['sh1']['price'], 'type': 'SH1'},
                {'idx': formation['sl1']['idx'], 'price': formation['sl1']['price'], 'type': 'SL1'},
                {'idx': formation['sh2']['idx'], 'price': formation['sh2']['price'], 'type': 'SH2'}
            ]
            controlling_swing = {'idx': formation['sh1']['idx'], 'price': formation['sh1']['price'], 'type': 'SH1 (Control)'}
        
        trend_info = {
            'id': i + 1,
            'type': formation['type'],
            'formation': formation,
            'termination': termination,
            'window_start': window_start,
            'window_end': window_end,
            'breakout_idx': breakout_idx,
            'swing_points': swing_points,
            'controlling_swing': controlling_swing,
            'duration': trend_duration,
            'formation_date': formation['formation_date'].strftime('%Y-%m-%d') if hasattr(formation['formation_date'], 'strftime') else str(formation['formation_date']),
            'status': 'Terminated' if termination else 'Active'
        }
        
        if termination:
            trend_info['termination_idx'] = termination['violation_idx']
            trend_info['violation_price'] = termination['violation_price']
            trend_info['termination_date'] = termination['violation_date'].strftime('%Y-%m-%d') if hasattr(termination['violation_date'], 'strftime') else str(termination['violation_date'])
        
        trend_data.append(trend_info)
    
    # Create the browseable HTML
    create_browseable_html(df, trend_data)

def create_browseable_html(df, trend_data):
    """Create interactive HTML with trend browser"""
    
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
    <title>USO Trend Browser - Individual Trend Analysis</title>
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
        .control-panel {{
            background: rgba(0,0,0,0.6);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 2px solid #333;
            display: grid;
            grid-template-columns: 300px 1fr 200px;
            gap: 20px;
            align-items: center;
        }}
        .trend-selector {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .trend-info {{
            background: rgba(0,50,0,0.3);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #00ff88;
        }}
        .trend-info h3 {{
            margin: 0 0 10px 0;
            color: #00ff88;
        }}
        .trend-info .info-item {{
            margin: 5px 0;
            font-size: 14px;
        }}
        .navigation {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .nav-btn {{
            background: #007bff;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }}
        .nav-btn:hover {{ 
            background: #0056b3; 
            transform: translateY(-2px);
        }}
        .nav-btn:disabled {{
            background: #555;
            cursor: not-allowed;
            transform: none;
        }}
        .trend-counter {{
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            color: #ffaa00;
        }}
        #chart {{ 
            margin-top: 20px;
            min-height: 600px;
        }}
        
        .uptrend {{ border-left: 4px solid #00ff00; }}
        .downtrend {{ border-left: 4px solid #ff0000; }}
        .terminated {{ background: rgba(100,0,0,0.1); }}
        .active {{ background: rgba(0,100,0,0.1); }}
        
        select {{
            background: #333;
            color: white;
            border: 1px solid #555;
            padding: 8px;
            border-radius: 4px;
            width: 100%;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 USO Trend Browser</h1>
        <p>Browse through {len(trend_data)} individual trend formations/terminations</p>
    </div>
    
    <div class="control-panel">
        <div class="trend-selector">
            <label style="color: #ffaa00; font-weight: bold;">Select Trend:</label>
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
            
            // Add swing points
            const swingDates = [];
            const swingPrices = [];
            const swingTexts = [];
            
            trend.swing_points.forEach(sp => {{
                if (sp.idx >= trend.window_start && sp.idx <= trend.window_end) {{
                    const candle = dfData[sp.idx];
                    swingDates.push(candle.datetime);
                    swingPrices.push(sp.price);
                    swingTexts.push(sp.type);
                }}
            }});
            
            if (swingDates.length > 0) {{
                traces.push({{
                    type: 'scatter',
                    mode: 'markers+text',
                    x: swingDates,
                    y: swingPrices,
                    marker: {{
                        color: trend.type === 'UPTREND' ? '#00ff00' : '#ff0000',
                        size: 12,
                        symbol: 'circle',
                        line: {{ color: 'white', width: 2 }}
                    }},
                    text: swingTexts,
                    textposition: 'top center',
                    textfont: {{ size: 12, color: 'white' }},
                    name: 'Swing Points'
                }});
            }}
            
            // Add breakout marker
            if (trend.breakout_idx >= trend.window_start && trend.breakout_idx <= trend.window_end) {{
                const breakoutCandle = dfData[trend.breakout_idx];
                traces.push({{
                    type: 'scatter',
                    mode: 'markers+text',
                    x: [breakoutCandle.datetime],
                    y: [trend.type === 'UPTREND' ? breakoutCandle.high : breakoutCandle.low],
                    marker: {{
                        color: '#ffaa00',
                        size: 15,
                        symbol: 'star',
                        line: {{ color: 'white', width: 2 }}
                    }},
                    text: ['BREAKOUT'],
                    textposition: 'top center',
                    textfont: {{ size: 12, color: '#ffaa00', family: 'Arial Black' }},
                    name: 'Breakout Origin'
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
                        color: '#ff6600',
                        size: 15,
                        symbol: 'x',
                        line: {{ color: 'white', width: 2 }}
                    }},
                    text: ['VIOLATION'],
                    textposition: 'bottom center',
                    textfont: {{ size: 12, color: '#ff6600', family: 'Arial Black' }},
                    name: 'Trend Termination'
                }});
            }}
            
            // Add controlling swing horizontal line
            if (trend.controlling_swing.idx >= trend.window_start && trend.controlling_swing.idx <= trend.window_end) {{
                const startDate = windowData[0].datetime;
                const endDate = windowData[windowData.length - 1].datetime;
                
                traces.push({{
                    type: 'scatter',
                    mode: 'lines',
                    x: [startDate, endDate],
                    y: [trend.controlling_swing.price, trend.controlling_swing.price],
                    line: {{
                        color: trend.type === 'UPTREND' ? '#00ff00' : '#ff0000',
                        width: 2,
                        dash: 'dash'
                    }},
                    name: 'Control Level',
                    showlegend: true
                }});
            }}
            
            const layout = {{
                title: {{
                    text: `${{trend.type}} #${{trend.id}} - ${{trend.formation_date}} (${{trend.status}})`,
                    font: {{ size: 20, color: 'white' }},
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
                <div class="info-item"><strong>Control Level:</strong> $${{trend.controlling_swing.price.toFixed(2)}}</div>
            `;
            
            if (trend.termination) {{
                detailsHTML += `
                    <div class="info-item"><strong>Termination:</strong> ${{trend.termination_date}}</div>
                    <div class="info-item"><strong>Violation:</strong> $${{trend.violation_price.toFixed(2)}}</div>
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
    filename = 'trend_browser.html'
    with open(filename, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Trend browser saved as: {filename}")
    print(f"🎯 Browse through {len(trend_data)} individual trends to see what's happening")

def main():
    """Create trend browser from existing analysis"""
    
    # Import the analysis results
    from complete_uso_trend_analysis import create_complete_uso_analysis
    
    print("🔍 Creating trend browser to examine individual formations...")
    
    # Get the raw (unfiltered) data to see all detected trends
    from uso_supply_demand_visualizer import SupplyDemandVisualizer
    from swing_point_detector import SwingPointDetector
    from complete_uso_trend_analysis import find_all_trend_formations, find_all_trend_terminations
    
    visualizer = SupplyDemandVisualizer()
    df_full = visualizer.fetch_uso_data(timeframe='daily')
    
    # Convert to format for swing detector  
    df = df_full.reset_index()
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})
    
    # Get swing points
    detector = SwingPointDetector(lookback_periods=1, equal_levels_count=False)
    swing_points = detector.detect_swing_points(df)
    
    # Get ALL formations (no filtering)
    formations = find_all_trend_formations(df, swing_points)
    terminations = find_all_trend_terminations(df, formations)
    
    print(f"📊 Found {len(formations)} formations and {len(terminations)} terminations")
    
    # Create the browser
    create_trend_browser(df, formations, terminations)

if __name__ == "__main__":
    main()