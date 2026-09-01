#!/usr/bin/env python3
"""
USO Unified Interactive Explorer
Single chart with dynamic timeframe switching - just like real trading platforms!
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import requests
import os
from datetime import datetime, timedelta
import json

# Dark theme
pio.templates.default = "plotly_dark"

class USOUnifiedExplorer:
    """Unified USO explorer with dynamic timeframe switching"""
    
    def __init__(self):
        self.api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        
    def fetch_all_data(self):
        """Fetch all timeframe data upfront for instant switching"""
        
        print("📡 Fetching all USO data for unified explorer...")
        
        intervals = {
            '15min': ('TIME_SERIES_INTRADAY', '15min'),
            '30min': ('TIME_SERIES_INTRADAY', '30min'),
            '60min': ('TIME_SERIES_INTRADAY', '60min'),
            'daily': ('TIME_SERIES_DAILY', None),
            'weekly': ('TIME_SERIES_WEEKLY', None)
        }
        
        all_data = {}
        
        for interval, (function, api_interval) in intervals.items():
            try:
                print(f"  📊 Fetching {interval} data...")
                
                url = "https://www.alphavantage.co/query"
                params = {
                    'function': function,
                    'symbol': 'USO',
                    'apikey': self.api_key,
                    'outputsize': 'full',
                    'entitlement': 'delayed'
                }
                
                if api_interval:
                    params['interval'] = api_interval
                
                response = requests.get(url, params=params, timeout=30)
                data = response.json()
                
                # Parse data
                if interval == 'daily':
                    time_series_key = 'Time Series (Daily)'
                elif interval == 'weekly':
                    time_series_key = 'Weekly Time Series'
                else:
                    time_series_key = f'Time Series ({api_interval})'
                
                time_series = data[time_series_key]
                
                # Convert to DataFrame
                df_data = []
                for datetime_str, values in time_series.items():
                    df_data.append({
                        'datetime': datetime_str,
                        'Open': float(values['1. open']),
                        'High': float(values['2. high']),
                        'Low': float(values['3. low']),
                        'Close': float(values['4. close']),
                        'Volume': int(values['5. volume'])
                    })
                
                df = pd.DataFrame(df_data)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                df.sort_index(inplace=True)
                
                # For intraday data, filter to regular trading hours
                if interval in ['15min', '30min', '60min']:
                    df_et = df.copy()
                    df_et.index = df_et.index.tz_localize('UTC').tz_convert('US/Eastern')
                    regular_hours = df_et.between_time('09:30', '16:00')
                    # Convert to CST for display
                    regular_hours.index = regular_hours.index.tz_convert('US/Central')
                    all_data[interval] = regular_hours
                else:
                    all_data[interval] = df
                
                print(f"    ✅ {len(all_data[interval]):,} candles")
                
            except Exception as e:
                print(f"    ❌ Error fetching {interval}: {e}")
                continue
        
        return all_data
    
    def create_unified_chart(self, all_data):
        """Create unified chart with JavaScript-powered timeframe switching"""
        
        print("🎯 Creating unified interactive chart...")
        
        # Convert all data to JSON for JavaScript
        chart_data = {}
        
        for interval, df in all_data.items():
            # Get different ranges for each timeframe
            if interval == '15min':
                display_data = df.tail(200)  # Last 200 15min candles
            elif interval == '30min':
                display_data = df.tail(150)  # Last 150 30min candles  
            elif interval == '60min':
                display_data = df.tail(100)  # Last 100 hourly candles
            elif interval == 'daily':
                display_data = df.tail(90)   # Last 90 daily candles
            else:  # weekly
                display_data = df.tail(52)   # Last 52 weekly candles
            
            # Convert to JavaScript-friendly format with both indices and labels
            chart_data[interval] = {
                'x': list(range(len(display_data))),  # Indices for no gaps
                'labels': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in display_data.index],  # Date labels
                'open': display_data['Open'].tolist(),
                'high': display_data['High'].tolist(), 
                'low': display_data['Low'].tolist(),
                'close': display_data['Close'].tolist(),
                'volume': display_data['Volume'].tolist(),
                'count': len(display_data)
            }
        
        # Create initial chart with daily data
        initial_data = all_data['daily'].tail(90)
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=False,  # Independent x-axes to avoid conflicts
            vertical_spacing=0.05,
            row_heights=[0.6, 0.25, 0.15],
            subplot_titles=[
                'USO Price - Unified Explorer',
                'Volume', 
                'Price Change Analytics'
            ]
        )
        
        # Add initial candlestick chart with no gaps but custom hover with dates
        hover_texts = []
        for i, (ts, row) in enumerate(initial_data.iterrows()):
            date_str = ts.strftime('%Y-%m-%d %H:%M')
            hover_text = f"Date: {date_str}<br>Index: {i}<br>O: ${row['Open']:.2f} H: ${row['High']:.2f}<br>L: ${row['Low']:.2f} C: ${row['Close']:.2f}"
            hover_texts.append(hover_text)
            
        fig.add_trace(
            go.Candlestick(
                x=list(range(len(initial_data))),
                open=initial_data['Open'],
                high=initial_data['High'],
                low=initial_data['Low'],
                close=initial_data['Close'],
                name='USO',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444',
                text=hover_texts,
                hoverinfo='text'
            ),
            row=1, col=1
        )
        
        # Add initial volume with enhanced visibility (no gaps)
        colors = ['#00ff88' if close >= open else '#ff4444' 
                 for close, open in zip(initial_data['Close'], initial_data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=list(range(len(initial_data))),
                y=initial_data['Volume'],
                name='Volume',
                marker=dict(
                    color=colors,
                    opacity=0.8,
                    line=dict(
                        color='rgba(255,255,255,0.1)',
                        width=0.5
                    )
                ),
                hovertemplate='<b>Volume</b><br>Time: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add price change analytics (no gaps)
        price_changes = initial_data['Close'].pct_change() * 100
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(initial_data))),
                y=price_changes,
                mode='markers+lines',
                name='Price Change %',
                marker=dict(
                    color=price_changes,
                    colorscale=[[0, '#ff4444'], [0.5, '#ffff44'], [1, '#00ff88']],
                    size=4
                ),
                line=dict(width=1, color='rgba(255,255,255,0.3)')
            ),
            row=3, col=1
        )
        
        # Create custom JavaScript for timeframe switching
        javascript_code = f"""
        // Chart data for all timeframes
        var chartData = {json.dumps(chart_data, indent=2)};
        var currentTimeframe = 'daily';
        
        // Drawing tools state
        var drawingMode = false;
        var drawnLines = {{}};  // Store lines per timeframe
        var currentLine = null;  // Line being drawn
        var snapPoints = [];     // Current candle snap points
        
        // Update chart function
        function updateChart(timeframe) {{
            if (timeframe === currentTimeframe) return;
            
            var data = chartData[timeframe];
            if (!data) return;
            
            // Use indices for no gaps but keep labels for display
            var xData = data.x;
            
            // Update candlestick with datetime hover info
            Plotly.restyle('chart', {{
                'x': [xData],
                'open': [data.open],
                'high': [data.high], 
                'low': [data.low],
                'close': [data.close],
                'customdata': [data.labels]
            }}, 0);
            
            // Simple red/green volume colors - no gaps complexity
            var volumeColors = data.close.map(function(close, i) {{
                return close >= data.open[i] ? '#00ff88' : '#ff4444';
            }});
            
            // Simple volume update - just restyle
            Plotly.restyle('chart', {{
                'x': [xData],
                'y': [data.volume],
                'marker.color': [volumeColors],
                'marker.opacity': [0.8]
            }}, 1);
            
            console.log('Volume updated with', volumeColors.length, 'red/green colors');
            
            // Calculate price changes
            var priceChanges = [];
            for (var i = 1; i < data.close.length; i++) {{
                var change = ((data.close[i] - data.close[i-1]) / data.close[i-1]) * 100;
                priceChanges.push(change);
            }}
            priceChanges.unshift(0); // First value is 0
            
            // Update price changes - rebuild trace
            Plotly.deleteTraces('chart', 2);  // Remove old price change trace
            
            // Add new price change trace
            var priceChangeTrace = {{
                x: xData,
                y: priceChanges,
                mode: 'markers+lines',
                type: 'scatter',
                name: 'Price Change %',
                marker: {{
                    color: priceChanges,
                    colorscale: [[0, '#ff4444'], [0.5, '#ffff44'], [1, '#00ff88']],
                    size: 4
                }},
                line: {{
                    width: 1,
                    color: 'rgba(255,255,255,0.3)'
                }},
                yaxis: 'y3',
                xaxis: 'x3'
            }};
            
            Plotly.addTraces('chart', priceChangeTrace, 2);
            
            // Update x-axis labels to show time labels
            var tickvals = [];
            var ticktext = [];
            var step = Math.max(1, Math.floor(data.x.length / 8)); // Show ~8 labels max
            for (var i = 0; i < data.x.length; i += step) {{
                tickvals.push(i);
                ticktext.push(data.labels[i].split(' ')[0]); // Just show date part
            }}
            
            // Update x-axis with custom labels
            Plotly.relayout('chart', {{
                'xaxis3.tickvals': tickvals,
                'xaxis3.ticktext': ticktext
            }});
            
            // Update title
            var title = 'USO Price - ' + timeframe.toUpperCase() + ' | ' + data.count + ' candles | Latest: $' + data.close[data.close.length-1].toFixed(2);
            Plotly.relayout('chart', {{'title.text': title}});
            
            // Update button styles
            document.querySelectorAll('.timeframe-btn').forEach(function(btn) {{
                btn.style.backgroundColor = '#333333';
                btn.style.color = '#ffffff';
            }});
            document.getElementById(timeframe + '-btn').style.backgroundColor = '#00ff88';
            document.getElementById(timeframe + '-btn').style.color = '#000000';
            
            currentTimeframe = timeframe;
            
            // Update snap points for new timeframe
            updateSnapPoints();
            
            // Redraw lines for this timeframe
            redrawLines();
        }}
        
        // Calculate snap points for current data
        function updateSnapPoints() {{
            var data = chartData[currentTimeframe];
            if (!data) return;
            
            snapPoints = [];
            for (var i = 0; i < data.x.length; i++) {{
                snapPoints.push({{
                    x: i,
                    high: data.high[i],
                    low: data.low[i],
                    open: data.open[i],
                    close: data.close[i],
                    timestamp: data.labels[i]
                }});
            }}
        }}
        
        // Find nearest snap point to clicked location
        function findNearestSnap(clickX, clickY) {{
            var minDistance = Infinity;
            var nearestPoint = null;
            var nearestType = '';
            
            for (var i = 0; i < snapPoints.length; i++) {{
                var point = snapPoints[i];
                var xDistance = Math.abs(point.x - clickX);
                
                // Check all four price points
                var pricePoints = [
                    {{type: 'high', price: point.high}},
                    {{type: 'low', price: point.low}},
                    {{type: 'open', price: point.open}},
                    {{type: 'close', price: point.close}}
                ];
                
                for (var j = 0; j < pricePoints.length; j++) {{
                    var pp = pricePoints[j];
                    var yDistance = Math.abs(pp.price - clickY);
                    var totalDistance = Math.sqrt(xDistance * xDistance + yDistance * yDistance);
                    
                    if (totalDistance < minDistance) {{
                        minDistance = totalDistance;
                        nearestPoint = {{
                            x: point.x,
                            y: pp.price,
                            timestamp: point.timestamp,
                            candleIndex: i
                        }};
                        nearestType = pp.type;
                    }}
                }}
            }}
            
            return {{point: nearestPoint, type: nearestType, distance: minDistance}};
        }}
        
        // Toggle drawing mode
        function toggleDrawing() {{
            drawingMode = !drawingMode;
            var btn = document.getElementById('drawing-btn');
            
            if (drawingMode) {{
                btn.style.backgroundColor = '#ff6b35';
                btn.style.color = '#000000';
                btn.innerHTML = 'EXIT DRAW';
                document.getElementById('chart').style.cursor = 'crosshair';
                updateSnapPoints();
                console.log('Drawing mode ON - Click candles to draw lines');
            }} else {{
                btn.style.backgroundColor = '#333333';
                btn.style.color = '#ffffff';
                btn.innerHTML = 'DRAW LINES';
                document.getElementById('chart').style.cursor = 'default';
                currentLine = null;
                console.log('Drawing mode OFF');
            }}
        }}
        
        // Handle chart clicks for drawing
        function handleChartClick(eventData) {{
            if (!drawingMode) return;
            
            var clickX = eventData.points[0].x;
            var clickY = eventData.points[0].y;
            
            // Find nearest snap point
            var snap = findNearestSnap(clickX, clickY);
            
            if (!snap.point) return;
            
            console.log('Snapped to', snap.type, 'at candle', snap.point.candleIndex, 'price', snap.point.y.toFixed(2));
            
            if (!currentLine) {{
                // Start new line
                currentLine = {{
                    start: snap.point,
                    startType: snap.type,
                    timeframe: currentTimeframe
                }};
                
                // Add temporary line trace
                var tempLine = {{
                    x: [snap.point.x, snap.point.x],
                    y: [snap.point.y, snap.point.y],
                    mode: 'lines+markers',
                    type: 'scatter',
                    name: 'Drawing...',
                    line: {{color: '#ffff00', width: 2, dash: 'dot'}},
                    marker: {{color: '#ffff00', size: 6}},
                    showlegend: false,
                    yaxis: 'y',
                    xaxis: 'x'
                }};
                
                Plotly.addTraces('chart', tempLine, 0);
                
            }} else {{
                // Complete the line
                currentLine.end = snap.point;
                currentLine.endType = snap.type;
                
                // Remove temporary line
                var traces = document.getElementById('chart').data;
                for (var i = traces.length - 1; i >= 0; i--) {{
                    if (traces[i].name === 'Drawing...') {{
                        Plotly.deleteTraces('chart', i);
                        break;
                    }}
                }}
                
                // Add permanent line
                if (!drawnLines[currentTimeframe]) {{
                    drawnLines[currentTimeframe] = [];
                }}
                
                var lineId = 'line_' + Date.now();
                var permanentLine = {{
                    id: lineId,
                    x: [currentLine.start.x, currentLine.end.x],
                    y: [currentLine.start.y, currentLine.end.y],
                    mode: 'lines+markers',
                    type: 'scatter',
                    name: currentLine.startType + ' → ' + currentLine.endType,
                    line: {{color: '#00ff88', width: 2}},
                    marker: {{color: '#00ff88', size: 4}},
                    showlegend: true,
                    yaxis: 'y',
                    xaxis: 'x',
                    hovertemplate: '<b>%{{fullData.name}}</b><br>' +
                                 'Start: %{{x[0]}} @ $%{{y[0]:.2f}}<br>' +
                                 'End: %{{x[1]}} @ $%{{y[1]:.2f}}<extra></extra>'
                }};
                
                drawnLines[currentTimeframe].push(permanentLine);
                Plotly.addTraces('chart', permanentLine, 0);
                
                console.log('Line completed:', currentLine.startType, 'to', currentLine.endType);
                currentLine = null;
            }}
        }}
        
        // Redraw all lines for current timeframe
        function redrawLines() {{
            if (!drawnLines[currentTimeframe]) return;
            
            for (var i = 0; i < drawnLines[currentTimeframe].length; i++) {{
                var line = drawnLines[currentTimeframe][i];
                Plotly.addTraces('chart', line, 0);
            }}
        }}
        
        // Clear all lines for current timeframe
        function clearLines() {{
            if (!drawnLines[currentTimeframe]) return;
            
            var traces = document.getElementById('chart').data;
            for (var i = traces.length - 1; i >= 0; i--) {{
                var trace = traces[i];
                if (trace.name && (trace.name.includes('→') || trace.name === 'Drawing...')) {{
                    Plotly.deleteTraces('chart', i);
                }}
            }}
            
            drawnLines[currentTimeframe] = [];
            console.log('Lines cleared for', currentTimeframe);
        }}
        
        // Gaps toggle removed for simplicity
        
        // Make info panel draggable
        function makeDraggable(element) {{
            var pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
            var header = element.querySelector('.drag-header') || element;
            header.onmousedown = dragMouseDown;
            
            function dragMouseDown(e) {{
                e = e || window.event;
                e.preventDefault();
                pos3 = e.clientX;
                pos4 = e.clientY;
                document.onmouseup = closeDragElement;
                document.onmousemove = elementDrag;
                element.style.cursor = 'grabbing';
            }}
            
            function elementDrag(e) {{
                e = e || window.event;
                e.preventDefault();
                pos1 = pos3 - e.clientX;
                pos2 = pos4 - e.clientY;
                pos3 = e.clientX;
                pos4 = e.clientY;
                var newTop = element.offsetTop - pos2;
                var newLeft = element.offsetLeft - pos1;
                
                // Keep within viewport
                newTop = Math.max(0, Math.min(newTop, window.innerHeight - element.offsetHeight));
                newLeft = Math.max(0, Math.min(newLeft, window.innerWidth - element.offsetWidth));
                
                element.style.top = newTop + "px";
                element.style.left = newLeft + "px";
            }}
            
            function closeDragElement() {{
                document.onmouseup = null;
                document.onmousemove = null;
                element.style.cursor = 'grab';
            }}
        }}
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            // Set initial active button
            document.getElementById('daily-btn').style.backgroundColor = '#00ff88';
            document.getElementById('daily-btn').style.color = '#000000';
            
            // Initialize snap points for default timeframe
            updateSnapPoints();
            
            // Make info panel draggable
            var infoPanel = document.querySelector('.info-panel');
            if (infoPanel) {{
                makeDraggable(infoPanel);
            }}
        }});
        """
        
        # Custom CSS for buttons
        css_code = """
        <style>
            .timeframe-container {
                position: fixed;
                top: 20px;
                left: 20px;
                z-index: 1000;
                display: flex;
                gap: 5px;
                background: rgba(30, 30, 30, 0.95);
                padding: 10px;
                border-radius: 8px;
                border: 1px solid rgba(100, 100, 100, 0.5);
            }
            
            .timeframe-btn {
                padding: 8px 15px;
                background: #333333;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
                font-size: 12px;
                transition: all 0.2s;
            }
            
            .timeframe-btn:hover {
                background: #555555;
                border-color: #00ff88;
            }
            
            .drawing-container {
                position: fixed;
                top: 20px;
                left: 420px;
                z-index: 1000;
                display: flex;
                gap: 5px;
                background: rgba(30, 30, 30, 0.95);
                padding: 10px;
                border-radius: 8px;
                border: 1px solid rgba(100, 100, 100, 0.5);
            }
            
            .drawing-btn {
                padding: 8px 15px;
                background: #333333;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
                font-size: 12px;
                transition: all 0.2s;
            }
            
            .drawing-btn:hover {
                background: #555555;
                border-color: #ff6b35;
            }
            
            .clear-btn {
                padding: 8px 15px;
                background: #333333;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
                font-size: 12px;
                transition: all 0.2s;
            }
            
            .clear-btn:hover {
                background: #666;
                border-color: #ff4444;
            }
            
            .gaps-btn {
                padding: 8px 15px;
                background: #333333;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
                font-size: 12px;
                transition: all 0.2s;
                margin-left: 10px;
            }
            
            .gaps-btn:hover {
                background: #555555;
                border-color: #ff6b35;
            }
            
            .info-panel {
                position: fixed;
                top: 100px;
                right: 20px;
                z-index: 1000;
                background: rgba(30, 30, 30, 0.95);
                padding: 15px;
                border-radius: 8px;
                border: 2px solid rgba(100, 100, 100, 0.5);
                color: white;
                font-size: 12px;
                max-width: 280px;
                cursor: grab;
                user-select: none;
            }
            
            .info-panel:active {
                cursor: grabbing;
            }
            
            .drag-header {
                color: #00ff88;
                font-weight: bold;
                margin-bottom: 8px;
                padding: 5px;
                border-radius: 4px;
                background: rgba(0, 255, 136, 0.1);
                cursor: grab;
            }
            
            .drag-header:active {
                cursor: grabbing;
            }
            
            .feature-list {
                margin: 10px 0;
                line-height: 1.4;
            }
            
            .minimize-btn {
                float: right;
                background: #666;
                color: white;
                border: none;
                border-radius: 3px;
                width: 20px;
                height: 20px;
                cursor: pointer;
                font-size: 12px;
                line-height: 1;
            }
            
            .minimize-btn:hover {
                background: #888;
            }
            
            body {
                margin: 0;
                background: #1a1a1a;
            }
        </style>
        """
        
        # HTML for timeframe buttons, drawing tools, and info panel
        html_controls = """
        <div class="timeframe-container">
            <button class="timeframe-btn" id="15min-btn" onclick="updateChart('15min')">15MIN</button>
            <button class="timeframe-btn" id="30min-btn" onclick="updateChart('30min')">30MIN</button>
            <button class="timeframe-btn" id="60min-btn" onclick="updateChart('60min')">1HOUR</button>
            <button class="timeframe-btn" id="daily-btn" onclick="updateChart('daily')">DAILY</button>
            <button class="timeframe-btn" id="weekly-btn" onclick="updateChart('weekly')">WEEKLY</button>
        </div>
        
        <div class="drawing-container">
            <button class="drawing-btn" id="drawing-btn" onclick="toggleDrawing()">DRAW LINES</button>
            <button class="clear-btn" id="clear-btn" onclick="clearLines()">CLEAR</button>
        </div>
        
        <div class="info-panel">
            <div class="drag-header">
                🎯 USO Unified Explorer
                <button class="minimize-btn" onclick="this.parentElement.parentElement.style.display='none'">×</button>
            </div>
            <div class="feature-list">
                <div>📊 <strong>Timeframes:</strong> Click buttons to switch instantly</div>
                <div>✏️ <strong>Draw Lines:</strong> Click DRAW LINES, then click candles</div>
                <div>🎯 <strong>Smart Snap:</strong> Lines snap to wick/body edges</div>
                <div>🗑️ <strong>Clear:</strong> Remove all lines for current timeframe</div>
                <div>🔍 <strong>Zoom:</strong> Drag to select area</div>
                <div>📱 <strong>Pan:</strong> Drag chart to navigate</div>
                <div>🖱️ <strong>Drag Panel:</strong> Click header to move</div>
                <div>💡 <strong>Hover:</strong> See candle details</div>
                <div>📷 <strong>Export:</strong> Camera icon (toolbar)</div>
            </div>
            <div style="font-size: 10px; color: #888; margin-top: 10px;">
                Data: Alpha Vantage | Times in CST | Click × to hide this panel
            </div>
        </div>
        """
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"USO Price - DAILY | {len(initial_data)} candles | Latest: ${initial_data['Close'].iloc[-1]:.2f}",
                font=dict(size=18, color='white'),
                x=0.5
            ),
            height=900,
            paper_bgcolor='rgba(15,15,15,1)',
            plot_bgcolor='rgba(25,25,25,1)',
            font=dict(color='white', size=11),
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            showlegend=False
        )
        
        # Update axes - only show x-axis labels on bottom chart
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, gridcolor='rgba(100,100,100,0.2)')
        fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor='rgba(100,100,100,0.2)')
        fig.update_yaxes(title_text="Change (%)", row=3, col=1, gridcolor='rgba(100,100,100,0.2)')
        
        # Hide x-axis labels on top two charts
        fig.update_xaxes(showticklabels=False, gridcolor='rgba(100,100,100,0.2)', row=1, col=1)
        fig.update_xaxes(showticklabels=False, gridcolor='rgba(100,100,100,0.2)', row=2, col=1)
        
        # Only show x-axis labels on bottom chart with custom time labels
        initial_indices = list(range(len(initial_data)))
        initial_labels = [ts.strftime('%Y-%m-%d') for ts in initial_data.index]
        step = max(1, len(initial_data) // 8)  # Show ~8 labels max
        tickvals = initial_indices[::step]
        ticktext = [initial_labels[i] for i in range(0, len(initial_labels), step)]
        
        fig.update_xaxes(
            title_text="Time (CST)", 
            gridcolor='rgba(100,100,100,0.2)', 
            tickvals=tickvals,
            ticktext=ticktext,
            row=3, col=1
        )
        
        # Save with custom HTML
        html_string = fig.to_html(include_plotlyjs=True, div_id="chart")
        
        # Insert custom elements
        html_string = html_string.replace('<head>', f'<head>{css_code}')
        html_string = html_string.replace('<body>', f'<body>{html_controls}')
        
        # Add the event listener setup after Plotly loads
        event_setup = '''
        // Wait for Plotly to fully load and then add event listeners
        setTimeout(function() {
            var chartDiv = document.getElementById('chart');
            if (chartDiv && chartDiv.on) {
                chartDiv.on('plotly_click', handleChartClick);
                console.log('Drawing event listener added successfully');
            } else {
                console.log('Chart not ready, retrying...');
                setTimeout(arguments.callee, 100);
            }
        }, 500);
        '''
        
        html_string = html_string.replace('</body>', f'<script>{javascript_code}</script><script>{event_setup}</script></body>')
        
        return html_string

def main():
    """Create the unified USO explorer"""
    
    print("🚀 Creating USO Unified Interactive Explorer")
    print("Single chart with dynamic timeframe switching!")
    print("=" * 60)
    
    explorer = USOUnifiedExplorer()
    
    try:
        # Fetch all data
        all_data = explorer.fetch_all_data()
        
        if not all_data:
            print("❌ No data fetched - cannot create chart")
            return
        
        # Create unified chart
        html_content = explorer.create_unified_chart(all_data)
        
        # Save to file
        with open('uso_unified_explorer.html', 'w') as f:
            f.write(html_content)
        
        print("\n✅ Unified Explorer Created!")
        print("💾 Saved as: uso_unified_explorer.html")
        print("\n🎯 Features:")
        print("  • Single chart with 5 timeframes (15min, 30min, 1hour, daily, weekly)")
        print("  • Instant timeframe switching (no reload needed)")
        print("  • Professional trading platform interface")
        print("  • Interactive zoom, pan, and drawing tools")
        print("  • Volume analysis with color-coded bars")
        print("  • Price change analytics")
        print("  • Dark theme with green/red candlesticks")
        print("  • Export to PNG capability")
        print("  • Hover tooltips with full data")
        
        # Print data summary
        print(f"\n📊 Data Summary:")
        for interval, df in all_data.items():
            print(f"  {interval:>6}: {len(df):,} candles")
        
    except Exception as e:
        print(f"❌ Error creating unified explorer: {e}")

if __name__ == "__main__":
    main()