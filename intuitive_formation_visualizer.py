#!/usr/bin/env python3
"""
Intuitive Formation Visualizer
Same interface as trend visualizer but for supply/demand formations:
- Intuitive slider to control time window
- Shows only formations with origins in the visible window
- Smooth, easy-to-use interface
- Formation type filtering (RBR, DBD, RBD, DBR)
"""

import plotly.graph_objects as go
import plotly.io as pio
from formation_detector import FormationDetector
from uso_supply_demand_visualizer import SupplyDemandVisualizer
import json
from datetime import datetime, timedelta

# Dark theme
pio.templates.default = "plotly_dark"

def create_intuitive_formation_visualizer():
    """
    Create an intuitive formation visualizer with proper slider controls
    """
    
    print("🎯 INTUITIVE FORMATION VISUALIZER")
    print("Creating user-friendly interface with slider controls")
    print("=" * 60)
    
    # Get the data (same as trend system)
    visualizer = SupplyDemandVisualizer()
    df_full = visualizer.fetch_uso_data(timeframe='daily')
    df = df_full.reset_index()
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})
    
    # The reset_index() already creates a datetime column from the original index
    if 'index' in df.columns:
        df = df.rename(columns={'index': 'datetime'})
    df = df.reset_index(drop=True)
    
    print(f"📊 Dataset: {len(df)} candles")
    
    # Find all formations
    detector = FormationDetector(decay_factor=0.7, min_leg_threshold=1.5)
    formations = detector.detect_formations(df)
    
    # Filter to valid formations only
    valid_formations = [f for f in formations if f['valid']]
    
    print(f"✅ Found {len(valid_formations)} valid formations")
    print(f"   • {len([f for f in valid_formations if f['type'] == 'RBR'])} RBR (Rally-Base-Rally)")
    print(f"   • {len([f for f in valid_formations if f['type'] == 'DBD'])} DBD (Drop-Base-Drop)") 
    print(f"   • {len([f for f in valid_formations if f['type'] == 'RBD'])} RBD (Rally-Base-Drop)")
    print(f"   • {len([f for f in valid_formations if f['type'] == 'DBR'])} DBR (Drop-Base-Rally)")
    
    # Prepare data for JavaScript
    df_json = df.to_dict('records')
    for i, record in enumerate(df_json):
        record['index'] = i
        if 'datetime' in record:
            record['datetime'] = record['datetime'].isoformat() if hasattr(record['datetime'], 'isoformat') else str(record['datetime'])
    
    # Prepare formation data
    formation_data = []
    for i, formation in enumerate(valid_formations):
        formation_info = {
            'id': i + 1,
            'type': formation['type'],
            'zone_type': formation['zone_type'],
            'start_idx': formation['start_idx'],
            'end_idx': formation['end_idx'],
            'leg_in_idx': formation['leg_in']['index'],
            'leg_out_idx': formation['leg_out']['index'],
            'base_candles': formation['base']['base_candles'],
            'base_range': formation['base']['base_range']['range'],
            'proximal_line': formation['base']['proximal_line'],
            'distal_line': formation['base']['distal_line'],
            'leg_in_movement': formation['validation_details']['leg_in_movement'],
            'leg_out_movement': formation['validation_details']['leg_out_movement'],
            'formation_date': df.iloc[formation['start_idx']]['datetime'].isoformat() if hasattr(df.iloc[formation['start_idx']]['datetime'], 'isoformat') else str(df.iloc[formation['start_idx']]['datetime'])
        }
        
        formation_data.append(formation_info)
    
    # Create the HTML interface
    html_content = create_html_interface(df_json, formation_data)
    
    # Save the visualizer
    with open('intuitive_formation_visualizer.html', 'w') as f:
        f.write(html_content)
    
    print(f"✅ Intuitive formation visualizer saved as: intuitive_formation_visualizer.html")
    print(f"🎯 Features:")
    print(f"   • Intuitive slider to control time window")
    print(f"   • Shows formations with origins in visible window")
    print(f"   • Formation type filtering (RBR, DBD, RBD, DBR)")
    print(f"   • Clean, user-friendly interface")
    print(f"   • Smooth drag-to-explore interaction")

def create_html_interface(df_data, formation_data):
    """Create the HTML interface with intuitive controls"""
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Intuitive Formation Visualizer</title>
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
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }}
        
        .time-slider::-moz-range-thumb {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #00ff88;
            cursor: pointer;
            border: none;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }}
        
        .filter-group {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
            margin: 15px 0;
        }}
        
        .filter-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            background: #333;
            padding: 8px 12px;
            border-radius: 20px;
            transition: background 0.2s;
        }}
        
        .filter-item:hover {{
            background: #444;
        }}
        
        .filter-item input[type="checkbox"] {{
            width: 18px;
            height: 18px;
            accent-color: #00ff88;
        }}
        
        .info-panel {{
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
        }}
        
        .info-item {{
            background: #333;
            padding: 10px 15px;
            border-radius: 8px;
            text-align: center;
            min-width: 120px;
        }}
        
        .chart-container {{
            background: #2a2a2a;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }}
        
        .formation-legend {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            justify-content: center;
            margin: 15px 0;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Intuitive Formation Visualizer</h1>
            <p>Interactive Supply & Demand Formation Analysis</p>
        </div>
        
        <div class="controls">
            <div class="slider-container">
                <label class="slider-label">Time Window Position (drag to explore)</label>
                <input type="range" id="positionSlider" class="time-slider" min="0" max="100" value="80">
            </div>
            
            <div class="slider-container">
                <label class="slider-label">Window Size (candles to display)</label>
                <input type="range" id="sizeSlider" class="time-slider" min="50" max="500" value="200">
            </div>
            
            <div class="slider-container">
                <label class="slider-label">Price Position (vertical positioning)</label>
                <input type="range" id="pricePositionSlider" class="time-slider" min="0" max="100" value="50">
            </div>
            
            <div class="slider-container">
                <label class="slider-label">Price Zoom (price range focus)</label>
                <input type="range" id="priceZoomSlider" class="time-slider" min="10" max="100" value="100">
            </div>
            
            <div class="filter-group">
                <strong>Formation Types:</strong>
                <div class="filter-item">
                    <input type="checkbox" id="showRBR" checked>
                    <label for="showRBR">RBR (Demand)</label>
                </div>
                <div class="filter-item">
                    <input type="checkbox" id="showDBD" checked>
                    <label for="showDBD">DBD (Supply)</label>
                </div>
                <div class="filter-item">
                    <input type="checkbox" id="showRBD" checked>
                    <label for="showRBD">RBD (Supply)</label>
                </div>
                <div class="filter-item">
                    <input type="checkbox" id="showDBR" checked>
                    <label for="showDBR">DBR (Demand)</label>
                </div>
            </div>
            
            <div class="info-panel">
                <div class="info-item">
                    <div id="formationCounter">0 formations displayed</div>
                </div>
                <div class="info-item">
                    <div id="windowInfo">Window: 0-0</div>
                </div>
                <div class="info-item">
                    <div id="priceRangeInfo">Price range: $0.00 - $0.00</div>
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="formation-legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #00ff88;"></div>
                    <span>RBR/DBR (Demand Zones)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff4444;"></div>
                    <span>RBD/DBD (Supply Zones)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ffaa00;"></div>
                    <span>Base Segments</span>
                </div>
            </div>
            <div id="chart" style="width: 100%; height: 600px;"></div>
        </div>
    </div>

    <script>
        // Data from Python
        const dfData = {json.dumps(df_data)};
        const formationData = {json.dumps(formation_data)};
        
        let currentWindowStart = 0;
        let currentWindowSize = 200;
        let currentPricePosition = 50;
        let currentPriceZoom = 100;
        
        // Initialize sliders
        document.getElementById('positionSlider').addEventListener('input', function() {{
            const position = parseInt(this.value);
            const maxStart = Math.max(0, dfData.length - currentWindowSize);
            currentWindowStart = Math.floor((position / 100) * maxStart);
            updateChart();
            updateWindowInfo();
        }});
        
        document.getElementById('sizeSlider').addEventListener('input', function() {{
            currentWindowSize = parseInt(this.value);
            // Adjust position if window extends beyond data
            const maxStart = Math.max(0, dfData.length - currentWindowSize);
            currentWindowStart = Math.min(currentWindowStart, maxStart);
            updateChart();
            updateWindowInfo();
        }});
        
        document.getElementById('pricePositionSlider').addEventListener('input', function() {{
            currentPricePosition = parseInt(this.value);
            updateChart();
            updatePriceRangeInfo();
        }});
        
        document.getElementById('priceZoomSlider').addEventListener('input', function() {{
            currentPriceZoom = parseInt(this.value);
            updateChart();
            updatePriceRangeInfo();
        }});
        
        // Formation type filters
        ['showRBR', 'showDBD', 'showRBD', 'showDBR'].forEach(id => {{
            document.getElementById(id).addEventListener('change', updateChart);
        }});
        
        function updateWindowInfo() {{
            const windowEnd = Math.min(currentWindowStart + currentWindowSize, dfData.length);
            const startDate = dfData[currentWindowStart].datetime.split('T')[0];
            const endDate = dfData[windowEnd - 1].datetime.split('T')[0];
            
            document.getElementById('windowInfo').textContent = 
                `Window: ${{startDate}} to ${{endDate}} (${{currentWindowSize}} candles)`;
        }}
        
        function updatePriceRangeInfo() {{
            const windowEnd = Math.min(currentWindowStart + currentWindowSize, dfData.length);
            const windowData = dfData.slice(currentWindowStart, windowEnd);
            
            if (windowData.length > 0) {{
                const prices = windowData.flatMap(d => [d.high, d.low]);
                const fullMinPrice = Math.min(...prices);
                const fullMaxPrice = Math.max(...prices);
                const fullRange = fullMaxPrice - fullMinPrice;
                
                // Calculate zoomed range
                const zoomFactor = currentPriceZoom / 100;
                const zoomWindowSize = fullRange * zoomFactor;
                
                // Calculate position within the available range
                const availableRange = fullRange - zoomWindowSize;
                const positionOffset = availableRange * (currentPricePosition / 100);
                
                const zoomedMin = fullMinPrice + positionOffset;
                const zoomedMax = zoomedMin + zoomWindowSize;
                
                document.getElementById('priceRangeInfo').textContent = 
                    `Price range: $${{zoomedMin.toFixed(2)}} - $${{zoomedMax.toFixed(2)}}`;
            }}
        }}
        
        function getVisibleFormations() {{
            const windowEnd = currentWindowStart + currentWindowSize;
            
            // Filter formations that have start within the visible window
            const visibleFormations = formationData.filter(formation => {{
                return formation.start_idx >= currentWindowStart && formation.start_idx < windowEnd;
            }});
            
            // Apply type filters
            const showRBR = document.getElementById('showRBR').checked;
            const showDBD = document.getElementById('showDBD').checked;
            const showRBD = document.getElementById('showRBD').checked;
            const showDBR = document.getElementById('showDBR').checked;
            
            const typeFilteredFormations = visibleFormations.filter(formation => {{
                if (formation.type === 'RBR') return showRBR;
                if (formation.type === 'DBD') return showDBD;
                if (formation.type === 'RBD') return showRBD;
                if (formation.type === 'DBR') return showDBR;
                return true;
            }});
            
            return typeFilteredFormations;
        }}
        
        function updateChart() {{
            const windowEnd = Math.min(currentWindowStart + currentWindowSize, dfData.length);
            const windowData = dfData.slice(currentWindowStart, windowEnd);
            const visibleFormations = getVisibleFormations();
            
            // Update formation counter
            document.getElementById('formationCounter').textContent = 
                `${{visibleFormations.length}} formations displayed`;
            
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
            
            // Add formation zones
            const formationColors = {{
                'RBR': '#00ff88',  // Green for demand
                'DBR': '#00ff88',  // Green for demand
                'RBD': '#ff4444',  // Red for supply  
                'DBD': '#ff4444'   // Red for supply
            }};
            
            visibleFormations.forEach(formation => {{
                const color = formationColors[formation.type];
                const startCandle = dfData[Math.max(formation.start_idx, currentWindowStart)];
                const endCandle = dfData[Math.min(formation.end_idx, windowEnd - 1)];
                
                // Formation zone rectangle
                traces.push({{
                    type: 'scatter',
                    mode: 'lines',
                    x: [startCandle.datetime, endCandle.datetime, endCandle.datetime, startCandle.datetime, startCandle.datetime],
                    y: [formation.proximal_line, formation.proximal_line, formation.distal_line, formation.distal_line, formation.proximal_line],
                    fill: 'tonexty',
                    fillcolor: color.replace(')', ', 0.2)').replace('rgb', 'rgba'),
                    line: {{ color: color, width: 2 }},
                    name: `${{formation.type}} #${{formation.id}}`,
                    hovertemplate: `<b>${{formation.type}} Formation #${{formation.id}}</b><br>` +
                                 `Zone: ${{formation.zone_type}}<br>` +
                                 `Base Range: $${{formation.base_range.toFixed(4)}}<br>` +
                                 `Proximal: $${{formation.proximal_line.toFixed(2)}}<br>` +
                                 `Distal: $${{formation.distal_line.toFixed(2)}}<br>` +
                                 `Leg In: $${{formation.leg_in_movement.toFixed(2)}}<br>` +
                                 `Leg Out: $${{formation.leg_out_movement.toFixed(2)}}<extra></extra>`
                }});
                
                // Base segment highlight
                formation.base_candles.forEach(baseIdx => {{
                    if (baseIdx >= currentWindowStart && baseIdx < windowEnd) {{
                        const baseCandle = dfData[baseIdx];
                        traces.push({{
                            type: 'scatter',
                            mode: 'markers',
                            x: [baseCandle.datetime],
                            y: [(baseCandle.high + baseCandle.low) / 2],
                            marker: {{ color: '#ffaa00', size: 8, symbol: 'square' }},
                            showlegend: false,
                            hovertemplate: `<b>Base Candle</b><br>Formation #${{formation.id}}<extra></extra>`
                        }});
                    }}
                }});
                
                // Formation markers
                if (formation.start_idx >= currentWindowStart && formation.start_idx < windowEnd) {{
                    traces.push({{
                        type: 'scatter',
                        mode: 'markers+text',
                        x: [dfData[formation.start_idx].datetime],
                        y: [formation.proximal_line],
                        marker: {{ color: color, size: 10, symbol: 'diamond' }},
                        text: [formation.type],
                        textposition: 'top center',
                        textfont: {{ size: 10, color: 'white', family: 'Arial Black' }},
                        showlegend: false,
                        hovertemplate: `<b>${{formation.type}} Start #${{formation.id}}</b><br>Date: ${{formation.formation_date.split('T')[0]}}<extra></extra>`
                    }});
                }}
            }});
            
            // Calculate price range for zoom
            const prices = windowData.flatMap(d => [d.high, d.low]);
            const fullMinPrice = Math.min(...prices);
            const fullMaxPrice = Math.max(...prices);
            const fullRange = fullMaxPrice - fullMinPrice;
            
            // Calculate zoomed range
            const zoomFactor = currentPriceZoom / 100;
            const zoomWindowSize = fullRange * zoomFactor;
            const availableRange = fullRange - zoomWindowSize;
            const positionOffset = availableRange * (currentPricePosition / 100);
            const zoomedMin = fullMinPrice + positionOffset;
            const zoomedMax = zoomedMin + zoomWindowSize;
            
            const layout = {{
                title: {{
                    text: 'USO Supply & Demand Formation Analysis',
                    font: {{ size: 18, color: 'white' }}
                }},
                xaxis: {{
                    title: 'Date',
                    gridcolor: '#444',
                    showgrid: true
                }},
                yaxis: {{
                    title: 'Price ($)',
                    gridcolor: '#444',
                    showgrid: true,
                    range: [zoomedMin, zoomedMax]
                }},
                plot_bgcolor: '#1a1a1a',
                paper_bgcolor: '#2a2a2a',
                font: {{ color: 'white' }},
                showlegend: false,
                margin: {{ t: 50, r: 30, b: 50, l: 60 }}
            }};
            
            const config = {{
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'resetScale2d', 'toImage']
            }};
            
            Plotly.newPlot('chart', traces, layout, config);
        }}
        
        // Initialize
        updateChart();
        updateWindowInfo();
        updatePriceRangeInfo();
    </script>
</body>
</html>
"""
    
    return html_content

if __name__ == "__main__":
    create_intuitive_formation_visualizer()