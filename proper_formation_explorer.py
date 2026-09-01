#!/usr/bin/env python3
"""
Proper Formation Explorer - Browse Individual Formations
Shows each formation with all details:
- Formation segments (Leg In, Base, Leg Out)
- Base range visualization
- Validation metrics
- Supply/demand zone levels
"""

import plotly.graph_objects as go
import plotly.io as pio
from two_pass_formation_detector import TwoPassFormationDetector
from uso_supply_demand_visualizer import SupplyDemandVisualizer
import json
from datetime import datetime

# Dark theme
pio.templates.default = "plotly_dark"

def create_proper_formation_explorer():
    """
    Create formation explorer that browses individual formations
    """
    
    print("🎯 PROPER FORMATION EXPLORER")
    print("Creating individual formation browser with complete analysis")
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
    
    # Find all formations using two-pass algorithm
    detector = TwoPassFormationDetector(decay_factor=0.7, min_leg_threshold=1.5)
    formations = detector.detect_formations(df)
    
    # Filter to valid formations only
    valid_formations = [f for f in formations if f['valid']]
    
    print(f"✅ Found {len(valid_formations)} valid formations")
    print(f"   • {len([f for f in valid_formations if f['type'] == 'RBR'])} RBR (Rally-Base-Rally)")
    print(f"   • {len([f for f in valid_formations if f['type'] == 'DBD'])} DBD (Drop-Base-Drop)") 
    print(f"   • {len([f for f in valid_formations if f['type'] == 'RBD'])} RBD (Rally-Base-Drop)")
    print(f"   • {len([f for f in valid_formations if f['type'] == 'DBR'])} DBR (Drop-Base-Rally)")
    
    # Save formation data to file with actual price/date details
    formation_details = []
    for i, formation in enumerate(valid_formations):
        # Get actual dates and prices for each segment
        leg_in_start_idx = formation['leg_in']['start_idx'] 
        leg_in_end_idx = formation['leg_in']['end_idx']
        base_start_idx = min(formation['base']['base_candles'])
        base_end_idx = max(formation['base']['base_candles'])
        leg_out_start_idx = formation['leg_out']['start_idx']
        leg_out_end_idx = formation['leg_out']['end_idx']
        
        # Get extended candle data for detailed analysis (5 candles before formation start, 5 after formation end)
        extended_start = max(0, formation['start_idx'] - 5)
        extended_end = min(len(df) - 1, formation['end_idx'] + 5)
        
        candle_details = []
        for idx in range(extended_start, extended_end + 1):
            candle = df.iloc[idx]
            candle_details.append({
                'index': idx,
                'date': candle['datetime'].strftime('%Y-%m-%d'),
                'open': f"${candle['open']:.2f}",
                'high': f"${candle['high']:.2f}", 
                'low': f"${candle['low']:.2f}",
                'close': f"${candle['close']:.2f}",
                'open_raw': candle['open'],
                'high_raw': candle['high'],
                'low_raw': candle['low'],
                'close_raw': candle['close']
            })
        
        formation_detail = {
            'id': i + 1,
            'type': formation['type'],
            'zone_type': formation['zone_type'],
            'leg_in': {
                'start_date': df.iloc[leg_in_start_idx]['datetime'].strftime('%Y-%m-%d'),
                'end_date': df.iloc[leg_in_end_idx]['datetime'].strftime('%Y-%m-%d'),
                'start_price': f"${df.iloc[leg_in_start_idx]['close']:.2f}",
                'end_price': f"${df.iloc[leg_in_end_idx]['close']:.2f}",
                'movement': f"${formation['validation_details']['leg_in_movement']:.2f}",
                'candles': f"{leg_in_start_idx}-{leg_in_end_idx}"
            },
            'base': {
                'start_date': df.iloc[base_start_idx]['datetime'].strftime('%Y-%m-%d'),
                'end_date': df.iloc[base_end_idx]['datetime'].strftime('%Y-%m-%d'),
                'high': f"${formation['base']['actual_high']:.2f}",
                'low': f"${formation['base']['actual_low']:.2f}",
                'weighted_high': f"${formation['base']['base_range']['high']:.2f}",
                'weighted_low': f"${formation['base']['base_range']['low']:.2f}",
                'range': f"${formation['base']['base_range']['range']:.2f}",
                'visual_range': f"${formation['base']['actual_high'] - formation['base']['actual_low']:.2f}",
                'candle_count': formation['base']['base_range']['candle_count'],
                'candles': f"{base_start_idx}-{base_end_idx}",
                'candle_list': formation['base']['base_candles']
            },
            'leg_out': {
                'start_date': df.iloc[leg_out_start_idx]['datetime'].strftime('%Y-%m-%d'),
                'end_date': df.iloc[leg_out_end_idx]['datetime'].strftime('%Y-%m-%d'), 
                'start_price': f"${df.iloc[leg_out_start_idx]['close']:.2f}",
                'end_price': f"${df.iloc[leg_out_end_idx]['close']:.2f}",
                'movement': f"${formation['validation_details']['leg_out_movement']:.2f}",
                'candles': f"{leg_out_start_idx}-{leg_out_end_idx}"
            },
            'validation': {
                'valid': formation['valid'],
                'leg_in_ratio': f"{formation['validation_details']['leg_in_movement'] / formation['validation_details']['base_range']:.2f}x",
                'leg_out_ratio': f"{formation['validation_details']['leg_out_movement'] / formation['validation_details']['base_range']:.2f}x"
            },
            'candle_details': candle_details
        }
        formation_details.append(formation_detail)
    
    # Save to JSON file
    import json
    with open('formation_details.json', 'w') as f:
        json.dump(formation_details, f, indent=2)
    print(f"💾 Saved detailed formation data to: formation_details.json")
    
    # Prepare data for JavaScript - SAVE THE RAW CANDLE DATA!
    df_json = df.to_dict('records')
    for i, record in enumerate(df_json):
        record['index'] = i
        if 'datetime' in record:
            record['datetime'] = record['datetime'].isoformat() if hasattr(record['datetime'], 'isoformat') else str(record['datetime'])
        # Ensure we have all the price data
        record['open'] = float(record['open']) if 'open' in record else 0.0
        record['high'] = float(record['high']) if 'high' in record else 0.0  
        record['low'] = float(record['low']) if 'low' in record else 0.0
        record['close'] = float(record['close']) if 'close' in record else 0.0
    
    # Prepare formation data for detailed analysis
    formation_data = []
    for i, formation in enumerate(valid_formations):
        formation_date = df.iloc[formation['start_idx']]['datetime']
        formation_date_str = formation_date.isoformat() if hasattr(formation_date, 'isoformat') else str(formation_date)
        
        formation_info = {
            'id': i + 1,
            'type': formation['type'],
            'zone_type': formation['zone_type'],
            'start_idx': formation['start_idx'],
            'end_idx': formation['end_idx'],
            'leg_in': {
                'start_idx': formation['leg_in']['start_idx'],
                'end_idx': formation['leg_in']['end_idx'],
                'sentiment': formation['leg_in']['direction'],
                'movement': formation['validation_details']['leg_in_movement'],
                'valid': formation['validation_details']['leg_in_valid']
            },
            'base': {
                'candles': formation['base']['base_candles'],
                'range': formation['base']['base_range']['range'],
                'candle_count': formation['base']['base_range']['candle_count'],
                'weight': formation['base']['base_range']['total_weight'],
                'proximal_line': formation['base']['proximal_line'],
                'distal_line': formation['base']['distal_line']
            },
            'leg_out': {
                'start_idx': formation['leg_out']['start_idx'],
                'end_idx': formation['leg_out']['end_idx'],
                'sentiment': formation['leg_out']['direction'],
                'movement': formation['validation_details']['leg_out_movement'],
                'valid': formation['validation_details']['leg_out_valid']
            },
            'validation': {
                'is_valid': formation['valid'],
                'leg_in_ratio': formation['validation_details']['leg_in_movement'] / max(formation['validation_details']['base_range'], 0.001),
                'leg_out_ratio': formation['validation_details']['leg_out_movement'] / max(formation['validation_details']['base_range'], 0.001),
                'threshold': 1.5
            },
            'formation_date': formation_date_str,
            'status': 'VALID' if formation['valid'] else 'INVALID'
        }
        
        formation_data.append(formation_info)
    
    # Create the HTML interface
    html_content = create_formation_browser_html(df_json, formation_data)
    
    # Save the explorer
    with open('proper_formation_explorer.html', 'w') as f:
        f.write(html_content)
    
    print(f"✅ Proper formation explorer saved as: proper_formation_explorer.html")
    print(f"🎯 Features:")
    print(f"   • Browse {len(valid_formations)} individual formations")
    print(f"   • Detailed formation analysis with validation metrics")
    print(f"   • Supply/demand zone visualization")
    print(f"   • Formation type filtering and navigation")
    print(f"   • Base segment and leg analysis")

def create_formation_browser_html(df_data, formation_data):
    """Create the HTML interface for browsing individual formations"""
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Proper Formation Explorer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a; 
            color: white; 
            margin: 0; 
            padding: 0;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
            text-align: center;
            padding: 20px;
            border-bottom: 2px solid #333;
        }}
        
        .header h1 {{
            margin: 0;
            color: #00ff88;
            font-size: 28px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header p {{
            margin: 8px 0 0 0;
            color: #ccc;
            font-size: 14px;
        }}
        
        .control-panel {{
            background: #2a2a2a;
            padding: 20px;
            border-bottom: 1px solid #333;
            display: grid;
            grid-template-columns: 1fr 2fr 1fr;
            gap: 20px;
            align-items: start;
        }}
        
        .formation-selector {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        
        .formation-info {{
            background: #333;
            border-radius: 8px;
            padding: 15px;
            border-left: 4px solid #00ff88;
        }}
        
        .formation-info h3 {{
            margin: 0 0 10px 0;
            color: #00ff88;
            font-size: 18px;
        }}
        
        .navigation {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .nav-btn {{
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.2s;
            text-align: center;
        }}
        
        .nav-btn:hover {{
            background: linear-gradient(135deg, #0056b3 0%, #007bff 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,123,255,0.3);
        }}
        
        .nav-btn:disabled {{
            background: #555;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }}
        
        .formation-counter {{
            background: #444;
            padding: 8px 12px;
            border-radius: 4px;
            text-align: center;
            font-weight: bold;
            color: #ffaa00;
            font-size: 12px;
            margin-bottom: 8px;
        }}
        
        .rbr {{ border-left: 4px solid #00ff88; }}
        .dbr {{ border-left: 4px solid #00ff88; }}
        .rbd {{ border-left: 4px solid #ff4444; }}
        .dbd {{ border-left: 4px solid #ff4444; }}
        
        select {{
            background: #333;
            color: white;
            border: 1px solid #555;
            padding: 8px;
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
            margin: 20px;
            min-height: 600px;
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
        }}
        
        .formation-details {{
            background: rgba(0,100,0,0.1);
            border: 1px solid #00ff88;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            font-size: 11px;
        }}
        
        .validation-details {{
            background: rgba(100,100,0,0.1);
            border: 1px solid #ffaa00;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            font-size: 11px;
        }}
        
        .detail-row {{
            display: flex;
            justify-content: space-between;
            margin: 3px 0;
        }}
        
        .detail-label {{
            color: #ccc;
            font-weight: bold;
        }}
        
        .detail-value {{
            color: white;
        }}
        
        .valid {{
            color: #00ff88;
            font-weight: bold;
        }}
        
        .invalid {{
            color: #ff4444;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Proper Formation Explorer</h1>
        <p>Browse through {len(formation_data)} valid formation setups with complete supply/demand analysis</p>
    </div>
    
    <div class="control-panel">
        <div class="formation-selector">
            <label>Filter by Formation Type:</label>
            <select id="typeFilter" onchange="filterFormationsByType()">
                <option value="ALL">All Formations</option>
                <option value="RBR">RBR (Rally-Base-Rally)</option>
                <option value="DBD">DBD (Drop-Base-Drop)</option>
                <option value="RBD">RBD (Rally-Base-Drop)</option>
                <option value="DBR">DBR (Drop-Base-Rally)</option>
            </select>
            
            <label>Select Formation:</label>
            <select id="formationSelect" onchange="loadFormation()">
                {chr(10).join([f'<option value="{i}">{formation["type"]} #{formation["id"]} ({formation["formation_date"].split("T")[0]}) - {formation["status"]}</option>' for i, formation in enumerate(formation_data)])}
            </select>
            
            <div class="formation-counter" id="formationCounter">
                Formation 1 of {len(formation_data)}
            </div>
        </div>
        
        <div class="formation-info" id="formationInfo">
            <h3 id="formationTitle">Loading...</h3>
            <div id="formationDetails"></div>
        </div>
        
        <div class="navigation">
            <button class="nav-btn" onclick="previousFormation()" id="prevBtn">← Previous</button>
            <button class="nav-btn" onclick="nextFormation()" id="nextBtn">Next →</button>
            <button class="nav-btn" onclick="jumpToRandom()" style="background: linear-gradient(135deg, #ff8c00 0%, #ff6b00 100%);">🎲 Random</button>
        </div>
    </div>
    
    <div id="chart"></div>
    
    <script>
        // Data from Python
        const dfData = {json.dumps(df_data)};
        const allFormationData = {json.dumps(formation_data, default=str)};
        let filteredFormationData = [...allFormationData];
        let currentFormationIndex = 0;
        
        function filterFormationsByType() {{
            const typeFilter = document.getElementById('typeFilter');
            const selectedType = typeFilter.value;
            
            if (selectedType === 'ALL') {{
                filteredFormationData = [...allFormationData];
            }} else {{
                filteredFormationData = allFormationData.filter(formation => formation.type === selectedType);
            }}
            
            // Update the formation selector dropdown
            updateFormationSelector();
            
            // Reset to first formation in filtered list
            currentFormationIndex = 0;
            loadFormation();
        }}
        
        function updateFormationSelector() {{
            const formationSelect = document.getElementById('formationSelect');
            formationSelect.innerHTML = '';
            
            filteredFormationData.forEach((formation, index) => {{
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `${{formation.type}} #${{formation.id}} (${{formation.formation_date.split('T')[0]}}) - ${{formation.status}}`;
                formationSelect.appendChild(option);
            }});
            
            // Update counter
            const counter = document.getElementById('formationCounter');
            counter.textContent = `Formation 1 of ${{filteredFormationData.length}}`;
        }}
        
        function loadFormation() {{
            const formationSelect = document.getElementById('formationSelect');
            currentFormationIndex = parseInt(formationSelect.value);
            
            const formation = filteredFormationData[currentFormationIndex];
            if (!formation) return;
            
            // Update info panel
            updateFormationInfo(formation);
            
            // Update chart
            updateChart(formation);
            
            // Update navigation buttons
            updateNavigation();
            
            // Update counter
            const counter = document.getElementById('formationCounter');
            counter.textContent = `Formation ${{currentFormationIndex + 1}} of ${{filteredFormationData.length}}`;
        }}
        
        function updateFormationInfo(formation) {{
            const title = document.getElementById('formationTitle');
            const details = document.getElementById('formationDetails');
            const info = document.getElementById('formationInfo');
            
            title.textContent = `${{formation.type}} Formation #${{formation.id}}`;
            
            // Set border color based on formation type
            info.className = `formation-info ${{formation.type.toLowerCase()}}`;
            
            details.innerHTML = `
                <div class="formation-details">
                    <div class="detail-row">
                        <span class="detail-label">Formation Date:</span>
                        <span class="detail-value">${{formation.formation_date.split('T')[0]}}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Zone Type:</span>
                        <span class="detail-value">${{formation.zone_type}}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Range:</span>
                        <span class="detail-value">Index ${{formation.start_idx}} - ${{formation.end_idx}}</span>
                    </div>
                </div>
                
                <div class="validation-details">
                    <div class="detail-row">
                        <span class="detail-label">Leg In Movement:</span>
                        <span class="detail-value ${{formation.leg_in.valid ? 'valid' : 'invalid'}}">
                            $${{formation.leg_in.movement.toFixed(3)}} (${{formation.validation.leg_in_ratio.toFixed(2)}}x)
                        </span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Base Range:</span>
                        <span class="detail-value">$${{formation.base.range.toFixed(4)}} (${{formation.base.candle_count}} candles)</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Leg Out Movement:</span>
                        <span class="detail-value ${{formation.leg_out.valid ? 'valid' : 'invalid'}}">
                            $${{formation.leg_out.movement.toFixed(3)}} (${{formation.validation.leg_out_ratio.toFixed(2)}}x)
                        </span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Validation Status:</span>
                        <span class="detail-value ${{formation.validation.is_valid ? 'valid' : 'invalid'}}">
                            ${{formation.validation.is_valid ? '✅ VALID' : '❌ INVALID'}}
                        </span>
                    </div>
                </div>
            `;
        }}
        
        function updateChart(formation) {{
            console.log('Debug: updateChart called with formation:', formation);
            
            // Get data window around the formation (20 candles before, 10 after)
            const contextBefore = 20;
            const contextAfter = 10;
            const startIdx = Math.max(0, formation.start_idx - contextBefore);
            const endIdx = Math.min(dfData.length - 1, formation.end_idx + contextAfter);
            
            console.log('Debug: Chart window:', startIdx, 'to', endIdx);
            
            const windowData = dfData.slice(startIdx, endIdx + 1);
            
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
            
            // Formation zone
            const zoneColor = formation.zone_type === 'SUPPLY' ? '#ff4444' : '#00ff88';
            
            // Formation boundaries for reference lines
            const startCandle = dfData[formation.start_idx];
            const endCandle = dfData[formation.end_idx];
            
            // Draw zone ONLY over base candles, not entire formation (with bounds checking)
            const baseStart = Math.min(...formation.base.candles);
            const baseEnd = Math.max(...formation.base.candles);
            const baseStartCandle = dfData[baseStart];
            const baseEndCandle = dfData[baseEnd];
            
            // Ensure base candles exist
            if (!baseStartCandle || !baseEndCandle) {{
                console.log('Debug: Base candles not found, skipping zone visualization');
                return;
            }}
            
            // Add formation zone rectangle (only over base segment)
            const fillColor = zoneColor === '#00ff88' ? 'rgba(0, 255, 136, 0.2)' : 'rgba(255, 68, 68, 0.2)';
            traces.push({{
                type: 'scatter',
                mode: 'lines',
                x: [baseStartCandle.datetime, baseEndCandle.datetime, baseEndCandle.datetime, baseStartCandle.datetime, baseStartCandle.datetime],
                y: [formation.base.proximal_line, formation.base.proximal_line, formation.base.distal_line, formation.base.distal_line, formation.base.proximal_line],
                fill: 'toself',
                fillcolor: fillColor,
                line: {{ color: zoneColor, width: 2 }},
                name: `${{formation.type}} Zone (Base Only)`,
                hovertemplate: `<b>${{formation.type}} Zone</b><br>Covers base candles only<br>Range: $${{formation.base.range.toFixed(4)}}<extra></extra>`
            }});
            
            // Leg In line segment (from actual start to actual end) - with bounds checking
            if (formation.leg_in.start_idx < dfData.length && formation.leg_in.end_idx < dfData.length) {{
                const legInStartCandle = dfData[formation.leg_in.start_idx];
                const legInEndCandle = dfData[formation.leg_in.end_idx];
                traces.push({{
                    type: 'scatter',
                    mode: 'lines',
                    x: [legInStartCandle.datetime, legInEndCandle.datetime],
                    y: [legInStartCandle.close, legInEndCandle.close],
                    line: {{ color: '#00ccff', width: 4 }},
                    name: 'Leg In Movement',
                    hovertemplate: `<b>Leg In Movement</b><br>Movement: $${{formation.leg_in.movement.toFixed(2)}}<extra></extra>`
                }});
            }}
            
            // Horizontal line at center of base range  
            const baseCenterPrice = (formation.base.proximal_line + formation.base.distal_line) / 2;
            traces.push({{
                type: 'scatter',
                mode: 'lines',
                x: [baseStartCandle.datetime, baseEndCandle.datetime],
                y: [baseCenterPrice, baseCenterPrice],
                line: {{ color: '#ffaa00', width: 3, dash: 'dash' }},
                name: 'Base Center Line',
                hovertemplate: `<b>Base Center</b><br>Price: $${{baseCenterPrice.toFixed(2)}}<extra></extra>`
            }});
            
            // Leg Out line segment (from actual start to actual end) - with bounds checking
            if (formation.leg_out.start_idx < dfData.length && formation.leg_out.end_idx < dfData.length) {{
                const legOutStartCandle = dfData[formation.leg_out.start_idx];
                const legOutEndCandle = dfData[formation.leg_out.end_idx];
                traces.push({{
                    type: 'scatter',
                    mode: 'lines',
                    x: [legOutStartCandle.datetime, legOutEndCandle.datetime],
                    y: [legOutStartCandle.close, legOutEndCandle.close],
                    line: {{ color: '#ff6600', width: 4 }},
                    name: 'Leg Out Movement',
                    hovertemplate: `<b>Leg Out Movement</b><br>Movement: $${{formation.leg_out.movement.toFixed(2)}}<extra></extra>`
                }});
            }}
            
            // Proximal line (entry level)
            traces.push({{
                type: 'scatter',
                mode: 'lines',
                x: [startCandle.datetime, endCandle.datetime],
                y: [formation.base.proximal_line, formation.base.proximal_line],
                line: {{ color: zoneColor, width: 3, dash: 'dash' }},
                name: 'Proximal Line',
                hovertemplate: `<b>Proximal Line</b><br>Price: $${{formation.base.proximal_line.toFixed(2)}}<extra></extra>`
            }});
            
            // Distal line (stop level)
            traces.push({{
                type: 'scatter',
                mode: 'lines',
                x: [startCandle.datetime, endCandle.datetime],
                y: [formation.base.distal_line, formation.base.distal_line],
                line: {{ color: zoneColor, width: 3 }},
                name: 'Distal Line',
                hovertemplate: `<b>Distal Line</b><br>Price: $${{formation.base.distal_line.toFixed(2)}}<extra></extra>`
            }});
            
            // Base candles markers
            formation.base.candles.forEach(baseIdx => {{
                if (baseIdx >= startIdx && baseIdx <= endIdx) {{
                    const baseCandle = dfData[baseIdx];
                    traces.push({{
                        type: 'scatter',
                        mode: 'markers',
                        x: [baseCandle.datetime],
                        y: [(baseCandle.high + baseCandle.low) / 2],
                        marker: {{ color: '#ffaa00', size: 10, symbol: 'square' }},
                        name: 'Base Candle',
                        hovertemplate: `<b>Base Candle</b><br>Index: ${{baseIdx}}<extra></extra>`,
                        showlegend: baseIdx === formation.base.candles[0]
                    }});
                }}
            }});
            
            // Leg markers - mark the start of each leg (with bounds checking)
            if (formation.leg_in.start_idx < dfData.length) {{
                const legInStartCandle = dfData[formation.leg_in.start_idx];
                traces.push({{
                    type: 'scatter',
                    mode: 'markers+text',
                    x: [legInStartCandle.datetime],
                    y: [formation.leg_in.sentiment === 'UP' ? legInStartCandle.high : legInStartCandle.low],
                    marker: {{ color: '#00ccff', size: 12, symbol: 'triangle-up' }},
                    text: ['LEG IN START'],
                    textposition: 'top center',
                    textfont: {{ size: 10, color: 'white', family: 'Arial Black' }},
                    name: 'Leg In Start',
                    hovertemplate: `<b>Leg In Start</b><br>Movement: $${{formation.leg_in.movement.toFixed(3)}}<br>Valid: ${{formation.leg_in.valid ? '✅' : '❌'}}<extra></extra>`
                }});
            }}
            
            if (formation.leg_out.start_idx < dfData.length) {{
                const legOutStartCandle = dfData[formation.leg_out.start_idx];
                traces.push({{
                    type: 'scatter',
                    mode: 'markers+text',
                    x: [legOutStartCandle.datetime],
                    y: [formation.leg_out.sentiment === 'UP' ? legOutStartCandle.high : legOutStartCandle.low],
                    marker: {{ color: '#ff6600', size: 12, symbol: 'triangle-down' }},
                    text: ['LEG OUT START'],
                    textposition: 'bottom center',
                    textfont: {{ size: 10, color: 'white', family: 'Arial Black' }},
                    name: 'Leg Out Start',
                    hovertemplate: `<b>Leg Out Start</b><br>Movement: $${{formation.leg_out.movement.toFixed(3)}}<br>Valid: ${{formation.leg_out.valid ? '✅' : '❌'}}<extra></extra>`
                }});
            }}
            
            const layout = {{
                title: {{
                    text: `${{formation.type}} Formation #${{formation.id}} - ${{formation.zone_type}} Zone (${{formation.validation.is_valid ? 'VALID' : 'INVALID'}})`,
                    font: {{ size: 16, color: 'white' }}
                }},
                xaxis: {{
                    title: 'Date',
                    gridcolor: '#444',
                    showgrid: true
                }},
                yaxis: {{
                    title: 'Price ($)',
                    gridcolor: '#444',
                    showgrid: true
                }},
                plot_bgcolor: '#1a1a1a',
                paper_bgcolor: '#2a2a2a',
                font: {{ color: 'white' }},
                showlegend: true,
                legend: {{
                    x: 0,
                    y: 1,
                    bgcolor: 'rgba(0,0,0,0.5)'
                }},
                margin: {{ t: 50, r: 30, b: 50, l: 60 }}
            }};
            
            const config = {{
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d']
            }};
            
            Plotly.newPlot('chart', traces, layout, config);
        }}
        
        function previousFormation() {{
            if (currentFormationIndex > 0) {{
                currentFormationIndex--;
                document.getElementById('formationSelect').value = currentFormationIndex;
                loadFormation();
            }}
        }}
        
        function nextFormation() {{
            if (currentFormationIndex < filteredFormationData.length - 1) {{
                currentFormationIndex++;
                document.getElementById('formationSelect').value = currentFormationIndex;
                loadFormation();
            }}
        }}
        
        function jumpToRandom() {{
            const randomIndex = Math.floor(Math.random() * filteredFormationData.length);
            currentFormationIndex = randomIndex;
            document.getElementById('formationSelect').value = currentFormationIndex;
            loadFormation();
        }}
        
        function updateNavigation() {{
            const prevBtn = document.getElementById('prevBtn');
            const nextBtn = document.getElementById('nextBtn');
            
            prevBtn.disabled = currentFormationIndex === 0;
            nextBtn.disabled = currentFormationIndex === filteredFormationData.length - 1;
        }}
        
        // Initialize with debugging
        console.log('Debug: Starting formation explorer...');
        console.log('Debug: dfData length:', dfData.length);
        console.log('Debug: allFormationData length:', allFormationData.length);
        
        if (allFormationData.length === 0) {{
            console.log('Debug: No formations found!');
            document.getElementById('chart').innerHTML = '<p style="color: red; text-align: center; padding: 50px;">No formations found</p>';
        }} else {{
            console.log('Debug: Found formations, loading first one...');
            loadFormation();
        }}
    </script>
</body>
</html>
"""
    
    return html_content

if __name__ == "__main__":
    create_proper_formation_explorer()