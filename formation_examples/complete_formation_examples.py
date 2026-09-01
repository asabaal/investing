#!/usr/bin/env python3
"""
Complete Formation Examples - All 4 Formation Types
Creates complete formation examples using correct monotonic run logic:
- RBR (Rally-Base-Rally) 
- DBD (Drop-Base-Drop)
- RBD (Rally-Base-Drop)  
- DBR (Drop-Base-Rally)

Each example shows:
- LEG IN: Monotonic run using BODY PRICES rule
- BASE: Consolidation period with smaller range
- LEG OUT: Monotonic run breaking out of base
- Validation: Leg movements > base range
"""

import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Dark theme
pio.templates.default = "plotly_dark"

def create_rbr_formation():
    """Rally-Base-Rally formation with proper monotonic runs"""
    dates = pd.date_range('2024-01-01', periods=18, freq='D')
    ohlc_data = [
        # RALLY IN (LEG IN) - Candles 0-5
        [100, 102, 99, 101],   # 0 - UP run starts: 100→101
        [102, 104, 101, 103],  # 1 - UP run continues: open 102 > prev close 101 ✓
        [103, 105, 102, 104],  # 2 - UP run continues: close 104 > prev close 103 ✓  
        [105, 106, 103, 105],  # 3 - UP run continues: open 105 > prev close 104 ✓
        [105, 108, 104, 107],  # 4 - UP run continues: close 107 > prev close 105 ✓
        [107, 109, 106, 108],  # 5 - UP run ENDS: close 108 > prev close 107 ✓
        
        # BASE (consolidation) - Candles 6-11  
        [108, 110, 107, 109],  # 6 - Base starts (violates UP run)
        [109, 110, 107, 108],  # 7 - Base continues
        [108, 111, 107, 109],  # 8 - Base continues
        [109, 111, 108, 110],  # 9 - Base continues
        [110, 111, 108, 109],  # 10 - Base continues
        [109, 112, 108, 111],  # 11 - Base ends
        
        # RALLY OUT (LEG OUT) - Candles 12-17
        [111, 114, 110, 113],  # 12 - UP run starts (breaks above base): 111→113
        [113, 116, 112, 115],  # 13 - UP run continues: close 115 > prev close 113 ✓
        [115, 118, 114, 117],  # 14 - UP run continues: close 117 > prev close 115 ✓
        [117, 119, 116, 118],  # 15 - UP run continues: close 118 > prev close 117 ✓
        [118, 121, 117, 120],  # 16 - UP run continues: close 120 > prev close 118 ✓
        [120, 123, 119, 122],  # 17 - UP run continues: close 122 > prev close 120 ✓
    ]
    return pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close']), dates

def create_dbd_formation():
    """Drop-Base-Drop formation with proper monotonic runs"""
    dates = pd.date_range('2024-01-01', periods=18, freq='D')
    ohlc_data = [
        # DROP IN (LEG IN) - Candles 0-5
        [120, 121, 118, 119],  # 0 - DOWN run starts: 120→119
        [118, 119, 116, 117],  # 1 - DOWN run continues: open 118 < prev close 119 ✓
        [117, 118, 115, 116],  # 2 - DOWN run continues: close 116 < prev close 117 ✓
        [115, 116, 113, 114],  # 3 - DOWN run continues: open 115 < prev close 116 ✓
        [114, 115, 112, 113],  # 4 - DOWN run continues: close 113 < prev close 114 ✓
        [113, 114, 111, 112],  # 5 - DOWN run ENDS: close 112 < prev close 113 ✓
        
        # BASE (consolidation) - Candles 6-11
        [112, 115, 111, 114],  # 6 - Base starts (violates DOWN run)
        [114, 116, 112, 113],  # 7 - Base continues
        [113, 116, 112, 115],  # 8 - Base continues
        [115, 116, 113, 114],  # 9 - Base continues
        [114, 116, 113, 115],  # 10 - Base continues
        [115, 116, 112, 113],  # 11 - Base ends
        
        # DROP OUT (LEG OUT) - Candles 12-17
        [113, 114, 110, 111],  # 12 - DOWN run starts (breaks below base): 113→111
        [111, 112, 108, 109],  # 13 - DOWN run continues: close 109 < prev close 111 ✓
        [109, 110, 106, 107],  # 14 - DOWN run continues: close 107 < prev close 109 ✓
        [107, 108, 104, 105],  # 15 - DOWN run continues: close 105 < prev close 107 ✓
        [105, 106, 102, 103],  # 16 - DOWN run continues: close 103 < prev close 105 ✓
        [103, 104, 100, 101],  # 17 - DOWN run continues: close 101 < prev close 103 ✓
    ]
    return pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close']), dates

def create_rbd_formation():
    """Rally-Base-Drop formation (reversal) with proper monotonic runs"""
    dates = pd.date_range('2024-01-01', periods=18, freq='D')
    ohlc_data = [
        # RALLY IN (LEG IN) - Candles 0-5
        [100, 102, 99, 101],   # 0 - UP run starts: 100→101
        [102, 104, 101, 103],  # 1 - UP run continues: open 102 > prev close 101 ✓
        [103, 105, 102, 104],  # 2 - UP run continues: close 104 > prev close 103 ✓
        [105, 106, 103, 105],  # 3 - UP run continues: open 105 > prev close 104 ✓
        [105, 108, 104, 107],  # 4 - UP run continues: close 107 > prev close 105 ✓
        [107, 109, 106, 108],  # 5 - UP run ENDS: close 108 > prev close 107 ✓
        
        # BASE (consolidation) - Candles 6-11
        [108, 110, 107, 109],  # 6 - Base starts (violates UP run)
        [109, 110, 107, 108],  # 7 - Base continues
        [108, 111, 107, 109],  # 8 - Base continues
        [109, 111, 108, 110],  # 9 - Base continues
        [110, 111, 108, 109],  # 10 - Base continues
        [109, 112, 108, 111],  # 11 - Base ends
        
        # DROP OUT (LEG OUT) - Candles 12-17 (REVERSAL!)
        [111, 112, 108, 109],  # 12 - DOWN run starts (breaks below base): 111→109
        [109, 110, 106, 107],  # 13 - DOWN run continues: close 107 < prev close 109 ✓
        [107, 108, 104, 105],  # 14 - DOWN run continues: close 105 < prev close 107 ✓
        [105, 106, 102, 103],  # 15 - DOWN run continues: close 103 < prev close 105 ✓
        [103, 104, 100, 101],  # 16 - DOWN run continues: close 101 < prev close 103 ✓
        [101, 102, 98, 99],    # 17 - DOWN run continues: close 99 < prev close 101 ✓
    ]
    return pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close']), dates

def create_dbr_formation():
    """Drop-Base-Rally formation (reversal) with proper monotonic runs"""
    dates = pd.date_range('2024-01-01', periods=18, freq='D')
    ohlc_data = [
        # DROP IN (LEG IN) - Candles 0-5
        [120, 121, 118, 119],  # 0 - DOWN run starts: 120→119
        [118, 119, 116, 117],  # 1 - DOWN run continues: open 118 < prev close 119 ✓
        [117, 118, 115, 116],  # 2 - DOWN run continues: close 116 < prev close 117 ✓
        [115, 116, 113, 114],  # 3 - DOWN run continues: open 115 < prev close 116 ✓
        [114, 115, 112, 113],  # 4 - DOWN run continues: close 113 < prev close 114 ✓
        [113, 114, 111, 112],  # 5 - DOWN run ENDS: close 112 < prev close 113 ✓
        
        # BASE (consolidation) - Candles 6-11
        [112, 115, 111, 114],  # 6 - Base starts (violates DOWN run)
        [114, 116, 112, 113],  # 7 - Base continues
        [113, 116, 112, 115],  # 8 - Base continues
        [115, 116, 113, 114],  # 9 - Base continues
        [114, 116, 113, 115],  # 10 - Base continues
        [115, 116, 112, 113],  # 11 - Base ends
        
        # RALLY OUT (LEG OUT) - Candles 12-17 (REVERSAL!)
        [113, 116, 112, 115],  # 12 - UP run starts (breaks above base): 113→115
        [115, 118, 114, 117],  # 13 - UP run continues: close 117 > prev close 115 ✓
        [117, 120, 116, 119],  # 14 - UP run continues: close 119 > prev close 117 ✓
        [119, 122, 118, 121],  # 15 - UP run continues: close 121 > prev close 119 ✓
        [121, 124, 120, 123],  # 16 - UP run continues: close 123 > prev close 121 ✓
        [123, 126, 122, 125],  # 17 - UP run continues: close 125 > prev close 123 ✓
    ]
    return pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close']), dates

def analyze_monotonic_runs(df, formation_type):
    """Analyze the monotonic runs in the formation data"""
    analysis = []
    
    if formation_type == 'RBR':
        # LEG IN: UP run (candles 0-5)
        for i in range(6):
            if i == 0:
                analysis.append({
                    'candle': i, 'segment': 'LEG_IN', 'run_type': 'UP',
                    'run_start': True, 'run_end': False,
                    'status': 'UP run starts'
                })
            elif i == 5:
                analysis.append({
                    'candle': i, 'segment': 'LEG_IN', 'run_type': 'UP', 
                    'run_start': False, 'run_end': True,
                    'status': 'UP run ENDS (leg in complete)'
                })
            else:
                analysis.append({
                    'candle': i, 'segment': 'LEG_IN', 'run_type': 'UP',
                    'run_start': False, 'run_end': False,
                    'status': 'UP run continues'
                })
        
        # BASE: No runs (candles 6-11)
        for i in range(6, 12):
            analysis.append({
                'candle': i, 'segment': 'BASE', 'run_type': None,
                'run_start': False, 'run_end': False, 
                'status': 'Base consolidation'
            })
        
        # LEG OUT: UP run (candles 12-17)
        for i in range(12, 18):
            if i == 12:
                analysis.append({
                    'candle': i, 'segment': 'LEG_OUT', 'run_type': 'UP',
                    'run_start': True, 'run_end': False,
                    'status': 'UP run starts (breakout above base)'
                })
            elif i == 17:
                analysis.append({
                    'candle': i, 'segment': 'LEG_OUT', 'run_type': 'UP',
                    'run_start': False, 'run_end': True,
                    'status': 'UP run continues (leg out continues)'
                })
            else:
                analysis.append({
                    'candle': i, 'segment': 'LEG_OUT', 'run_type': 'UP',
                    'run_start': False, 'run_end': False,
                    'status': 'UP run continues'
                })
    
    elif formation_type == 'DBD':
        # LEG IN: DOWN run (candles 0-5)  
        for i in range(6):
            if i == 0:
                analysis.append({
                    'candle': i, 'segment': 'LEG_IN', 'run_type': 'DOWN',
                    'run_start': True, 'run_end': False,
                    'status': 'DOWN run starts'
                })
            elif i == 5:
                analysis.append({
                    'candle': i, 'segment': 'LEG_IN', 'run_type': 'DOWN',
                    'run_start': False, 'run_end': True,
                    'status': 'DOWN run ENDS (leg in complete)'
                })
            else:
                analysis.append({
                    'candle': i, 'segment': 'LEG_IN', 'run_type': 'DOWN',
                    'run_start': False, 'run_end': False,
                    'status': 'DOWN run continues'
                })
        
        # BASE: No runs (candles 6-11)
        for i in range(6, 12):
            analysis.append({
                'candle': i, 'segment': 'BASE', 'run_type': None,
                'run_start': False, 'run_end': False,
                'status': 'Base consolidation'
            })
        
        # LEG OUT: DOWN run (candles 12-17)
        for i in range(12, 18):
            if i == 12:
                analysis.append({
                    'candle': i, 'segment': 'LEG_OUT', 'run_type': 'DOWN',
                    'run_start': True, 'run_end': False,
                    'status': 'DOWN run starts (breakout below base)'
                })
            elif i == 17:
                analysis.append({
                    'candle': i, 'segment': 'LEG_OUT', 'run_type': 'DOWN',
                    'run_start': False, 'run_end': True,
                    'status': 'DOWN run continues (leg out continues)'
                })
            else:
                analysis.append({
                    'candle': i, 'segment': 'LEG_OUT', 'run_type': 'DOWN',
                    'run_start': False, 'run_end': False,
                    'status': 'DOWN run continues'
                })
    
    elif formation_type == 'RBD':
        # LEG IN: UP run (candles 0-5)
        for i in range(6):
            if i == 0:
                analysis.append({
                    'candle': i, 'segment': 'LEG_IN', 'run_type': 'UP',
                    'run_start': True, 'run_end': False,
                    'status': 'UP run starts'
                })
            elif i == 5:
                analysis.append({
                    'candle': i, 'segment': 'LEG_IN', 'run_type': 'UP',
                    'run_start': False, 'run_end': True,
                    'status': 'UP run ENDS (leg in complete)'
                })
            else:
                analysis.append({
                    'candle': i, 'segment': 'LEG_IN', 'run_type': 'UP',
                    'run_start': False, 'run_end': False,
                    'status': 'UP run continues'
                })
        
        # BASE: No runs (candles 6-11)
        for i in range(6, 12):
            analysis.append({
                'candle': i, 'segment': 'BASE', 'run_type': None,
                'run_start': False, 'run_end': False,
                'status': 'Base consolidation'
            })
        
        # LEG OUT: DOWN run (candles 12-17) - REVERSAL
        for i in range(12, 18):
            if i == 12:
                analysis.append({
                    'candle': i, 'segment': 'LEG_OUT', 'run_type': 'DOWN',
                    'run_start': True, 'run_end': False,
                    'status': 'DOWN run starts (REVERSAL - breakout below base)'
                })
            elif i == 17:
                analysis.append({
                    'candle': i, 'segment': 'LEG_OUT', 'run_type': 'DOWN',
                    'run_start': False, 'run_end': True,
                    'status': 'DOWN run continues (leg out continues)'
                })
            else:
                analysis.append({
                    'candle': i, 'segment': 'LEG_OUT', 'run_type': 'DOWN',
                    'run_start': False, 'run_end': False,
                    'status': 'DOWN run continues'
                })
    
    elif formation_type == 'DBR':
        # LEG IN: DOWN run (candles 0-5)
        for i in range(6):
            if i == 0:
                analysis.append({
                    'candle': i, 'segment': 'LEG_IN', 'run_type': 'DOWN',
                    'run_start': True, 'run_end': False,
                    'status': 'DOWN run starts'
                })
            elif i == 5:
                analysis.append({
                    'candle': i, 'segment': 'LEG_IN', 'run_type': 'DOWN',
                    'run_start': False, 'run_end': True,
                    'status': 'DOWN run ENDS (leg in complete)'
                })
            else:
                analysis.append({
                    'candle': i, 'segment': 'LEG_IN', 'run_type': 'DOWN',
                    'run_start': False, 'run_end': False,
                    'status': 'DOWN run continues'
                })
        
        # BASE: No runs (candles 6-11)
        for i in range(6, 12):
            analysis.append({
                'candle': i, 'segment': 'BASE', 'run_type': None,
                'run_start': False, 'run_end': False,
                'status': 'Base consolidation'
            })
        
        # LEG OUT: UP run (candles 12-17) - REVERSAL
        for i in range(12, 18):
            if i == 12:
                analysis.append({
                    'candle': i, 'segment': 'LEG_OUT', 'run_type': 'UP',
                    'run_start': True, 'run_end': False,
                    'status': 'UP run starts (REVERSAL - breakout above base)'
                })
            elif i == 17:
                analysis.append({
                    'candle': i, 'segment': 'LEG_OUT', 'run_type': 'UP',
                    'run_start': False, 'run_end': True,
                    'status': 'UP run continues (leg out continues)'
                })
            else:
                analysis.append({
                    'candle': i, 'segment': 'LEG_OUT', 'run_type': 'UP',
                    'run_start': False, 'run_end': False,
                    'status': 'UP run continues'
                })
    
    return analysis

def validate_formation(df, formation_type, leg_in_range, base_range, leg_out_range):
    """Validate the formation using the fundamental rule"""
    
    # Calculate leg in movement
    leg_in_start = df.iloc[leg_in_range[0]]['close']
    leg_in_end = df.iloc[leg_in_range[1]]['close']
    leg_in_movement = abs(leg_in_end - leg_in_start)
    
    # Calculate base range
    base_data = df.iloc[base_range[0]:base_range[1]+1]
    base_high = base_data['high'].max()
    base_low = base_data['low'].min()
    base_consolidation_range = base_high - base_low
    
    # Calculate leg out movement
    leg_out_start = df.iloc[leg_out_range[0]]['close']
    leg_out_end = df.iloc[leg_out_range[1]]['close']
    leg_out_movement = abs(leg_out_end - leg_out_start)
    
    # Validation: Both legs must be > base range
    leg_in_valid = leg_in_movement > base_consolidation_range
    leg_out_valid = leg_out_movement > base_consolidation_range
    formation_valid = leg_in_valid and leg_out_valid
    
    # Determine zone type
    if formation_type in ['RBR', 'DBR']:
        zone_type = 'DEMAND'
    else:  # RBD, DBD
        zone_type = 'SUPPLY'
    
    return {
        'valid': formation_valid,
        'leg_in_movement': leg_in_movement,
        'leg_out_movement': leg_out_movement, 
        'base_range': base_consolidation_range,
        'leg_in_valid': leg_in_valid,
        'leg_out_valid': leg_out_valid,
        'zone_type': zone_type,
        'base_high': base_high,
        'base_low': base_low,
        'validation_details': f"Leg In: {leg_in_movement:.1f}, Base: {base_consolidation_range:.1f}, Leg Out: {leg_out_movement:.1f}"
    }

def create_formation_visualization(df, dates, formation_type, analysis, validation):
    """Create comprehensive formation visualization"""
    
    df['datetime'] = dates[:len(df)]
    
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ))
    
    # Color coding for segments
    segment_colors = {
        'LEG_IN': 'rgba(0, 255, 136, 0.3)',
        'BASE': 'rgba(255, 170, 0, 0.4)',
        'LEG_OUT': 'rgba(0, 170, 255, 0.3)'
    }
    
    # Add segment backgrounds
    leg_in_start, leg_in_end = 0, 5
    base_start, base_end = 6, 11
    leg_out_start, leg_out_end = 12, 17
    
    # LEG IN background
    fig.add_shape(
        type="rect",
        x0=df['datetime'].iloc[leg_in_start],
        x1=df['datetime'].iloc[leg_in_end],
        y0=df['low'].min() * 0.97,
        y1=df['high'].max() * 1.03,
        fillcolor=segment_colors['LEG_IN'],
        opacity=0.3,
        line=dict(width=2, color='#00ff88')
    )
    
    # BASE background
    fig.add_shape(
        type="rect",
        x0=df['datetime'].iloc[base_start],
        x1=df['datetime'].iloc[base_end],
        y0=df['low'].min() * 0.97,
        y1=df['high'].max() * 1.03,
        fillcolor=segment_colors['BASE'],
        opacity=0.4,
        line=dict(width=3, color='#ffaa00')
    )
    
    # LEG OUT background
    fig.add_shape(
        type="rect",
        x0=df['datetime'].iloc[leg_out_start],
        x1=df['datetime'].iloc[leg_out_end],
        y0=df['low'].min() * 0.97,
        y1=df['high'].max() * 1.03,
        fillcolor=segment_colors['LEG_OUT'],
        opacity=0.3,
        line=dict(width=2, color='#00aaff')
    )
    
    # Add supply/demand zone over BASE only
    zone_color = '#00ff88' if validation['zone_type'] == 'DEMAND' else '#ff4444'
    zone_fill = 'rgba(0, 255, 136, 0.2)' if validation['zone_type'] == 'DEMAND' else 'rgba(255, 68, 68, 0.2)'
    
    fig.add_shape(
        type="rect",
        x0=df['datetime'].iloc[base_start],
        x1=df['datetime'].iloc[base_end],
        y0=validation['base_low'],
        y1=validation['base_high'],
        fillcolor=zone_fill,
        opacity=0.6,
        line=dict(width=3, color=zone_color)
    )
    
    # Add formation validation annotations
    status = '✅ VALID' if validation['valid'] else '❌ INVALID'
    fig.add_annotation(
        x=df['datetime'].iloc[9],  # Middle of formation
        y=df['high'].max() * 1.05,
        text=f"<b>{formation_type} FORMATION {status}</b><br>" +
             f"{validation['zone_type']} Zone<br>" +
             f"Validation: {validation['validation_details']}",
        showarrow=False,
        font=dict(size=14, color='cyan'),
        bgcolor='rgba(0,0,0,0.8)',
        bordercolor='cyan',
        borderwidth=2
    )
    
    # Add segment labels
    fig.add_annotation(
        x=df['datetime'].iloc[2],
        y=df['high'].max() * 1.01,
        text=f"<b>LEG IN</b><br>Movement: {validation['leg_in_movement']:.1f}",
        showarrow=False,
        font=dict(size=12, color='#00ff88'),
        bgcolor='rgba(0,0,0,0.7)'
    )
    
    fig.add_annotation(
        x=df['datetime'].iloc[9],
        y=df['low'].min() * 0.96,
        text=f"<b>BASE</b><br>Range: {validation['base_range']:.1f}",
        showarrow=False,
        font=dict(size=12, color='#ffaa00'),
        bgcolor='rgba(0,0,0,0.7)'
    )
    
    fig.add_annotation(
        x=df['datetime'].iloc[15],
        y=df['high'].max() * 1.01,
        text=f"<b>LEG OUT</b><br>Movement: {validation['leg_out_movement']:.1f}",
        showarrow=False,
        font=dict(size=12, color='#00aaff'),
        bgcolor='rgba(0,0,0,0.7)'
    )
    
    # Add run start/end markers
    for item in analysis:
        candle_idx = item['candle']
        candle_time = df['datetime'].iloc[candle_idx]
        candle_data = df.iloc[candle_idx]
        
        if item['run_start']:
            fig.add_shape(
                type="line",
                x0=candle_time, x1=candle_time,
                y0=candle_data['low'] - 2, y1=candle_data['high'] + 2,
                line=dict(color='#00ccff', width=4, dash='dash')
            )
            fig.add_annotation(
                x=candle_time,
                y=candle_data['high'] + 4,
                text="▲ RUN START",
                showarrow=False,
                font=dict(size=10, color='#00ccff'),
                bgcolor='rgba(0,204,255,0.2)'
            )
        
        if item['run_end']:
            fig.add_shape(
                type="line",
                x0=candle_time, x1=candle_time,
                y0=candle_data['low'] - 2, y1=candle_data['high'] + 2,
                line=dict(color='#ff6600', width=4, dash='dot')
            )
            fig.add_annotation(
                x=candle_time,
                y=candle_data['low'] - 4,
                text="▼ RUN END",
                showarrow=False,
                font=dict(size=10, color='#ff6600'),
                bgcolor='rgba(255,102,0,0.2)'
            )
    
    fig.update_layout(
        title=f'{formation_type} Formation - Complete Example with Monotonic Runs',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#2a2a2a',
        font=dict(color='white'),
        height=800,
        showlegend=False
    )
    
    return fig

def main():
    """Generate complete formation examples for all 4 types"""
    
    print("🎯 GENERATING COMPLETE FORMATION EXAMPLES")
    print("=" * 60)
    print("Using correct monotonic run logic with BODY PRICES rule")
    print("Each formation shows: LEG IN → BASE → LEG OUT structure")
    
    formations = [
        ('RBR', create_rbr_formation(), "Rally-Base-Rally (Continuation)"),
        ('DBD', create_dbd_formation(), "Drop-Base-Drop (Continuation)"),
        ('RBD', create_rbd_formation(), "Rally-Base-Drop (Reversal)"),
        ('DBR', create_dbr_formation(), "Drop-Base-Rally (Reversal)")
    ]
    
    for formation_type, (df, dates), description in formations:
        print(f"\n📊 Creating {formation_type} formation example...")
        print(f"   {description}")
        
        # Analyze monotonic runs
        analysis = analyze_monotonic_runs(df, formation_type)
        
        # Validate formation
        leg_in_range = (0, 5)
        base_range = (6, 11)
        leg_out_range = (12, 17)
        validation = validate_formation(df, formation_type, leg_in_range, base_range, leg_out_range)
        
        # Create visualization
        fig = create_formation_visualization(df, dates, formation_type, analysis, validation)
        
        # Save
        filename = f"complete_{formation_type.lower()}_formation.html"
        fig.write_html(f"formation_examples/{filename}")
        print(f"   ✅ Saved: formation_examples/{filename}")
        print(f"      Validation: {validation['validation_details']}")
        print(f"      Status: {'✅ VALID' if validation['valid'] else '❌ INVALID'}")
    
    print(f"\n🎯 COMPLETE! Generated all 4 formation types:")
    print(f"   • complete_rbr_formation.html - Rally-Base-Rally (Demand zone)")
    print(f"   • complete_dbd_formation.html - Drop-Base-Drop (Supply zone)")
    print(f"   • complete_rbd_formation.html - Rally-Base-Drop (Supply zone)")
    print(f"   • complete_dbr_formation.html - Drop-Base-Rally (Demand zone)")
    print(f"\nEach example shows:")
    print(f"   ✓ Proper monotonic runs using BODY PRICES rule")
    print(f"   ✓ LEG movements > BASE consolidation range validation")
    print(f"   ✓ Run start/end markers")
    print(f"   ✓ Supply/demand zone identification")
    print(f"   ✓ Formation type classification")

if __name__ == "__main__":
    main()