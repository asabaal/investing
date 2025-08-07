#!/usr/bin/env python3
"""
Sideways Trend Formation Progression
Step-by-step demonstration of how sideways trends are detected and formed
"""

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Dark theme
pio.templates.default = "plotly_dark"

def create_sideways_formation_progression():
    """Create step-by-step sideways trend formation demonstration"""
    
    print("🎯 Creating sideways trend formation progression...")
    
    # Create synthetic data that demonstrates sideways formation
    dates = pd.date_range(start='2023-01-01', periods=40, freq='D')
    
    # Create multiple examples showing different scenarios
    examples = create_sideways_examples(dates)
    
    # Create individual example charts
    for i, example in enumerate(examples):
        create_example_chart(example, i+1)
    
    # Create corrected rules overview
    create_corrected_sideways_overview()
    
    print(f"✅ Sideways trend progression created!")
    print(f"   📁 Files created:")
    for i, example in enumerate(examples):
        print(f"     • sideways_example_{i+1}_{example['name'].lower().replace(' ', '_')}.html")
    print(f"     • sideways_corrected_rules.html")

def create_sideways_examples(dates):
    """Create multiple sideways trend examples showing different behaviors"""
    
    examples = []
    
    # Example 1: Range Contraction - Range gets smaller over time
    example1 = [
        # SIDEWAYS TREND STARTS HERE (Origin = SH1)
        {'date': dates[0], 'open': 48.0, 'high': 52.0, 'low': 47.5, 'close': 51.5, 'type': 'swing_high_1', 'origin': True},  # ORIGIN POINT
        {'date': dates[1], 'open': 51.5, 'high': 51.8, 'low': 49.0, 'close': 49.5, 'type': 'normal'},
        {'date': dates[2], 'open': 49.5, 'high': 50.0, 'low': 46.0, 'close': 47.8, 'type': 'swing_low_1'},   # SL1 - Range: 46.0-52.0
        {'date': dates[3], 'open': 47.8, 'high': 49.5, 'low': 47.2, 'close': 48.9, 'type': 'normal'},
        {'date': dates[4], 'open': 48.9, 'high': 51.0, 'low': 48.0, 'close': 50.2, 'type': 'swing_high_2'}, # SH2 - Range: 46.0-51.0 (contracts up)
        {'date': dates[5], 'open': 50.2, 'high': 50.5, 'low': 49.8, 'close': 50.1, 'type': 'normal'},
        {'date': dates[6], 'open': 50.1, 'high': 50.8, 'low': 47.0, 'close': 48.0, 'type': 'swing_low_2'},   # SL2 - Range: 47.0-51.0 (contracts down)
        {'date': dates[7], 'open': 48.0, 'high': 49.8, 'low': 47.8, 'close': 49.2, 'type': 'normal'},
        {'date': dates[8], 'open': 49.2, 'high': 50.5, 'low': 49.0, 'close': 50.0, 'type': 'swing_high_3'}, # SH3 - Range: 47.0-50.5 (contracts up)
        {'date': dates[9], 'open': 50.0, 'high': 50.2, 'low': 49.5, 'close': 49.8, 'type': 'normal'},
        {'date': dates[10], 'open': 49.8, 'high': 50.0, 'low': 47.5, 'close': 48.2, 'type': 'swing_low_3'}, # SL3 - Range: 47.5-50.5 (contracts down)
        {'date': dates[11], 'open': 48.2, 'high': 49.8, 'low': 47.9, 'close': 49.1, 'type': 'normal'},
        # VIOLATION - breaks above most recent range (47.5-50.5)
        {'date': dates[12], 'open': 49.1, 'high': 51.8, 'low': 49.0, 'close': 51.2, 'type': 'violation'},
    ]
    
    examples.append({
        'name': 'Range Contraction',
        'data': example1,
        'description': 'Range gets progressively smaller - supported behavior'
    })
    
    # Example 2: Balanced Expansion - Range expands equally on both sides
    example2 = [
        # SIDEWAYS TREND STARTS HERE (Origin = SH1)
        {'date': dates[0], 'open': 48.0, 'high': 51.0, 'low': 47.5, 'close': 50.5, 'type': 'swing_high_1', 'origin': True},  # ORIGIN POINT
        {'date': dates[1], 'open': 50.5, 'high': 50.8, 'low': 49.0, 'close': 49.5, 'type': 'normal'},
        {'date': dates[2], 'open': 49.5, 'high': 50.0, 'low': 48.0, 'close': 48.5, 'type': 'swing_low_1'},   # SL1 - Range: 48.0-51.0 (3.0)
        {'date': dates[3], 'open': 48.5, 'high': 49.5, 'low': 48.2, 'close': 48.9, 'type': 'normal'},
        {'date': dates[4], 'open': 48.9, 'high': 51.5, 'low': 48.0, 'close': 51.0, 'type': 'swing_high_2'}, # SH2 - Range: 48.0-51.5 (3.5) +0.5 up
        {'date': dates[5], 'open': 51.0, 'high': 51.2, 'low': 49.8, 'close': 50.1, 'type': 'normal'},
        {'date': dates[6], 'open': 50.1, 'high': 50.8, 'low': 47.5, 'close': 48.0, 'type': 'swing_low_2'},   # SL2 - Range: 47.5-51.5 (4.0) +0.5 down (balanced)
        {'date': dates[7], 'open': 48.0, 'high': 49.8, 'low': 47.8, 'close': 49.2, 'type': 'normal'},
        {'date': dates[8], 'open': 49.2, 'high': 52.0, 'low': 49.0, 'close': 51.5, 'type': 'swing_high_3'}, # SH3 - Range: 47.5-52.0 (4.5) +0.5 up
        {'date': dates[9], 'open': 51.5, 'high': 51.8, 'low': 50.2, 'close': 50.8, 'type': 'normal'},
        {'date': dates[10], 'open': 50.8, 'high': 51.0, 'low': 47.0, 'close': 47.8, 'type': 'swing_low_3'}, # SL3 - Range: 47.0-52.0 (5.0) +0.5 down (balanced)
        {'date': dates[11], 'open': 47.8, 'high': 49.8, 'low': 47.2, 'close': 49.1, 'type': 'normal'},
        # VIOLATION - breaks above most recent range (47.0-52.0)
        {'date': dates[12], 'open': 49.1, 'high': 53.2, 'low': 49.0, 'close': 52.8, 'type': 'violation'},
    ]
    
    examples.append({
        'name': 'Balanced Expansion',
        'data': example2,
        'description': 'Range expands equally on both sides - supported behavior'
    })
    
    # Example 3: Imbalanced Expansion - Should terminate sideways trend
    example3 = [
        # SIDEWAYS TREND STARTS HERE (Origin = SH1)  
        {'date': dates[0], 'open': 48.0, 'high': 51.0, 'low': 47.5, 'close': 50.5, 'type': 'swing_high_1', 'origin': True},  # ORIGIN POINT
        {'date': dates[1], 'open': 50.5, 'high': 50.8, 'low': 49.0, 'close': 49.5, 'type': 'normal'},
        {'date': dates[2], 'open': 49.5, 'high': 50.0, 'low': 48.0, 'close': 48.5, 'type': 'swing_low_1'},   # SL1 - Range: 48.0-51.0 (3.0)
        {'date': dates[3], 'open': 48.5, 'high': 49.5, 'low': 48.2, 'close': 48.9, 'type': 'normal'},
        {'date': dates[4], 'open': 48.9, 'high': 53.0, 'low': 48.0, 'close': 52.5, 'type': 'swing_high_2'}, # SH2 - Range: 48.0-53.0 (5.0) +2.0 up
        {'date': dates[5], 'open': 52.5, 'high': 52.8, 'low': 51.0, 'close': 51.5, 'type': 'normal'},
        {'date': dates[6], 'open': 51.5, 'high': 52.0, 'low': 47.8, 'close': 48.2, 'type': 'swing_low_2'},   # SL2 - Range: 47.8-53.0 (5.2) +0.2 down - IMBALANCED!
        # This should END the sideways trend due to imbalanced expansion
        {'date': dates[7], 'open': 48.2, 'high': 49.8, 'low': 47.8, 'close': 49.2, 'type': 'trend_ended'},
    ]
    
    examples.append({
        'name': 'Imbalanced Expansion',
        'data': example3,
        'description': 'Range expands much more upward - sideways trend should END'
    })
    
    return examples

def create_example_chart(example, example_num):
    """Create chart for a specific example"""
    
    df = pd.DataFrame(example['data'])
    
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ))
    
    # Mark origin point
    origin_data = df[df.get('origin', False) == True]
    if len(origin_data) > 0:
        fig.add_trace(go.Scatter(
            x=origin_data['date'],
            y=origin_data['high'],
            mode='markers+text',
            marker=dict(color='#ffaa00', size=16, symbol='diamond'),
            text=['ORIGIN'],
            textposition='top center',
            textfont=dict(size=12, color='white'),
            name='Sideways Origin'
        ))
    
    # Add all swing points
    swing_highs = df[df['type'].str.contains('swing_high', na=False)]
    swing_lows = df[df['type'].str.contains('swing_low', na=False)]
    
    if len(swing_highs) > 0:
        fig.add_trace(go.Scatter(
            x=swing_highs['date'],
            y=swing_highs['high'],
            mode='markers+text',
            marker=dict(color='#ffaa00', size=10, symbol='triangle-down'),
            text=[f'SH{i+1}' for i in range(len(swing_highs))],
            textposition='top center',
            textfont=dict(size=9, color='white'),
            name='Swing Highs'
        ))
    
    if len(swing_lows) > 0:
        fig.add_trace(go.Scatter(
            x=swing_lows['date'],
            y=swing_lows['low'],
            mode='markers+text',
            marker=dict(color='#00aaff', size=10, symbol='triangle-up'),
            text=[f'SL{i+1}' for i in range(len(swing_lows))],
            textposition='bottom center',
            textfont=dict(size=9, color='white'),
            name='Swing Lows'
        ))
    
    # Show dynamic range evolution (most recent two swings)
    if len(swing_highs) >= 1 and len(swing_lows) >= 1:
        # Get most recent swing high and low
        recent_high = swing_highs.iloc[-1]['high'] if len(swing_highs) > 0 else None
        recent_low = swing_lows.iloc[-1]['low'] if len(swing_lows) > 0 else None
        
        if recent_high and recent_low:
            center_level = (recent_high + recent_low) / 2
            
            # Add range boundaries
            fig.add_hline(y=recent_high, line_dash="dot", line_color="#ffaa00", line_width=2,
                         annotation_text=f"Current Range High: ${recent_high:.1f}")
            fig.add_hline(y=recent_low, line_dash="dot", line_color="#00aaff", line_width=2,
                         annotation_text=f"Current Range Low: ${recent_low:.1f}")
            
            # Add center line (sideways trend line)
            if len(df) > 4:  # Show trend line after confirmation
                trend_start = df.iloc[0]['date']
                trend_end = df[df['type'] != 'violation'].iloc[-1]['date']
                
                fig.add_trace(go.Scatter(
                    x=[trend_start, trend_end],
                    y=[center_level, center_level],
                    mode='lines',
                    line=dict(color='#ffaa00', width=4),
                    name='Sideways Trend Line'
                ))
    
    # Mark violation if present
    violation_data = df[df['type'] == 'violation']
    if len(violation_data) > 0:
        fig.add_trace(go.Scatter(
            x=violation_data['date'],
            y=violation_data['high'],
            mode='markers+text',
            marker=dict(color='#ff0000', size=16, symbol='x'),
            text=['VIOLATION'],
            textposition='top center',
            textfont=dict(size=12, color='#ff0000'),
            name='Range Violation'
        ))
    
    # Layout
    fig.update_layout(
        title={
            'text': f"Example {example_num}: {example['name']}<br><sub>{example['description']}</sub>",
            'font': {'size': 18, 'color': 'white'},
            'x': 0.5
        },
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_dark',
        height=600,
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0.5)')
    )
    
    # Save the chart
    filename = f"sideways_example_{example_num}_{example['name'].lower().replace(' ', '_')}.html"
    filepath = f"/home/asabaal/asabaal_ventures/repos/investing/trend_progression/{filename}"
    fig.write_html(filepath)

def create_corrected_sideways_overview():
    """Create overview with corrected sideways trend rules"""
    
    fig = go.Figure()
    
    # Add text-only overview of corrected rules
    rules_text = """
    <b>CORRECTED Sideways Trend Rules:</b><br><br>
    
    <b>📍 ORIGIN POINT:</b><br>
    • Sideways trend STARTS at the 1st swing point (like directional trends)<br>
    • Not at the 4th swing point - that's just confirmation<br><br>
    
    <b>📊 DYNAMIC RANGE:</b><br>
    • Use the 2 MOST RECENT swing points to define current range<br>
    • NOT the original 4 swing points<br>
    • Range evolves as new swings form<br><br>
    
    <b>✅ SUPPORTED BEHAVIORS:</b><br>
    • Range Contraction: Range gets smaller over time<br>
    • Balanced Expansion: Range expands equally on both sides (within 5%)<br><br>
    
    <b>❌ UNSUPPORTED BEHAVIORS:</b><br>
    • Imbalanced Expansion: Range expands much more in one direction<br>
    • This indicates directional trending, not sideways consolidation<br><br>
    
    <b>🎯 TERMINATION:</b><br>
    • ANY candle breaks outside the CURRENT range (most recent 2 swings)<br>
    • High violation: candle.high > current_range_high<br>
    • Low violation: candle.low < current_range_low<br>
    • Trend ends at last candle BEFORE violation<br><br>
    
    <b>📈 VISUALIZATION:</b><br>
    • Horizontal line through center of CURRENT range<br>
    • Line starts at origin (1st swing)<br>
    • Line ends at termination (before violation)
    """
    
    fig.add_annotation(
        text=rules_text,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        xanchor="center", yanchor="middle",
        bgcolor="rgba(30,30,30,0.9)",
        bordercolor="white",
        borderwidth=2,
        font=dict(size=14, color="white")
    )
    
    # Layout
    fig.update_layout(
        title={
            'text': "Corrected Sideways Trend Detection Rules<br><sub>Based on your feedback and requirements</sub>",
            'font': {'size': 22, 'color': 'white'},
            'x': 0.5
        },
        template='plotly_dark',
        height=800,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    # Save the overview
    filepath = "/home/asabaal/asabaal_ventures/repos/investing/trend_progression/sideways_corrected_rules.html"
    fig.write_html(filepath)

def create_stage_chart(df, stage):
    """Create individual stage chart"""
    
    # Get data up to this stage
    stage_data = df.iloc[:stage['end_idx'] + 1].copy()
    
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=stage_data['date'],
        open=stage_data['open'],
        high=stage_data['high'],
        low=stage_data['low'],
        close=stage_data['close'],
        name='Price',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ))
    
    # Identify swing points for this stage
    swing_highs = stage_data[stage_data['type'].str.contains('swing_high', na=False)]
    swing_lows = stage_data[stage_data['type'].str.contains('swing_low', na=False)]
    
    # Add swing points
    if len(swing_highs) > 0:
        fig.add_trace(go.Scatter(
            x=swing_highs['date'],
            y=swing_highs['high'],
            mode='markers+text',
            marker=dict(color='#ffaa00', size=12, symbol='triangle-down'),
            text=[f'SH{i+1}' for i in range(len(swing_highs))],
            textposition='top center',
            textfont=dict(size=10, color='white'),
            name='Swing Highs'
        ))
    
    if len(swing_lows) > 0:
        fig.add_trace(go.Scatter(
            x=swing_lows['date'],
            y=swing_lows['low'],
            mode='markers+text',
            marker=dict(color='#00aaff', size=12, symbol='triangle-up'),
            text=[f'SL{i+1}' for i in range(len(swing_lows))],
            textposition='bottom center',
            textfont=dict(size=10, color='white'),
            name='Swing Lows'
        ))
    
    # Add sideways range if we have enough swings
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        high_level = max(swing_highs['high'].max(), 52.0)  # Current range high
        low_level = min(swing_lows['low'].min(), 47.0)     # Current range low
        center_level = (high_level + low_level) / 2
        
        # Add range boundaries
        fig.add_hline(y=high_level, line_dash="dot", line_color="#ffaa00", 
                     annotation_text=f"Range High: ${high_level:.1f}")
        fig.add_hline(y=low_level, line_dash="dot", line_color="#00aaff", 
                     annotation_text=f"Range Low: ${low_level:.1f}")
        
        # Add center line if in active stage
        if 'active' in stage['name'].lower() or 'violation' in stage['name'].lower():
            fig.add_hline(y=center_level, line_dash="solid", line_color="#ffaa00", line_width=3,
                         annotation_text=f"Sideways Center: ${center_level:.1f}")
    
    # Mark violation if present
    violation_data = stage_data[stage_data['type'] == 'violation']
    if len(violation_data) > 0:
        fig.add_trace(go.Scatter(
            x=violation_data['date'],
            y=violation_data['high'],
            mode='markers+text',
            marker=dict(color='#ff0000', size=16, symbol='x'),
            text=['VIOLATION'],
            textposition='top center',
            textfont=dict(size=12, color='#ff0000'),
            name='Range Violation'
        ))
    
    # Layout
    fig.update_layout(
        title={
            'text': f"{stage['name']}<br><sub>{stage['description']}</sub>",
            'font': {'size': 18, 'color': 'white'},
            'x': 0.5
        },
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_dark',
        height=600,
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0.5)')
    )
    
    # Save the chart
    filepath = f"/home/asabaal/asabaal_ventures/repos/investing/trend_progression/{stage['file']}"
    fig.write_html(filepath)

def create_sideways_overview(df):
    """Create comprehensive overview of sideways trend rules"""
    
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ))
    
    # Add all swing points
    swing_highs = df[df['type'].str.contains('swing_high', na=False)]
    swing_lows = df[df['type'].str.contains('swing_low', na=False)]
    
    fig.add_trace(go.Scatter(
        x=swing_highs['date'],
        y=swing_highs['high'],
        mode='markers',
        marker=dict(color='#ffaa00', size=8, symbol='triangle-down'),
        name='Swing Highs'
    ))
    
    fig.add_trace(go.Scatter(
        x=swing_lows['date'],
        y=swing_lows['low'],
        mode='markers',
        marker=dict(color='#00aaff', size=8, symbol='triangle-up'),
        name='Swing Lows'
    ))
    
    # Add sideways range
    high_level = 52.0
    low_level = 47.0
    center_level = (high_level + low_level) / 2
    
    # Range boundaries
    fig.add_hline(y=high_level, line_dash="dot", line_color="#ffaa00", 
                 annotation_text=f"Range High: ${high_level:.1f}")
    fig.add_hline(y=low_level, line_dash="dot", line_color="#00aaff", 
                 annotation_text=f"Range Low: ${low_level:.1f}")
    
    # Sideways trend line (center)
    sideways_start = df[df['type'] == 'swing_low_2']['date'].iloc[0]
    sideways_end = df[df['type'] == 'pre_violation']['date'].iloc[-1]
    
    fig.add_trace(go.Scatter(
        x=[sideways_start, sideways_end],
        y=[center_level, center_level],
        mode='lines',
        line=dict(color='#ffaa00', width=4),
        name='Sideways Trend'
    ))
    
    # Mark violation
    violation_data = df[df['type'] == 'violation']
    fig.add_trace(go.Scatter(
        x=violation_data['date'],
        y=violation_data['high'],
        mode='markers+text',
        marker=dict(color='#ff0000', size=16, symbol='x'),
        text=['VIOLATION'],
        textposition='top center',
        textfont=dict(size=12, color='#ff0000'),
        name='Range Violation'
    ))
    
    # Add rules text
    rules_text = """
    <b>Sideways Trend Formation Rules:</b><br>
    1. Need minimum 4 swing points (2 highs + 2 lows)<br>
    2. Range expansion must be balanced within 5%<br>
    3. Most recent pair of swings defines current range<br>
    4. Continues until ANY candle violates the range<br><br>
    <b>Termination:</b><br>
    • High violation: candle.high > range_high<br>
    • Low violation: candle.low < range_low<br>
    • Trend ends at last candle BEFORE violation
    """
    
    fig.add_annotation(
        text=rules_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        bgcolor="rgba(0,0,0,0.8)",
        bordercolor="white",
        borderwidth=1,
        font=dict(size=12, color="white")
    )
    
    # Layout
    fig.update_layout(
        title={
            'text': "Sideways Trend Detection - Complete Overview<br><sub>Formation Rules and Violation Logic</sub>",
            'font': {'size': 20, 'color': 'white'},
            'x': 0.5
        },
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_dark',
        height=700,
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0.5)')
    )
    
    # Save the overview
    filepath = "/home/asabaal/asabaal_ventures/repos/investing/trend_progression/sideways_trend_overview.html"
    fig.write_html(filepath)

if __name__ == "__main__":
    create_sideways_formation_progression()