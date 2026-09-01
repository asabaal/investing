#!/usr/bin/env python3
"""
Monotonic Run Examples for Formation Detection
Creates specific examples to nail down the monotonic run logic
"""

import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Dark theme
pio.templates.default = "plotly_dark"

def create_up_monotonic_with_down_candles():
    """UP monotonic run with mixed candle directions"""
    dates = pd.date_range('2024-01-01', periods=8, freq='D')
    ohlc_data = [
        # UP MONOTONIC RUN - overall price keeps going UP despite individual down candles
        [100, 102, 99, 102],   # 0 - UP candle: $100 → $102 (run starts at $102)
        [102, 103, 100, 101],  # 1 - DOWN candle: $102 → $101 BUT run continues (still above $100 start)
        [101, 105, 101, 104],  # 2 - UP candle: $101 → $104 (run progresses to $104)  
        [104, 106, 103, 103],  # 3 - DOWN candle: $104 → $103 BUT run continues (still above start, making progress)
        [103, 108, 103, 107],  # 4 - UP candle: $103 → $107 (run progresses to $107)
        [107, 108, 105, 106],  # 5 - DOWN candle: $107 → $106 BUT run continues (still well above start)
        
        # RUN ENDS HERE - next candle breaks monotonicity
        [106, 107, 98, 99],    # 6 - DOWN candle: $106 → $99 (BREAKS run - below start of $100)
        [99, 101, 98, 100],    # 7 - New potential run starts
    ]
    return pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close']), dates

def create_down_monotonic_with_up_candles():
    """DOWN monotonic run with mixed candle directions"""
    dates = pd.date_range('2024-01-01', periods=8, freq='D')
    ohlc_data = [
        # DOWN MONOTONIC RUN - overall price keeps going DOWN despite individual up candles
        [120, 121, 117, 118],  # 0 - DOWN candle: $120 → $118 (run starts at $118)
        [118, 119, 115, 117],  # 1 - UP candle: $118 → $117 BUT run continues (still below $120 start)
        [117, 118, 113, 114],  # 2 - DOWN candle: $117 → $114 (run progresses to $114)
        [114, 116, 113, 115],  # 3 - UP candle: $114 → $115 BUT run continues (still below start, making progress down)
        [115, 116, 111, 112],  # 4 - DOWN candle: $115 → $112 (run progresses to $112)
        [112, 114, 111, 113],  # 5 - UP candle: $112 → $113 BUT run continues (still well below start)
        
        # RUN ENDS HERE - next candle breaks monotonicity  
        [113, 122, 113, 121],  # 6 - UP candle: $113 → $121 (BREAKS run - above start of $120)
        [121, 123, 120, 122],  # 7 - New potential run starts
    ]
    return pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close']), dates

def create_edge_case_examples():
    """Edge cases for monotonic run detection"""
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    ohlc_data = [
        # CASE 1: UP run with sideways movement
        [100, 102, 99, 101],   # 0 - UP candle: $100 → $101 (run starts)
        [101, 103, 100, 102],  # 1 - UP candle: $101 → $102 (slight progress)
        [102, 103, 101, 102],  # 2 - FLAT candle: $102 → $102 (no progress but doesn't break?)
        [102, 104, 101, 103],  # 3 - UP candle: $102 → $103 (continues)
        
        # CASE 2: When does "no progress" become "run ended"?
        [103, 104, 100, 101],  # 4 - DOWN candle: $103 → $101 (still above start but significant retreat)
        [101, 102, 99, 100],   # 5 - DOWN candle: $101 → $100 (back to start level - does run end?)
        [100, 101, 98, 99],    # 6 - DOWN candle: $100 → $99 (below start - run definitely ends)
        
        # CASE 3: New run or continuation?
        [99, 103, 99, 102],    # 7 - UP candle: new run or old run revival?
        [102, 105, 101, 104],  # 8 - UP candle
        [104, 107, 103, 106],  # 9 - UP candle
    ]
    return pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close']), dates

def create_question_examples():
    """Specific questionable scenarios to test logic"""
    dates = pd.date_range('2024-01-01', periods=6, freq='D')
    ohlc_data = [
        # QUESTION 1: Does opening matter?
        [100, 103, 99, 102],   # 0 - UP candle: close $102
        [105, 106, 103, 104],  # 1 - DOWN candle: opens ABOVE previous close, closes above start
        
        # QUESTION 2: What about gaps?
        [104, 105, 101, 103],  # 2 - DOWN candle: gaps down on open
        [98, 108, 98, 107],    # 3 - UP candle: gaps down but closes much higher
        
        # QUESTION 3: Which price matters for monotonicity?
        [107, 109, 95, 96],    # 4 - DOWN candle: high continues up, close breaks down significantly  
        [96, 98, 94, 97],      # 5 - UP candle: trying to recover
    ]
    return pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close']), dates

def create_monotonic_visualization(df, dates, title, run_analysis):
    """Create visualization showing monotonic run analysis"""
    
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
    
    # Color code candles by run participation
    for i, analysis in enumerate(run_analysis):
        candle_time = df['datetime'].iloc[i]
        candle = df.iloc[i]
        
        # Add annotation for each candle
        color = '#00ff88' if analysis['in_run'] else '#ff4444'
        
        fig.add_annotation(
            x=candle_time,
            y=candle['high'] + 2,
            text=f"#{i}<br>{analysis['status']}<br>Close: ${candle['close']:.0f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=color,
            font=dict(size=10, color=color),
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor=color,
            borderwidth=1
        )
    
    # Add run start/end markers
    run_starts = [i for i, a in enumerate(run_analysis) if a.get('run_start')]
    run_ends = [i for i, a in enumerate(run_analysis) if a.get('run_end')]
    
    for start_idx in run_starts:
        fig.add_shape(
            type="line",
            x0=df['datetime'].iloc[start_idx], x1=df['datetime'].iloc[start_idx],
            y0=df['low'].min() * 0.95, y1=df['high'].max() * 1.05,
            line=dict(color='#00ccff', width=3, dash='dash'),
        )
        fig.add_annotation(
            x=df['datetime'].iloc[start_idx],
            y=df['high'].max() * 1.06,
            text="RUN START",
            showarrow=False,
            font=dict(size=12, color='#00ccff'),
            bgcolor='rgba(0,204,255,0.2)'
        )
    
    for end_idx in run_ends:
        fig.add_shape(
            type="line", 
            x0=df['datetime'].iloc[end_idx], x1=df['datetime'].iloc[end_idx],
            y0=df['low'].min() * 0.95, y1=df['high'].max() * 1.05,
            line=dict(color='#ff6600', width=3, dash='dash'),
        )
        fig.add_annotation(
            x=df['datetime'].iloc[end_idx],
            y=df['high'].max() * 1.06,
            text="RUN END",
            showarrow=False,
            font=dict(size=12, color='#ff6600'),
            bgcolor='rgba(255,102,0,0.2)'
        )
    
    fig.update_layout(
        title=f'{title}',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#2a2a2a',
        font=dict(color='white'),
        height=700,
        showlegend=False
    )
    
    return fig

def main():
    """Generate monotonic run examples to nail down the logic"""
    
    print("🔧 GENERATING MONOTONIC RUN EXAMPLES")
    print("=" * 50)
    
    # Example 1: UP run with down candles
    print("\n📊 Example 1: UP Monotonic Run with DOWN Candles")
    df1, dates1 = create_up_monotonic_with_down_candles()
    
    # Manual analysis - QUESTION: What should the logic be?
    run_analysis_1 = [
        {'in_run': True, 'run_start': True, 'status': 'UP RUN START\n100→102', 'run_type': 'UP'},
        {'in_run': True, 'status': 'UP RUN continues\nOpen 102 ≥ prev close 102 ✓', 'run_type': 'UP'},
        {'in_run': True, 'status': 'UP RUN continues\nClose 104 > prev close 101 ✓', 'run_type': 'UP'},
        {'in_run': True, 'status': 'UP RUN continues\nOpen 104 ≥ prev close 104 ✓', 'run_type': 'UP'},
        {'in_run': True, 'status': 'UP RUN continues\nClose 107 > prev close 103 ✓', 'run_type': 'UP'},
        {'in_run': True, 'status': 'UP RUN continues\nOpen 107 ≥ prev close 107 ✓\n(Close 106 < prev close, but open continues)', 'run_type': 'UP'},
        {'in_run': True, 'run_start': True, 'run_end': True, 'status': 'BOTH RUN END + START\nENDS UP run (violates it)\nSTARTS DOWN run\nNEITHER open nor close ≥ prev close 106', 'run_type': 'DOWN'},
        {'in_run': True, 'status': 'DOWN RUN continues\nCandles 6-7: 106→99→100', 'run_type': 'DOWN'},
    ]
    
    fig1 = create_monotonic_visualization(df1, dates1, "UP Monotonic Run with DOWN Candles", run_analysis_1)
    fig1.write_html("formation_examples/up_monotonic_with_down_candles.html")
    print("   ✅ Saved: formation_examples/up_monotonic_with_down_candles.html")
    
    # Example 2: DOWN run with up candles  
    print("\n📊 Example 2: DOWN Monotonic Run with UP Candles")
    df2, dates2 = create_down_monotonic_with_up_candles()
    
    run_analysis_2 = [
        {'in_run': True, 'run_start': True, 'status': 'DOWN RUN START\n120→118', 'run_type': 'DOWN'},
        {'in_run': True, 'status': 'DOWN RUN continues\nOpen 118 ≤ prev close 118 ✓', 'run_type': 'DOWN'},
        {'in_run': True, 'status': 'DOWN RUN continues\nClose 114 < prev close 117 ✓', 'run_type': 'DOWN'},
        {'in_run': True, 'status': 'DOWN RUN continues\nOpen 114 ≤ prev close 114 ✓', 'run_type': 'DOWN'},
        {'in_run': True, 'status': 'DOWN RUN continues\nClose 112 < prev close 115 ✓', 'run_type': 'DOWN'},
        {'in_run': True, 'run_start': True, 'run_end': True, 'status': 'BOTH RUN END + START\nENDS DOWN run\nSTARTS UP run\nOpen 112 ≤ prev close 112 ✓', 'run_type': 'UP'},
        {'in_run': True, 'status': 'UP RUN continues\nClose 121 > prev close 113 ✓', 'run_type': 'UP'},
        {'in_run': True, 'status': 'UP RUN continues\nCandles 6-7: 113→121→122', 'run_type': 'UP'},
    ]
    
    fig2 = create_monotonic_visualization(df2, dates2, "DOWN Monotonic Run with UP Candles", run_analysis_2)
    fig2.write_html("formation_examples/down_monotonic_with_up_candles.html")
    print("   ✅ Saved: formation_examples/down_monotonic_with_up_candles.html")
    
    # Example 3: Edge cases
    print("\n📊 Example 3: Edge Cases")
    df3, dates3 = create_edge_case_examples()
    
    run_analysis_3 = [
        {'in_run': True, 'run_start': True, 'status': 'UP RUN START\n100→101', 'run_type': 'UP'},
        {'in_run': True, 'status': 'UP RUN continues\nClose 102 > prev close 101 ✓', 'run_type': 'UP'},
        {'in_run': True, 'status': 'UP RUN continues\nOpen 102 = prev close 102 ✓ (flat)', 'run_type': 'UP'},
        {'in_run': True, 'run_end': True, 'status': 'UP RUN ENDS HERE\nLAST candle in run\nClose 103 > prev close 102 ✓', 'run_type': 'UP'},
        {'in_run': True, 'run_start': True, 'status': 'DOWN RUN STARTS\nViolates UP run + starts DOWN\n103→101 breaks UP', 'run_type': 'DOWN'},
        {'in_run': True, 'status': 'DOWN RUN continues\nClose 100 < prev close 101 ✓', 'run_type': 'DOWN'},
        {'in_run': True, 'run_end': True, 'status': 'DOWN RUN ENDS HERE\nLAST candle in run\nClose 99 < prev close 100 ✓', 'run_type': 'DOWN'},
        {'in_run': True, 'run_start': True, 'status': 'UP RUN STARTS\nViolates DOWN run + starts UP\n99→102 breaks DOWN', 'run_type': 'UP'},
        {'in_run': True, 'status': 'UP RUN continues\nClose 104 > prev close 102 ✓', 'run_type': 'UP'},
        {'in_run': True, 'status': 'UP RUN continues\nClose 106 > prev close 104 ✓', 'run_type': 'UP'},
    ]
    
    fig3 = create_monotonic_visualization(df3, dates3, "Edge Cases for Monotonic Runs", run_analysis_3)
    fig3.write_html("formation_examples/edge_cases_monotonic.html")  
    print("   ✅ Saved: formation_examples/edge_cases_monotonic.html")
    
    # Example 4: Specific questions
    print("\n📊 Example 4: Specific Questions")
    df4, dates4 = create_question_examples()
    
    run_analysis_4 = [
        {'in_run': True, 'run_start': True, 'status': 'UP RUN START\n100→102', 'run_type': 'UP'},
        {'in_run': True, 'run_start': True, 'run_end': True, 'status': 'BOTH RUN END + START\nENDS UP run\nSTARTS DOWN run\nOpen 105 > prev close 102 ✓', 'run_type': 'DOWN'},
        {'in_run': True, 'run_end': True, 'status': 'DOWN RUN ENDS HERE\nBreaks DOWN run\nNEITHER open nor close ≤ prev close 104'},
        {'in_run': False, 'status': 'NO RUN - Big gap\n98→107 (expansion, not monotonic run)'},
        {'in_run': True, 'run_start': True, 'status': 'DOWN RUN STARTS\n107→96', 'run_type': 'DOWN'},
        {'in_run': True, 'run_end': True, 'status': 'DOWN RUN ENDS HERE\nLAST candle in run\nOpen 96 ≤ prev close 96 ✓', 'run_type': 'DOWN'},
    ]
    
    fig4 = create_monotonic_visualization(df4, dates4, "Questions About Monotonic Run Logic", run_analysis_4)
    fig4.write_html("formation_examples/questions_monotonic.html")
    print("   ✅ Saved: formation_examples/questions_monotonic.html")
    
    print(f"\n🔧 COMPLETE! Generated 4 monotonic run examples:")
    print(f"   • up_monotonic_with_down_candles.html")
    print(f"   • down_monotonic_with_up_candles.html") 
    print(f"   • edge_cases_monotonic.html")
    print(f"   • questions_monotonic.html")
    print(f"\n❓ KEY QUESTIONS TO RESOLVE:")
    print(f"   1. Does a DOWN candle continue an UP run if close > run_start?")
    print(f"   2. Which price matters: open, close, high, low?")
    print(f"   3. How much retreat is allowed before run ends?")
    print(f"   4. What happens at flat/sideways candles?")
    print(f"   5. When is it a new run vs continuation of old run?")

if __name__ == "__main__":
    main()