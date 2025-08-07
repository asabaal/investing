#!/usr/bin/env python3
"""
Concrete Trend Formation/Termination Examples
Using daily USO data to identify specific candles where trends form and terminate
Based on: 3 swing points + BREAKOUT candle = formation, VIOLATION = termination
"""

from daily_trend_examples import analyze_daily_uso_trends
import plotly.graph_objects as go
import plotly.io as pio

# Dark theme
pio.templates.default = "plotly_dark"

def identify_trend_examples():
    """Identify concrete examples of trend formation and termination"""
    
    print("🎯 Concrete Trend Formation/Termination Examples")
    print("=" * 60)
    
    # Get the daily data and swing points
    df, swing_points = analyze_daily_uso_trends()
    
    print(f"\n🔍 ANALYZING TREND FORMATION EXAMPLES:")
    print("Looking for: 3 swing points + BREAKOUT candle")
    
    # Let's examine potential uptrend formations
    examples = []
    
    # Example 1: Look for uptrend formation around April
    print(f"\n📈 UPTREND FORMATION EXAMPLE 1:")
    print("Swing points around April 2025:")
    
    april_swings = [
        {'index': 11, 'date': '2025-03-28', 'type': 'LOW', 'price': 74.54},  # SL1
        {'index': 13, 'date': '2025-04-01', 'type': 'HIGH', 'price': 78.01}, # SH1
        {'index': 19, 'date': '2025-04-09', 'type': 'LOW', 'price': 60.67},  # SL2 - Wait, this is LOWER low!
    ]
    
    print("SL1 (Mar 28): $74.54")
    print("SH1 (Apr 01): $78.01") 
    print("SL2 (Apr 09): $60.67")
    print("❌ This is NOT an uptrend setup - SL2 < SL1 (lower low, not higher low)")
    
    # Example 2: Look for uptrend formation in May
    print(f"\n📈 UPTREND FORMATION EXAMPLE 2:")
    print("Swing points in May 2025:")
    
    may_swings = [
        {'index': 36, 'date': '2025-05-05', 'type': 'LOW', 'price': 61.75},   # SL1
        {'index': 37, 'date': '2025-05-06', 'type': 'HIGH', 'price': 65.41},  # SH1
        {'index': 38, 'date': '2025-05-07', 'type': 'LOW', 'price': 63.26},   # SL2
    ]
    
    print("SL1 (May 05): $61.75")
    print("SH1 (May 06): $65.41")
    print("SL2 (May 07): $63.26")
    print("✅ Higher low: SL2 ($63.26) > SL1 ($61.75) ✓")
    print("✅ We have the 3 swing points for potential uptrend!")
    
    # Now look for breakout candle after SL2
    print("\n🚀 Looking for BREAKOUT candle after SL2 (candle 38):")
    print("Breakout condition: Price must break ABOVE SH1 ($65.41)")
    
    # Check subsequent candles
    for i in range(39, min(45, len(df))):
        candle = df.iloc[i]
        if candle['high'] > 65.41:
            print(f"✅ BREAKOUT CANDLE FOUND: Candle {i} ({candle['datetime'].strftime('%Y-%m-%d')})")
            print(f"   High: ${candle['high']:.2f} > ${65.41} (SH1)")
            print(f"   🎯 UPTREND FORMED at candle {i}!")
            examples.append({
                'type': 'UPTREND_FORMATION',
                'formation_candle': i,
                'swing_points': may_swings,
                'breakout_price': candle['high']
            })
            break
    
    # Example 3: Look for downtrend formation
    print(f"\n📉 DOWNTREND FORMATION EXAMPLE:")
    print("Looking for: SH1 → SL1 → SH2 (where SH2 < SH1 = lower high)")
    
    # Check around June when we had that big drop
    june_swings = [
        {'index': 64, 'date': '2025-06-13', 'type': 'HIGH', 'price': 81.14},  # SH1
        {'index': 65, 'date': '2025-06-16', 'type': 'LOW', 'price': 76.25},   # SL1
        {'index': 69, 'date': '2025-06-23', 'type': 'HIGH', 'price': 83.57},  # SH2
    ]
    
    print("SH1 (Jun 13): $81.14")
    print("SL1 (Jun 16): $76.25")
    print("SH2 (Jun 23): $83.57")
    print("❌ This is NOT a downtrend setup - SH2 > SH1 (higher high, not lower high)")
    
    # Let's try later in June/July
    july_swings = [
        {'index': 69, 'date': '2025-06-23', 'type': 'HIGH', 'price': 83.57},  # SH1
        {'index': 70, 'date': '2025-06-24', 'type': 'LOW', 'price': 71.97},   # SL1  
        {'index': 72, 'date': '2025-06-26', 'type': 'HIGH', 'price': 74.68},  # SH2
    ]
    
    print(f"\n📉 DOWNTREND FORMATION EXAMPLE 2:")
    print("SH1 (Jun 23): $83.57")
    print("SL1 (Jun 24): $71.97")
    print("SH2 (Jun 26): $74.68")
    print("✅ Lower high: SH2 ($74.68) < SH1 ($83.57) ✓")
    print("✅ We have the 3 swing points for potential downtrend!")
    
    # Look for breakout (breakdown) candle
    print("\n🚀 Looking for BREAKOUT candle after SH2 (candle 72):")
    print("Breakout condition: Price must break BELOW SL1 ($71.97)")
    
    for i in range(73, min(80, len(df))):
        candle = df.iloc[i]
        if candle['low'] < 71.97:
            print(f"✅ BREAKOUT CANDLE FOUND: Candle {i} ({candle['datetime'].strftime('%Y-%m-%d')})")
            print(f"   Low: ${candle['low']:.2f} < ${71.97} (SL1)")
            print(f"   🎯 DOWNTREND FORMED at candle {i}!")
            examples.append({
                'type': 'DOWNTREND_FORMATION',
                'formation_candle': i,
                'swing_points': july_swings,
                'breakout_price': candle['low']
            })
            break
        else:
            print(f"   Candle {i}: Low ${candle['low']:.2f} (still above $71.97)")
    
    # Example 4: Look for trend termination (violation)
    print(f"\n🛑 TREND TERMINATION EXAMPLES:")
    print("Termination requires VIOLATION of controlling swing point")
    
    # If we had the uptrend from May, when did it terminate?
    if examples and examples[0]['type'] == 'UPTREND_FORMATION':
        uptrend_example = examples[0]
        controlling_low = min(uptrend_example['swing_points'], key=lambda x: x['price'])
        
        print(f"\nUptrend controlling swing: SL1 at ${controlling_low['price']} (candle {controlling_low['index']})")
        print(f"Looking for violation: Price breaks BELOW ${controlling_low['price']}")
        
        # Check for violation after uptrend formed
        formation_candle = uptrend_example['formation_candle']
        for i in range(formation_candle + 1, len(df)):
            candle = df.iloc[i]
            if candle['low'] < controlling_low['price']:
                print(f"✅ VIOLATION FOUND: Candle {i} ({candle['datetime'].strftime('%Y-%m-%d')})")
                print(f"   Low: ${candle['low']:.2f} < ${controlling_low['price']} (controlling swing)")
                print(f"   🛑 UPTREND TERMINATED at candle {i}!")
                break
    
    return df, swing_points, examples

def create_trend_example_visualization(df, swing_points, examples):
    """Create visualization highlighting trend formation/termination examples"""
    
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='USO Daily',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ))
    
    # Add swing points
    swing_highs = [s for s in swing_points if s['type'] == 'HIGH']
    if swing_highs:
        fig.add_trace(go.Scatter(
            x=[s['datetime'] for s in swing_highs],
            y=[s['price'] for s in swing_highs],
            mode='markers+text',
            marker=dict(color='#ffaa00', size=8, symbol='triangle-up'),
            text=[f"SH{i+1}" for i in range(len(swing_highs))],
            textposition='top center',
            textfont=dict(size=8),
            name='Swing Highs'
        ))
    
    swing_lows = [s for s in swing_points if s['type'] == 'LOW']
    if swing_lows:
        fig.add_trace(go.Scatter(
            x=[s['datetime'] for s in swing_lows],
            y=[s['price'] for s in swing_lows],
            mode='markers+text',
            marker=dict(color='#00aaff', size=8, symbol='triangle-down'),
            text=[f"SL{i+1}" for i in range(len(swing_lows))],
            textposition='bottom center',
            textfont=dict(size=8),
            name='Swing Lows'
        ))
    
    # Highlight trend formation/termination examples
    for example in examples:
        formation_candle = df.iloc[example['formation_candle']]
        
        if example['type'] == 'UPTREND_FORMATION':
            color = '#00ff00'
            symbol = 'arrow-up'
            text = 'UPTREND\nFORMS'
        else:
            color = '#ff0000'
            symbol = 'arrow-down' 
            text = 'DOWNTREND\nFORMS'
        
        fig.add_trace(go.Scatter(
            x=[formation_candle['datetime']],
            y=[formation_candle['high'] + 2],
            mode='markers+text',
            marker=dict(color=color, size=15, symbol=symbol),
            text=[text],
            textposition='top center',
            textfont=dict(color=color, size=10),
            name=f'{example["type"].replace("_", " ").title()}',
            showlegend=False
        ))
    
    fig.update_layout(
        title='USO Daily - Trend Formation Examples: 3 Swing Points + Breakout',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_dark',
        height=700,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def main():
    """Main function"""
    print("🎯 Concrete Trend Formation/Termination Examples")
    print("Based on: 3 swing points + BREAKOUT = formation, VIOLATION = termination")
    print("=" * 60)
    
    try:
        df, swings, examples = identify_trend_examples()
        
        # Create visualization
        fig = create_trend_example_visualization(df, swings, examples)
        filename = 'trend_formation_examples.html'
        fig.write_html(filename)
        
        print(f"\n💾 Trend formation examples chart saved as: {filename}")
        print(f"\n🎯 Summary of examples found:")
        for example in examples:
            print(f"   • {example['type'].replace('_', ' ')}: Candle {example['formation_candle']}")
        
        print(f"\n✅ Ready to implement the 3-swing-points + breakout logic!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()