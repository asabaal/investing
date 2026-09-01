#!/usr/bin/env python3
"""
Show Swing Points on the Same USO Data We've Been Working With
Uses the exact same data source and timeframe as our existing visualizations
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime, timedelta
from swing_point_detector import SwingPointDetector
from uso_supply_demand_visualizer import SupplyDemandVisualizer
import os

# Dark theme
pio.templates.default = "plotly_dark"

def show_uso_swing_points_on_existing_data():
    """Show swing points on the exact same USO data we've been using"""
    
    print("🚀 USO Swing Points on Our Existing Data")
    print("=" * 60)
    
    # Initialize the same visualizer we've been using
    visualizer = SupplyDemandVisualizer()
    
    if not visualizer.api_key:
        print("❌ No Alpha Vantage API key found")
        return
    
    # Fetch the same data - let's try both timeframes to see which one works
    timeframes_to_try = ['weekly', 'daily']
    df = None
    used_timeframe = None
    
    for timeframe in timeframes_to_try:
        try:
            print(f"\n📡 Trying to fetch USO {timeframe} data...")
            df = visualizer.fetch_uso_data(timeframe=timeframe)
            used_timeframe = timeframe
            print(f"✅ Successfully loaded {timeframe} data")
            break
        except Exception as e:
            print(f"❌ Failed to fetch {timeframe} data: {e}")
            continue
    
    if df is None or df.empty:
        print("❌ Could not fetch any USO data")
        return
    
    print(f"\n📊 Loaded {len(df)} USO {used_timeframe} candles")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
    
    # Convert to format expected by swing detector
    df_for_swing = df.reset_index()
    df_for_swing = df_for_swing.rename(columns={
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close'
    })
    
    # Take recent data for better visualization (same as our existing charts)
    display_candles = 100 if used_timeframe == 'daily' else 52
    df_recent = df_for_swing.tail(display_candles).reset_index(drop=True)
    
    print(f"📊 Analyzing recent {len(df_recent)} candles for swing points")
    
    # Initialize swing detector
    detector = SwingPointDetector(lookback_periods=1, equal_levels_count=False)
    
    # Detect swing points
    swing_points = detector.detect_swing_points(df_recent)
    
    print(f"\n🎯 Swing Point Detection Results:")
    print(f"   Total swing points found: {len(swing_points)}")
    
    if swing_points:
        swing_highs = [s for s in swing_points if s['type'] == 'HIGH']
        swing_lows = [s for s in swing_points if s['type'] == 'LOW']
        
        print(f"   • Swing highs: {len(swing_highs)}")
        print(f"   • Swing lows: {len(swing_lows)}")
        print(f"   • Swing frequency: {len(swing_points)/len(df_recent)*100:.1f}% of candles")
        
        print(f"\n📋 All swing points found:")
        for i, swing in enumerate(swing_points):
            candle_date = df_recent.iloc[swing['index']]['datetime'].strftime('%Y-%m-%d')
            print(f"   {i+1:2d}. Candle {swing['index']:2d} ({candle_date}): {swing['type']:4s} at ${swing['price']:6.2f}")
        
        # Show pattern
        pattern = [s['type'][0] for s in swing_points]  # First letter of type
        pattern_str = ' → '.join(pattern)
        print(f"\n📈 Swing pattern: {pattern_str}")
        
        # Show detailed analysis for middle section
        mid_start = max(0, len(df_recent)//4)
        mid_end = min(len(df_recent), mid_start + 20)
        print(f"\n🔍 Detailed Analysis (Candles {mid_start}-{mid_end-1}):")
        detector.print_swing_analysis(df_recent, swing_points, start_idx=mid_start, end_idx=mid_end)
    
    # Create visualization
    fig = create_comprehensive_swing_visualization(df_recent, swing_points, used_timeframe)
    
    filename = f'uso_swing_points_{used_timeframe}_existing_data.html'
    fig.write_html(filename)
    print(f"\n💾 Swing point visualization saved as: {filename}")
    
    # Summary statistics
    if swing_points:
        price_range = df_recent['high'].max() - df_recent['low'].min()
        avg_swing_distance = price_range / len(swing_points) if swing_points else 0
        
        print(f"\n📊 Summary Statistics:")
        print(f"   • Data timeframe: {used_timeframe.upper()}")
        print(f"   • Candles analyzed: {len(df_recent)}")
        print(f"   • Swing points detected: {len(swing_points)}")
        print(f"   • Detection sensitivity: {len(swing_points)/len(df_recent)*100:.1f}% of candles are swing points")
        print(f"   • Average price between swings: ${avg_swing_distance:.2f}")
        
        # Analysis of sensitivity
        if len(swing_points)/len(df_recent) > 0.3:
            print(f"\n⚠️  HIGH SENSITIVITY DETECTED:")
            print(f"   • {len(swing_points)/len(df_recent)*100:.1f}% of candles are swing points")
            print(f"   • This suggests our swing detection might be too aggressive")
            print(f"   • Consider increasing lookback_periods or adding minimum threshold")
        else:
            print(f"\n✅ REASONABLE SENSITIVITY:")
            print(f"   • {len(swing_points)/len(df_recent)*100:.1f}% of candles are swing points")
            print(f"   • This looks like a reasonable swing point frequency")
    
    return df_recent, swing_points, used_timeframe

def create_comprehensive_swing_visualization(df, swing_points, timeframe):
    """Create comprehensive swing point visualization matching our existing style"""
    
    fig = go.Figure()
    
    # Add candlestick chart with same styling as existing charts
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=f'USO {timeframe.upper()}',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ))
    
    # Add swing highs with clear markers
    swing_highs = [s for s in swing_points if s['type'] == 'HIGH']
    if swing_highs:
        fig.add_trace(go.Scatter(
            x=[df.iloc[s['index']]['datetime'] for s in swing_highs],
            y=[s['price'] for s in swing_highs],
            mode='markers+text',
            marker=dict(
                color='#ffaa00',
                size=12,
                symbol='triangle-up',
                line=dict(color='white', width=1)
            ),
            text=[f"SH{i+1}" for i in range(len(swing_highs))],
            textposition='top center',
            textfont=dict(color='#ffaa00', size=8, family='monospace'),
            name='Swing Highs',
            hovertemplate='Swing HIGH #%{text}<br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
        ))
    
    # Add swing lows with clear markers
    swing_lows = [s for s in swing_points if s['type'] == 'LOW']
    if swing_lows:
        fig.add_trace(go.Scatter(
            x=[df.iloc[s['index']]['datetime'] for s in swing_lows],
            y=[s['price'] for s in swing_lows],
            mode='markers+text',
            marker=dict(
                color='#00aaff',
                size=12,
                symbol='triangle-down',
                line=dict(color='white', width=1)
            ),
            text=[f"SL{i+1}" for i in range(len(swing_lows))],
            textposition='bottom center',
            textfont=dict(color='#00aaff', size=8, family='monospace'),
            name='Swing Lows',
            hovertemplate='Swing LOW #%{text}<br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
        ))
    
    # Connect swing points to show structure
    if len(swing_points) > 1:
        fig.add_trace(go.Scatter(
            x=[df.iloc[s['index']]['datetime'] for s in swing_points],
            y=[s['price'] for s in swing_points],
            mode='lines',
            line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dot'),
            name='Swing Structure',
            hoverinfo='skip',
            showlegend=True
        ))
    
    # Update layout to match our existing style
    detection_freq = len(swing_points)/len(df)*100 if df is not None and len(df) > 0 else 0
    fig.update_layout(
        title=dict(
            text=f'USO {timeframe.upper()} - Swing Point Analysis<br><sub>Range-Based Detection: {len(swing_points)} swing points ({detection_freq:.1f}% frequency)</sub>',
            font=dict(size=16, color='white'),
            x=0.5
        ),
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_dark',
        showlegend=True,
        height=700,
        xaxis_rangeslider_visible=False,
        paper_bgcolor='rgba(15,15,15,1)',
        plot_bgcolor='rgba(25,25,25,1)'
    )
    
    # Add explanation annotation
    explanation_text = (
        f"🎯 Swing Point Logic:<br>"
        f"• SH = Swing High (price exceeds neighboring candles)<br>"
        f"• SL = Swing Low (price falls below neighboring candles)<br>"
        f"• Lookback: 1 candle each direction<br>"
        f"• Frequency: {detection_freq:.1f}% of candles are swing points"
    )
    
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text=explanation_text,
        showarrow=False,
        font=dict(color='white', size=10),
        bgcolor='rgba(0,0,0,0.8)',
        bordercolor='rgba(255,255,255,0.3)',
        borderwidth=1,
        align='left'
    )
    
    return fig

def main():
    """Main function"""
    print("🎯 USO Swing Point Analysis on Existing Data")
    print("Using the exact same data source as our supply/demand visualizations")
    print("=" * 60)
    
    # Check for API key
    if not os.environ.get('ALPHA_VANTAGE_API_KEY'):
        print("❌ No ALPHA_VANTAGE_API_KEY environment variable found")
        print("   This is the same API we've been using for USO data")
        return
    
    # Run the analysis
    try:
        df, swing_points, timeframe = show_uso_swing_points_on_existing_data()
        print(f"\n✅ Analysis completed successfully!")
        print(f"   Timeframe: {timeframe}")
        print(f"   Swing points: {len(swing_points)}")
        print(f"   Chart saved as: uso_swing_points_{timeframe}_existing_data.html")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()