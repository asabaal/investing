#!/usr/bin/env python3
"""
Show Swing Points on Real USO Data
Let's see what swing points we're actually detecting on the USO data we've been working with
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime, timedelta
from swing_point_detector import SwingPointDetector
import glob
import os

# Dark theme
pio.templates.default = "plotly_dark"

def load_uso_data():
    """Load existing USO data from our previous work"""
    
    # Look for existing USO data files
    data_files = glob.glob("uso_*candles*.csv")
    
    if data_files:
        print(f"📁 Found existing data files: {data_files}")
        # Use the most recent one
        data_file = sorted(data_files)[-1]
        print(f"📊 Loading: {data_file}")
        
        df = pd.read_csv(data_file)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'Datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['Datetime'])
            df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})
        
        return df
    
    # If no CSV files, create some sample data based on our previous work
    print("📊 No existing data files found, creating sample USO-like data")
    return create_uso_sample_data()

def create_uso_sample_data():
    """Create realistic USO-like sample data"""
    
    # Create 100 candles of realistic USO price action
    np.random.seed(42)  # For reproducible results
    
    base_price = 72.0
    dates = []
    start_date = datetime(2025, 1, 1, 9, 30)
    
    candles = []
    current_price = base_price
    
    for i in range(100):
        # Create realistic price movement
        daily_volatility = 0.02  # 2% daily volatility
        price_change = np.random.normal(0, daily_volatility * current_price / 10)
        current_price += price_change
        
        # Create OHLC for this candle
        high_wick = np.random.uniform(0.1, 0.5)
        low_wick = np.random.uniform(0.1, 0.5) 
        body_size = np.random.uniform(0.1, 0.8)
        
        if np.random.random() > 0.5:  # Bullish candle
            open_price = current_price - body_size/2
            close_price = current_price + body_size/2
        else:  # Bearish candle
            open_price = current_price + body_size/2
            close_price = current_price - body_size/2
            
        high_price = max(open_price, close_price) + high_wick
        low_price = min(open_price, close_price) - low_wick
        
        candles.append({
            'datetime': start_date + timedelta(minutes=15*i),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price
        })
        
        current_price = close_price
    
    return pd.DataFrame(candles)

def analyze_uso_swing_points():
    """Analyze swing points on USO data"""
    
    print("🚀 USO Swing Point Analysis")
    print("=" * 60)
    
    # Load data
    df = load_uso_data()
    
    if df.empty:
        print("❌ No data available")
        return
    
    print(f"📊 Loaded {len(df)} USO candles")
    print(f"   Date range: {df.iloc[0]['datetime']} to {df.iloc[-1]['datetime']}")
    print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    # Initialize swing detector
    detector = SwingPointDetector(lookback_periods=1, equal_levels_count=False)
    
    # Detect swing points
    swing_points = detector.detect_swing_points(df)
    
    print(f"\n🎯 Swing Point Detection Results:")
    print(f"   Total swing points found: {len(swing_points)}")
    
    if swing_points:
        swing_highs = [s for s in swing_points if s['type'] == 'HIGH']
        swing_lows = [s for s in swing_points if s['type'] == 'LOW']
        
        print(f"   • Swing highs: {len(swing_highs)}")
        print(f"   • Swing lows: {len(swing_lows)}")
        
        print(f"\n📋 First 10 swing points:")
        for i, swing in enumerate(swing_points[:10]):
            print(f"   {i+1:2d}. Candle {swing['index']:2d}: {swing['type']:4s} at ${swing['price']:6.2f}")
        
        if len(swing_points) > 10:
            print(f"   ... and {len(swing_points)-10} more")
        
        # Show detailed analysis for a section
        print(f"\n🔍 Detailed Analysis (Candles 20-40):")
        if len(df) >= 40:
            detector.print_swing_analysis(df, swing_points, start_idx=20, end_idx=40)
    
    # Create visualization
    fig = create_uso_swing_visualization(df, swing_points)
    
    filename = 'uso_swing_points_analysis.html'
    fig.write_html(filename)
    print(f"\n💾 Swing point visualization saved as: {filename}")
    
    return df, swing_points

def create_uso_swing_visualization(df, swing_points):
    """Create comprehensive swing point visualization"""
    
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='USO 15min',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ))
    
    # Add swing highs
    swing_highs = [s for s in swing_points if s['type'] == 'HIGH']
    if swing_highs:
        fig.add_trace(go.Scatter(
            x=[s['datetime'] for s in swing_highs],
            y=[s['price'] for s in swing_highs],
            mode='markers+text',
            marker=dict(color='#ffaa00', size=10, symbol='triangle-up'),
            text=['SH'] * len(swing_highs),
            textposition='top center',
            textfont=dict(color='#ffaa00', size=8),
            name='Swing Highs',
            hovertemplate='Swing HIGH: $%{y:.2f}<br>%{x}<extra></extra>'
        ))
    
    # Add swing lows  
    swing_lows = [s for s in swing_points if s['type'] == 'LOW']
    if swing_lows:
        fig.add_trace(go.Scatter(
            x=[s['datetime'] for s in swing_lows],
            y=[s['price'] for s in swing_lows],
            mode='markers+text',
            marker=dict(color='#00aaff', size=10, symbol='triangle-down'),
            text=['SL'] * len(swing_lows),
            textposition='bottom center',
            textfont=dict(color='#00aaff', size=8),
            name='Swing Lows',
            hovertemplate='Swing LOW: $%{y:.2f}<br>%{x}<extra></extra>'
        ))
    
    # Connect swing points with lines to show the structure
    if len(swing_points) > 1:
        fig.add_trace(go.Scatter(
            x=[s['datetime'] for s in swing_points],
            y=[s['price'] for s in swing_points],
            mode='lines',
            line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot'),
            name='Swing Structure',
            hoverinfo='skip'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'USO 15min - Swing Point Detection (Range-Based Logic)',
        xaxis_title='Time',
        yaxis_title='Price ($)',
        template='plotly_dark',
        showlegend=True,
        height=700,
        xaxis_rangeslider_visible=False
    )
    
    # Add annotations explaining the logic
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text="Swing Points: Local extrema that exceed neighboring candle ranges<br>" +
             "SH = Swing High (triangles up) | SL = Swing Low (triangles down)",
        showarrow=False,
        font=dict(color='white', size=10),
        bgcolor='rgba(0,0,0,0.7)',
        bordercolor='white',
        borderwidth=1,
        align='left'
    )
    
    return fig

def main():
    """Main function"""
    print("🎯 USO Swing Point Analysis")
    print("Let's see what swing points we're detecting on real USO data")
    print("=" * 60)
    
    # Analyze swing points
    df, swing_points = analyze_uso_swing_points()
    
    if swing_points:
        # Calculate some statistics
        price_range = df['high'].max() - df['low'].min()
        avg_swing_distance = price_range / len(swing_points) if swing_points else 0
        
        print(f"\n📊 Swing Point Statistics:")
        print(f"   • Total candles analyzed: {len(df)}")
        print(f"   • Swing points found: {len(swing_points)}")
        print(f"   • Swing point frequency: {len(swing_points)/len(df)*100:.1f}% of candles")
        print(f"   • Average distance between swings: ${avg_swing_distance:.2f}")
        
        # Show swing point alternation pattern
        pattern = [s['type'][0] for s in swing_points[:20]]  # First letter of type
        pattern_str = ' → '.join(pattern)
        print(f"   • Swing pattern (first 20): {pattern_str}")
        if len(swing_points) > 20:
            print("     ... (truncated)")

if __name__ == "__main__":
    import numpy as np
    main()