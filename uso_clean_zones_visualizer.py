#!/usr/bin/env python3
"""
USO Clean Supply & Demand Zone Visualizer
Simple, focused visualization with:
- Horizontal rectangular zones
- Clear swing point trend lines
- Minimal clutter
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import requests
import os
from datetime import datetime, timedelta

# Dark theme
pio.templates.default = "plotly_dark"

class CleanZoneVisualizer:
    """Clean, simple supply & demand zone visualization"""
    
    def __init__(self):
        self.api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        
    def classify_candle_sentiment(self, candle_data):
        """
        Simple 1/3-2/3 sentiment classification from your course
        BASE: Close between 1/3 and 2/3 of range
        """
        high = candle_data['High']
        low = candle_data['Low']
        close = candle_data['Close']
        open_price = candle_data['Open']
        
        total_range = high - low
        if total_range == 0:
            return 'BASE'
        
        # Close position within range (0 = bottom, 1 = top)
        close_position = (close - low) / total_range
        
        # BASE: Close between 1/3 and 2/3 of range
        if 1/3 <= close_position <= 2/3:
            return 'BASE'
        
        # LEG: Determine direction by close vs open
        return 'LEG_BULLISH' if close > open_price else 'LEG_BEARISH'
    
    def detect_multi_candle_zones(self, df):
        """
        Enhanced zone detection: multi-candle LEG sequences → multi-candle BASE sequences → multi-candle LEG sequences
        Returns zones with proper formation leg data for visualization
        """
        zones = []
        candle_types = []
        
        # Classify all candles
        for i, row in df.iterrows():
            candle_data = {
                'High': row['High'],
                'Low': row['Low'], 
                'Open': row['Open'],
                'Close': row['Close']
            }
            sentiment = self.classify_candle_sentiment(candle_data)
            candle_types.append(sentiment)
        
        print(f"🔍 Classified {len(candle_types)} candles")
        
        # Find LEG sequence → BASE sequence → LEG sequence patterns
        i = 0
        while i < len(candle_types) - 4:  # Need at least 5 candles minimum
            
            # Look for entry LEG sequence (1-3 consecutive LEG candles of same type)
            entry_leg_start = i
            entry_leg_type = None
            entry_leg_length = 0
            
            if candle_types[i].startswith('LEG'):
                entry_leg_type = candle_types[i]
                entry_leg_length = 1
                
                # Extend entry leg sequence (up to 3 candles)
                for j in range(i + 1, min(i + 4, len(candle_types))):
                    if candle_types[j] == entry_leg_type:
                        entry_leg_length += 1
                    else:
                        break
            
            if entry_leg_length == 0:
                i += 1
                continue
                
            entry_leg_end = entry_leg_start + entry_leg_length - 1
            base_start = entry_leg_end + 1
            
            if base_start >= len(candle_types):
                break
            
            # Look for BASE sequence (1-5 consecutive BASE candles)
            base_length = 0
            for j in range(base_start, min(base_start + 6, len(candle_types))):
                if candle_types[j] == 'BASE':
                    base_length += 1
                else:
                    break
            
            if base_length == 0:
                i += 1
                continue
                
            base_end = base_start + base_length - 1
            exit_leg_start = base_end + 1
            
            if exit_leg_start >= len(candle_types):
                break
            
            # Look for exit LEG sequence (1-3 consecutive LEG candles of same type)
            exit_leg_type = None
            exit_leg_length = 0
            
            if exit_leg_start < len(candle_types) and candle_types[exit_leg_start].startswith('LEG'):
                exit_leg_type = candle_types[exit_leg_start]
                exit_leg_length = 1
                
                # Extend exit leg sequence (up to 3 candles)
                for j in range(exit_leg_start + 1, min(exit_leg_start + 4, len(candle_types))):
                    if candle_types[j] == exit_leg_type:
                        exit_leg_length += 1
                    else:
                        break
            
            if exit_leg_length == 0:
                i += 1
                continue
            
            exit_leg_end = exit_leg_start + exit_leg_length - 1
            
            # Determine formation type
            zone_type = None
            formation_type = None
            
            if entry_leg_type == 'LEG_BULLISH' and exit_leg_type == 'LEG_BULLISH':
                formation_type = 'RBR'
                zone_type = 'DEMAND'
            elif entry_leg_type == 'LEG_BEARISH' and exit_leg_type == 'LEG_BEARISH':
                formation_type = 'DBD'
                zone_type = 'SUPPLY'
            elif entry_leg_type == 'LEG_BULLISH' and exit_leg_type == 'LEG_BEARISH':
                formation_type = 'RBD'
                zone_type = 'SUPPLY'
            elif entry_leg_type == 'LEG_BEARISH' and exit_leg_type == 'LEG_BULLISH':
                formation_type = 'DBR'
                zone_type = 'DEMAND'
            
            if zone_type:
                # Get base sequence data for zone boundaries
                base_candles = df.iloc[base_start:base_end+1]
                zone_high = base_candles['High'].max()
                zone_low = base_candles['Low'].min()
                
                # Get entry and exit leg endpoint data
                entry_leg_last_candle = df.iloc[entry_leg_end]
                exit_leg_first_candle = df.iloc[exit_leg_start]
                
                zone_info = {
                    'type': formation_type,
                    'zone_type': zone_type,
                    'entry_leg_start_idx': entry_leg_start,
                    'entry_leg_end_idx': entry_leg_end,
                    'base_start_idx': base_start,
                    'base_end_idx': base_end,
                    'exit_leg_start_idx': exit_leg_start,
                    'exit_leg_end_idx': exit_leg_end,
                    'zone_high': zone_high,
                    'zone_low': zone_low,
                    'zone_mid': (zone_high + zone_low) / 2,
                    # Proper endpoint data for line segments
                    'entry_leg_endpoint': self.get_leg_endpoint(entry_leg_last_candle, entry_leg_type, formation_type),
                    'exit_leg_endpoint': self.get_leg_endpoint(exit_leg_first_candle, exit_leg_type, formation_type),
                    'formation_summary': f"R({entry_leg_start}-{entry_leg_end}) B({base_start}-{base_end}) {'R' if exit_leg_type=='LEG_BULLISH' else 'D'}({exit_leg_start}-{exit_leg_end})"
                }
                
                zones.append(zone_info)
                print(f"   Found {formation_type}: {zone_info['formation_summary']}")
            
            # Move past this formation
            i = exit_leg_end + 1
        
        return zones, candle_types
    
    def get_leg_endpoint(self, candle_row, leg_type, formation_type):
        """
        Get the correct price endpoint for a leg candle based on formation context
        
        For RBR: Entry leg (rally) connects from HIGH, Exit leg (rally) connects from HIGH
        For DBD: Entry leg (drop) connects from LOW, Exit leg (drop) connects from LOW  
        For RBD: Entry leg (rally) connects from HIGH, Exit leg (drop) connects from LOW
        For DBR: Entry leg (drop) connects from LOW, Exit leg (rally) connects from HIGH
        """
        if leg_type == 'LEG_BULLISH':
            return candle_row['High']  # Always use high for bullish legs (top of rally)
        else:  # LEG_BEARISH
            return candle_row['Low']   # Always use low for bearish legs (bottom of drop)
    
    def find_swing_points(self, df, lookback=3):
        """
        Find swing highs and lows for trend lines
        Simple approach: local extrema with lookback
        """
        swing_points = []
        
        for i in range(lookback, len(df) - lookback):
            high = df.iloc[i]['High']
            low = df.iloc[i]['Low']
            
            # Check if this is a swing high
            is_swing_high = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and df.iloc[j]['High'] >= high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_points.append({
                    'idx': i,
                    'price': high,
                    'type': 'HIGH',
                    'timestamp': df.index[i]
                })
            
            # Check if this is a swing low
            is_swing_low = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and df.iloc[j]['Low'] <= low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_points.append({
                    'idx': i,
                    'price': low,
                    'type': 'LOW',
                    'timestamp': df.index[i]
                })
        
        return swing_points
    
    def fetch_uso_data(self, timeframe='weekly'):
        """Fetch USO data"""
        
        print(f"📡 Fetching USO {timeframe} data...")
        
        if timeframe == 'weekly':
            function = 'TIME_SERIES_WEEKLY'
            params = {
                'function': function,
                'symbol': 'USO',
                'apikey': self.api_key
            }
        else:
            function = 'TIME_SERIES_DAILY'
            params = {
                'function': function,
                'symbol': 'USO',
                'apikey': self.api_key,
                'outputsize': 'full'
            }
        
        response = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
        data = response.json()
        
        # Parse data
        if timeframe == 'weekly':
            time_series = data['Weekly Time Series']
        else:
            time_series = data['Time Series (Daily)']
        
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
        
        print(f"✅ Loaded {len(df):,} {timeframe} candles")
        return df
    
    def create_clean_chart(self, df, zones, timeframe='weekly'):
        """Create clean chart with zones over base candles and formation segments"""
        
        print("🎯 Creating clean formation visualization...")
        
        # Use recent data for display
        display_data = df.tail(52) if timeframe == 'weekly' else df.tail(100)
        start_offset = len(df) - len(display_data)
        
        # Adjust zones for display window
        display_zones = []
        for zone in zones:
            if zone['entry_leg_start_idx'] >= start_offset:
                adj_zone = zone.copy()
                adj_zone['entry_leg_start_idx'] -= start_offset
                adj_zone['entry_leg_end_idx'] -= start_offset
                adj_zone['base_start_idx'] -= start_offset
                adj_zone['base_end_idx'] -= start_offset
                adj_zone['exit_leg_start_idx'] -= start_offset
                adj_zone['exit_leg_end_idx'] -= start_offset
                display_zones.append(adj_zone)
        
        # Create figure
        fig = go.Figure()
        
        # Add candlesticks (simple, no gaps)
        fig.add_trace(
            go.Candlestick(
                x=list(range(len(display_data))),
                open=display_data['Open'],
                high=display_data['High'],
                low=display_data['Low'],
                close=display_data['Close'],
                name='USO',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            )
        )
        
        # Add zones and formation segments
        for zone in display_zones:
            zone_color = 'rgba(255, 68, 68, 0.3)' if zone['zone_type'] == 'SUPPLY' else 'rgba(0, 255, 136, 0.3)'
            zone_line_color = '#ff4444' if zone['zone_type'] == 'SUPPLY' else '#00ff88'
            segment_color = '#ff6b35' if zone['zone_type'] == 'SUPPLY' else '#00d4aa'
            
            # Zone rectangle over the BASE candle sequence
            base_center = (zone['base_start_idx'] + zone['base_end_idx']) / 2
            base_width = zone['base_end_idx'] - zone['base_start_idx'] + 0.8  # Cover all base candles
            
            fig.add_shape(
                type="rect",
                x0=zone['base_start_idx'] - 0.4,
                y0=zone['zone_low'],
                x1=zone['base_end_idx'] + 0.4,
                y1=zone['zone_high'],
                fillcolor=zone_color,
                line=dict(color=zone_line_color, width=2),
                layer="below"
            )
            
            # Entry leg segment - connecting from entry leg END to zone
            fig.add_trace(
                go.Scatter(
                    x=[zone['entry_leg_end_idx'], base_center],
                    y=[zone['entry_leg_endpoint'], zone['zone_mid']],
                    mode='lines+markers',
                    line=dict(color=segment_color, width=3),
                    marker=dict(color=segment_color, size=8),
                    name=f"Entry Leg ({zone['type']})" if zone == display_zones[0] else None,
                    showlegend=zone == display_zones[0],
                    hovertemplate=f'Entry Leg<br>{zone["type"]} Formation<br>Price: $%{{y:.2f}}<extra></extra>'
                )
            )
            
            # Exit leg segment - connecting from zone to exit leg START  
            fig.add_trace(
                go.Scatter(
                    x=[base_center, zone['exit_leg_start_idx']],
                    y=[zone['zone_mid'], zone['exit_leg_endpoint']],
                    mode='lines+markers',
                    line=dict(color=segment_color, width=3),
                    marker=dict(color=segment_color, size=8),
                    name=f"Exit Leg ({zone['type']})" if zone == display_zones[0] else None,
                    showlegend=zone == display_zones[0],
                    hovertemplate=f'Exit Leg<br>{zone["type"]} Formation<br>Price: $%{{y:.2f}}<extra></extra>'
                )
            )
            
            # Zone label with formation summary
            fig.add_annotation(
                x=base_center,
                y=zone['zone_high'] + (zone['zone_high'] - zone['zone_low']) * 0.3,  # Above zone
                text=f"{zone['type']}<br>{zone['formation_summary']}<br>${zone['zone_low']:.2f}-${zone['zone_high']:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowcolor=zone_line_color,
                font=dict(size=9, color=zone_line_color),
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor=zone_line_color,
                borderwidth=1
            )
        
        # Update layout - clean and simple
        fig.update_layout(
            title=dict(
                text=f"USO Formation Analysis - {timeframe.upper()} | {len(display_zones)} Formations",
                font=dict(size=16, color='white'),
                x=0.5
            ),
            height=700,
            paper_bgcolor='rgba(15,15,15,1)',
            plot_bgcolor='rgba(25,25,25,1)',
            font=dict(color='white', size=11),
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Clean axes
        fig.update_yaxes(
            title_text="Price ($)",
            gridcolor='rgba(100,100,100,0.2)'
        )
        
        fig.update_xaxes(
            title_text="Candle Index",
            gridcolor='rgba(100,100,100,0.2)'
        )
        
        return fig
    
    def print_simple_summary(self, zones):
        """Print clean summary of findings"""
        
        print(f"\n📊 FORMATION ANALYSIS")
        print("=" * 40)
        
        if zones:
            supply_zones = [z for z in zones if z['zone_type'] == 'SUPPLY']
            demand_zones = [z for z in zones if z['zone_type'] == 'DEMAND']
            
            print(f"✅ Total Formations: {len(zones)}")
            print(f"   🔴 Supply Zones: {len(supply_zones)}")
            print(f"   🟢 Demand Zones: {len(demand_zones)}")
            
            # Count by formation type
            formation_counts = {}
            for zone in zones:
                ftype = zone['type']
                formation_counts[ftype] = formation_counts.get(ftype, 0) + 1
            
            print(f"\n📊 FORMATION TYPES:")
            for ftype, count in formation_counts.items():
                print(f"   {ftype}: {count}")
            
            print(f"\n🎯 RECENT FORMATIONS:")
            for zone in zones[-5:]:  # Show last 5 zones
                zone_width = zone['zone_high'] - zone['zone_low']
                print(f"   {zone['type']} | Base: ${zone['zone_low']:.2f}-${zone['zone_high']:.2f} | Width: ${zone_width:.2f}")
        else:
            print("❌ No formations detected")

def main():
    """Create clean formation visualization"""
    
    print("🚀 USO Formation Analysis")
    print("Zone rectangles over base candles + formation leg segments")
    print("=" * 55)
    
    visualizer = CleanZoneVisualizer()
    
    try:
        # Fetch data
        timeframe = 'weekly'
        df = visualizer.fetch_uso_data(timeframe)
        
        # Enhanced multi-candle zone detection
        zones, candle_types = visualizer.detect_multi_candle_zones(df)
        
        # Print summary
        visualizer.print_simple_summary(zones)
        
        # Create clean chart
        fig = visualizer.create_clean_chart(df, zones, timeframe)
        
        # Save chart
        filename = f'uso_formation_analysis_{timeframe}.html'
        fig.write_html(filename)
        
        print(f"\n✅ Formation Analysis Chart Created!")
        print(f"💾 Saved as: {filename}")
        print(f"\n🎯 Features:")
        print("  • Zone rectangles ONLY over base candles")
        print("  • Entry/exit leg segments showing formation pattern")
        print("  • Clear RBR/DBD/RBD/DBR formation identification")
        print("  • Supply zones (red) & demand zones (green)")
        print("  • Clean, focused visualization")
        print("  • No support/resistance trend lines")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()