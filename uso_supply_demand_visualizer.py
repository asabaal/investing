#!/usr/bin/env python3
"""
USO Supply & Demand Zone Visualizer
Focused visual system implementing the 4 classic formations:
- RBR (Rally-Base-Rally) = DEMAND ZONE
- DBD (Drop-Base-Drop) = SUPPLY ZONE  
- RBD (Rally-Base-Drop) = SUPPLY ZONE
- DBR (Drop-Base-Rally) = DEMAND ZONE

Key Concepts:
- BASE candles: 1/3-2/3 closed within range (consolidation)
- LEG candles: Strong directional movement
- Proximal line: Inside edge (least extreme body edge of base)
- Distal line: Outside edge (most extreme wick of formation)
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
from swing_point_detector import SwingPointDetector

# Dark theme
pio.templates.default = "plotly_dark"

class SupplyDemandVisualizer:
    """Supply & Demand zone detection and visualization"""
    
    def __init__(self):
        self.api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        self.swing_detector = SwingPointDetector(lookback_periods=1, equal_levels_count=False)
        
    def classify_candle_sentiment_class_method(self, candle_data):
        """
        EXACT class method: BASE candles have 1/3 to 2/3 closed within range
        This is the precise system taught in your supply/demand class.
        
        Args:
            candle_data: Dict with OHLC values
            
        Returns:
            str: 'BASE', 'LEG_BULLISH', 'LEG_BEARISH'
        """
        high = candle_data['High']
        low = candle_data['Low']
        open_price = candle_data['Open']
        close = candle_data['Close']
        
        total_range = high - low
        if total_range == 0:
            return 'BASE'
        
        # Calculate 1/3 and 2/3 levels of the total range
        one_third_level = low + (total_range * 1/3)
        two_thirds_level = low + (total_range * 2/3)
        
        # Check where the close is relative to the range
        close_position = (close - low) / total_range  # 0 = bottom, 1 = top
        
        # BASE candle: Close is between 1/3 and 2/3 of the range
        if 1/3 <= close_position <= 2/3:
            return 'BASE'
        
        # LEG candles: Close is outside the middle third
        # Determine bullish vs bearish by close vs open
        if close > open_price:
            return 'LEG_BULLISH'
        else:
            return 'LEG_BEARISH'
    
    def classify_candle_sentiment_body_ratio(self, candle_data):
        """
        Alternative method: Body-to-wick ratio analysis
        BASE: Small body relative to total range (consolidation)
        LEG: Large body relative to range (directional movement)
        
        Args:
            candle_data: Dict with OHLC values
            
        Returns:
            str: 'BASE', 'LEG_BULLISH', 'LEG_BEARISH'
        """
        high = candle_data['High']
        low = candle_data['Low']
        open_price = candle_data['Open']
        close = candle_data['Close']
        
        total_range = high - low
        if total_range == 0:
            return 'BASE'
        
        # Body size relative to total range
        body_size = abs(close - open_price)
        body_to_range_ratio = body_size / total_range
        
        # BASE candle: Small body relative to total range
        if body_to_range_ratio < 0.6:  # Body is less than 60% of total range
            return 'BASE'
        
        # LEG candle: Large body, determine direction
        if close > open_price:
            return 'LEG_BULLISH'
        else:
            return 'LEG_BEARISH'
    
    def classify_candle_sentiment(self, candle_data, method='class'):
        """
        Wrapper method to choose between classification approaches
        
        Args:
            candle_data: Dict with OHLC values
            method: 'class' (exact 1/3-2/3 method) or 'body_ratio' (body/wick analysis)
            
        Returns:
            str: 'BASE', 'LEG_BULLISH', 'LEG_BEARISH'
        """
        if method == 'class':
            return self.classify_candle_sentiment_class_method(candle_data)
        elif method == 'body_ratio':
            return self.classify_candle_sentiment_body_ratio(candle_data)
        else:
            raise ValueError("Method must be 'class' or 'body_ratio'")
    
    def detect_swing_point_formations(self, df, sentiment_method='class'):
        """
        Detect supply/demand formations using proper swing point foundation
        
        Logic:
        1. Find all swing points using range-based detection
        2. Look for alternating swing high → swing low → swing high (or vice versa)
        3. Classify the candles between swing points as BASE or LEG
        4. Identify valid RBR/DBD/RBD/DBR patterns
        
        Args:
            df: DataFrame with OHLC data (must have columns: datetime, open, high, low, close)
            sentiment_method: 'class' or 'body_ratio' for candle classification
            
        Returns:
            list: Detected formations with zone information
        """
        formations = []
        
        print(f"🎯 Using swing point based formation detection")
        print(f"   📊 Sentiment method: {sentiment_method.upper()}")
        
        # Step 1: Find swing points
        swing_points = self.swing_detector.detect_swing_points(df)
        
        if len(swing_points) < 2:
            print(f"   ❌ Not enough swing points found ({len(swing_points)})")
            return formations
            
        print(f"   🎯 Found {len(swing_points)} swing points")
        
        # Step 2: Look for alternating swing point sequences
        for i in range(len(swing_points) - 1):
            current_swing = swing_points[i]
            next_swing = swing_points[i + 1]
            
            # Must be alternating (high → low or low → high)
            if current_swing['type'] == next_swing['type']:
                continue
                
            # Get candles between these swing points
            start_idx = current_swing['index']
            end_idx = next_swing['index']
            
            if end_idx - start_idx < 2:  # Need at least some candles between
                continue
                
            # Analyze this segment for formations
            formation = self._analyze_swing_segment(
                df, start_idx, end_idx, current_swing, next_swing, sentiment_method
            )
            
            if formation:
                formations.append(formation)
        
        print(f"   ✅ Detected {len(formations)} swing point based formations")
        return formations
    
    def _analyze_swing_segment(self, df, start_idx, end_idx, start_swing, end_swing, sentiment_method):
        """
        Analyze candles between two swing points to identify formation
        
        Args:
            df: DataFrame with OHLC data
            start_idx: Starting candle index
            end_idx: Ending candle index  
            start_swing: First swing point dict
            end_swing: Second swing point dict
            sentiment_method: Classification method
            
        Returns:
            dict: Formation info or None if no valid formation
        """
        segment_length = end_idx - start_idx + 1
        
        # Classify candles in this segment
        base_candles = []
        leg_candles = []
        
        for i in range(start_idx, end_idx + 1):
            row = df.iloc[i]
            candle_data = {
                'High': row['high'],
                'Low': row['low'],
                'Open': row['open'], 
                'Close': row['close']
            }
            sentiment = self.classify_candle_sentiment(candle_data, method=sentiment_method)
            
            if sentiment == 'BASE':
                base_candles.append(i)
            elif sentiment in ['LEG_BULLISH', 'LEG_BEARISH']:
                leg_candles.append(i)
        
        # Determine formation type based on swing point sequence
        if start_swing['type'] == 'HIGH' and end_swing['type'] == 'LOW':
            # High to Low = potential RBD or DBD
            formation_type = self._classify_high_to_low_formation(
                df, start_idx, end_idx, base_candles, leg_candles
            )
        elif start_swing['type'] == 'LOW' and end_swing['type'] == 'HIGH':
            # Low to High = potential DBR or RBR
            formation_type = self._classify_low_to_high_formation(
                df, start_idx, end_idx, base_candles, leg_candles
            )
        else:
            return None
            
        if formation_type:
            return {
                'type': formation_type,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'base_candles': base_candles,
                'leg_candles': leg_candles,
                'start_swing': start_swing,
                'end_swing': end_swing,
                'zone_high': max(df.iloc[start_idx:end_idx+1]['high']),
                'zone_low': min(df.iloc[start_idx:end_idx+1]['low']),
                'datetime_start': df.iloc[start_idx]['datetime'],
                'datetime_end': df.iloc[end_idx]['datetime']
            }
        
        return None
    
    def _classify_high_to_low_formation(self, df, start_idx, end_idx, base_candles, leg_candles):
        """Classify High → Low swing sequence as RBD or DBD"""
        # For now, simplified logic - can be refined later
        if len(base_candles) > 0:
            # Has consolidation = RBD (Rally-Base-Drop)
            return 'RBD'  # SUPPLY ZONE
        else:
            # No consolidation = DBD (Drop-Base-Drop) 
            return 'DBD'  # SUPPLY ZONE
    
    def _classify_low_to_high_formation(self, df, start_idx, end_idx, base_candles, leg_candles):
        """Classify Low → High swing sequence as DBR or RBR"""
        # For now, simplified logic - can be refined later
        if len(base_candles) > 0:
            # Has consolidation = DBR (Drop-Base-Rally)
            return 'DBR'  # DEMAND ZONE
        else:
            # No consolidation = RBR (Rally-Base-Rally)
            return 'RBR'  # DEMAND ZONE

    def detect_formations(self, df, min_base_candles=1, max_base_candles=3, sentiment_method='class'):
        """
        Detect the 4 classic supply/demand formations:
        RBR, DBD, RBD, DBR
        
        Args:
            df: DataFrame with OHLC data
            min_base_candles: Minimum base candles in formation
            max_base_candles: Maximum base candles in formation
            sentiment_method: 'class' (1/3-2/3 method) or 'body_ratio' (body/wick analysis)
            
        Returns:
            list: Detected formations with zone information
        """
        formations = []
        candle_types = []
        
        print(f"🔍 Using sentiment method: {sentiment_method.upper()}")
        if sentiment_method == 'class':
            print("   📚 Class method: BASE = close between 1/3 and 2/3 of range")
        else:
            print("   📊 Body ratio method: BASE = small body relative to total range")
        
        # Classify all candles first
        for i, row in df.iterrows():
            candle_data = {
                'High': row['High'],
                'Low': row['Low'], 
                'Open': row['Open'],
                'Close': row['Close']
            }
            sentiment = self.classify_candle_sentiment(candle_data, method=sentiment_method)
            candle_types.append(sentiment)
        
        # Look for LEG → BASE → LEG patterns
        for i in range(len(candle_types) - 2):
            # Need at least 3 candles for a formation
            if i + 2 >= len(candle_types):
                break
                
            # Look for different base lengths
            for base_length in range(min_base_candles, max_base_candles + 1):
                if i + 1 + base_length >= len(candle_types):
                    continue
                    
                # Check pattern: LEG → BASE(s) → LEG
                entry_leg = candle_types[i]
                base_candles = candle_types[i+1:i+1+base_length]
                exit_leg_idx = i + 1 + base_length
                
                if exit_leg_idx >= len(candle_types):
                    continue
                    
                exit_leg = candle_types[exit_leg_idx]
                
                # All base candles must be BASE type
                if not all(candle == 'BASE' for candle in base_candles):
                    continue
                
                # Entry and exit must be LEG types
                if not (entry_leg.startswith('LEG') and exit_leg.startswith('LEG')):
                    continue
                
                # Determine formation type and zone type
                formation_type = None
                zone_type = None
                
                if entry_leg == 'LEG_BULLISH' and exit_leg == 'LEG_BULLISH':
                    formation_type = 'RBR'  # Rally-Base-Rally
                    zone_type = 'DEMAND'
                elif entry_leg == 'LEG_BEARISH' and exit_leg == 'LEG_BEARISH':
                    formation_type = 'DBD'  # Drop-Base-Drop  
                    zone_type = 'SUPPLY'
                elif entry_leg == 'LEG_BULLISH' and exit_leg == 'LEG_BEARISH':
                    formation_type = 'RBD'  # Rally-Base-Drop
                    zone_type = 'SUPPLY'
                elif entry_leg == 'LEG_BEARISH' and exit_leg == 'LEG_BULLISH':
                    formation_type = 'DBR'  # Drop-Base-Rally
                    zone_type = 'DEMAND'
                
                if formation_type:
                    # Calculate zone boundaries
                    base_start_idx = i + 1
                    base_end_idx = i + base_length
                    
                    # Get base candle data for zone calculation
                    base_data = df.iloc[base_start_idx:base_end_idx+1]
                    formation_data = df.iloc[i:exit_leg_idx+1]  # Full formation including legs
                    
                    # Calculate proximal and distal lines
                    if zone_type == 'SUPPLY':
                        # Supply zone: price expected to go down
                        # Distal = highest point of formation (most extreme wick)
                        distal_line = formation_data['High'].max()
                        # Proximal = lowest body edge of base candles  
                        base_body_bottoms = []
                        for _, candle in base_data.iterrows():
                            body_bottom = min(candle['Open'], candle['Close'])
                            base_body_bottoms.append(body_bottom)
                        proximal_line = min(base_body_bottoms)
                    else:
                        # Demand zone: price expected to go up
                        # Distal = lowest point of formation (most extreme wick)
                        distal_line = formation_data['Low'].min()
                        # Proximal = highest body edge of base candles
                        base_body_tops = []
                        for _, candle in base_data.iterrows():
                            body_top = max(candle['Open'], candle['Close'])
                            base_body_tops.append(body_top)
                        proximal_line = max(base_body_tops)
                    
                    formation_info = {
                        'type': formation_type,
                        'zone_type': zone_type,
                        'start_idx': i,
                        'end_idx': exit_leg_idx,
                        'base_start_idx': base_start_idx,
                        'base_end_idx': base_end_idx,
                        'base_length': base_length,
                        'proximal_line': proximal_line,
                        'distal_line': distal_line,
                        'entry_leg': entry_leg,
                        'exit_leg': exit_leg,
                        'strength': base_length,  # More base candles = stronger zone
                        'formation_range': distal_line - formation_data['Low'].min() if zone_type == 'SUPPLY' else formation_data['High'].max() - distal_line
                    }
                    
                    formations.append(formation_info)
        
        return formations, candle_types
    
    def fetch_uso_data(self, timeframe='daily'):
        """Fetch USO data from Alpha Vantage"""
        
        print(f"📡 Fetching USO {timeframe} data for supply/demand analysis...")
        
        if timeframe == 'daily':
            function = 'TIME_SERIES_DAILY'
            params = {
                'function': function,
                'symbol': 'USO',
                'apikey': self.api_key,
                'outputsize': 'full'
            }
        elif timeframe == 'weekly':
            function = 'TIME_SERIES_WEEKLY'
            params = {
                'function': function,
                'symbol': 'USO', 
                'apikey': self.api_key
            }
        else:
            raise ValueError("Timeframe must be 'daily' or 'weekly'")
        
        response = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
        data = response.json()
        
        # Parse data
        if timeframe == 'daily':
            time_series = data['Time Series (Daily)']
        else:
            time_series = data['Weekly Time Series']
        
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
    
    def create_supply_demand_chart(self, df, formations, candle_types, timeframe='daily', sentiment_method='class'):
        """Create comprehensive supply & demand visualization"""
        
        print("🎯 Creating Supply & Demand visualization...")
        
        # Use recent data for better visualization
        display_data = df.tail(100) if timeframe == 'daily' else df.tail(52)
        
        # Adjust formations and candle_types for the display window
        start_offset = len(df) - len(display_data)
        display_formations = []
        for formation in formations:
            if formation['start_idx'] >= start_offset:
                # Adjust indices for display window
                adj_formation = formation.copy()
                adj_formation['start_idx'] -= start_offset
                adj_formation['end_idx'] -= start_offset
                adj_formation['base_start_idx'] -= start_offset
                adj_formation['base_end_idx'] -= start_offset
                display_formations.append(adj_formation)
        
        display_candle_types = candle_types[start_offset:] if start_offset < len(candle_types) else []
        
        # Create subplots
        method_label = "Class Method (1/3-2/3)" if sentiment_method == 'class' else "Body Ratio Method"
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.2, 0.1],
            subplot_titles=[
                f'USO Supply & Demand Zones - {timeframe.upper()} - {method_label}',
                'Formation Analysis',
                'Candle Classification'
            ]
        )
        
        # Main candlestick chart with zones
        hover_texts = []
        for i, (ts, row) in enumerate(display_data.iterrows()):
            date_str = ts.strftime('%Y-%m-%d')
            candle_type = display_candle_types[i] if i < len(display_candle_types) else 'UNKNOWN'
            hover_text = f"Date: {date_str}<br>Index: {i}<br>{candle_type}<br>O: ${row['Open']:.2f} H: ${row['High']:.2f}<br>L: ${row['Low']:.2f} C: ${row['Close']:.2f}"
            hover_texts.append(hover_text)
        
        # Add candlesticks
        fig.add_trace(
            go.Candlestick(
                x=list(range(len(display_data))),
                open=display_data['Open'],
                high=display_data['High'],
                low=display_data['Low'],
                close=display_data['Close'],
                name='USO',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444',
                text=hover_texts,
                hoverinfo='text'
            ),
            row=1, col=1
        )
        
        # Add supply and demand zones
        zone_colors = {
            'SUPPLY': 'rgba(255, 68, 68, 0.3)',    # Red for supply
            'DEMAND': 'rgba(0, 255, 136, 0.3)'     # Green for demand
        }
        
        zone_line_colors = {
            'SUPPLY': '#ff4444',
            'DEMAND': '#00ff88'
        }
        
        for formation in display_formations:
            zone_type = formation['zone_type']
            proximal = formation['proximal_line']
            distal = formation['distal_line']
            start_idx = formation['start_idx']
            end_idx = formation['end_idx']
            
            # Zone rectangle
            x_coords = [start_idx, end_idx, end_idx, start_idx, start_idx]
            y_coords = [proximal, proximal, distal, distal, proximal]
            
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    fill='tonexty' if zone_type == 'SUPPLY' else 'tozeroy',
                    fillcolor=zone_colors[zone_type],
                    line=dict(color=zone_line_colors[zone_type], width=2),
                    name=f"{formation['type']} Zone",
                    text=f"{formation['type']}<br>Proximal: ${proximal:.2f}<br>Distal: ${distal:.2f}",
                    hoverinfo='text',
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Proximal line (dashed)
            fig.add_trace(
                go.Scatter(
                    x=[start_idx-2, end_idx+5], 
                    y=[proximal, proximal],
                    mode='lines',
                    line=dict(color=zone_line_colors[zone_type], width=1, dash='dash'),
                    name=f"Proximal (${proximal:.2f})",
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            # Distal line (solid)
            fig.add_trace(
                go.Scatter(
                    x=[start_idx-2, end_idx+5],
                    y=[distal, distal], 
                    mode='lines',
                    line=dict(color=zone_line_colors[zone_type], width=2),
                    name=f"Distal (${distal:.2f})",
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
        
        # Formation analysis chart
        formation_y_values = []
        formation_colors = []
        formation_names = []
        
        for i in range(len(display_data)):
            found_formation = None
            for formation in display_formations:
                if formation['start_idx'] <= i <= formation['end_idx']:
                    found_formation = formation
                    break
            
            if found_formation:
                formation_y_values.append(1)
                if found_formation['zone_type'] == 'SUPPLY':
                    formation_colors.append('#ff4444')
                else:
                    formation_colors.append('#00ff88')
                formation_names.append(found_formation['type'])
            else:
                formation_y_values.append(0)
                formation_colors.append('#666666')
                formation_names.append('None')
        
        fig.add_trace(
            go.Bar(
                x=list(range(len(display_data))),
                y=formation_y_values,
                marker=dict(color=formation_colors),
                name='Formations',
                text=formation_names,
                hovertemplate='<b>%{text}</b><br>Index: %{x}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Candle classification chart  
        candle_y_values = []
        candle_colors = []
        
        for i, candle_type in enumerate(display_candle_types[:len(display_data)]):
            if candle_type == 'BASE':
                candle_y_values.append(1)
                candle_colors.append('#ffff00')  # Yellow for base
            elif candle_type == 'LEG_BULLISH':
                candle_y_values.append(1)
                candle_colors.append('#00ff88')  # Green for bullish leg
            elif candle_type == 'LEG_BEARISH':
                candle_y_values.append(1)
                candle_colors.append('#ff4444')  # Red for bearish leg
            else:
                candle_y_values.append(0)
                candle_colors.append('#666666')
        
        fig.add_trace(
            go.Bar(
                x=list(range(len(display_data))),
                y=candle_y_values,
                marker=dict(color=candle_colors),
                name='Candle Types',
                text=display_candle_types[:len(display_data)],
                hovertemplate='<b>%{text}</b><br>Index: %{x}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"USO Supply & Demand Analysis - {timeframe.upper()} ({method_label}) | {len(display_formations)} Formations Detected",
                font=dict(size=18, color='white'),
                x=0.5
            ),
            height=1000,
            paper_bgcolor='rgba(15,15,15,1)',
            plot_bgcolor='rgba(25,25,25,1)', 
            font=dict(color='white', size=11),
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        
        # Update axes
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, gridcolor='rgba(100,100,100,0.2)')
        fig.update_yaxes(title_text="Formation", row=2, col=1, gridcolor='rgba(100,100,100,0.2)', tickmode='array', tickvals=[0, 1], ticktext=['None', 'Active'])
        fig.update_yaxes(title_text="Candle Type", row=3, col=1, gridcolor='rgba(100,100,100,0.2)', tickmode='array', tickvals=[0, 1], ticktext=['Other', 'Classified'])
        
        # Only show x-axis labels on bottom chart
        fig.update_xaxes(showticklabels=False, gridcolor='rgba(100,100,100,0.2)', row=1, col=1)
        fig.update_xaxes(showticklabels=False, gridcolor='rgba(100,100,100,0.2)', row=2, col=1)
        fig.update_xaxes(title_text="Candle Index", gridcolor='rgba(100,100,100,0.2)', row=3, col=1)
        
        return fig
    
    def print_formation_summary(self, formations, candle_types):
        """Print detailed formation analysis"""
        
        print(f"\n📊 SUPPLY & DEMAND ANALYSIS SUMMARY")
        print("=" * 50)
        
        if not formations:
            print("❌ No formations detected")
            return
        
        # Count by type
        formation_counts = {}
        for formation in formations:
            ftype = formation['type']
            if ftype not in formation_counts:
                formation_counts[ftype] = 0
            formation_counts[ftype] += 1
        
        print(f"✅ Total Formations: {len(formations)}")
        for ftype, count in formation_counts.items():
            zone_type = 'SUPPLY' if ftype in ['DBD', 'RBD'] else 'DEMAND'
            print(f"   {ftype} ({zone_type}): {count}")
        
        print(f"\n📈 CANDLE CLASSIFICATION:")
        candle_counts = {}
        for ctype in candle_types:
            if ctype not in candle_counts:
                candle_counts[ctype] = 0
            candle_counts[ctype] += 1
        
        total_candles = len(candle_types)
        for ctype, count in candle_counts.items():
            percentage = (count / total_candles) * 100
            print(f"   {ctype}: {count} ({percentage:.1f}%)")
        
        print(f"\n🎯 FORMATION DETAILS:")
        for i, formation in enumerate(formations, 1):
            print(f"\n{i}. {formation['type']} Zone ({formation['zone_type']})")
            print(f"   Range: Index {formation['start_idx']}-{formation['end_idx']}")
            print(f"   Base: Index {formation['base_start_idx']}-{formation['base_end_idx']} ({formation['base_length']} candles)")
            print(f"   Proximal: ${formation['proximal_line']:.2f}")
            print(f"   Distal: ${formation['distal_line']:.2f}")
            print(f"   Zone Width: ${abs(formation['distal_line'] - formation['proximal_line']):.2f}")
            print(f"   Entry Leg: {formation['entry_leg']}")
            print(f"   Exit Leg: {formation['exit_leg']}")

def main():
    """Create Supply & Demand visualization with both methods"""
    
    print("🚀 USO Supply & Demand Zone Analysis")
    print("Detecting RBR, DBD, RBD, DBR formations with proximal/distal zones")
    print("=" * 70)
    
    visualizer = SupplyDemandVisualizer()
    
    try:
        # Fetch data
        timeframe = 'weekly'  # Start with weekly for cleaner patterns
        df = visualizer.fetch_uso_data(timeframe)
        
        # Generate analysis with BOTH methods
        methods = [
            ('class', 'Class Method (Exact 1/3-2/3 from your course)'),
            ('body_ratio', 'Body Ratio Method (Alternative approach)')
        ]
        
        for method_key, method_name in methods:
            print(f"\n{'='*50}")
            print(f"📊 ANALYSIS USING: {method_name}")
            print(f"{'='*50}")
            
            # Detect formations using this method
            formations, candle_types = visualizer.detect_formations(df, sentiment_method=method_key)
            
            # Print analysis
            visualizer.print_formation_summary(formations, candle_types)
            
            # Create visualization
            fig = visualizer.create_supply_demand_chart(df, formations, candle_types, timeframe, method_key)
            
            # Save chart
            filename = f'uso_supply_demand_{timeframe}_{method_key}_method.html'
            fig.write_html(filename)
            
            print(f"\n✅ Chart Created with {method_name}!")
            print(f"💾 Saved as: {filename}")
        
        print(f"\n🎯 SUMMARY:")
        print("  📚 Class Method: Uses exact 1/3-2/3 range closure rule from your course")
        print("  📊 Body Ratio Method: Uses body-to-wick ratio for classification")
        print("  🔄 Compare both charts to see the differences in zone detection")
        print("  ✅ Both implement full RBR/DBD/RBD/DBR formation detection")
        print("  🎯 Proximal/Distal zone lines calculated for each method")
        print("  🎨 Color-coded supply (red) & demand (green) zones")
        print("  📱 Interactive hover with formation details")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()