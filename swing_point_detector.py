#!/usr/bin/env python3
"""
Swing Point Detection Implementation
Based on approved examples - uses range-based logic where swing points 
only occur when extrema exceed neighboring candle ranges
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from market_data_database import MarketDataDatabase
from datetime import datetime, timedelta

# Dark theme
pio.templates.default = "plotly_dark"

class SwingPointDetector:
    def __init__(self, lookback_periods=1, equal_levels_count=False):
        """
        Initialize swing point detector
        
        Args:
            lookback_periods: How many candles on each side to check (default 1)
            equal_levels_count: Whether equal levels should count as swing points (default False)
        """
        self.lookback_periods = lookback_periods
        self.equal_levels_count = equal_levels_count
    
    def detect_swing_points(self, df):
        """
        Detect swing points using range-based logic with proper alternation constraint
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            List of swing point dictionaries (guaranteed to alternate HIGH/LOW)
        """
        # Step 1: Find all potential swing points using range-based logic
        potential_swings = []
        
        # Need at least (lookback * 2 + 1) candles to detect swing points
        min_candles = (self.lookback_periods * 2) + 1
        if len(df) < min_candles:
            return potential_swings
        
        for i in range(self.lookback_periods, len(df) - self.lookback_periods):
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            # Check for swing high
            is_swing_high = self._is_swing_high(df, i, current_high)
            if is_swing_high:
                potential_swings.append({
                    'index': i,
                    'datetime': df.iloc[i]['datetime'],
                    'type': 'HIGH',
                    'price': current_high,
                    'reason': f'High {current_high} exceeds neighboring ranges'
                })
            
            # Check for swing low
            is_swing_low = self._is_swing_low(df, i, current_low)
            if is_swing_low:
                potential_swings.append({
                    'index': i,
                    'datetime': df.iloc[i]['datetime'],
                    'type': 'LOW',
                    'price': current_low,
                    'reason': f'Low {current_low} below neighboring ranges'
                })
        
        # Step 2: Enforce alternation constraint - select best swing points that alternate
        swing_points = self._enforce_alternation(potential_swings)
        
        return swing_points
    
    def _enforce_alternation(self, potential_swings):
        """
        Enforce alternation constraint: swing points must alternate HIGH/LOW
        
        Logic:
        1. Sort potential swings by index (time order)
        2. Walk through chronologically, ensuring alternation
        3. When consecutive same-type swings are found, pick the most extreme
        
        Args:
            potential_swings: List of potential swing point dicts
            
        Returns:
            List of valid alternating swing points
        """
        if len(potential_swings) == 0:
            return []
        
        # Sort by index (time order)
        sorted_swings = sorted(potential_swings, key=lambda x: x['index'])
        
        if len(sorted_swings) == 1:
            return sorted_swings
        
        alternating_swings = []
        
        # Start with the first swing
        current_group = [sorted_swings[0]]
        current_type = sorted_swings[0]['type']
        
        # Walk through remaining swings
        for i in range(1, len(sorted_swings)):
            swing = sorted_swings[i]
            
            if swing['type'] == current_type:
                # Same type as current group - add to group
                current_group.append(swing)
            else:
                # Different type - finalize current group and start new one
                # Pick the most extreme from current group
                if current_type == 'HIGH':
                    best_swing = max(current_group, key=lambda x: x['price'])
                else:
                    best_swing = min(current_group, key=lambda x: x['price'])
                
                alternating_swings.append(best_swing)
                
                # Start new group
                current_group = [swing]
                current_type = swing['type']
        
        # Handle the last group
        if current_group:
            if current_type == 'HIGH':
                best_swing = max(current_group, key=lambda x: x['price'])
            else:
                best_swing = min(current_group, key=lambda x: x['price'])
            alternating_swings.append(best_swing)
        
        return alternating_swings
    
    def _is_swing_high(self, df, center_idx, center_high):
        """
        Check if center candle is a swing high based on range comparison
        
        Logic: High is swing high if it EXCEEDS the high of neighboring candles
        """
        for offset in range(1, self.lookback_periods + 1):
            left_idx = center_idx - offset
            right_idx = center_idx + offset
            
            left_high = df.iloc[left_idx]['high']
            right_high = df.iloc[right_idx]['high']
            
            # Must exceed ALL neighboring highs
            if self.equal_levels_count:
                if center_high < left_high or center_high < right_high:
                    return False
            else:
                if center_high <= left_high or center_high <= right_high:
                    return False
        
        return True
    
    def _is_swing_low(self, df, center_idx, center_low):
        """
        Check if center candle is a swing low based on range comparison
        
        Logic: Low is swing low if it falls BELOW the low of neighboring candles
        """
        for offset in range(1, self.lookback_periods + 1):
            left_idx = center_idx - offset
            right_idx = center_idx + offset
            
            left_low = df.iloc[left_idx]['low']
            right_low = df.iloc[right_idx]['low']
            
            # Must be below ALL neighboring lows
            if self.equal_levels_count:
                if center_low > left_low or center_low > right_low:
                    return False
            else:
                if center_low >= left_low or center_low >= right_low:
                    return False
        
        return True
    
    def print_swing_analysis(self, df, swing_points, start_idx=0, end_idx=None):
        """Print detailed analysis of swing point detection"""
        
        if end_idx is None:
            end_idx = len(df)
        
        print(f"\n🎯 SWING POINT ANALYSIS (Candles {start_idx}-{end_idx-1})")
        print("=" * 60)
        
        # Show candle data
        print(f"\n📊 Candle Data:")
        for i in range(start_idx, min(end_idx, len(df))):
            row = df.iloc[i]
            range_size = row['high'] - row['low']
            
            # Mark if this is a swing point
            swing_marker = ""
            for swing in swing_points:
                if swing['index'] == i:
                    swing_marker = f" ← SWING {swing['type']} ({swing['price']})"
                    break
            
            print(f"  {i:2d}: O={row['open']:5.2f} H={row['high']:5.2f} L={row['low']:5.2f} C={row['close']:5.2f} | Range={range_size:4.2f}{swing_marker}")
        
        # Show swing points found
        relevant_swings = [s for s in swing_points if start_idx <= s['index'] < end_idx]
        if relevant_swings:
            print(f"\n🎯 Detected Swing Points:")
            for swing in relevant_swings:
                print(f"  • Candle {swing['index']}: {swing['type']} at {swing['price']:.2f} - {swing['reason']}")
        else:
            print(f"\n❌ No swing points detected in range {start_idx}-{end_idx-1}")

def create_swing_point_visualization(df, swing_points, symbol="USO", timeframe="15min"):
    """Create visualization showing detected swing points"""
    
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=f"{symbol} {timeframe}",
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ))
    
    # Add swing highs
    swing_highs = [s for s in swing_points if s['type'] == 'HIGH']
    if swing_highs:
        fig.add_trace(go.Scatter(
            x=[s['datetime'] for s in swing_highs],
            y=[s['price'] for s in swing_highs],
            mode='markers',
            marker=dict(color='#ffaa00', size=10, symbol='triangle-up'),
            name='Swing Highs',
            hovertemplate='Swing HIGH: %{y}<br>%{x}<extra></extra>'
        ))
    
    # Add swing lows
    swing_lows = [s for s in swing_points if s['type'] == 'LOW']
    if swing_lows:
        fig.add_trace(go.Scatter(
            x=[s['datetime'] for s in swing_lows],
            y=[s['price'] for s in swing_lows],
            mode='markers',
            marker=dict(color='#00aaff', size=10, symbol='triangle-down'),
            name='Swing Lows',
            hovertemplate='Swing LOW: %{y}<br>%{x}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} {timeframe} - Swing Point Detection',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_dark',
        showlegend=True,
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def test_swing_detection():
    """Test swing point detection on recent USO data"""
    
    print("🚀 Testing Swing Point Detection on Real Data")
    print("=" * 60)
    
    # Get recent data
    db = MarketDataDatabase()
    
    # Calculate date range for last 100 candles
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Get more than needed
    
    df = db.get_data('USO', '15min', start_date, end_date)
    
    if df.empty:
        print("❌ No data available")
        return
    
    # Take last 50 candles for testing
    df = df.tail(50).reset_index(drop=True)
    
    print(f"📊 Loaded {len(df)} candles for analysis")
    print(f"Date range: {df.iloc[0]['datetime']} to {df.iloc[-1]['datetime']}")
    
    # Initialize detector
    detector = SwingPointDetector(lookback_periods=1, equal_levels_count=False)
    
    # Detect swing points
    swing_points = detector.detect_swing_points(df)
    
    print(f"\n🎯 Found {len(swing_points)} swing points:")
    for swing in swing_points:
        print(f"  • Candle {swing['index']}: {swing['type']} at {swing['price']:.2f}")
    
    # Print detailed analysis for a subset
    if len(df) >= 20:
        detector.print_swing_analysis(df, swing_points, start_idx=10, end_idx=25)
    
    # Create visualization
    fig = create_swing_point_visualization(df, swing_points)
    
    filename = 'swing_point_detection_test.html'
    fig.write_html(filename)
    print(f"\n✅ Visualization saved as: {filename}")
    
    return df, swing_points

def main():
    """Main function"""
    print("🎯 Swing Point Detector")
    print("Range-based detection using approved examples")
    print("=" * 60)
    
    # Test on real data
    df, swing_points = test_swing_detection()
    
    print(f"\n🎯 Summary:")
    print(f"• Algorithm: Range-based comparison")
    print(f"• Lookback periods: 1 candle each side")  
    print(f"• Equal levels count: False")
    print(f"• Swing points found: {len(swing_points)}")
    
    print(f"\n✅ Swing point detection implemented successfully!")
    print("Next: Apply this to supply/demand formation detection")

if __name__ == "__main__":
    main()