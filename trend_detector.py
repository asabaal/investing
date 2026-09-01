#!/usr/bin/env python3
"""
Trend Detection Based on Proper Alternating Swing Points
Uses the swing point foundation to identify uptrends, downtrends, and consolidation
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
from swing_point_detector import SwingPointDetector

# Dark theme
pio.templates.default = "plotly_dark"

class TrendDetector:
    def __init__(self):
        self.swing_detector = SwingPointDetector(lookback_periods=1, equal_levels_count=False)
    
    def detect_trends(self, df, min_swings_for_trend=4):
        """
        Detect trends using alternating swing point analysis
        
        Logic:
        - UPTREND: Higher highs AND higher lows
        - DOWNTREND: Lower highs AND lower lows  
        - CONSOLIDATION: Mixed or insufficient pattern
        
        Args:
            df: DataFrame with OHLC data
            min_swings_for_trend: Minimum swing points needed to establish trend
            
        Returns:
            dict: Trend analysis with segments and breakouts
        """
        print(f"🎯 Detecting trends using swing point analysis")
        
        # Step 1: Get swing points
        swing_points = self.swing_detector.detect_swing_points(df)
        
        if len(swing_points) < min_swings_for_trend:
            print(f"   ❌ Not enough swing points ({len(swing_points)} < {min_swings_for_trend})")
            return {'trends': [], 'swing_points': swing_points, 'breakouts': []}
        
        print(f"   🎯 Found {len(swing_points)} alternating swing points")
        
        # Step 2: Analyze trend segments
        trend_segments = self._analyze_trend_segments(swing_points, min_swings_for_trend)
        
        # Step 3: Identify breakouts (where trend changes)
        breakouts = self._identify_breakouts(swing_points, trend_segments)
        
        print(f"   ✅ Detected {len(trend_segments)} trend segments")
        print(f"   🚀 Found {len(breakouts)} potential breakouts")
        
        return {
            'trends': trend_segments,
            'swing_points': swing_points, 
            'breakouts': breakouts
        }
    
    def _analyze_trend_segments(self, swing_points, min_swings):
        """
        Analyze swing points to identify trend segments
        
        Args:
            swing_points: List of alternating swing points
            min_swings: Minimum swings needed for trend
            
        Returns:
            List of trend segment dicts
        """
        if len(swing_points) < min_swings:
            return []
        
        trend_segments = []
        
        # Use sliding window to identify trend segments
        for start_idx in range(len(swing_points) - min_swings + 1):
            for end_idx in range(start_idx + min_swings - 1, len(swing_points)):
                segment_swings = swing_points[start_idx:end_idx + 1]
                
                # Analyze this segment
                trend_type = self._classify_trend_segment(segment_swings)
                
                if trend_type != 'CONSOLIDATION':
                    # Check if this extends an existing trend or starts a new one
                    segment = {
                        'type': trend_type,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'start_candle': segment_swings[0]['index'],
                        'end_candle': segment_swings[-1]['index'],
                        'start_date': segment_swings[0]['datetime'],
                        'end_date': segment_swings[-1]['datetime'],
                        'swing_count': len(segment_swings),
                        'price_start': segment_swings[0]['price'],
                        'price_end': segment_swings[-1]['price'],
                        'price_change': segment_swings[-1]['price'] - segment_swings[0]['price'],
                        'strength': self._calculate_trend_strength(segment_swings)
                    }
                    
                    # Only add if it doesn't overlap too much with existing segments
                    if not self._overlaps_significantly(segment, trend_segments):
                        trend_segments.append(segment)
        
        # Sort by start position and remove redundant segments
        trend_segments = sorted(trend_segments, key=lambda x: x['start_idx'])
        trend_segments = self._consolidate_overlapping_trends(trend_segments)
        
        return trend_segments
    
    def _classify_trend_segment(self, swing_points):
        """
        Classify a segment of swing points as uptrend, downtrend, or consolidation
        
        Args:
            swing_points: List of swing points in chronological order
            
        Returns:
            str: 'UPTREND', 'DOWNTREND', or 'CONSOLIDATION'
        """
        if len(swing_points) < 4:
            return 'CONSOLIDATION'
        
        # Separate highs and lows
        highs = [sp for sp in swing_points if sp['type'] == 'HIGH']
        lows = [sp for sp in swing_points if sp['type'] == 'LOW']
        
        if len(highs) < 2 or len(lows) < 2:
            return 'CONSOLIDATION'
        
        # Check if highs are rising (higher highs)
        higher_highs = all(highs[i]['price'] > highs[i-1]['price'] for i in range(1, len(highs)))
        
        # Check if lows are rising (higher lows)  
        higher_lows = all(lows[i]['price'] > lows[i-1]['price'] for i in range(1, len(lows)))
        
        # Check if highs are falling (lower highs)
        lower_highs = all(highs[i]['price'] < highs[i-1]['price'] for i in range(1, len(highs)))
        
        # Check if lows are falling (lower lows)
        lower_lows = all(lows[i]['price'] < lows[i-1]['price'] for i in range(1, len(lows)))
        
        # Determine trend
        if higher_highs and higher_lows:
            return 'UPTREND'
        elif lower_highs and lower_lows:
            return 'DOWNTREND'
        else:
            return 'CONSOLIDATION'
    
    def _calculate_trend_strength(self, swing_points):
        """
        Calculate trend strength based on consistency and magnitude
        
        Args:
            swing_points: List of swing points in the trend
            
        Returns:
            float: Trend strength score (0.0 to 1.0)
        """
        if len(swing_points) < 2:
            return 0.0
        
        # Base strength on price change and consistency
        total_move = abs(swing_points[-1]['price'] - swing_points[0]['price'])
        time_span = swing_points[-1]['index'] - swing_points[0]['index'] + 1
        
        # Normalize by price level (percentage move)
        avg_price = sum(sp['price'] for sp in swing_points) / len(swing_points)
        pct_move = total_move / avg_price if avg_price > 0 else 0
        
        # Factor in number of confirming swing points
        swing_bonus = min(len(swing_points) / 10.0, 0.5)  # Cap at 0.5
        
        # Combine factors
        strength = min(pct_move * 2 + swing_bonus, 1.0)  # Cap at 1.0
        
        return strength
    
    def _overlaps_significantly(self, segment, existing_segments, overlap_threshold=0.7):
        """Check if segment overlaps significantly with existing segments"""
        
        for existing in existing_segments:
            # Calculate overlap
            overlap_start = max(segment['start_idx'], existing['start_idx'])
            overlap_end = min(segment['end_idx'], existing['end_idx'])
            
            if overlap_end > overlap_start:
                overlap_size = overlap_end - overlap_start
                segment_size = segment['end_idx'] - segment['start_idx']
                
                if overlap_size / segment_size > overlap_threshold:
                    return True
        
        return False
    
    def _consolidate_overlapping_trends(self, trend_segments):
        """Remove redundant overlapping trend segments, keeping the strongest"""
        
        if len(trend_segments) <= 1:
            return trend_segments
        
        consolidated = []
        
        for segment in trend_segments:
            # Check if this segment should replace or be added to consolidated list
            should_add = True
            
            for i, existing in enumerate(consolidated):
                if self._segments_overlap(segment, existing):
                    # Keep the stronger/longer trend
                    if segment['strength'] > existing['strength'] or segment['swing_count'] > existing['swing_count']:
                        consolidated[i] = segment
                        should_add = False
                        break
                    else:
                        should_add = False
                        break
            
            if should_add:
                consolidated.append(segment)
        
        return consolidated
    
    def _segments_overlap(self, seg1, seg2):
        """Check if two segments overlap in swing point indices"""
        return not (seg1['end_idx'] < seg2['start_idx'] or seg2['end_idx'] < seg1['start_idx'])
    
    def _identify_breakouts(self, swing_points, trend_segments):
        """
        Identify potential breakout points where trends change
        
        Args:
            swing_points: List of swing points
            trend_segments: List of identified trend segments
            
        Returns:
            List of breakout dicts
        """
        breakouts = []
        
        if len(trend_segments) < 2:
            return breakouts
        
        # Look for trend transitions
        for i in range(len(trend_segments) - 1):
            current_trend = trend_segments[i]
            next_trend = trend_segments[i + 1]
            
            # Skip if trends are the same type
            if current_trend['type'] == next_trend['type']:
                continue
            
            # Find the breakout swing point (where trend changes)
            breakout_swing_idx = current_trend['end_idx']
            if breakout_swing_idx < len(swing_points):
                breakout_swing = swing_points[breakout_swing_idx]
                
                breakout = {
                    'swing_point': breakout_swing,
                    'from_trend': current_trend['type'],
                    'to_trend': next_trend['type'],
                    'candle_index': breakout_swing['index'],
                    'datetime': breakout_swing['datetime'],
                    'price': breakout_swing['price'],
                    'significance': self._calculate_breakout_significance(current_trend, next_trend)
                }
                
                breakouts.append(breakout)
        
        return breakouts
    
    def _calculate_breakout_significance(self, from_trend, to_trend):
        """
        Calculate the significance of a trend breakout
        
        Args:
            from_trend: Previous trend segment
            to_trend: New trend segment
            
        Returns:
            float: Significance score (0.0 to 1.0)
        """
        # Base significance on strength of both trends
        avg_strength = (from_trend['strength'] + to_trend['strength']) / 2
        
        # Boost significance for major trend reversals
        if (from_trend['type'] == 'UPTREND' and to_trend['type'] == 'DOWNTREND') or \
           (from_trend['type'] == 'DOWNTREND' and to_trend['type'] == 'UPTREND'):
            reversal_bonus = 0.3
        else:
            reversal_bonus = 0.1
        
        significance = min(avg_strength + reversal_bonus, 1.0)
        return significance

def create_trend_visualization(df, trend_analysis, symbol="USO", timeframe="weekly"):
    """Create comprehensive trend visualization with swing points and breakouts"""
    
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=f'{symbol} {timeframe.upper()}',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ))
    
    swing_points = trend_analysis['swing_points']
    trend_segments = trend_analysis['trends']
    breakouts = trend_analysis['breakouts']
    
    # Add swing points
    swing_highs = [s for s in swing_points if s['type'] == 'HIGH']
    if swing_highs:
        fig.add_trace(go.Scatter(
            x=[s['datetime'] for s in swing_highs],
            y=[s['price'] for s in swing_highs],
            mode='markers',
            marker=dict(color='#ffaa00', size=8, symbol='triangle-up'),
            name='Swing Highs',
            hovertemplate='Swing HIGH: $%{y:.2f}<br>%{x}<extra></extra>'
        ))
    
    swing_lows = [s for s in swing_points if s['type'] == 'LOW']
    if swing_lows:
        fig.add_trace(go.Scatter(
            x=[s['datetime'] for s in swing_lows],
            y=[s['price'] for s in swing_lows],
            mode='markers',
            marker=dict(color='#00aaff', size=8, symbol='triangle-down'),
            name='Swing Lows',
            hovertemplate='Swing LOW: $%{y:.2f}<br>%{x}<extra></extra>'
        ))
    
    # Add trend lines
    trend_colors = {'UPTREND': '#00ff00', 'DOWNTREND': '#ff0000', 'CONSOLIDATION': '#ffff00'}
    
    for i, trend in enumerate(trend_segments):
        start_swing = swing_points[trend['start_idx']]
        end_swing = swing_points[trend['end_idx']]
        
        color = trend_colors.get(trend['type'], '#ffffff')
        
        fig.add_trace(go.Scatter(
            x=[start_swing['datetime'], end_swing['datetime']],
            y=[start_swing['price'], end_swing['price']],
            mode='lines',
            line=dict(color=color, width=3, dash='solid'),
            name=f'{trend["type"]} {i+1}',
            hovertemplate=f'{trend["type"]}<br>Strength: {trend["strength"]:.2f}<br>Swings: {trend["swing_count"]}<extra></extra>'
        ))
    
    # Add breakout markers
    for breakout in breakouts:
        size = 15 + (breakout['significance'] * 10)  # Size based on significance
        
        fig.add_trace(go.Scatter(
            x=[breakout['datetime']],
            y=[breakout['price']],
            mode='markers',
            marker=dict(
                color='#ff00ff',
                size=size,
                symbol='star',
                line=dict(color='white', width=2)
            ),
            name=f'Breakout: {breakout["from_trend"]}→{breakout["to_trend"]}',
            hovertemplate=f'BREAKOUT<br>{breakout["from_trend"]} → {breakout["to_trend"]}<br>Price: $%{{y:.2f}}<br>Significance: {breakout["significance"]:.2f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} {timeframe.upper()} - Trend Analysis with Swing Points',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_dark',
        showlegend=True,
        height=700,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def main():
    """Test trend detection on USO data"""
    print("🎯 Trend Detection Using Proper Swing Points")
    print("=" * 60)
    
    # This will be implemented when we test it
    pass

if __name__ == "__main__":
    main()