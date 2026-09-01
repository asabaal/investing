#!/usr/bin/env python3
"""
Real Market Data Pattern Explorer
Searches through actual market data to find interesting supply/demand zone formations
"""

import sys
sys.path.append('/home/asabaal/asabaal_ventures/repos/investing')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from market_data_database import MarketDataDatabase
from trade_scoring_system import TradeScorer, Zone, ZoneType, MarketContext, TrendDirection, FreshnessStatus

pio.templates.default = 'plotly_dark'

class PatternExplorer:
    def __init__(self):
        self.db = MarketDataDatabase()
        self.scorer = TradeScorer()
    
    def find_interesting_patterns(self, symbols=['SPY', 'AAPL', 'NVDA', 'TSLA', 'GOOGL'], 
                                interval='daily', days_back=500):
        """
        Search through recent market data for interesting patterns
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        interesting_patterns = []
        
        for symbol in symbols:
            print(f"\n🔍 Analyzing {symbol}...")
            
            try:
                # Get data
                df = self.db.get_data(symbol, start_date, end_date, interval)
                if df.empty:
                    print(f"  No data found for {symbol}")
                    continue
                
                print(f"  Got {len(df)} candles from {df.index[0]} to {df.index[-1]}")
                
                # Look for patterns in different time windows
                patterns = self._scan_for_patterns(df, symbol)
                interesting_patterns.extend(patterns)
                
            except Exception as e:
                print(f"  Error analyzing {symbol}: {e}")
                continue
        
        return interesting_patterns
    
    def _scan_for_patterns(self, df, symbol):
        """
        Scan dataframe for interesting supply/demand patterns
        """
        patterns = []
        
        # Use sliding window to find formations
        window_size = 40  # Look at ~40 candle windows
        
        for i in range(len(df) - window_size):
            window_data = df.iloc[i:i+window_size].copy()
            
            # Look for different formation types
            formations = self._identify_formations(window_data, symbol, i)
            patterns.extend(formations)
        
        return patterns
    
    def _identify_formations(self, df, symbol, start_idx):
        """
        Identify potential supply/demand formations in the window
        """
        formations = []
        
        # Simple approach: look for significant swings and consolidations
        highs = df['High'].values
        lows = df['Low'].values
        
        # Find local extremes using simple peak/trough detection
        peaks = []
        troughs = []
        
        for i in range(2, len(df)-2):
            # Peak: higher than surrounding 2 candles on each side
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                peaks.append(i)
            
            # Trough: lower than surrounding 2 candles on each side  
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                troughs.append(i)
        
        # Look for formations with at least 2 peaks or 2 troughs
        if len(peaks) >= 2:
            formation = self._analyze_supply_zone(df, peaks, symbol, start_idx)
            if formation:
                formations.append(formation)
                
        if len(troughs) >= 2:
            formation = self._analyze_demand_zone(df, troughs, symbol, start_idx)
            if formation:
                formations.append(formation)
        
        return formations
    
    def _analyze_supply_zone(self, df, peaks, symbol, start_idx):
        """
        Analyze potential supply zone formation
        """
        if len(peaks) < 2:
            return None
        
        # Take first two significant peaks
        peak1_idx, peak2_idx = peaks[0], peaks[1]
        
        # Check if there's meaningful price action between peaks
        zone_high = max(df['High'].iloc[peak1_idx], df['High'].iloc[peak2_idx])
        zone_low = zone_high - (zone_high * 0.02)  # 2% zone depth
        
        # Look for consolidation between peaks
        consolidation_start = min(peak1_idx, peak2_idx)
        consolidation_end = max(peak1_idx, peak2_idx)
        
        # Need at least 3 candles for base
        if consolidation_end - consolidation_start < 3:
            return None
        
        # Check for leg out after second peak
        leg_out_start = consolidation_end
        leg_out_end = min(len(df) - 1, leg_out_start + 8)
        
        # Calculate leg out movement
        leg_out_move = df['close'].iloc[leg_out_start] - df['low'].iloc[leg_out_start:leg_out_end+1].min()
        zone_range = zone_high - zone_low
        
        # Score the formation
        zone = Zone(
            zone_type=ZoneType.SUPPLY,
            high=zone_high,
            low=zone_low,
            base_candles=consolidation_end - consolidation_start,
            leg_out_start=df['close'].iloc[leg_out_start],
            leg_out_end=df['low'].iloc[leg_out_start:leg_out_end+1].min(),
            opposing_zones=[]
        )
        
        # Determine trend context
        trend = self._determine_trend(df, consolidation_start)
        
        context = MarketContext(
            current_trend=trend,
            current_price=df['close'].iloc[-1],
            long_term_high=df['high'].max(),
            long_term_low=df['low'].min(),
            freshness_status=FreshnessStatus.UNTESTED,  # Simplified for now
            penetration_percentage=0.0,
            target_zone_distance=leg_out_move
        )
        
        scores = self.scorer.calculate_total_score(zone, context)
        
        return {
            'symbol': symbol,
            'type': 'supply',
            'formation': 'RBR/DBD',  # Simplified classification
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d'),
            'window_start_idx': start_idx,
            'zone': zone,
            'context': context,
            'scores': scores,
            'data': df.copy(),
            'zone_high': zone_high,
            'zone_low': zone_low,
            'consolidation_start': consolidation_start,
            'consolidation_end': consolidation_end
        }
    
    def _analyze_demand_zone(self, df, troughs, symbol, start_idx):
        """
        Analyze potential demand zone formation  
        """
        if len(troughs) < 2:
            return None
        
        # Take first two significant troughs
        trough1_idx, trough2_idx = troughs[0], troughs[1]
        
        # Check if there's meaningful price action between troughs
        zone_low = min(df['low'].iloc[trough1_idx], df['low'].iloc[trough2_idx])
        zone_high = zone_low + (zone_low * 0.02)  # 2% zone depth
        
        # Look for consolidation between troughs
        consolidation_start = min(trough1_idx, trough2_idx)
        consolidation_end = max(trough1_idx, trough2_idx)
        
        # Need at least 3 candles for base
        if consolidation_end - consolidation_start < 3:
            return None
        
        # Check for leg out after second trough
        leg_out_start = consolidation_end
        leg_out_end = min(len(df) - 1, leg_out_start + 8)
        
        # Calculate leg out movement
        leg_out_move = df['high'].iloc[leg_out_start:leg_out_end+1].max() - df['close'].iloc[leg_out_start]
        zone_range = zone_high - zone_low
        
        # Score the formation
        zone = Zone(
            zone_type=ZoneType.DEMAND,
            high=zone_high,
            low=zone_low,
            base_candles=consolidation_end - consolidation_start,
            leg_out_start=df['close'].iloc[leg_out_start],
            leg_out_end=df['high'].iloc[leg_out_start:leg_out_end+1].max(),
            opposing_zones=[]
        )
        
        # Determine trend context
        trend = self._determine_trend(df, consolidation_start)
        
        context = MarketContext(
            current_trend=trend,
            current_price=df['close'].iloc[-1],
            long_term_high=df['high'].max(),
            long_term_low=df['low'].min(),
            freshness_status=FreshnessStatus.UNTESTED,  # Simplified for now
            penetration_percentage=0.0,
            target_zone_distance=leg_out_move
        )
        
        scores = self.scorer.calculate_total_score(zone, context)
        
        return {
            'symbol': symbol,
            'type': 'demand',
            'formation': 'RBR/DBR',  # Simplified classification
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d'),
            'window_start_idx': start_idx,
            'zone': zone,
            'context': context,
            'scores': scores,
            'data': df.copy(),
            'zone_high': zone_high,
            'zone_low': zone_low,
            'consolidation_start': consolidation_start,
            'consolidation_end': consolidation_end
        }
    
    def _determine_trend(self, df, reference_idx):
        """
        Simple trend determination based on price movement
        """
        if reference_idx < 10:
            return TrendDirection.SIDEWAYS
            
        # Compare price 10 candles before reference point to reference point
        before_price = df['close'].iloc[reference_idx - 10]
        reference_price = df['close'].iloc[reference_idx]
        
        change_pct = (reference_price - before_price) / before_price
        
        if change_pct > 0.02:  # 2% up
            return TrendDirection.UP
        elif change_pct < -0.02:  # 2% down
            return TrendDirection.DOWN
        else:
            return TrendDirection.SIDEWAYS
    
    def visualize_pattern(self, pattern, filename=None):
        """
        Create visualization for an interesting pattern
        """
        df = pattern['data']
        zone = pattern['zone']
        scores = pattern['scores']
        
        fig = go.Figure()
        
        # Add candlesticks
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'], 
            low=df['low'],
            close=df['close'],
            name=pattern['symbol']
        ))
        
        # Add zone
        fig.add_hrect(
            y0=pattern['zone_low'],
            y1=pattern['zone_high'],
            fillcolor="rgba(255, 255, 0, 0.2)",
            layer="below",
            line_width=2,
            line_color="yellow",
        )
        
        # Add annotations
        title = (f"{pattern['symbol']} {pattern['type'].title()} Zone - "
                f"Score: {scores['total']:.1f}/10.0<br>"
                f"Zone Strength: {scores['zone_strength']:.1f}/2.0, "
                f"Position: {scores['price_position']:.1f}/1.0, "
                f"Trend: {scores['trend_alignment']:.1f}/2.0<br>"
                f"Freshness: {scores['freshness']:.1f}/2.0, "
                f"Time: {scores['time_base']:.1f}/1.0, "
                f"R:R: {scores['profit_potential']:.1f}/2.0")
        
        fig.update_layout(
            title=title,
            yaxis_title='Price',
            xaxis_title='Date',
            showlegend=False,
            template='plotly_dark'
        )
        
        if filename:
            fig.write_html(filename)
            print(f"  💾 Saved visualization: {filename}")
        
        return fig
    
    def rank_patterns_by_score(self, patterns):
        """
        Sort patterns by their total scores and categorize them
        """
        # Sort by total score
        sorted_patterns = sorted(patterns, key=lambda p: p['scores']['total'], reverse=True)
        
        # Categorize
        excellent = [p for p in sorted_patterns if p['scores']['total'] >= 8.0]
        good = [p for p in sorted_patterns if 6.0 <= p['scores']['total'] < 8.0] 
        moderate = [p for p in sorted_patterns if 4.0 <= p['scores']['total'] < 6.0]
        poor = [p for p in sorted_patterns if p['scores']['total'] < 4.0]
        
        return {
            'all': sorted_patterns,
            'excellent': excellent,
            'good': good,
            'moderate': moderate,
            'poor': poor
        }

if __name__ == "__main__":
    explorer = PatternExplorer()
    
    print("🚀 Starting Real Market Data Pattern Exploration...")
    
    # Find patterns
    patterns = explorer.find_interesting_patterns(
        symbols=['SPY', 'AAPL', 'NVDA', 'TSLA', 'GOOGL'],
        interval='daily',
        days_back=200  # Look at last 200 days
    )
    
    print(f"\n📊 Found {len(patterns)} potential formations!")
    
    if patterns:
        # Rank patterns
        ranked = explorer.rank_patterns_by_score(patterns)
        
        print(f"\n🏆 Pattern Categories:")
        print(f"  Excellent (8.0+): {len(ranked['excellent'])}")  
        print(f"  Good (6.0-7.9): {len(ranked['good'])}")
        print(f"  Moderate (4.0-5.9): {len(ranked['moderate'])}")
        print(f"  Poor (<4.0): {len(ranked['poor'])}")
        
        # Show top patterns from each category
        categories = ['excellent', 'good', 'moderate', 'poor']
        
        for category in categories:
            patterns_in_cat = ranked[category]
            if patterns_in_cat:
                print(f"\n🎯 Top {category.upper()} Patterns:")
                
                for i, pattern in enumerate(patterns_in_cat[:3]):  # Show top 3 from each category
                    scores = pattern['scores']
                    print(f"  {i+1}. {pattern['symbol']} {pattern['type']} zone:")
                    print(f"     Total: {scores['total']:.1f}/10.0")
                    print(f"     Strength: {scores['zone_strength']:.1f}/2.0")
                    print(f"     Position: {scores['price_position']:.1f}/1.0") 
                    print(f"     Trend: {scores['trend_alignment']:.1f}/2.0")
                    print(f"     Freshness: {scores['freshness']:.1f}/2.0")
                    print(f"     Time: {scores['time_base']:.1f}/1.0")
                    print(f"     R:R: {scores['profit_potential']:.1f}/2.0")
                    print(f"     Date Range: {pattern['start_date']} to {pattern['end_date']}")
                    
                    # Create visualization
                    filename = f"{pattern['symbol']}_{pattern['type']}_{category}_{i+1}.html"
                    explorer.visualize_pattern(pattern, filename)
                    print()
    else:
        print("No patterns found. Try adjusting search parameters.")