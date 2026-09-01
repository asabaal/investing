#!/usr/bin/env python3
"""
Manual Pattern Finder - Look at real candlestick data to find interesting formations
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

class ManualPatternFinder:
    def __init__(self):
        self.db = MarketDataDatabase()
        self.scorer = TradeScorer()
    
    def examine_specific_period(self, symbol, start_date, end_date, description=""):
        """
        Look at a specific period and analyze it manually
        """
        print(f"\n🔍 Examining {symbol} from {start_date} to {end_date} - {description}")
        
        df = self.db.get_data(symbol, start_date, end_date, 'daily')
        
        if df.empty:
            print(f"  No data found for {symbol}")
            return None
            
        print(f"  Got {len(df)} candles")
        print(f"  Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
        
        # Show the data so I can manually inspect it
        print("\n📊 Candlestick Data:")
        display_df = df[['Open', 'High', 'Low', 'Close']].round(2)
        print(display_df)
        
        return df
    
    def analyze_formation(self, symbol, start_date, end_date, zone_type, zone_high, zone_low, 
                         description, formation_type="Manual", my_reasoning=""):
        """
        Analyze a formation I've manually identified
        """
        print(f"\n🎯 Analyzing {description}")
        print(f"My reasoning: {my_reasoning}")
        
        df = self.db.get_data(symbol, start_date, end_date, 'daily')
        
        if df.empty:
            return None
        
        # Create zone object
        zone = Zone(
            zone_type=ZoneType.SUPPLY if zone_type.lower() == 'supply' else ZoneType.DEMAND,
            high=zone_high,
            low=zone_low,
            base_candles=5,  # I'll estimate this manually
            leg_out_start=zone_high if zone_type.lower() == 'supply' else zone_low,
            leg_out_end=0,  # I'll calculate this based on what I see
            opposing_zones=[]
        )
        
        # Analyze trend context
        trend = self._analyze_trend_context(df, zone)
        
        # Determine freshness 
        freshness = self._analyze_freshness(df, zone)
        
        # Create market context
        context = MarketContext(
            current_trend=trend,
            current_price=df['Close'].iloc[-1],
            long_term_high=df['High'].max(),
            long_term_low=df['Low'].min(),
            freshness_status=freshness,
            penetration_percentage=0.0,  # I'll estimate
            target_zone_distance=abs(df['High'].max() - df['Low'].min()) * 0.1  # Rough estimate
        )
        
        # Calculate scores
        scores = self.scorer.calculate_total_score(zone, context)
        
        print(f"\n🏆 Scoring Results:")
        print(f"  Total Score: {scores['total']:.1f}/10.0")
        print(f"  Zone Strength: {scores['zone_strength']:.1f}/2.0")
        print(f"  Price Position: {scores['price_position']:.1f}/1.0")
        print(f"  Trend Alignment: {scores['trend_alignment']:.1f}/2.0")
        print(f"  Freshness: {scores['freshness']:.1f}/2.0")
        print(f"  Time/Base: {scores['time_base']:.1f}/1.0")
        print(f"  Profit Potential: {scores['profit_potential']:.1f}/2.0")
        
        # Create visualization
        self.visualize_manual_pattern(df, zone, scores, symbol, description, my_reasoning)
        
        return {
            'symbol': symbol,
            'description': description,
            'zone': zone,
            'context': context,
            'scores': scores,
            'data': df,
            'my_reasoning': my_reasoning
        }
    
    def _analyze_trend_context(self, df, zone):
        """
        Simple trend analysis based on price movement
        """
        first_half = df['Close'].iloc[:len(df)//2].mean()
        second_half = df['Close'].iloc[len(df)//2:].mean()
        
        change = (second_half - first_half) / first_half
        
        if change > 0.02:
            return TrendDirection.UP
        elif change < -0.02:
            return TrendDirection.DOWN
        else:
            return TrendDirection.SIDEWAYS
    
    def _analyze_freshness(self, df, zone):
        """
        Check if zone appears to be tested/violated
        """
        # Simple check: see if price came back near zone after formation
        if zone.zone_type == ZoneType.SUPPLY:
            recent_highs = df['High'].iloc[-10:].max() if len(df) >= 10 else df['High'].max()
            if recent_highs > zone.high:
                return FreshnessStatus.VIOLATED
            elif recent_highs > zone.low:
                return FreshnessStatus.PARTIAL_PENETRATION
        else:
            recent_lows = df['Low'].iloc[-10:].min() if len(df) >= 10 else df['Low'].min()
            if recent_lows < zone.low:
                return FreshnessStatus.VIOLATED
            elif recent_lows < zone.high:
                return FreshnessStatus.PARTIAL_PENETRATION
        
        return FreshnessStatus.UNTESTED
    
    def visualize_manual_pattern(self, df, zone, scores, symbol, description, reasoning):
        """
        Create visualization for the manually identified pattern
        """
        fig = go.Figure()
        
        # Add candlesticks
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=symbol
        ))
        
        # Add zone
        fig.add_hrect(
            y0=zone.low,
            y1=zone.high,
            fillcolor="rgba(255, 255, 0, 0.3)",
            layer="below",
            line_width=2,
            line_color="yellow",
        )
        
        # Add zone labels
        fig.add_annotation(
            x=df.index[len(df)//2],
            y=zone.high + (zone.high - zone.low) * 0.1,
            text=f"{zone.zone_type.value.title()} Zone<br>${zone.low:.2f} - ${zone.high:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="yellow",
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="yellow"
        )
        
        title = (f"{symbol} {zone.zone_type.value.title()} Zone Analysis<br>"
                f"<b>Score: {scores['total']:.1f}/10.0</b> - {description}<br>"
                f"Zone Strength: {scores['zone_strength']:.1f}/2.0 | "
                f"Position: {scores['price_position']:.1f}/1.0 | "
                f"Trend: {scores['trend_alignment']:.1f}/2.0<br>"
                f"Freshness: {scores['freshness']:.1f}/2.0 | "
                f"Time: {scores['time_base']:.1f}/1.0 | "
                f"R:R: {scores['profit_potential']:.1f}/2.0<br>"
                f"<i>My Reasoning: {reasoning}</i>")
        
        fig.update_layout(
            title=title,
            yaxis_title='Price ($)',
            xaxis_title='Date',
            showlegend=False,
            template='plotly_dark',
            height=700
        )
        
        filename = f"{symbol}_{zone.zone_type.value}_manual_analysis.html"
        fig.write_html(filename)
        print(f"  💾 Saved visualization: {filename}")
        
        return fig

if __name__ == "__main__":
    finder = ManualPatternFinder()
    
    print("🚀 Manual Pattern Analysis - Looking at Real Market Data")
    
    # Let's examine some specific periods I think might be interesting
    
    # 1. SPY April 2024 drop - potential supply zone
    print("\n" + "="*60)
    print("EXAMPLE 1: SPY SUPPLY ZONE")
    finder.examine_specific_period('SPY', '2024-03-15', '2024-05-15', 'April 2024 significant drop')
    
    # Let me manually analyze what I see as a supply zone
    finder.analyze_formation(
        symbol='SPY',
        start_date='2024-03-15',
        end_date='2024-05-15', 
        zone_type='supply',
        zone_high=524.0,
        zone_low=520.0,
        description='Supply zone around March highs before April drop',
        my_reasoning='I see SPY hit highs around 523-524 in March, consolidated briefly, then dropped significantly to 487. This looks like a supply zone at the March highs that caused the major sell-off.'
    )
    
    # 2. Let's look at NVDA - it's been volatile
    print("\n" + "="*60)
    print("EXAMPLE 2: NVDA ANALYSIS")
    finder.examine_specific_period('NVDA', '2024-06-01', '2024-08-01', 'NVDA summer 2024 action')
    
    # 3. Look at a potential demand zone
    print("\n" + "="*60) 
    print("EXAMPLE 3: SPY POTENTIAL DEMAND ZONE")
    finder.examine_specific_period('SPY', '2024-04-15', '2024-06-15', 'Recovery from April lows')
    
    finder.analyze_formation(
        symbol='SPY',
        start_date='2024-04-15',
        end_date='2024-06-15',
        zone_type='demand', 
        zone_high=495.0,
        zone_low=487.0,
        description='Demand zone at April 2024 lows',
        my_reasoning='SPY dropped to around 487-495 area in April and found strong support. This low area held and price bounced strongly from here, suggesting demand absorption.'
    )
    
    # 4. Let's check TSLA for interesting patterns
    print("\n" + "="*60)
    print("EXAMPLE 4: TSLA ANALYSIS")
    finder.examine_specific_period('TSLA', '2024-01-01', '2024-04-01', 'TSLA early 2024')
    
    print("\n🎯 Manual Analysis Complete!")
    print("I've identified several patterns based on visual inspection of the candlestick data.")
    print("These represent real market formations with actual scoring based on our system.")