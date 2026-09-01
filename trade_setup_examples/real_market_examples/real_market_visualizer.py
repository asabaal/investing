#!/usr/bin/env python3
"""
Real Market Data Visualizer
Creates comprehensive visualizations of actual supply/demand zones found in real market data
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

class RealMarketVisualizer:
    def __init__(self):
        self.db = MarketDataDatabase()
        self.scorer = TradeScorer()
        self.output_dir = "/home/asabaal/asabaal_ventures/repos/investing/trade_setup_examples/real_market_examples/"
    
    def create_spy_supply_march_2024(self):
        """
        SPY Supply Zone - March 2024 highs before major drop
        Score: 6.5/10.0
        """
        print("\n🎯 Creating SPY Supply Zone (March 2024) Visualization...")
        
        # Get broader context to show the full rally and decline
        df = self.db.get_data('SPY', '2024-01-01', '2024-05-15', 'daily')
        
        # Define the zone based on my analysis
        zone = Zone(
            zone_type=ZoneType.SUPPLY,
            high=524.0,
            low=520.0,
            base_candles=5,  # Estimated from consolidation period
            leg_out_start=523.0,
            leg_out_end=487.5,  # The major low
            opposing_zones=[]
        )
        
        # Market context
        context = MarketContext(
            current_trend=TrendDirection.SIDEWAYS,  # Mixed trend context
            current_price=515.21,  # End price
            long_term_high=df['High'].max(),
            long_term_low=df['Low'].min(),
            freshness_status=FreshnessStatus.UNTESTED,  # Zone held during this period
            penetration_percentage=0.0,
            target_zone_distance=abs(523.0 - 487.5)
        )
        
        scores = self.scorer.calculate_total_score(zone, context)
        
        # Create visualization
        fig = self._create_detailed_visualization(
            df, zone, scores, context,
            title="SPY Supply Zone - March 2024 Market Top",
            description="Real Market Example: Major resistance at March highs led to 7% decline",
            my_analysis=(
                "🔍 <b>My Analysis:</b><br>"
                "• SPY rallied from $480 (Jan 2024) to $524 (Mar 2024) = +9%<br>"
                "• Multiple tests of $523-524 level showed resistance<br>"
                "• Brief consolidation (5-7 days) before major breakdown<br>"
                "• Decline from $523 → $487 = -7% in 3 weeks<br>"
                "• Zone formed at new yearly highs - perfect positioning<br>"
                "<br><b>Context Shows:</b><br>"
                "• 3-month rally created overbought conditions<br>"
                "• Multiple rejection candles at the highs<br>"
                "• Clear institutional distribution pattern<br>"
                "• Zone held as resistance during decline"
            ),
            key_levels={
                'Supply Zone': (zone.low, zone.high),
                'Rally Start': 480.0,  # January low  
                'Peak High': 524.61,
                'Major Low': 487.5
            }
        )
        
        filename = f"{self.output_dir}SPY_supply_march_2024.html"
        fig.write_html(filename)
        print(f"  💾 Saved: {filename}")
        
        return scores
    
    def create_spy_demand_april_2024(self):
        """
        SPY Demand Zone - April 2024 lows with strong recovery
        Score: 8.5/10.0 ⭐
        """
        print("\n🎯 Creating SPY Demand Zone (April 2024) Visualization...")
        
        # Get broader context to show the full decline and recovery
        df = self.db.get_data('SPY', '2024-03-01', '2024-07-01', 'daily')
        
        # Define the zone
        zone = Zone(
            zone_type=ZoneType.DEMAND,
            high=495.0,
            low=487.0,
            base_candles=4,  # Consolidation at lows
            leg_out_start=490.0,
            leg_out_end=534.39,  # Strong recovery high
            opposing_zones=[]
        )
        
        # Market context - this is why it scores so well
        context = MarketContext(
            current_trend=TrendDirection.UP,  # Recovery trend - excellent alignment
            current_price=534.39,
            long_term_high=df['High'].max(),
            long_term_low=df['Low'].min(),
            freshness_status=FreshnessStatus.UNTESTED,  # Zone held perfectly
            penetration_percentage=0.0,
            target_zone_distance=abs(534.39 - 490.0)
        )
        
        scores = self.scorer.calculate_total_score(zone, context)
        
        # Create visualization
        fig = self._create_detailed_visualization(
            df, zone, scores, context,
            title="SPY Demand Zone - April 2024 Market Bottom ⭐",
            description="Real Market Example: Strong support at April lows led to 9% rally",
            my_analysis=(
                "🔍 <b>My Analysis:</b><br>"
                "• SPY declined from $524 (Mar peak) to $487 (Apr low) = -7%<br>"
                "• Strong support established in $487-495 zone<br>"
                "• Multiple tests of the zone showed buying interest<br>"
                "• Rally from $490 → $534 (+9%) over 2 months<br>"
                "• Zone never violated - perfect freshness maintained<br>"
                "<br><b>Why this scores 8.5/10.0:</b><br>"
                "• Perfect trend alignment - demand zone in recovery (2.0/2.0)<br>"
                "• Perfect freshness - never retested (2.0/2.0)<br>"
                "• Excellent profit potential from the bounce (2.0/2.0)<br>"
                "• Context shows clear reversal from oversold conditions"
            ),
            key_levels={
                'Demand Zone': (zone.low, zone.high),
                'Previous High': 524.0,  # March peak
                'Recovery High': 534.39,
                'Initial Low': 487.51
            }
        )
        
        filename = f"{self.output_dir}SPY_demand_april_2024.html"
        fig.write_html(filename)
        print(f"  💾 Saved: {filename}")
        
        return scores
    
    def create_tsla_supply_january_2024(self):
        """
        TSLA Supply Zone - January 2024 peak before major decline
        Score: 8.5/10.0 
        """
        print("\n🎯 Creating TSLA Supply Zone (January 2024) Visualization...")
        
        # Get more context - show the run-up to the peak and the full decline
        df = self.db.get_data('TSLA', '2023-11-01', '2024-04-01', 'daily')
        
        # Define the zone
        zone = Zone(
            zone_type=ZoneType.SUPPLY,
            high=251.0,
            low=245.0,
            base_candles=3,  # Brief consolidation at peak
            leg_out_start=248.0,
            leg_out_end=175.01,  # Major decline
            opposing_zones=[]
        )
        
        # Market context
        context = MarketContext(
            current_trend=TrendDirection.DOWN,  # Trend alignment hurt by downtrend context
            current_price=188.71,
            long_term_high=df['High'].max(),
            long_term_low=df['Low'].min(),
            freshness_status=FreshnessStatus.UNTESTED,
            penetration_percentage=0.0,
            target_zone_distance=abs(248.0 - 175.01)
        )
        
        scores = self.scorer.calculate_total_score(zone, context)
        
        # Create visualization with full context
        fig = self._create_detailed_visualization(
            df, zone, scores, context,
            title="TSLA Supply Zone - January 2024 Peak with Full Context",
            description="Real Market Example: Rally exhaustion at yearly highs led to 30% decline",
            my_analysis=(
                "🔍 <b>My Analysis:</b><br>"
                "• TSLA rallied from $195 (Nov 2023) to $251 (Jan 2024) = +29%<br>"
                "• Peak at $251 on January 2nd showed immediate rejection<br>"
                "• Very brief consolidation (2-3 days) before massive breakdown<br>"
                "• Decline from $248 → $175 = -30% in 6 weeks<br>"
                "• Zone formed at absolute high of 5-month range<br>"
                "<br><b>Context Shows:</b><br>"
                "• Strong rally into resistance makes peak more significant<br>"
                "• Volume exhaustion at the highs<br>"
                "• Clear distribution pattern at yearly peaks<br>"
                "• Perfect supply zone at range extreme"
            ),
            key_levels={
                'Supply Zone': (zone.low, zone.high),
                'Rally Start': 195.0,  # November low
                'Peak High': 251.25,
                'Major Low': 175.01
            }
        )
        
        filename = f"{self.output_dir}TSLA_supply_january_2024.html"
        fig.write_html(filename)
        print(f"  💾 Saved: {filename}")
        
        return scores
    
    def create_spy_zone_violation_example(self):
        """
        SPY Zone Violation - March supply zone eventually broken
        Shows how zones can be overcome
        """
        print("\n🎯 Creating SPY Zone Violation Example...")
        
        df = self.db.get_data('SPY', '2024-03-15', '2024-10-01', 'daily')
        
        # Original zone that was later violated
        zone = Zone(
            zone_type=ZoneType.SUPPLY,
            high=524.0,
            low=520.0,
            base_candles=5,
            leg_out_start=523.0,
            leg_out_end=487.5,
            opposing_zones=[]
        )
        
        # Context shows violation
        context = MarketContext(
            current_trend=TrendDirection.UP,
            current_price=568.46,  # Well above original zone
            long_term_high=df['High'].max(),
            long_term_low=df['Low'].min(),
            freshness_status=FreshnessStatus.VIOLATED,  # Zone was broken
            penetration_percentage=100.0,  # Complete violation
            target_zone_distance=0.0  # No longer valid for trading
        )
        
        scores = self.scorer.calculate_total_score(zone, context)
        
        # Create visualization
        fig = go.Figure()
        
        # Add candlesticks - use Unadjusted_Close for proper coloring
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df.columns else 'Close'
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df[close_col],
            name='SPY'
        ))
        
        # Add original supply zone
        fig.add_hrect(
            y0=zone.low,
            y1=zone.high,
            fillcolor="rgba(255, 0, 0, 0.2)",
            layer="below",
            line_width=2,
            line_color="red",
            annotation_text="Original Supply Zone<br>Later VIOLATED",
            annotation_position="top left"
        )
        
        # Add violation level
        violation_date = df[df['High'] > zone.high].index[0] if len(df[df['High'] > zone.high]) > 0 else df.index[-1]
        
        fig.add_annotation(
            x=violation_date,
            y=zone.high + 10,
            text="🚨 ZONE VIOLATED<br>Supply Overcome",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            bgcolor="rgba(255,0,0,0.8)",
            bordercolor="red"
        )
        
        title = (f"SPY Supply Zone VIOLATION - Market Evolution<br>"
                f"<b>Score: {scores['total']:.1f}/10.0</b> (Zone no longer valid)<br>"
                f"Original Zone: ${zone.low}-${zone.high} | Current: ${context.current_price:.2f}<br>"
                f"<i>Shows how markets can overcome previous resistance levels</i>")
        
        fig.update_layout(
            title=title,
            yaxis_title='Price ($)',
            xaxis_title='Date',
            showlegend=False,
            template='plotly_dark',
            height=700
        )
        
        filename = f"{self.output_dir}SPY_zone_violation_example.html"
        fig.write_html(filename)
        print(f"  💾 Saved: {filename}")
        
        return scores
    
    def _create_detailed_visualization(self, df, zone, scores, context, title, description, my_analysis, key_levels):
        """
        Create detailed visualization with scoring breakdown
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=['Price Chart with Zone', 'Scoring Breakdown'],
            row_heights=[0.7, 0.3]
        )
        
        # Main candlestick chart - use Unadjusted_Close for proper coloring
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df.columns else 'Close'
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df[close_col],
            name='Price'
        ), row=1, col=1)
        
        # Add zone
        zone_color = "rgba(255, 255, 0, 0.3)" if zone.zone_type == ZoneType.SUPPLY else "rgba(0, 255, 255, 0.3)"
        zone_line_color = "orange" if zone.zone_type == ZoneType.SUPPLY else "cyan"
        
        fig.add_hrect(
            y0=zone.low,
            y1=zone.high,
            fillcolor=zone_color,
            layer="below",
            line_width=3,
            line_color=zone_line_color,
            row=1, col=1
        )
        
        # Add key level annotations
        for level_name, level_value in key_levels.items():
            if isinstance(level_value, tuple):  # Zone range
                mid_point = (level_value[0] + level_value[1]) / 2
                fig.add_annotation(
                    x=df.index[len(df)//2],
                    y=mid_point,
                    text=f"{level_name}<br>${level_value[0]:.2f}-${level_value[1]:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=zone_line_color,
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor=zone_line_color,
                    row=1, col=1
                )
            elif level_name == 'Decline %':
                continue  # Skip percentage annotations
            else:  # Single level
                fig.add_hline(
                    y=level_value,
                    line_dash="dash",
                    line_color="white",
                    opacity=0.7,
                    row=1, col=1
                )
        
        # Scoring breakdown chart
        score_components = ['Zone Strength', 'Price Position', 'Trend Alignment', 'Freshness', 'Time/Base', 'Profit Potential']
        score_values = [
            scores['zone_strength'],
            scores['price_position'],
            scores['trend_alignment'],
            scores['freshness'],
            scores['time_base'],
            scores['profit_potential']
        ]
        max_values = [2.0, 1.0, 2.0, 2.0, 1.0, 2.0]
        
        # Create scoring bars
        fig.add_trace(go.Bar(
            x=score_components,
            y=score_values,
            name='Actual Score',
            marker_color='lightblue',
            text=[f"{v:.1f}" for v in score_values],
            textposition='auto'
        ), row=2, col=1)
        
        # Add max score reference
        fig.add_trace(go.Bar(
            x=score_components,
            y=max_values,
            name='Max Score',
            marker_color='rgba(128,128,128,0.3)',
            text=[f"/{m:.1f}" for m in max_values],
            textposition='auto'
        ), row=2, col=1)
        
        # Update layout
        main_title = (f"{title}<br>"
                     f"<b>Total Score: {scores['total']:.1f}/10.0</b><br>"
                     f"{description}<br>"
                     f"{my_analysis}")
        
        fig.update_layout(
            title=main_title,
            template='plotly_dark',
            height=1000,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        return fig
    
    def generate_all_examples(self):
        """
        Generate all real market examples
        """
        print("🚀 Generating All Real Market Data Examples...")
        
        results = {}
        
        # Generate each example
        results['spy_supply_march'] = self.create_spy_supply_march_2024()
        results['spy_demand_april'] = self.create_spy_demand_april_2024() 
        results['tsla_supply_january'] = self.create_tsla_supply_january_2024()
        results['spy_violation'] = self.create_spy_zone_violation_example()
        
        # Summary
        print("\n" + "="*60)
        print("📊 REAL MARKET EXAMPLES SUMMARY")
        print("="*60)
        
        for name, scores in results.items():
            print(f"{name.upper().replace('_', ' ')}: {scores['total']:.1f}/10.0")
        
        print(f"\n✅ All visualizations saved to: {self.output_dir}")
        
        return results

if __name__ == "__main__":
    visualizer = RealMarketVisualizer()
    results = visualizer.generate_all_examples()