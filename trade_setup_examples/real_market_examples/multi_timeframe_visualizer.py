#!/usr/bin/env python3
"""
Multi-Timeframe Visualizer
Creates true multi-scale visualizations showing the same formation at different timeframes
"""

import sys
sys.path.append('/home/asabaal/asabaal_ventures/repos/investing')

import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from market_data_database import MarketDataDatabase
from trade_scoring_system import TradeScorer, Zone, ZoneType, MarketContext, TrendDirection, FreshnessStatus

pio.templates.default = 'plotly_dark'

class MultiTimeframeVisualizer:
    def __init__(self):
        self.db = MarketDataDatabase()
        self.scorer = TradeScorer()
        self.output_dir = "/home/asabaal/asabaal_ventures/repos/investing/trade_setup_examples/real_market_examples/"
    
    def create_multi_scale_tsla_rbd(self):
        """
        TSLA RBD Formation across 3 timeframes
        """
        print("\n🎯 Creating Multi-Scale TSLA RBD Visualization...")
        
        # Get data for each timeframe
        df_yearly = self.db.get_data('TSLA', '2023-01-01', '2024-04-01', 'daily')
        df_formation = self.db.get_data('TSLA', '2023-10-01', '2024-04-01', 'daily') 
        df_base = self.db.get_data('TSLA', '2023-12-20', '2024-01-15', 'daily')
        
        # Create 3-panel subplot
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.08,
            subplot_titles=[
                '📅 SCALE 1: Yearly Context - Peak at 83% of Annual Range',
                '📅 SCALE 2: Formation Context - Clear RBD Structure', 
                '📅 SCALE 3: Base Detail - Supply Zone Formation'
            ],
            row_heights=[0.35, 0.35, 0.3]
        )
        
        # Scale 1: Yearly context
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df_yearly.columns else 'Close'
        fig.add_trace(go.Candlestick(
            x=df_yearly.index,
            open=df_yearly['Open'],
            high=df_yearly['High'],
            low=df_yearly['Low'],
            close=df_yearly[close_col],
            name='TSLA Yearly',
            showlegend=False
        ), row=1, col=1)
        
        # Add yearly high/low annotations
        yearly_high = df_yearly['High'].max()
        yearly_low = df_yearly['Low'].min()
        our_peak = 265.13
        
        fig.add_hline(y=our_peak, line_dash="solid", line_color="orange", 
                     annotation_text=f"Our Peak: ${our_peak:.0f} (83% of range)", 
                     row=1, col=1)
        fig.add_hline(y=yearly_high, line_dash="dash", line_color="red",
                     annotation_text=f"Year High: ${yearly_high:.0f}",
                     row=1, col=1)
        fig.add_hline(y=yearly_low, line_dash="dash", line_color="green", 
                     annotation_text=f"Year Low: ${yearly_low:.0f}",
                     row=1, col=1)
        
        # Scale 2: Formation context  
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df_formation.columns else 'Close'
        fig.add_trace(go.Candlestick(
            x=df_formation.index,
            open=df_formation['Open'],
            high=df_formation['High'],
            low=df_formation['Low'],
            close=df_formation[close_col],
            name='TSLA Formation',
            showlegend=False
        ), row=2, col=1)
        
        # Add RBD phase annotations
        fig.add_annotation(
            x=pd.to_datetime('2023-11-15'),
            y=220,
            text="📈 RALLY<br>$195 → $265",
            showarrow=True,
            bgcolor="green",
            bordercolor="white",
            row=2, col=1
        )
        
        fig.add_annotation(
            x=pd.to_datetime('2024-01-02'),
            y=280,
            text="🔄 BASE<br>Supply Zone",
            showarrow=True,
            bgcolor="orange", 
            bordercolor="white",
            row=2, col=1
        )
        
        fig.add_annotation(
            x=pd.to_datetime('2024-01-20'),
            y=200,
            text="📉 DROP<br>$248 → $175",
            showarrow=True,
            bgcolor="red",
            bordercolor="white",
            row=2, col=1
        )
        
        # Scale 3: Base detail
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df_base.columns else 'Close'
        fig.add_trace(go.Candlestick(
            x=df_base.index,
            open=df_base['Open'],
            high=df_base['High'],
            low=df_base['Low'],
            close=df_base[close_col],
            name='TSLA Base',
            showlegend=False
        ), row=3, col=1)
        
        # Add supply zone to base detail
        supply_zone_high = 265
        supply_zone_low = 245
        
        fig.add_hrect(
            y0=supply_zone_low,
            y1=supply_zone_high,
            fillcolor="rgba(255, 165, 0, 0.3)",
            layer="below",
            line_width=2,
            line_color="orange",
            row=3, col=1
        )
        
        fig.add_annotation(
            x=pd.to_datetime('2024-01-08'),
            y=supply_zone_high + 5,
            text=f"Supply Zone<br>${supply_zone_low}-${supply_zone_high}",
            showarrow=True,
            bgcolor="orange",
            bordercolor="white",
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=(
                "TSLA RBD Formation - Multi-Timeframe Analysis<br>"
                "<b>Same Formation Viewed at 3 Different Scales</b><br>"
                "🔍 Each timeframe reveals different aspects of the same pattern"
            ),
            height=1200,
            template='plotly_dark',
            xaxis=dict(rangeslider=dict(visible=False)),
            xaxis2=dict(rangeslider=dict(visible=False)),
            xaxis3=dict(rangeslider=dict(visible=False))
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=3, col=1)
        
        # Update x-axis titles
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        filename = f"{self.output_dir}TSLA_multi_timeframe_rbd.html"
        fig.write_html(filename)
        print(f"  💾 Saved: {filename}")
        
        return fig
    
    def create_multi_scale_spy_supply(self):
        """
        SPY Supply Zone across 3 timeframes
        """
        print("\n🎯 Creating Multi-Scale SPY Supply Visualization...")
        
        # Get data for each timeframe
        df_quarterly = self.db.get_data('SPY', '2023-10-01', '2024-05-01', 'daily')
        df_monthly = self.db.get_data('SPY', '2024-01-01', '2024-04-01', 'daily')
        df_weekly = self.db.get_data('SPY', '2024-03-01', '2024-04-01', 'daily')
        
        # Create 3-panel subplot
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.08,
            subplot_titles=[
                '📅 SCALE 1: Quarterly Context - 4-Month +28% Bull Run',
                '📅 SCALE 2: Monthly Context - Rally to New Highs',
                '📅 SCALE 3: Weekly Detail - Multiple Rejections at Resistance'
            ],
            row_heights=[0.35, 0.35, 0.3]
        )
        
        # Scale 1: Quarterly context
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df_quarterly.columns else 'Close'
        fig.add_trace(go.Candlestick(
            x=df_quarterly.index,
            open=df_quarterly['Open'],
            high=df_quarterly['High'],
            low=df_quarterly['Low'],
            close=df_quarterly[close_col],
            name='SPY Quarterly',
            showlegend=False
        ), row=1, col=1)
        
        # Add quarterly rally annotations
        q4_low = 409
        q1_high = 525
        
        fig.add_annotation(
            x=pd.to_datetime('2023-11-01'),
            y=430,
            text=f"Q4 Low<br>${q4_low}",
            showarrow=True,
            bgcolor="green",
            bordercolor="white",
            row=1, col=1
        )
        
        fig.add_annotation(
            x=pd.to_datetime('2024-03-15'),
            y=540,
            text=f"Q1 High<br>${q1_high}<br>+28% Rally",
            showarrow=True,
            bgcolor="red",
            bordercolor="white",
            row=1, col=1
        )
        
        # Scale 2: Monthly context
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df_monthly.columns else 'Close'
        fig.add_trace(go.Candlestick(
            x=df_monthly.index,
            open=df_monthly['Open'],
            high=df_monthly['High'],
            low=df_monthly['Low'],
            close=df_monthly[close_col],
            name='SPY Monthly',
            showlegend=False
        ), row=2, col=1)
        
        # Add resistance level
        fig.add_hline(y=520, line_dash="solid", line_color="orange",
                     annotation_text="Resistance at $520",
                     row=2, col=1)
        
        # Scale 3: Weekly detail
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df_weekly.columns else 'Close'
        fig.add_trace(go.Candlestick(
            x=df_weekly.index,
            open=df_weekly['Open'],
            high=df_weekly['High'],
            low=df_weekly['Low'],
            close=df_weekly[close_col],
            name='SPY Weekly',
            showlegend=False
        ), row=3, col=1)
        
        # Add supply zone to weekly detail
        supply_zone_high = 525
        supply_zone_low = 520
        
        fig.add_hrect(
            y0=supply_zone_low,
            y1=supply_zone_high,
            fillcolor="rgba(255, 165, 0, 0.3)",
            layer="below",
            line_width=2,
            line_color="orange",
            row=3, col=1
        )
        
        # Mark rejection days
        rejection_dates = ['2024-03-21', '2024-03-28']
        for date in rejection_dates:
            fig.add_annotation(
                x=pd.to_datetime(date),
                y=530,
                text="❌",
                showarrow=False,
                font=dict(size=20, color="red"),
                row=3, col=1
            )
        
        fig.update_layout(
            title=(
                "SPY Supply Zone - Multi-Timeframe Analysis<br>"
                "<b>Resistance Forms at End of Major Bull Run</b><br>"
                "🔍 Each scale shows different context for the same resistance level"
            ),
            height=1200,
            template='plotly_dark',
            xaxis=dict(rangeslider=dict(visible=False)),
            xaxis2=dict(rangeslider=dict(visible=False)),
            xaxis3=dict(rangeslider=dict(visible=False))
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=3, col=1)
        
        # Update x-axis titles
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        filename = f"{self.output_dir}SPY_multi_timeframe_supply.html"
        fig.write_html(filename)
        print(f"  💾 Saved: {filename}")
        
        return fig
    
    def create_multi_scale_spy_demand(self):
        """
        SPY Demand Zone across 3 timeframes
        """
        print("\n🎯 Creating Multi-Scale SPY Demand Visualization...")
        
        # Get data for each timeframe
        df_cycle = self.db.get_data('SPY', '2024-02-01', '2024-06-01', 'daily')
        df_decline = self.db.get_data('SPY', '2024-03-15', '2024-05-01', 'daily')
        df_support = self.db.get_data('SPY', '2024-04-15', '2024-04-30', 'daily')
        
        # Create 3-panel subplot
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.08,
            subplot_titles=[
                '📅 SCALE 1: Full Cycle - V-Shaped Recovery Pattern',
                '📅 SCALE 2: Decline Context - 4-Week Selling Pressure',
                '📅 SCALE 3: Support Detail - Demand Zone Formation'
            ],
            row_heights=[0.35, 0.35, 0.3]
        )
        
        # Scale 1: Full cycle
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df_cycle.columns else 'Close'
        fig.add_trace(go.Candlestick(
            x=df_cycle.index,
            open=df_cycle['Open'],
            high=df_cycle['High'],
            low=df_cycle['Low'],
            close=df_cycle[close_col],
            name='SPY Cycle',
            showlegend=False
        ), row=1, col=1)
        
        # Add cycle annotations
        fig.add_annotation(
            x=pd.to_datetime('2024-03-20'),
            y=540,
            text="Peak<br>$525",
            showarrow=True,
            bgcolor="red",
            bordercolor="white",
            row=1, col=1
        )
        
        fig.add_annotation(
            x=pd.to_datetime('2024-04-19'),
            y=480,
            text="Trough<br>$494",
            showarrow=True,
            bgcolor="green",
            bordercolor="white",
            row=1, col=1
        )
        
        fig.add_annotation(
            x=pd.to_datetime('2024-05-20'),
            y=550,
            text="Recovery<br>$533",
            showarrow=True,
            bgcolor="blue",
            bordercolor="white",
            row=1, col=1
        )
        
        # Scale 2: Decline context
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df_decline.columns else 'Close'
        fig.add_trace(go.Candlestick(
            x=df_decline.index,
            open=df_decline['Open'],
            high=df_decline['High'],
            low=df_decline['Low'],
            close=df_decline[close_col],
            name='SPY Decline',
            showlegend=False
        ), row=2, col=1)
        
        # Scale 3: Support detail
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df_support.columns else 'Close'
        fig.add_trace(go.Candlestick(
            x=df_support.index,
            open=df_support['Open'],
            high=df_support['High'],
            low=df_support['Low'],
            close=df_support[close_col],
            name='SPY Support',
            showlegend=False
        ), row=3, col=1)
        
        # Add demand zone to support detail
        demand_zone_high = 502
        demand_zone_low = 494
        
        fig.add_hrect(
            y0=demand_zone_low,
            y1=demand_zone_high,
            fillcolor="rgba(0, 255, 255, 0.3)",
            layer="below",
            line_width=2,
            line_color="cyan",
            row=3, col=1
        )
        
        fig.add_annotation(
            x=pd.to_datetime('2024-04-22'),
            y=510,
            text=f"Demand Zone<br>${demand_zone_low}-${demand_zone_high}",
            showarrow=True,
            bgcolor="cyan",
            bordercolor="white",
            row=3, col=1
        )
        
        fig.update_layout(
            title=(
                "SPY Demand Zone - Multi-Timeframe Analysis<br>"
                "<b>Support Forms After Selling Climax</b><br>"
                "🔍 V-shaped recovery validates demand zone effectiveness"
            ),
            height=1200,
            template='plotly_dark',
            xaxis=dict(rangeslider=dict(visible=False)),
            xaxis2=dict(rangeslider=dict(visible=False)),
            xaxis3=dict(rangeslider=dict(visible=False))
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=3, col=1)
        
        # Update x-axis titles
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        filename = f"{self.output_dir}SPY_multi_timeframe_demand.html"
        fig.write_html(filename)
        print(f"  💾 Saved: {filename}")
        
        return fig
    
    def create_comparison_summary(self):
        """
        Create a summary showing why multi-timeframe matters
        """
        print("\n🎯 Creating Multi-Timeframe Summary...")
        
        # Create a text-based summary figure
        fig = go.Figure()
        
        fig.add_annotation(
            text=(
                "<b>🎯 WHY MULTI-TIMEFRAME ANALYSIS MATTERS</b><br><br>"
                
                "<b>📊 SINGLE TIMEFRAME VIEW:</b><br>"
                "• TSLA: '6 daily candles in consolidation'<br>"
                "• SPY Supply: '4 days of resistance'<br>"
                "• SPY Demand: '2 weeks of support'<br><br>"
                
                "<b>🔍 MULTI-TIMEFRAME REVEALS:</b><br>"
                "• TSLA: Peak at 83% of yearly range after +29% rally<br>"
                "• SPY Supply: Resistance after 4-month +28% bull run<br>"
                "• SPY Demand: V-shaped recovery validates zone effectiveness<br><br>"
                
                "<b>💡 KEY INSIGHTS:</b><br>"
                "1. <b>Context Matters:</b> Same zone, different significance at different scales<br>"
                "2. <b>Pattern Recognition:</b> Formations emerge across timeframes<br>"
                "3. <b>Validation:</b> Multiple scales must align for high-confidence setups<br>"
                "4. <b>Entry Timing:</b> Long-term identifies, short-term executes<br>"
                "5. <b>Risk Assessment:</b> Broader context reveals true risk/reward<br><br>"
                
                "<b>🎯 TRADING IMPLICATIONS:</b><br>"
                "• Higher timeframe = Higher confidence<br>"
                "• Multiple timeframe alignment = Better probability<br>"
                "• Single timeframe analysis = Incomplete picture"
            ),
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14),
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="white",
            borderwidth=1
        )
        
        fig.update_layout(
            title="Multi-Timeframe Analysis Summary",
            template='plotly_dark',
            height=800,
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        filename = f"{self.output_dir}multi_timeframe_summary.html"
        fig.write_html(filename)
        print(f"  💾 Saved: {filename}")
        
        return fig
    
    def generate_all_multi_timeframe_examples(self):
        """
        Generate all multi-timeframe visualizations
        """
        print("🚀 Generating Multi-Timeframe Visualizations...")
        print("Creating TRUE multi-scale analysis showing same formations at different timeframes")
        
        # Generate each multi-timeframe example
        self.create_multi_scale_tsla_rbd()
        self.create_multi_scale_spy_supply()
        self.create_multi_scale_spy_demand()
        self.create_comparison_summary()
        
        print("\n" + "="*60)
        print("✅ MULTI-TIMEFRAME VISUALIZATIONS COMPLETE!")
        print("="*60)
        print("Each file now shows the SAME formation at 3 different scales:")
        print("  • Long-term context (why it matters)")
        print("  • Formation context (how it developed)")  
        print("  • Detail context (precise entry/exit levels)")
        print(f"\nFiles saved to: {self.output_dir}")

if __name__ == "__main__":
    visualizer = MultiTimeframeVisualizer()
    visualizer.generate_all_multi_timeframe_examples()