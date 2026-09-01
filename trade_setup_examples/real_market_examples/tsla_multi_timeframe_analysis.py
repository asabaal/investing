#!/usr/bin/env python3
"""
Multi-Timeframe TSLA Analysis
Demonstrates how to analyze the same formation across different timeframes
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

pio.templates.default = 'plotly_dark'

class MultiTimeframeAnalysis:
    def __init__(self):
        self.db = MarketDataDatabase()
        self.supply_zone_high = 251.0
        self.supply_zone_low = 245.0
    
    def analyze_tsla_formation(self):
        """
        Step 1: Long-term view to identify the RBD formation
        """
        print("🎯 STEP 1: Long-Term Analysis (Daily Timeframe)")
        print("=" * 60)
        
        # Get long-term daily data
        df_daily = self.db.get_data('TSLA', '2023-11-01', '2024-04-01', 'daily')
        
        print(f"📊 Daily Data: {len(df_daily)} candles")
        print(f"Price Range: ${df_daily['Low'].min():.2f} - ${df_daily['High'].max():.2f}")
        
        # Identify the RBD components
        rally_phase = df_daily['2023-11-01':'2023-12-28']  # Rally to peak
        base_phase = df_daily['2023-12-28':'2024-01-05']   # Consolidation
        drop_phase = df_daily['2024-01-05':'2024-02-15']   # Major decline
        
        print(f"\n🔍 RBD Formation Breakdown:")
        print(f"Rally Phase: {rally_phase['Close'].iloc[0]:.2f} → {rally_phase['High'].max():.2f} ({len(rally_phase)} days)")
        print(f"Base Phase: {base_phase['High'].max():.2f} - {base_phase['Low'].min():.2f} ({len(base_phase)} days)")
        print(f"Drop Phase: {drop_phase['Close'].iloc[0]:.2f} → {drop_phase['Low'].min():.2f} ({len(drop_phase)} days)")
        
        # Create long-term visualization
        fig = self.create_long_term_chart(df_daily)
        fig.write_html('tsla_long_term_rbd_formation.html')
        print(f"\n💾 Saved: tsla_long_term_rbd_formation.html")
        
        return df_daily
    
    def simulate_trade_setup(self, df_daily):
        """
        Step 2: If today were April 1, 2024, what would we be watching for?
        """
        print(f"\n🎯 STEP 2: Trade Setup Analysis")
        print("=" * 60)
        
        current_price = df_daily['Close'].iloc[-1]
        distance_to_zone = self.supply_zone_low - current_price
        distance_pct = (distance_to_zone / current_price) * 100
        
        print(f"Current Price (Apr 1, 2024): ${current_price:.2f}")
        print(f"Supply Zone: ${self.supply_zone_low:.2f} - ${self.supply_zone_high:.2f}")
        print(f"Distance to Zone: ${distance_to_zone:.2f} ({distance_pct:.1f}%)")
        
        if distance_pct > 15:
            setup_status = "🟡 MONITOR - Price too far from zone"
        elif distance_pct > 5:
            setup_status = "🟠 WATCH - Price approaching zone"
        elif distance_pct > -2:
            setup_status = "🔴 ACTIVE - Price in/near zone for short entry"
        else:
            setup_status = "❌ MISSED - Price above zone"
            
        print(f"Setup Status: {setup_status}")
        
        trade_plan = f"""
        📋 TRADE PLAN (if price returns to zone):
        
        Entry Zone: ${self.supply_zone_low:.2f} - ${self.supply_zone_high:.2f}
        Entry Type: Short (expecting rejection from supply)
        
        Stop Loss: ${self.supply_zone_high + 5:.2f} (above zone + buffer)
        Target 1: ${current_price:.2f} (recent low)
        Target 2: ${df_daily['Low'].min():.2f} (major low)
        
        Risk/Reward: ~1:3 if targeting major low
        
        What to watch for:
        • Price rally back toward ${self.supply_zone_low:.2f}
        • Rejection signals at the zone (spinning tops, shooting stars)
        • Volume exhaustion on approach to zone
        """
        
        print(trade_plan)
        
        return {
            'current_price': current_price,
            'distance_pct': distance_pct,
            'status': setup_status
        }
    
    def analyze_smaller_timeframe(self):
        """
        Step 3: Look at shorter timeframe to see if formation is detectable
        """
        print(f"\n🎯 STEP 3: Smaller Timeframe Analysis")
        print("=" * 60)
        
        try:
            # Try to get 1-minute data for the base period
            df_1min = self.db.get_data('TSLA', '2023-12-28', '2024-01-05', '1min')
            
            if not df_1min.empty:
                print(f"📊 1-Minute Data: {len(df_1min)} candles")
                
                # Look for micro-formations within the base
                # This would show if our system could detect the formation on smaller TF
                self.analyze_micro_patterns(df_1min)
                
            else:
                print("❌ No 1-minute data available - using daily approximation")
                
        except Exception as e:
            print(f"❌ Cannot get intraday data: {e}")
            print("💡 Using daily data to simulate smaller timeframe view")
            
            # Focus just on the base period in daily data
            df_daily = self.db.get_data('TSLA', '2023-12-20', '2024-01-15', 'daily')
            self.analyze_base_period_daily(df_daily)
    
    def analyze_micro_patterns(self, df_1min):
        """
        Analyze 1-minute patterns within the base period
        """
        print("🔍 Micro-Pattern Analysis (1-minute timeframe):")
        
        # Look for smaller RBD patterns within the larger base
        # This is where our system might detect tradeable formations
        
        peaks = []
        troughs = []
        
        # Simple peak/trough detection on 1-minute data
        for i in range(10, len(df_1min)-10):
            if (df_1min['High'].iloc[i] > df_1min['High'].iloc[i-5:i].max() and
                df_1min['High'].iloc[i] > df_1min['High'].iloc[i+1:i+6].max()):
                peaks.append(i)
                
            if (df_1min['Low'].iloc[i] < df_1min['Low'].iloc[i-5:i].min() and
                df_1min['Low'].iloc[i] < df_1min['Low'].iloc[i+1:i+6].min()):
                troughs.append(i)
        
        print(f"Found {len(peaks)} micro-peaks and {len(troughs)} micro-troughs")
        print("💡 These could be detected by our system as smaller formations")
        
        # Create micro-timeframe chart
        fig = self.create_micro_chart(df_1min, peaks, troughs)
        fig.write_html('tsla_micro_patterns.html')
        print(f"💾 Saved: tsla_micro_patterns.html")
    
    def analyze_base_period_daily(self, df_daily):
        """
        Analyze just the base period on daily timeframe
        """
        print("🔍 Base Period Analysis (Daily focus):")
        
        base_data = df_daily['2023-12-25':'2024-01-10']
        
        print(f"Base Period: {len(base_data)} days")
        print(f"High: ${base_data['High'].max():.2f}")
        print(f"Low: ${base_data['Low'].min():.2f}")
        print(f"Range: ${base_data['High'].max() - base_data['Low'].min():.2f}")
        
        # Check if this looks like a consolidation pattern our system could detect
        volatility = base_data['High'].std()
        print(f"Volatility: ${volatility:.2f}")
        
        if volatility < 5:
            print("✅ Low volatility - good base for supply zone")
        else:
            print("⚠️ High volatility - may be harder to define clear zone")
    
    def create_long_term_chart(self, df):
        """
        Create long-term RBD formation chart
        """
        fig = go.Figure()
        
        # Use unadjusted close for proper coloring
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df.columns else 'Close'
        
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df[close_col],
            name='TSLA'
        ))
        
        # Add supply zone
        fig.add_hrect(
            y0=self.supply_zone_low,
            y1=self.supply_zone_high,
            fillcolor="rgba(255, 165, 0, 0.3)",
            layer="below",
            line_width=2,
            line_color="orange",
        )
        
        # Add RBD phase annotations
        fig.add_annotation(
            x=pd.to_datetime('2023-12-01'),
            y=220,
            text="R: RALLY<br>$195 → $251",
            showarrow=True,
            bgcolor="green",
            bordercolor="white"
        )
        
        fig.add_annotation(
            x=pd.to_datetime('2024-01-02'),
            y=self.supply_zone_high + 5,
            text="B: BASE<br>Supply Zone",
            showarrow=True,
            bgcolor="orange",
            bordercolor="white"
        )
        
        fig.add_annotation(
            x=pd.to_datetime('2024-01-20'),
            y=200,
            text="D: DROP<br>$248 → $175",
            showarrow=True,
            bgcolor="red",
            bordercolor="white"
        )
        
        fig.update_layout(
            title="TSLA RBD Formation - Long Term Analysis<br>Supply Zone at Peak of Rally Before Major Drop",
            yaxis_title='Price ($)',
            xaxis_title='Date',
            template='plotly_dark',
            height=600
        )
        
        return fig
    
    def create_micro_chart(self, df_1min, peaks, troughs):
        """
        Create micro-timeframe analysis chart
        """
        fig = go.Figure()
        
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df_1min.columns else 'Close'
        
        fig.add_trace(go.Candlestick(
            x=df_1min.index,
            open=df_1min['Open'],
            high=df_1min['High'],
            low=df_1min['Low'],
            close=df_1min[close_col],
            name='TSLA 1min'
        ))
        
        # Mark peaks and troughs
        for peak in peaks[:10]:  # Show first 10
            fig.add_trace(go.Scatter(
                x=[df_1min.index[peak]],
                y=[df_1min['High'].iloc[peak]],
                mode='markers',
                marker=dict(color='red', size=8, symbol='triangle-down'),
                name='Micro Peak',
                showlegend=False
            ))
            
        for trough in troughs[:10]:  # Show first 10
            fig.add_trace(go.Scatter(
                x=[df_1min.index[trough]],
                y=[df_1min['Low'].iloc[trough]],
                mode='markers',
                marker=dict(color='green', size=8, symbol='triangle-up'),
                name='Micro Trough',
                showlegend=False
            ))
        
        fig.update_layout(
            title="TSLA Micro-Patterns Within Base Period<br>Shows smaller formations our system could detect",
            yaxis_title='Price ($)',
            xaxis_title='Time',
            template='plotly_dark',
            height=600
        )
        
        return fig

if __name__ == "__main__":
    analyzer = MultiTimeframeAnalysis()
    
    print("🚀 TSLA Multi-Timeframe Formation Analysis")
    print("Demonstrating how formations span multiple scales")
    print()
    
    # Step 1: Identify the formation on long timeframe
    df_daily = analyzer.analyze_tsla_formation()
    
    # Step 2: Determine current trading setup
    trade_info = analyzer.simulate_trade_setup(df_daily)
    
    # Step 3: Look at smaller timeframe for detection
    analyzer.analyze_smaller_timeframe()
    
    print("\n" + "="*60)
    print("🎯 KEY INSIGHTS:")
    print("="*60)
    print("1. Formation identification requires multiple timeframes")
    print("2. Daily view shows the RBD structure clearly")
    print("3. Smaller timeframes reveal micro-patterns")
    print("4. Our system would need multi-scale analysis")
    print("5. Trade setup depends on current price vs zone")
    print("\n✅ Analysis complete - check generated HTML files!")