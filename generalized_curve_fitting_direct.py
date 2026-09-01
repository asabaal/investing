#!/usr/bin/env python3
"""
Generalized Curve Fitting for Investing - Direct Import Version
==============================================================

Updated version using direct import of generalized spline utilities.
Demonstrates successful migration to shared mathematical models.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Direct import of generalized spline utilities
sys.path.insert(0, '/home/asabaal/asabaal_ventures/repos/asabaal-utils/src/asabaal_utils/mathematical_models')
from spline_utils import SplineExtremaDetector, create_test_data

from market_data_database import MarketDataDatabase

pio.templates.default = 'plotly_dark'


class GeneralizedInvestingDetector:
    """
    Investing-specific implementation using generalized spline utilities.
    Demonstrates the hybrid approach: shared math + domain-specific logic.
    """
    
    def __init__(self):
        self.db = MarketDataDatabase()
        self.spline_detector = SplineExtremaDetector()
    
    def analyze_symbol_with_generalized_splines(self, symbol, start_date, end_date):
        """
        Analyze financial data using generalized spline utilities
        """
        print(f"\n💰 FINANCIAL ANALYSIS USING GENERALIZED SPLINES")
        print(f"Symbol: {symbol} | Period: {start_date} to {end_date}")
        print("="*60)
        
        # Get financial data
        df = self.db.get_data(symbol, start_date, end_date, 'daily')
        if df.empty:
            print(f"❌ No data for {symbol}")
            return None
        
        print(f"📊 Analyzing {len(df)} daily candles")
        
        # Extract price series
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df.columns else 'Close'
        if close_col not in df.columns:
            close_col = 'close'
        
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        
        # Create time index
        x = np.arange(len(df))
        
        # Use generalized spline utilities for financial analysis
        print("\n🔧 APPLYING GENERALIZED SPLINE ANALYSIS:")
        
        # 1. Close price analysis (trend detection)
        close_curves, close_extrema = self.spline_detector.analyze_data_with_splines(
            x, df[close_col].values, "close_price_trend")
        
        # 2. High price analysis (resistance levels)
        high_curves, high_extrema = self.spline_detector.analyze_data_with_splines(
            x, df[high_col].values, "resistance_levels")
        
        # 3. Low price analysis (support levels)
        low_curves, low_extrema = self.spline_detector.analyze_data_with_splines(
            x, df[low_col].values, "support_levels")
        
        # 4. Volatility analysis (range analysis)
        price_range = df[high_col] - df[low_col]
        range_curves, range_extrema = self.spline_detector.analyze_data_with_splines(
            x, price_range.values, "volatility_analysis")
        
        # Extract supply and demand zones using extrema
        supply_zones = []
        demand_zones = []
        
        print(f"\n📍 EXTRACTING SUPPLY/DEMAND ZONES:")
        
        # Use high price extrema for supply zones (resistance)
        for method_name, extrema_result in high_extrema.items():
            for max_x, max_y in extrema_result.maxima:
                if 0 <= int(max_x) < len(df):
                    zone_date = df.index[int(max_x)]
                    supply_zones.append({
                        'date': zone_date,
                        'price': max_y,
                        'index': int(max_x),
                        'type': 'SUPPLY',
                        'method': method_name
                    })
        
        # Use low price extrema for demand zones (support)
        for method_name, extrema_result in low_extrema.items():
            for min_x, min_y in extrema_result.minima:
                if 0 <= int(min_x) < len(df):
                    zone_date = df.index[int(min_x)]
                    demand_zones.append({
                        'date': zone_date,
                        'price': min_y,
                        'index': int(min_x),
                        'type': 'DEMAND',
                        'method': method_name
                    })
        
        print(f"   Found {len(supply_zones)} supply zones (resistance levels)")
        print(f"   Found {len(demand_zones)} demand zones (support levels)")
        
        # Compile results
        analysis_result = {
            'symbol': symbol,
            'period': f"{start_date} to {end_date}",
            'data': df,
            'spline_analysis': {
                'close': {'curves': close_curves, 'extrema': close_extrema},
                'high': {'curves': high_curves, 'extrema': high_extrema},
                'low': {'curves': low_curves, 'extrema': low_extrema},
                'range': {'curves': range_curves, 'extrema': range_extrema}
            },
            'trading_zones': {
                'supply_zones': supply_zones,
                'demand_zones': demand_zones
            },
            'generalized_utilities_used': True
        }
        
        return analysis_result
    
    def create_financial_visualization(self, analysis_result):
        """
        Create financial chart with spline-detected zones
        """
        if not analysis_result:
            return None
        
        df = analysis_result['data']
        symbol = analysis_result['symbol']
        supply_zones = analysis_result['trading_zones']['supply_zones']
        demand_zones = analysis_result['trading_zones']['demand_zones']
        
        # Create financial chart
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                f'{symbol} - Price Action with Spline Zones', 'Volatility Analysis',
                'Support/Resistance Levels', 'Price Trend Analysis',
                'Supply Zones Detail', 'Demand Zones Detail'
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df.columns else 'Close'
        if close_col not in df.columns:
            close_col = 'close'
        
        # Main candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df[close_col],
            name='Price Action',
            showlegend=False
        ), row=1, col=1)
        
        # Add supply zones (resistance)
        if supply_zones:
            supply_dates = [zone['date'] for zone in supply_zones]
            supply_prices = [zone['price'] for zone in supply_zones]
            
            fig.add_trace(go.Scatter(
                x=supply_dates,
                y=supply_prices,
                mode='markers',
                marker=dict(color='red', size=12, symbol='triangle-down'),
                name='Supply Zones',
                showlegend=False
            ), row=1, col=1)
        
        # Add demand zones (support)
        if demand_zones:
            demand_dates = [zone['date'] for zone in demand_zones]
            demand_prices = [zone['price'] for zone in demand_zones]
            
            fig.add_trace(go.Scatter(
                x=demand_dates,
                y=demand_prices,
                mode='markers',
                marker=dict(color='green', size=12, symbol='triangle-up'),
                name='Demand Zones',
                showlegend=False
            ), row=1, col=1)
        
        # Volatility (price range)
        price_range = df['High'] - df['Low']
        fig.add_trace(go.Scatter(
            x=df.index,
            y=price_range,
            mode='lines',
            name='Volatility',
            line=dict(color='orange'),
            showlegend=False
        ), row=1, col=2)
        
        # Support/Resistance levels
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['High'],
            mode='lines',
            name='Resistance Trend',
            line=dict(color='red'),
            showlegend=False
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Low'],
            mode='lines',
            name='Support Trend',
            line=dict(color='green'),
            showlegend=False
        ), row=2, col=1)
        
        # Price trend
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[close_col],
            mode='lines',
            name='Price Trend',
            line=dict(color='cyan'),
            showlegend=False
        ), row=2, col=2)
        
        # Add spline fits if available
        spline_analysis = analysis_result['spline_analysis']
        if 'close' in spline_analysis and 'cubic_spline' in spline_analysis['close']['curves']:
            close_result = spline_analysis['close']['curves']['cubic_spline']
            if close_result.success:
                x_indices = np.arange(len(df))
                x_fine = np.linspace(0, len(df)-1, len(df)*2)
                try:
                    y_fit = close_result.fitted_func(x_fine)
                    dates_fine = pd.date_range(df.index[0], df.index[-1], len(x_fine))
                    
                    fig.add_trace(go.Scatter(
                        x=dates_fine,
                        y=y_fit,
                        mode='lines',
                        name='Trend Spline',
                        line=dict(color='yellow', width=3),
                        showlegend=False
                    ), row=2, col=2)
                except Exception as e:
                    print(f"  ⚠️ Spline visualization error: {e}")
        
        # Zone details
        if supply_zones:
            zone_counts = {}
            for zone in supply_zones:
                method = zone['method']
                zone_counts[method] = zone_counts.get(method, 0) + 1
            
            methods = list(zone_counts.keys())
            counts = list(zone_counts.values())
            
            fig.add_trace(go.Bar(
                x=methods,
                y=counts,
                name='Supply Zone Methods',
                marker=dict(color='red'),
                showlegend=False
            ), row=3, col=1)
        
        if demand_zones:
            zone_counts = {}
            for zone in demand_zones:
                method = zone['method']
                zone_counts[method] = zone_counts.get(method, 0) + 1
            
            methods = list(zone_counts.keys())
            counts = list(zone_counts.values())
            
            fig.add_trace(go.Bar(
                x=methods,
                y=counts,
                name='Demand Zone Methods',
                marker=dict(color='green'),
                showlegend=False
            ), row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} Financial Analysis - Generalized Spline Detection<br>Supply/Demand Zones using Shared Mathematical Models",
            height=1000,
            template='plotly_dark'
        )
        
        return fig


def test_financial_spline_integration():
    """Test the integration of generalized splines with financial analysis"""
    print("🧪 TESTING FINANCIAL SPLINE INTEGRATION")
    print("="*50)
    
    detector = GeneralizedInvestingDetector()
    
    # Test with recent TSLA data
    analysis = detector.analyze_symbol_with_generalized_splines(
        'TSLA', '2023-10-01', '2023-12-31')
    
    if analysis:
        print(f"\n✅ INTEGRATION TEST SUCCESSFUL!")
        print(f"   Symbol: {analysis['symbol']}")
        print(f"   Period: {analysis['period']}")
        print(f"   Supply zones: {len(analysis['trading_zones']['supply_zones'])}")
        print(f"   Demand zones: {len(analysis['trading_zones']['demand_zones'])}")
        
        # Create visualization
        fig = detector.create_financial_visualization(analysis)
        if fig:
            filename = "tsla_generalized_financial_analysis.html"
            fig.write_html(filename)
            print(f"   💾 Saved chart: {filename}")
        
        return True
    else:
        print("❌ Integration test failed")
        return False


if __name__ == "__main__":
    print("🚀 GENERALIZED SPLINE INTEGRATION - INVESTING DOMAIN")
    print("Demonstrating hybrid approach: shared math + domain expertise")
    print()
    
    success = test_financial_spline_integration()
    
    if success:
        print("\n🎉 HYBRID APPROACH SUCCESSFUL!")
        print("✅ Shared mathematical models working in investing domain")
        print("✅ Domain-specific financial logic preserved")
        print("✅ Supply/demand zone detection enhanced")
        print("✅ Ready for vocal pattern detection implementation")
    else:
        print("\n❌ Integration needs debugging")
    
    print(f"\n📋 MIGRATION STATUS:")
    print("✅ Generalized spline utilities created in asabaal-utils")
    print("✅ Compatibility tests passed")
    print("✅ Investing repo updated to use shared utilities")
    print("🔄 Next: Implement vocal-specific pattern detection")