#!/usr/bin/env python3
"""
Create Multiple Real Market Examples
Shows curve fitting supply/demand analysis on different stocks
"""

import sys
sys.path.append('/home/asabaal/asabaal_ventures/repos/investing')

from simple_curve_example import create_simple_example
from market_data_database import MarketDataDatabase
from curve_fitting_extrema_detector import CurveFittingExtemaDetector
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = 'plotly_dark'

def create_example_for_symbol(symbol, start_date, end_date, description):
    """Create curve fitting example for specific symbol"""
    
    print(f"\n📈 ANALYZING {symbol} - {description}")
    print("="*50)
    
    db = MarketDataDatabase()
    detector = CurveFittingExtemaDetector()
    
    # Run analysis
    analysis = detector.analyze_symbol(symbol, start_date, end_date, window_size=35)
    
    if not analysis or not analysis['window_results']:
        print(f"❌ No data for {symbol}")
        return None
    
    # Get best window (middle one)
    middle_idx = len(analysis['window_results']) // 2
    window = analysis['window_results'][middle_idx]
    window_df = window['raw_data']
    transformed_df = window['transformed_data']
    
    print(f"📊 Analyzing {len(window_df)} candles from {window_df.index[0].strftime('%Y-%m-%d')} to {window_df.index[-1].strftime('%Y-%m-%d')}")
    
    # Create focused visualization 
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'{symbol} Price with Curve Fit',
            'Supply/Demand Zones (Extrema)',
            'Inflection Points (Trend Changes)',
            'Analysis Summary'
        ],
        vertical_spacing=0.15
    )
    
    close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in window_df.columns else 'Close'
    if close_col not in window_df.columns:
        close_col = 'close'
    
    # Main price chart with curve fit
    fig.add_trace(go.Candlestick(
        x=window_df.index,
        open=window_df['Open'],
        high=window_df['High'], 
        low=window_df['Low'],
        close=window_df[close_col],
        name=f'{symbol}',
        showlegend=False
    ), row=1, col=1)
    
    # Add cubic spline if available
    if 'cubic_spline' in window['traditional_curves']:
        func = window['traditional_curves']['cubic_spline']['fitted_func']
        try:
            x_fine = np.linspace(0, len(window_df)-1, len(window_df)*2)
            y_fit = func(x_fine)
            dates_fine = pd.date_range(window_df.index[0], window_df.index[-1], len(x_fine))
            
            fig.add_trace(go.Scatter(
                x=dates_fine,
                y=y_fit,
                mode='lines',
                name='Curve Fit',
                line=dict(color='yellow', width=2),
                showlegend=False
            ), row=1, col=1)
        except:
            pass
    
    # Extrema analysis
    supply_zones = []
    demand_zones = []
    inflection_points = []
    
    for curve_name, extrema_data in window['traditional_extrema'].items():
        if 'numerical' in extrema_data:
            maxima = extrema_data['numerical']['maxima']
            minima = extrema_data['numerical']['minima']
            inflections = extrema_data['numerical']['inflection_points']
            
            # Supply zones (maxima)
            if maxima:
                for max_x, max_y in maxima:
                    if int(max_x) < len(window_df):
                        idx = int(max_x)
                        supply_zones.append((window_df.index[idx], window_df[close_col].iloc[idx]))
            
            # Demand zones (minima)
            if minima:
                for min_x, min_y in minima:
                    if int(min_x) < len(window_df):
                        idx = int(min_x)
                        demand_zones.append((window_df.index[idx], window_df[close_col].iloc[idx]))
            
            # Inflection points
            if inflections:
                for infl_x, infl_y in inflections:
                    if int(infl_x) < len(window_df):
                        idx = int(infl_x)
                        inflection_points.append((window_df.index[idx], window_df[close_col].iloc[idx]))
            
            break  # Use first curve
    
    # Plot supply/demand zones
    fig.add_trace(go.Scatter(
        x=window_df.index,
        y=window_df[close_col],
        mode='lines',
        line=dict(color='white', width=1),
        showlegend=False
    ), row=1, col=2)
    
    if supply_zones:
        dates, prices = zip(*supply_zones)
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='markers',
            marker=dict(color='red', size=12, symbol='triangle-down'),
            name='Supply',
            showlegend=False
        ), row=1, col=2)
    
    if demand_zones:
        dates, prices = zip(*demand_zones)
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='markers',
            marker=dict(color='green', size=12, symbol='triangle-up'),
            name='Demand',
            showlegend=False
        ), row=1, col=2)
    
    # Plot inflection points
    fig.add_trace(go.Scatter(
        x=window_df.index,
        y=window_df[close_col],
        mode='lines',
        line=dict(color='gray', width=1),
        showlegend=False
    ), row=2, col=1)
    
    if inflection_points:
        dates, prices = zip(*inflection_points)
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='markers',
            marker=dict(color='yellow', size=8, symbol='diamond'),
            name='Inflection',
            showlegend=False
        ), row=2, col=1)
    
    # Summary stats
    price_change = ((window_df[close_col].iloc[-1] / window_df[close_col].iloc[0]) - 1) * 100
    volatility = window_df[close_col].pct_change().std() * np.sqrt(252) * 100  # Annualized
    
    stats_text = f"""
<b>{symbol} Analysis Results</b><br><br>
📅 Period: {window_df.index[0].strftime('%b %d')} - {window_df.index[-1].strftime('%b %d %Y')}<br>
💰 Price: ${window_df[close_col].iloc[0]:.2f} → ${window_df[close_col].iloc[-1]:.2f}<br>
📈 Change: {price_change:+.1f}%<br>
📊 Volatility: {volatility:.0f}%<br><br>

<b>🎯 Detected Zones:</b><br>
🔴 Supply Zones: {len(supply_zones)}<br>
🟢 Demand Zones: {len(demand_zones)}<br>
🟡 Inflection Points: {len(inflection_points)}<br><br>

<b>📋 Trading Insights:</b><br>
• Red triangles = Resistance levels<br>
• Green triangles = Support levels<br>
• Yellow diamonds = Trend changes<br>
• Yellow curve = Mathematical fit<br><br>

<i>{description}</i>
    """
    
    fig.add_annotation(
        text=stats_text,
        xref="x domain", yref="y domain",
        x=0.05, y=0.95,
        showarrow=False,
        font=dict(size=11),
        bgcolor="rgba(0,0,0,0.8)",
        bordercolor="white",
        borderwidth=1,
        row=2, col=2
    )
    
    fig.update_layout(
        title=f"{symbol} Supply/Demand Zone Detection via Curve Fitting<br>{description}",
        height=750,
        template='plotly_dark',
        showlegend=False
    )
    
    # Remove range sliders 
    for i in range(1, 5):
        fig.update_layout(**{f'xaxis{i}': dict(rangeslider=dict(visible=False))})
    
    # Save example
    filename = f"/home/asabaal/asabaal_ventures/repos/investing/trade_setup_examples/real_market_examples/curve_fitting_examples/{symbol.lower()}_supply_demand_zones.html"
    fig.write_html(filename)
    
    print(f"💾 Saved: {symbol.lower()}_supply_demand_zones.html")
    print(f"✅ Found {len(supply_zones)} supply, {len(demand_zones)} demand zones, {len(inflection_points)} trend changes")
    
    return filename

def create_all_examples():
    """Create comprehensive set of examples"""
    
    print("🚀 CREATING COMPREHENSIVE CURVE FITTING EXAMPLES")
    print("Real market supply/demand zone detection")
    print("="*70)
    
    examples = [
        {
            'symbol': 'SPY',
            'start': '2024-03-01', 
            'end': '2024-07-01',
            'description': 'S&P 500 ETF - Market trending upward with supply zones'
        },
        {
            'symbol': 'AAPL',
            'start': '2023-09-01',
            'end': '2024-01-01',
            'description': 'Apple stock - Tech volatility with clear demand levels'
        },
        {
            'symbol': 'NVDA',
            'start': '2023-05-01',
            'end': '2023-09-01', 
            'description': 'Nvidia - High volatility AI stock with strong zones'
        },
        {
            'symbol': 'TSLA',
            'start': '2023-08-01',
            'end': '2023-12-01',
            'description': 'Tesla - High-beta stock with frequent zone tests'
        }
    ]
    
    results = []
    for example in examples:
        try:
            result = create_example_for_symbol(
                example['symbol'],
                example['start'],
                example['end'], 
                example['description']
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Failed to create {example['symbol']}: {e}")
    
    print(f"\n🎯 CREATED {len(results)} REAL MARKET EXAMPLES!")
    print("📂 Location: trade_setup_examples/real_market_examples/curve_fitting_examples/")
    print("\n✨ Each example demonstrates:")
    print("  • Curve fitting on actual market data")
    print("  • Mathematical extrema = supply/demand zones") 
    print("  • Inflection points = trend change signals")
    print("  • Practical trading zone identification")
    print("  • Both traditional and spacetime coordinate analysis")
    
    return results

if __name__ == "__main__":
    create_all_examples()