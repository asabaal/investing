#!/usr/bin/env python3
"""
Supply/Demand Zone Detection Using Curve Fitting
Creates real market examples showing how extrema and inflection points identify trading zones
"""

import sys
sys.path.append('/home/asabaal/asabaal_ventures/repos/investing')
sys.path.append('/home/asabaal/asabaal_ventures/repos/investing/trade_setup_examples')

import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from market_data_database import MarketDataDatabase
from curve_fitting_extrema_detector import CurveFittingExtemaDetector

# Import scoring system
from trade_setup_visualizer import TradeSetupVisualizer
from trade_scoring_system import Zone, ZoneType

pio.templates.default = 'plotly_dark'

class SupplyDemandCurveDetector:
    def __init__(self):
        self.db = MarketDataDatabase()
        self.curve_detector = CurveFittingExtemaDetector()
        self.zone_visualizer = TradeSetupVisualizer()
    
    def detect_zones_from_extrema(self, extrema_data, df, price_col='Close'):
        """
        Convert detected extrema and inflection points into supply/demand zones
        """
        zones = []
        
        for curve_name, curve_extrema in extrema_data.items():
            if 'numerical' not in curve_extrema:
                continue
                
            maxima = curve_extrema['numerical']['maxima']
            minima = curve_extrema['numerical']['minima']
            inflections = curve_extrema['numerical']['inflection_points']
            
            # Create supply zones from maxima
            for max_x, max_y in maxima:
                if int(max_x) < len(df):
                    idx = int(max_x)
                    price = df[price_col].iloc[idx]
                    high = df['High'].iloc[idx] if 'High' in df.columns else price * 1.002
                    low = price * 0.998  # Small zone around the extrema
                    
                    zone = Zone(
                        zone_type=ZoneType.SUPPLY,
                        high=high,
                        low=low,
                        base_candles=1,
                        leg_out_start=df.index[max(0, idx-3)],
                        leg_out_end=df.index[min(len(df)-1, idx+3)],
                        opposing_zones=[]
                    )
                    zones.append((zone, f"{curve_name}_supply_extrema"))
            
            # Create demand zones from minima  
            for min_x, min_y in minima:
                if int(min_x) < len(df):
                    idx = int(min_x)
                    price = df[price_col].iloc[idx]
                    low = df['Low'].iloc[idx] if 'Low' in df.columns else price * 0.998
                    high = price * 1.002  # Small zone around the extrema
                    
                    zone = Zone(
                        zone_type=ZoneType.DEMAND,
                        high=high,
                        low=low,
                        base_candles=1,
                        leg_out_start=df.index[max(0, idx-3)],
                        leg_out_end=df.index[min(len(df)-1, idx+3)],
                        opposing_zones=[]
                    )
                    zones.append((zone, f"{curve_name}_demand_extrema"))
        
        return zones
    
    def create_supply_demand_example(self, symbol, start_date, end_date, window_size=50, example_name="curve_fitting_zones"):
        """
        Create comprehensive supply/demand zone example using curve fitting
        """
        print(f"\n🎯 CREATING SUPPLY/DEMAND ZONES - {symbol}")
        print(f"📅 Period: {start_date} to {end_date}")
        print("="*60)
        
        # Get market data
        df = self.db.get_data(symbol, start_date, end_date, 'daily')
        if df.empty:
            print(f"❌ No data for {symbol}")
            return None
        
        # Run curve fitting analysis
        analysis = self.curve_detector.analyze_symbol(symbol, start_date, end_date, window_size)
        if not analysis or not analysis['window_results']:
            print(f"❌ Curve fitting failed for {symbol}")
            return None
        
        print(f"📊 Analyzed {len(df)} candles in {len(analysis['window_results'])} windows")
        
        # Get the middle window for best results
        middle_window_idx = len(analysis['window_results']) // 2
        window_result = analysis['window_results'][middle_window_idx]
        window_df = window_result['raw_data']
        
        # Detect zones from extrema
        traditional_zones = self.detect_zones_from_extrema(
            window_result['traditional_extrema'], window_df
        )
        spacetime_zones = self.detect_zones_from_extrema(
            window_result['spacetime_extrema'], window_df
        )
        
        print(f"🔍 Detected {len(traditional_zones)} traditional zones, {len(spacetime_zones)} spacetime zones")
        
        # Create comprehensive visualization
        fig = self.create_comprehensive_visualization(
            symbol, df, analysis, middle_window_idx, traditional_zones, spacetime_zones
        )
        
        # Save example
        filename = f"/home/asabaal/asabaal_ventures/repos/investing/trade_setup_examples/real_market_examples/curve_fitting_examples/{symbol.lower()}_{example_name}.html"
        fig.write_html(filename)
        print(f"💾 Saved: {filename}")
        
        return {
            'symbol': symbol,
            'data': df,
            'analysis': analysis,
            'traditional_zones': traditional_zones,
            'spacetime_zones': spacetime_zones,
            'filename': filename
        }
    
    def create_comprehensive_visualization(self, symbol, df, analysis, window_idx, traditional_zones, spacetime_zones):
        """
        Create detailed visualization showing curve fitting + supply/demand zones
        """
        window = analysis['window_results'][window_idx]
        window_df = window['raw_data']
        transformed_df = window['transformed_data']
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Traditional Curve Fitting + Supply/Demand Zones',
                'Curved Spacetime Coordinates',
                'Detected Extrema (Red=Max, Green=Min, Yellow=Inflection)',
                'Zone Strength Analysis',
                'Traditional vs Spacetime Comparison',
                'Trading Signals Summary'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"colspan": 2}, None]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in window_df.columns else 'Close'
        if close_col not in window_df.columns:
            close_col = 'close'
        
        # Row 1, Col 1: Traditional OHLC with zones
        fig.add_trace(go.Candlestick(
            x=window_df.index,
            open=window_df['Open'],
            high=window_df['High'],
            low=window_df['Low'],
            close=window_df[close_col],
            name='Price Data',
            showlegend=False
        ), row=1, col=1)
        
        # Add supply/demand zones from traditional analysis
        for zone, zone_source in traditional_zones[:5]:  # Limit to 5 zones for clarity
            color = 'rgba(255,0,0,0.2)' if zone.zone_type == ZoneType.SUPPLY else 'rgba(0,255,0,0.2)'
            fig.add_shape(
                type="rect",
                x0=zone.leg_out_start,
                x1=zone.leg_out_end,
                y0=zone.low,
                y1=zone.high,
                fillcolor=color,
                line=dict(color=color.replace('0.2', '0.8'), width=1),
                row=1, col=1
            )
        
        # Add curve fit
        if 'cubic_spline' in window['traditional_extrema']:
            for curve_name, curve_data in window['traditional_curves'].items():
                if curve_name == 'cubic_spline':
                    try:
                        func = curve_data['fitted_func']
                        x_fine = np.linspace(0, len(window_df)-1, len(window_df)*3)
                        y_fit = func(x_fine)
                        dates_fine = pd.date_range(window_df.index[0], window_df.index[-1], len(x_fine))
                        
                        fig.add_trace(go.Scatter(
                            x=dates_fine,
                            y=y_fit,
                            mode='lines',
                            name='Cubic Spline Fit',
                            line=dict(color='yellow', width=2),
                            showlegend=False
                        ), row=1, col=1)
                        break
                    except:
                        pass
        
        # Row 1, Col 2: Spacetime coordinates
        fig.add_trace(go.Scatter(
            x=window_df.index,
            y=transformed_df['low'],
            mode='lines+markers',
            name='Spacetime Low',
            line=dict(color='cyan'),
            showlegend=False
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=window_df.index,
            y=transformed_df['range'],
            mode='lines+markers',
            name='Range',
            line=dict(color='orange'),
            showlegend=False
        ), row=1, col=2)
        
        # Row 2: Extrema visualization
        for curve_name, extrema_data in window['traditional_extrema'].items():
            if 'numerical' in extrema_data:
                maxima = extrema_data['numerical']['maxima']
                minima = extrema_data['numerical']['minima'] 
                inflections = extrema_data['numerical']['inflection_points']
                
                # Plot extrema on price chart
                if maxima:
                    max_x, max_y = zip(*maxima)
                    max_dates = [window_df.index[int(xi)] for xi in max_x if int(xi) < len(window_df)]
                    max_prices = [window_df[close_col].iloc[int(xi)] for xi in max_x if int(xi) < len(window_df)]
                    
                    fig.add_trace(go.Scatter(
                        x=max_dates,
                        y=max_prices,
                        mode='markers',
                        marker=dict(color='red', size=12, symbol='triangle-down'),
                        name='Supply Zones (Maxima)',
                        showlegend=False
                    ), row=2, col=1)
                
                if minima:
                    min_x, min_y = zip(*minima)
                    min_dates = [window_df.index[int(xi)] for xi in min_x if int(xi) < len(window_df)]
                    min_prices = [window_df[close_col].iloc[int(xi)] for xi in min_x if int(xi) < len(window_df)]
                    
                    fig.add_trace(go.Scatter(
                        x=min_dates,
                        y=min_prices,
                        mode='markers',
                        marker=dict(color='green', size=12, symbol='triangle-up'),
                        name='Demand Zones (Minima)',
                        showlegend=False
                    ), row=2, col=1)
                
                if inflections:
                    infl_x, infl_y = zip(*inflections)
                    infl_dates = [window_df.index[int(xi)] for xi in infl_x if int(xi) < len(window_df)]
                    infl_prices = [window_df[close_col].iloc[int(xi)] for xi in infl_x if int(xi) < len(window_df)]
                    
                    fig.add_trace(go.Scatter(
                        x=infl_dates,
                        y=infl_prices,
                        mode='markers',
                        marker=dict(color='yellow', size=8, symbol='diamond'),
                        name='Inflection Points',
                        showlegend=False
                    ), row=2, col=1)
                
                break  # Use first curve's extrema
        
        # Row 2, Col 2: Zone analysis
        zone_count_supply = len([z for z, _ in traditional_zones if z.zone_type == ZoneType.SUPPLY])
        zone_count_demand = len([z for z, _ in traditional_zones if z.zone_type == ZoneType.DEMAND])
        
        fig.add_trace(go.Bar(
            x=['Supply Zones', 'Demand Zones'],
            y=[zone_count_supply, zone_count_demand],
            marker_color=['red', 'green'],
            name='Zone Count',
            showlegend=False
        ), row=2, col=2)
        
        # Row 3: Summary statistics
        total_extrema = 0
        total_inflections = 0
        for curve_name, extrema_data in window['traditional_extrema'].items():
            if 'numerical' in extrema_data:
                total_extrema += len(extrema_data['numerical']['maxima']) + len(extrema_data['numerical']['minima'])
                total_inflections += len(extrema_data['numerical']['inflection_points'])
        
        summary_text = f"""
        <b>{symbol} Curve Fitting Supply/Demand Analysis</b><br>
        📊 Analysis Period: {window_df.index[0].strftime('%Y-%m-%d')} to {window_df.index[-1].strftime('%Y-%m-%d')}<br>
        🔍 Extrema Detected: {total_extrema} (Supply + Demand zones)<br>
        📈 Inflection Points: {total_inflections} (Trend change signals)<br>
        🎯 Supply Zones: {zone_count_supply}<br>
        🎯 Demand Zones: {zone_count_demand}<br><br>
        
        <b>Key Insights:</b><br>
        • Red triangles = Local maxima (potential supply zones)<br>
        • Green triangles = Local minima (potential demand zones)<br>
        • Yellow diamonds = Inflection points (momentum shifts)<br>
        • Curve fitting reveals mathematical turning points<br>
        • Both traditional OHLC and spacetime coordinates analyzed
        """
        
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.5, y=0.02,
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="white",
            borderwidth=1
        )
        
        fig.update_layout(
            title=f"{symbol} Supply/Demand Zone Detection via Curve Fitting<br>Extrema + Inflection Points → Trading Zones",
            height=1000,
            template='plotly_dark',
            showlegend=True
        )
        
        # Remove range selectors
        for i in range(1, 5):
            fig.update_layout(**{f'xaxis{i}': dict(rangeslider=dict(visible=False))})
        
        return fig

def create_examples():
    """Create multiple real market examples"""
    detector = SupplyDemandCurveDetector()
    
    examples = [
        {
            'symbol': 'TSLA',
            'start': '2023-08-01',
            'end': '2023-12-01',
            'name': 'tsla_supply_demand_curve_analysis',
            'description': 'Tesla major supply/demand zones via curve fitting'
        },
        {
            'symbol': 'SPY',
            'start': '2024-01-01', 
            'end': '2024-06-01',
            'name': 'spy_market_turning_points',
            'description': 'SPY extrema and inflection analysis'
        },
        {
            'symbol': 'AAPL',
            'start': '2023-10-01',
            'end': '2024-02-01', 
            'name': 'aapl_curve_fitting_zones',
            'description': 'Apple supply/demand detection'
        }
    ]
    
    print("🚀 CREATING REAL MARKET SUPPLY/DEMAND EXAMPLES")
    print("Using curve fitting to detect extrema & inflection points")
    print("="*70)
    
    results = []
    for example in examples:
        print(f"\n📈 Processing {example['symbol']}...")
        result = detector.create_supply_demand_example(
            example['symbol'],
            example['start'], 
            example['end'],
            window_size=40,
            example_name=example['name']
        )
        if result:
            results.append(result)
            print(f"✅ {example['description']} - COMPLETE")
        else:
            print(f"❌ {example['symbol']} failed")
    
    print(f"\n🎯 CREATED {len(results)} REAL MARKET EXAMPLES")
    print("📂 Location: trade_setup_examples/real_market_examples/curve_fitting_examples/")
    print("\nEach example shows:")
    print("  • Curve fitting on real market data")
    print("  • Mathematical extrema detection (supply/demand zones)")
    print("  • Inflection points (momentum shifts)")
    print("  • Both traditional OHLC and spacetime coordinate analysis")
    
    return results

if __name__ == "__main__":
    results = create_examples()