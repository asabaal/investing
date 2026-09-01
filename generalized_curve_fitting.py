#!/usr/bin/env python3
"""
Generalized Curve Fitting for Investing
=======================================

Updated version using the generalized spline utilities from asabaal-utils.
Replaces the original curve_fitting_extrema_detector.py with shared implementation.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Import generalized spline utilities
sys.path.insert(0, '/home/asabaal/asabaal_ventures/repos/asabaal-utils/src')
from asabaal_utils.mathematical_models import SplineExtremaDetector, create_test_data

from market_data_database import MarketDataDatabase

pio.templates.default = 'plotly_dark'


class GeneralizedCurveFittingDetector:
    """
    Updated curve fitting detector using generalized spline utilities.
    Maintains API compatibility with original implementation.
    """
    
    def __init__(self):
        self.db = MarketDataDatabase()
        self.spline_detector = SplineExtremaDetector()
    
    def transform_to_curved_spacetime(self, df):
        """
        Transform OHLC to curved spacetime coordinates
        (Maintained from original implementation)
        """
        transformed = pd.DataFrame(index=df.index)
        
        # Base coordinates
        transformed['low'] = df['Low'] if 'Low' in df.columns else df['low']
        transformed['high'] = df['High'] if 'High' in df.columns else df['high']
        transformed['open'] = df['Open'] if 'Open' in df.columns else df['open']
        
        # Use unadjusted close for proper body calculation
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in df.columns else 'Close'
        if close_col not in df.columns:
            close_col = 'close'
        transformed['close'] = df[close_col]
        
        # Curved spacetime coordinates
        transformed['range'] = transformed['high'] - transformed['low']
        transformed['body'] = abs(transformed['close'] - transformed['open'])
        transformed['body_ratio'] = transformed['body'] / (transformed['range'] + 1e-10)
        
        # Upper wick ratio
        transformed['upper_wick'] = np.where(
            transformed['close'] > transformed['open'],
            transformed['high'] - transformed['close'],
            transformed['high'] - transformed['open']
        )
        transformed['upper_wick_ratio'] = transformed['upper_wick'] / (transformed['range'] + 1e-10)
        
        return transformed
    
    def analyze_data_directly(self, df, window_size=50):
        """
        Complete analysis using provided dataframe - for intraday data
        Updated to use generalized spline utilities.
        """
        print(f"\n🎯 ANALYZING PROVIDED DATA - GENERALIZED CURVE FITTING")
        print("="*60)
        
        if df.empty:
            print(f"❌ No data provided")
            return None
        
        print(f"📊 Analyzing {len(df)} candles")
        return self._analyze_dataframe(df, window_size)
    
    def analyze_symbol(self, symbol, start_date, end_date, window_size=50):
        """
        Complete analysis of a symbol using generalized curve fitting
        """
        print(f"\n🎯 ANALYZING {symbol} - GENERALIZED CURVE FITTING")
        print("="*60)
        
        # Get data
        df = self.db.get_data(symbol, start_date, end_date, 'daily')
        if df.empty:
            print(f"❌ No data for {symbol}")
            return None
        
        print(f"📊 Got {len(df)} daily candles")
        return self._analyze_dataframe(df, window_size)
    
    def _analyze_dataframe(self, df, window_size=50):
        """
        Common analysis logic using generalized spline utilities
        """
        # Transform to curved spacetime
        transformed = self.transform_to_curved_spacetime(df)
        
        # Use sliding window approach
        results = []
        
        for i in range(window_size, len(df), window_size//2):
            window_start = max(0, i - window_size)
            window_end = min(len(df), i)
            
            if window_end - window_start < 20:
                continue
            
            print(f"\n📈 Analyzing window {window_start} to {window_end}")
            
            # Extract window data
            window_df = df.iloc[window_start:window_end].copy()
            window_transformed = transformed.iloc[window_start:window_end].copy()
            
            # Create x-axis (time indices)
            x = np.arange(len(window_df))
            
            # Use generalized spline utilities for both coordinate systems
            traditional_results = {}
            spacetime_results = {}
            
            # Traditional OHLC analysis
            close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in window_df.columns else 'Close'
            if close_col not in window_df.columns:
                close_col = 'close'
            
            high_col = 'High' if 'High' in window_df.columns else 'high'
            low_col = 'Low' if 'Low' in window_df.columns else 'low'
            
            # Analyze traditional OHLC data
            print("  🔧 Analyzing traditional OHLC coordinates...")
            close_curves, close_extrema = self.spline_detector.analyze_data_with_splines(
                x, window_df[close_col].values, "close_price")
            high_curves, high_extrema = self.spline_detector.analyze_data_with_splines(
                x, window_df[high_col].values, "high_price")
            low_curves, low_extrema = self.spline_detector.analyze_data_with_splines(
                x, window_df[low_col].values, "low_price")
            
            traditional_results.update({
                'close_curves': close_curves, 'close_extrema': close_extrema,
                'high_curves': high_curves, 'high_extrema': high_extrema,
                'low_curves': low_curves, 'low_extrema': low_extrema
            })
            
            # Analyze curved spacetime coordinates
            print("  🌌 Analyzing curved spacetime coordinates...")
            low_st_curves, low_st_extrema = self.spline_detector.analyze_data_with_splines(
                x, window_transformed['low'].values, "spacetime_low")
            range_curves, range_extrema = self.spline_detector.analyze_data_with_splines(
                x, window_transformed['range'].values, "spacetime_range")
            body_curves, body_extrema = self.spline_detector.analyze_data_with_splines(
                x, window_transformed['body_ratio'].values, "spacetime_body_ratio")
            wick_curves, wick_extrema = self.spline_detector.analyze_data_with_splines(
                x, window_transformed['upper_wick_ratio'].values, "spacetime_upper_wick")
            
            spacetime_results.update({
                'low_curves': low_st_curves, 'low_extrema': low_st_extrema,
                'range_curves': range_curves, 'range_extrema': range_extrema,
                'body_curves': body_curves, 'body_extrema': body_extrema,
                'wick_curves': wick_curves, 'wick_extrema': wick_extrema
            })
            
            # Store results in format compatible with original implementation
            window_result = {
                'window_start': window_start,
                'window_end': window_end,
                'dates': window_df.index,
                'traditional_analysis': traditional_results,
                'spacetime_analysis': spacetime_results,
                'raw_data': window_df,
                'transformed_data': window_transformed
            }
            
            results.append(window_result)
        
        return {
            'symbol': 'GENERALIZED_ANALYSIS',
            'full_data': df,
            'transformed_data': transformed,
            'window_results': results
        }
    
    def create_analysis_visualization(self, analysis_result, window_idx=0):
        """
        Create visualization using generalized spline results
        Enhanced from original implementation.
        """
        if not analysis_result or not analysis_result['window_results']:
            return None
        
        window = analysis_result['window_results'][window_idx]
        symbol = analysis_result['symbol']
        
        # Create enhanced subplot
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'Traditional OHLC Data', 'Spacetime Low Coordinate',
                'Close Price Spline Fits', 'Spacetime Range Analysis', 
                'Detected Extrema Points', 'Spacetime Body Ratio',
                'Supply/Demand Zones', 'Spacetime Upper Wick Analysis'
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        window_df = window['raw_data']
        transformed_df = window['transformed_data']
        traditional = window['traditional_analysis']
        spacetime = window['spacetime_analysis']
        
        # Get close column name
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in window_df.columns else 'Close'
        if close_col not in window_df.columns:
            close_col = 'close'
        
        # Row 1: Original OHLC data
        fig.add_trace(go.Candlestick(
            x=window_df.index,
            open=window_df['Open'],
            high=window_df['High'],
            low=window_df['Low'],
            close=window_df[close_col],
            name='OHLC',
            showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=window_df.index,
            y=transformed_df['low'],
            mode='lines+markers',
            name='Spacetime Low',
            line=dict(color='cyan'),
            showlegend=False
        ), row=1, col=2)
        
        # Row 2: Spline fits
        if 'close_curves' in traditional and 'cubic_spline' in traditional['close_curves']:
            result = traditional['close_curves']['cubic_spline']
            if result.success:
                x_indices = np.arange(len(window_df))
                x_fine = np.linspace(0, len(window_df)-1, len(window_df)*3)
                try:
                    y_fit = result.fitted_func(x_fine)
                    dates_fine = pd.date_range(window_df.index[0], window_df.index[-1], len(x_fine))
                    fig.add_trace(go.Scatter(
                        x=dates_fine,
                        y=y_fit,
                        mode='lines',
                        name='Close Spline Fit',
                        line=dict(color='yellow', width=2),
                        showlegend=False
                    ), row=2, col=1)
                except Exception as e:
                    print(f"  ⚠️ Spline visualization error: {e}")
        
        # Add range analysis
        fig.add_trace(go.Scatter(
            x=window_df.index,
            y=transformed_df['range'],
            mode='lines+markers',
            name='Price Range',
            line=dict(color='orange'),
            showlegend=False
        ), row=2, col=2)
        
        # Row 3: Extrema points
        fig.add_trace(go.Scatter(
            x=window_df.index,
            y=window_df[close_col],
            mode='lines',
            name='Close Price',
            line=dict(color='white'),
            showlegend=False
        ), row=3, col=1)
        
        # Add extrema markers if available
        if 'close_extrema' in traditional:
            for method_name, extrema_result in traditional['close_extrema'].items():
                if extrema_result.maxima:
                    max_indices = [int(pt[0]) for pt in extrema_result.maxima if int(pt[0]) < len(window_df)]
                    if max_indices:
                        max_dates = [window_df.index[i] for i in max_indices]
                        max_prices = [window_df[close_col].iloc[i] for i in max_indices]
                        
                        fig.add_trace(go.Scatter(
                            x=max_dates,
                            y=max_prices,
                            mode='markers',
                            marker=dict(color='red', size=10, symbol='triangle-down'),
                            name='Supply Zones',
                            showlegend=False
                        ), row=3, col=1)
                
                if extrema_result.minima:
                    min_indices = [int(pt[0]) for pt in extrema_result.minima if int(pt[0]) < len(window_df)]
                    if min_indices:
                        min_dates = [window_df.index[i] for i in min_indices]
                        min_prices = [window_df[close_col].iloc[i] for i in min_indices]
                        
                        fig.add_trace(go.Scatter(
                            x=min_dates,
                            y=min_prices,
                            mode='markers',
                            marker=dict(color='green', size=10, symbol='triangle-up'),
                            name='Demand Zones',
                            showlegend=False
                        ), row=3, col=1)
                break  # Just use first extrema result
        
        # Add body ratio analysis
        fig.add_trace(go.Scatter(
            x=window_df.index,
            y=transformed_df['body_ratio'],
            mode='lines+markers',
            name='Body Ratio',
            line=dict(color='purple'),
            showlegend=False
        ), row=3, col=2)
        
        # Row 4: Advanced analysis
        fig.add_trace(go.Scatter(
            x=window_df.index,
            y=window_df[close_col],
            mode='lines',
            name='Price with Zones',
            line=dict(color='lightblue'),
            showlegend=False
        ), row=4, col=1)
        
        fig.add_trace(go.Scatter(
            x=window_df.index,
            y=transformed_df['upper_wick_ratio'],
            mode='lines+markers',
            name='Upper Wick Ratio',
            line=dict(color='red'),
            showlegend=False
        ), row=4, col=2)
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} Generalized Curve Fitting Analysis - Window {window_idx+1}<br>Using Asabaal-Utils Mathematical Models",
            height=1200,
            template='plotly_dark'
        )
        
        return fig


if __name__ == "__main__":
    print("🚀 GENERALIZED CURVE FITTING FOR INVESTING")
    print("Using shared spline utilities from asabaal-utils")
    print()
    
    detector = GeneralizedCurveFittingDetector()
    
    # Test with TSLA data
    analysis = detector.analyze_symbol('TSLA', '2023-09-01', '2024-01-01', window_size=30)
    
    if analysis:
        print("\n🎯 GENERALIZED ANALYSIS COMPLETE!")
        print(f"Found {len(analysis['window_results'])} windows of analysis")
        
        # Create visualization
        fig = detector.create_analysis_visualization(analysis, window_idx=0)
        if fig:
            filename = "tsla_generalized_curve_fitting.html"
            fig.write_html(filename)
            print(f"💾 Saved visualization: {filename}")
        
        print("\n✅ MIGRATION SUCCESSFUL!")
        print("Investing repo now uses generalized spline utilities from asabaal-utils:")
        print("  • Shared mathematical models across domains")
        print("  • Consistent API and results")
        print("  • Enhanced curve fitting capabilities")
        print("  • Both traditional OHLC and curved spacetime analysis")
    else:
        print("❌ Analysis failed - check data availability")