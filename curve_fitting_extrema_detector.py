#!/usr/bin/env python3
"""
Curve Fitting Extrema Detector
Practical system for finding supply/demand zones using curve fitting and differentiation
Uses both traditional OHLC and curved spacetime coordinates
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

# Curve fitting imports
from scipy.interpolate import UnivariateSpline, CubicSpline
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks, argrelextrema
import warnings
warnings.filterwarnings('ignore')

# Prophet import (if available)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet not available - using splines only")

pio.templates.default = 'plotly_dark'

class CurveFittingExtemaDetector:
    def __init__(self):
        self.db = MarketDataDatabase()
    
    def transform_to_curved_spacetime(self, df):
        """
        Transform OHLC to curved spacetime coordinates
        Using: low, range, body_ratio, upper_wick_ratio
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
        transformed['body_ratio'] = transformed['body'] / (transformed['range'] + 1e-10)  # Avoid division by zero
        
        # Upper wick ratio
        transformed['upper_wick'] = np.where(
            transformed['close'] > transformed['open'],
            transformed['high'] - transformed['close'],  # Bullish candle
            transformed['high'] - transformed['open']    # Bearish candle
        )
        transformed['upper_wick_ratio'] = transformed['upper_wick'] / (transformed['range'] + 1e-10)
        
        return transformed
    
    def fit_curves_multiple_methods(self, x, y, label="price"):
        """
        Fit curves using multiple methods and return results
        """
        results = {}
        
        print(f"🔧 Fitting curves for {label}...")
        
        # Method 1: Cubic Spline
        try:
            cs = CubicSpline(x, y)
            results['cubic_spline'] = {
                'fitted_func': cs,
                'method': 'cubic_spline',
                'params': 'natural cubic spline'
            }
            print(f"  ✅ Cubic spline fitted")
        except Exception as e:
            print(f"  ❌ Cubic spline failed: {e}")
        
        # Method 2: Univariate Spline (smoothing)
        try:
            us = UnivariateSpline(x, y, s=len(x)*0.1)  # Some smoothing
            results['univariate_spline'] = {
                'fitted_func': us,
                'method': 'univariate_spline', 
                'params': f'smoothing factor: {len(x)*0.1}'
            }
            print(f"  ✅ Univariate spline fitted")
        except Exception as e:
            print(f"  ❌ Univariate spline failed: {e}")
        
        # Method 3: Polynomial (degree 5-8)
        for degree in [5, 6, 7, 8]:
            try:
                if len(x) > degree + 1:  # Need enough points
                    poly_coeffs = np.polyfit(x, y, degree)
                    poly_func = np.poly1d(poly_coeffs)
                    results[f'poly_{degree}'] = {
                        'fitted_func': poly_func,
                        'method': 'polynomial',
                        'params': f'degree {degree}',
                        'coefficients': poly_coeffs
                    }
                    print(f"  ✅ Polynomial degree {degree} fitted")
                    break  # Use first successful polynomial
            except Exception as e:
                continue
        
        # Method 4: Prophet (if available and appropriate)
        if PROPHET_AVAILABLE and len(y) > 10:
            try:
                # Prepare data for Prophet
                prophet_df = pd.DataFrame({
                    'ds': pd.to_datetime(x, unit='D', origin='2020-01-01'),  # Convert to dates
                    'y': y
                })
                
                model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.1
                )
                model.fit(prophet_df)
                
                results['prophet'] = {
                    'fitted_model': model,
                    'method': 'prophet',
                    'params': 'facebook prophet with trend detection'
                }
                print(f"  ✅ Prophet fitted")
            except Exception as e:
                print(f"  ⚠️ Prophet failed: {e}")
        
        return results
    
    def find_extrema_and_inflection_points(self, fitted_curves, x_range):
        """
        Find extrema and inflection points using differentiation
        """
        results = {}
        
        for curve_name, curve_data in fitted_curves.items():
            print(f"\n🔍 Analyzing {curve_name} for extrema...")
            
            if curve_data['method'] == 'prophet':
                continue  # Skip Prophet for now - different approach needed
            
            func = curve_data['fitted_func']
            extrema_results = {}
            
            try:
                # Create fine-grained x values for analysis
                x_fine = np.linspace(x_range[0], x_range[-1], len(x_range) * 10)
                y_fine = func(x_fine)
                
                # Method 1: Numerical differentiation
                if hasattr(func, 'derivative'):
                    # For splines that have analytical derivatives
                    first_deriv = func.derivative(1)
                    second_deriv = func.derivative(2)
                    
                    # Find roots of first derivative (extrema)
                    extrema_candidates = []
                    for i in range(len(x_fine)-1):
                        if (first_deriv(x_fine[i]) * first_deriv(x_fine[i+1])) < 0:
                            # Sign change indicates root
                            root = minimize_scalar(
                                lambda x: abs(first_deriv(x)),
                                bounds=(x_fine[i], x_fine[i+1]),
                                method='bounded'
                            )
                            if root.success:
                                extrema_candidates.append(root.x)
                    
                    # Classify extrema as maxima or minima
                    maxima = []
                    minima = []
                    for x_ext in extrema_candidates:
                        if second_deriv(x_ext) < 0:
                            maxima.append((x_ext, func(x_ext)))
                        elif second_deriv(x_ext) > 0:
                            minima.append((x_ext, func(x_ext)))
                    
                    # Find inflection points (roots of second derivative)
                    inflection_candidates = []
                    for i in range(len(x_fine)-1):
                        if (second_deriv(x_fine[i]) * second_deriv(x_fine[i+1])) < 0:
                            root = minimize_scalar(
                                lambda x: abs(second_deriv(x)),
                                bounds=(x_fine[i], x_fine[i+1]),
                                method='bounded'
                            )
                            if root.success:
                                inflection_candidates.append((root.x, func(root.x)))
                    
                    extrema_results['analytical'] = {
                        'maxima': maxima,
                        'minima': minima,
                        'inflection_points': inflection_candidates
                    }
                
                # Method 2: Numerical peak finding (always available)
                peaks_max, _ = find_peaks(y_fine, height=np.percentile(y_fine, 70))
                peaks_min, _ = find_peaks(-y_fine, height=np.percentile(-y_fine, 70))
                
                numerical_maxima = [(x_fine[i], y_fine[i]) for i in peaks_max]
                numerical_minima = [(x_fine[i], y_fine[i]) for i in peaks_min]
                
                # Simple inflection point detection (curvature changes)
                second_diff = np.diff(np.diff(y_fine))
                inflection_indices = np.where(np.diff(np.sign(second_diff)))[0] + 1
                numerical_inflections = [(x_fine[i], y_fine[i]) for i in inflection_indices if i < len(x_fine)]
                
                extrema_results['numerical'] = {
                    'maxima': numerical_maxima,
                    'minima': numerical_minima,
                    'inflection_points': numerical_inflections
                }
                
                print(f"  📊 Found {len(numerical_maxima)} maxima, {len(numerical_minima)} minima, {len(numerical_inflections)} inflection points")
                
            except Exception as e:
                print(f"  ❌ Analysis failed: {e}")
                continue
            
            results[curve_name] = extrema_results
        
        return results
    
    def analyze_data_directly(self, df, window_size=50):
        """
        Complete analysis using provided dataframe - for intraday data
        """
        print(f"\n🎯 ANALYZING PROVIDED DATA - CURVE FITTING APPROACH")
        print("="*60)
        
        if df.empty:
            print(f"❌ No data provided")
            return None
        
        print(f"📊 Analyzing {len(df)} candles")
        
        # Use the same analysis logic as analyze_symbol but with provided data
        return self._analyze_dataframe(df, window_size)
    
    def analyze_symbol(self, symbol, start_date, end_date, window_size=50):
        """
        Complete analysis of a symbol using curve fitting
        """
        print(f"\n🎯 ANALYZING {symbol} - CURVE FITTING APPROACH")
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
        Common analysis logic for both methods
        """
        
        # Transform to curved spacetime
        transformed = self.transform_to_curved_spacetime(df)
        
        # Use sliding window approach
        results = []
        
        for i in range(window_size, len(df), window_size//2):  # 50% overlap
            window_start = max(0, i - window_size)
            window_end = min(len(df), i)
            
            if window_end - window_start < 20:  # Need minimum data
                continue
            
            print(f"\n📈 Analyzing window {window_start} to {window_end}")
            
            # Extract window data
            window_df = df.iloc[window_start:window_end].copy()
            window_transformed = transformed.iloc[window_start:window_end].copy()
            
            # Create x-axis (time indices)
            x = np.arange(len(window_df))
            
            # Fit curves for both coordinate systems
            traditional_curves = {}
            spacetime_curves = {}
            
            # Traditional OHLC fitting - handle different column name conventions
            close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in window_df.columns else 'Close'
            if close_col not in window_df.columns:
                close_col = 'close'
            
            high_col = 'High' if 'High' in window_df.columns else 'high'
            low_col = 'Low' if 'Low' in window_df.columns else 'low'
            
            traditional_curves.update(self.fit_curves_multiple_methods(
                x, window_df[close_col].values, "close_price"
            ))
            traditional_curves.update(self.fit_curves_multiple_methods(
                x, window_df[high_col].values, "high_price"
            ))
            traditional_curves.update(self.fit_curves_multiple_methods(
                x, window_df[low_col].values, "low_price"
            ))
            
            # Curved spacetime fitting
            spacetime_curves.update(self.fit_curves_multiple_methods(
                x, window_transformed['low'].values, "spacetime_low"
            ))
            spacetime_curves.update(self.fit_curves_multiple_methods(
                x, window_transformed['range'].values, "spacetime_range"
            ))
            spacetime_curves.update(self.fit_curves_multiple_methods(
                x, window_transformed['body_ratio'].values, "spacetime_body_ratio"
            ))
            spacetime_curves.update(self.fit_curves_multiple_methods(
                x, window_transformed['upper_wick_ratio'].values, "spacetime_upper_wick"
            ))
            
            # Find extrema for both systems
            traditional_extrema = self.find_extrema_and_inflection_points(traditional_curves, x)
            spacetime_extrema = self.find_extrema_and_inflection_points(spacetime_curves, x)
            
            # Store results
            window_result = {
                'window_start': window_start,
                'window_end': window_end,
                'dates': window_df.index,
                'traditional_curves': traditional_curves,
                'spacetime_curves': spacetime_curves,
                'traditional_extrema': traditional_extrema,
                'spacetime_extrema': spacetime_extrema,
                'raw_data': window_df,
                'transformed_data': window_transformed
            }
            
            results.append(window_result)
        
        return {
            'symbol': 'UNKNOWN',  # Generic symbol for direct data analysis
            'full_data': df,
            'transformed_data': transformed,
            'window_results': results
        }
    
    def create_analysis_visualization(self, analysis_result, window_idx=0):
        """
        Create visualization showing curve fits and detected extrema
        """
        if not analysis_result or not analysis_result['window_results']:
            return None
        
        window = analysis_result['window_results'][window_idx]
        symbol = analysis_result['symbol']
        
        # Create subplot
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'Traditional OHLC - Price Data', 'Curved Spacetime - Low Coordinate',
                'Traditional - Curve Fits', 'Spacetime - Range Coordinate', 
                'Extrema + Inflection Points', 'Spacetime - Body Ratio',
                'Comparison View', 'Spacetime - Upper Wick Ratio'
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        window_df = window['raw_data']
        transformed_df = window['transformed_data']
        x = np.arange(len(window_df))
        
        # Row 1: Original data
        close_col = 'Unadjusted_Close' if 'Unadjusted_Close' in window_df.columns else 'Close'
        if close_col not in window_df.columns:
            close_col = 'close'
            
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
        
        # Row 2: Curve fits
        if 'cubic_spline' in window['traditional_curves']:
            func = window['traditional_curves']['cubic_spline']['fitted_func']
            x_fine = np.linspace(0, len(window_df)-1, len(window_df)*5)
            try:
                y_fit = func(x_fine)
                dates_fine = pd.date_range(window_df.index[0], window_df.index[-1], len(x_fine))
                fig.add_trace(go.Scatter(
                    x=dates_fine,
                    y=y_fit,
                    mode='lines',
                    name='Cubic Spline Fit',
                    line=dict(color='yellow'),
                    showlegend=False
                ), row=2, col=1)
            except:
                pass
        
        fig.add_trace(go.Scatter(
            x=window_df.index,
            y=transformed_df['range'],
            mode='lines+markers',
            name='Range',
            line=dict(color='orange'),
            showlegend=False
        ), row=2, col=2)
        
        # Row 3: Extrema and Inflection Points
        # Add extrema points if found
        for curve_name, extrema_data in window['traditional_extrema'].items():
            if 'numerical' in extrema_data:
                maxima = extrema_data['numerical']['maxima']
                minima = extrema_data['numerical']['minima']
                inflections = extrema_data['numerical']['inflection_points']
                
                if maxima:
                    max_x, max_y = zip(*maxima)
                    max_dates = [window_df.index[int(xi)] for xi in max_x if int(xi) < len(window_df)]
                    max_prices = [window_df[close_col].iloc[int(xi)] for xi in max_x if int(xi) < len(window_df)]
                    
                    fig.add_trace(go.Scatter(
                        x=max_dates,
                        y=max_prices,
                        mode='markers',
                        marker=dict(color='red', size=12, symbol='triangle-down'),
                        name='Maxima',
                        showlegend=False
                    ), row=3, col=1)
                
                if minima:
                    min_x, min_y = zip(*minima)
                    min_dates = [window_df.index[int(xi)] for xi in min_x if int(xi) < len(window_df)]
                    min_prices = [window_df[close_col].iloc[int(xi)] for xi in min_x if int(xi) < len(window_df)]
                    
                    fig.add_trace(go.Scatter(
                        x=min_dates,
                        y=min_prices,
                        mode='markers',
                        marker=dict(color='green', size=12, symbol='triangle-up'),
                        name='Minima',
                        showlegend=False
                    ), row=3, col=1)
                
                # Add inflection points
                if inflections:
                    infl_x, infl_y = zip(*inflections)
                    infl_dates = [window_df.index[int(xi)] for xi in infl_x if int(xi) < len(window_df)]
                    infl_prices = [window_df[close_col].iloc[int(xi)] for xi in infl_x if int(xi) < len(window_df)]
                    
                    fig.add_trace(go.Scatter(
                        x=infl_dates,
                        y=infl_prices,
                        mode='markers',
                        marker=dict(color='yellow', size=10, symbol='diamond'),
                        name='Inflection Points',
                        showlegend=False
                    ), row=3, col=1)
                
                break  # Just use first curve's extrema
        
        fig.add_trace(go.Scatter(
            x=window_df.index,
            y=transformed_df['body_ratio'],
            mode='lines+markers',
            name='Body Ratio',
            line=dict(color='purple'),
            showlegend=False
        ), row=3, col=2)
        
        # Row 4: Comparison
        fig.add_trace(go.Scatter(
            x=window_df.index,
            y=window_df[close_col],
            mode='lines',
            name='Close Price',
            line=dict(color='white'),
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
        
        fig.update_layout(
            title=f"{symbol} Curve Fitting Analysis - Window {window_idx+1}<br>Traditional OHLC vs Curved Spacetime Coordinates",
            height=1200,
            template='plotly_dark',
            xaxis=dict(rangeslider=dict(visible=False)),
            xaxis2=dict(rangeslider=dict(visible=False)),
            xaxis3=dict(rangeslider=dict(visible=False)),
            xaxis4=dict(rangeslider=dict(visible=False)),
            xaxis5=dict(rangeslider=dict(visible=False)),
            xaxis6=dict(rangeslider=dict(visible=False)),
            xaxis7=dict(rangeslider=dict(visible=False)),
            xaxis8=dict(rangeslider=dict(visible=False))
        )
        
        return fig

if __name__ == "__main__":
    detector = CurveFittingExtemaDetector()
    
    print("🚀 CURVE FITTING EXTREMA DETECTOR")
    print("Finding supply/demand zones using curve fitting + differentiation")
    print()
    
    # Test with TSLA data - longer period for better analysis
    analysis = detector.analyze_symbol('TSLA', '2023-06-01', '2024-03-01', window_size=30)
    
    if analysis:
        print("\n🎯 ANALYSIS COMPLETE!")
        print(f"Found {len(analysis['window_results'])} windows of analysis")
        
        # Create visualization for first window
        fig = detector.create_analysis_visualization(analysis, window_idx=0)
        if fig:
            filename = "tsla_curve_fitting_analysis.html"
            fig.write_html(filename)
            print(f"💾 Saved visualization: {filename}")
        
        print("\n✅ PRACTICAL SYSTEM READY!")
        print("This approach gives you:")
        print("  • Multiple curve fitting methods (cubic spline, univariate, polynomial)")
        print("  • Both traditional OHLC and curved spacetime coordinates")
        print("  • Analytical differentiation for exact extrema")
        print("  • Numerical methods as backup")
        print("  • Inflection point detection")
        print("  • Fast sliding window analysis")
    else:
        print("❌ Analysis failed - check data availability")