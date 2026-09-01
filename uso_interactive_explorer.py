#!/usr/bin/env python3
"""
USO Interactive Chart Explorer
Professional trading platform-style interface with dynamic timeframes and ranges
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import requests
import os
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional

# Dark theme
pio.templates.default = "plotly_dark"

class USOInteractiveExplorer:
    """Interactive USO chart explorer with trading platform features"""
    
    def __init__(self):
        self.api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        self.data_cache = {}
        self.supported_intervals = {
            '1min': ('TIME_SERIES_INTRADAY', '1min'),
            '5min': ('TIME_SERIES_INTRADAY', '5min'),
            '15min': ('TIME_SERIES_INTRADAY', '15min'),
            '30min': ('TIME_SERIES_INTRADAY', '30min'),
            '60min': ('TIME_SERIES_INTRADAY', '60min'),
            'daily': ('TIME_SERIES_DAILY', None),
            'weekly': ('TIME_SERIES_WEEKLY', None),
            'monthly': ('TIME_SERIES_MONTHLY', None)
        }
        
        # Predefined ranges
        self.range_presets = {
            '1D': {'days': 1, 'best_interval': '15min'},
            '3D': {'days': 3, 'best_interval': '30min'},
            '1W': {'days': 7, 'best_interval': '60min'},
            '1M': {'days': 30, 'best_interval': 'daily'},
            '3M': {'days': 90, 'best_interval': 'daily'},
            '6M': {'days': 180, 'best_interval': 'daily'},
            '1Y': {'days': 365, 'best_interval': 'weekly'},
            '2Y': {'days': 730, 'best_interval': 'weekly'}
        }
    
    def fetch_uso_data(self, interval: str, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch and cache USO data for given interval"""
        
        # Check cache first
        if interval in self.data_cache and not force_refresh:
            print(f"📊 Using cached {interval} data")
            return self.data_cache[interval]
        
        if interval not in self.supported_intervals:
            raise ValueError(f"Unsupported interval: {interval}")
        
        function, api_interval = self.supported_intervals[interval]
        
        print(f"📡 Fetching {interval} USO data from Alpha Vantage...")
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': function,
            'symbol': 'USO',
            'apikey': self.api_key,
            'outputsize': 'full',
            'entitlement': 'delayed'
        }
        
        if api_interval:
            params['interval'] = api_interval
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Check for errors
        if 'Error Message' in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        
        if 'Note' in data:
            raise ValueError(f"API Rate Limited: {data['Note']}")
        
        # Parse data
        time_series_key = self._get_time_series_key(function, api_interval)
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        df_data = []
        for datetime_str, values in time_series.items():
            df_data.append({
                'datetime': datetime_str,
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['5. volume'])
            })
        
        df = pd.DataFrame(df_data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # Cache the data
        self.data_cache[interval] = df
        
        print(f"✅ Fetched {len(df):,} {interval} candles")
        print(f"📅 Range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def _get_time_series_key(self, function: str, interval: Optional[str]) -> str:
        """Get the appropriate time series key from API response"""
        if function == 'TIME_SERIES_DAILY':
            return 'Time Series (Daily)'
        elif function == 'TIME_SERIES_WEEKLY':
            return 'Weekly Time Series'
        elif function == 'TIME_SERIES_MONTHLY':
            return 'Monthly Time Series'
        else:
            return f'Time Series ({interval})'
    
    def filter_data_by_range(self, df: pd.DataFrame, days: int, interval: str) -> pd.DataFrame:
        """Filter data based on specified date range"""
        
        # For intraday data, filter to regular trading hours first
        if interval in ['1min', '5min', '15min', '30min', '60min']:
            # Convert to ET timezone for proper filtering
            if df.index.tz is None:
                df_et = df.copy()
                df_et.index = df_et.index.tz_localize('UTC').tz_convert('US/Eastern')
            else:
                df_et = df.tz_convert('US/Eastern')
            
            # Filter for regular trading hours (9:30 AM - 4:00 PM ET)
            regular_hours = df_et.between_time('09:30', '16:00')
        else:
            regular_hours = df
        
        # Get data for specified number of days
        if days == 1:
            # For 1 day, get the most recent trading day
            filtered_data = regular_hours.tail(100)  # Last 100 periods
        else:
            # For longer periods, filter by date
            end_date = regular_hours.index.max()
            start_date = end_date - timedelta(days=days)
            filtered_data = regular_hours[regular_hours.index >= start_date]
        
        return filtered_data
    
    def create_interactive_chart(self, interval: str = 'daily', range_days: int = 30, 
                               include_volume: bool = True, include_indicators: bool = False) -> go.Figure:
        """Create interactive chart with specified parameters"""
        
        print(f"\n🎯 Creating interactive chart: {interval}, {range_days} days")
        
        # Fetch data
        raw_data = self.fetch_uso_data(interval)
        
        # Filter by range
        chart_data = self.filter_data_by_range(raw_data, range_days, interval)
        
        # Convert to CST for display
        if chart_data.index.tz is None and interval in ['1min', '5min', '15min', '30min', '60min']:
            chart_data.index = chart_data.index.tz_localize('UTC').tz_convert('US/Central')
        elif chart_data.index.tz is not None:
            chart_data.index = chart_data.index.tz_convert('US/Central')
        
        # Create subplots
        if include_volume:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.25, 0.15],
                subplot_titles=[
                    f'USO Price ({interval.upper()}) - Interactive Explorer',
                    'Volume',
                    'Analytics Panel'
                ]
            )
        else:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.8, 0.2],
                subplot_titles=[
                    f'USO Price ({interval.upper()}) - Interactive Explorer',
                    'Analytics Panel'
                ]
            )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=chart_data.index,
                open=chart_data['Open'],
                high=chart_data['High'],
                low=chart_data['Low'],
                close=chart_data['Close'],
                name='USO',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444',
                hovertext=[f"{ts.strftime('%Y-%m-%d %H:%M CST')}" for ts in chart_data.index]
            ),
            row=1, col=1
        )
        
        # Add volume if requested
        if include_volume:
            colors = ['#00ff88' if close >= open else '#ff4444' 
                     for close, open in zip(chart_data['Close'], chart_data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=chart_data.index,
                    y=chart_data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7,
                    hovertemplate='<b>Volume</b><br>' +
                                 'Time: %{x}<br>' +
                                 'Volume: %{y:,.0f}<br>' +
                                 '<extra></extra>'
                ),
                row=2, col=1
            )
            analytics_row = 3
        else:
            analytics_row = 2
        
        # Add basic indicators if requested
        if include_indicators:
            # Simple moving averages
            chart_data['SMA_20'] = chart_data['Close'].rolling(20).mean()
            chart_data['SMA_50'] = chart_data['Close'].rolling(50).mean()
            
            # Add SMAs to main chart
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='#ffaa00', width=2),
                    hovertemplate='SMA 20: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='#ff6b35', width=2),
                    hovertemplate='SMA 50: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add analytics panel (price change info)
        price_changes = chart_data['Close'].pct_change() * 100
        
        fig.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=price_changes,
                mode='markers+lines',
                name='Price Change %',
                marker=dict(
                    color=price_changes,
                    colorscale=[[0, '#ff4444'], [0.5, '#ffff44'], [1, '#00ff88']],
                    size=6,
                    showscale=True,
                    colorbar=dict(title="Change %", x=1.02)
                ),
                line=dict(width=1, color='rgba(255,255,255,0.3)'),
                hovertemplate='Change: %{y:.2f}%<extra></extra>'
            ),
            row=analytics_row, col=1
        )
        
        # Update layout with professional styling
        fig.update_layout(
            title=dict(
                text=f"🎯 USO Interactive Explorer - {interval.upper()} | {range_days}D Range<br>" +
                     f"<sup>Latest: ${chart_data['Close'].iloc[-1]:.2f} | " +
                     f"Range: ${chart_data['Low'].min():.2f} - ${chart_data['High'].max():.2f} | " +
                     f"Candles: {len(chart_data):,}</sup>",
                font=dict(size=18, color='white'),
                x=0.5
            ),
            height=900,
            paper_bgcolor='rgba(15,15,15,1)',
            plot_bgcolor='rgba(25,25,25,1)',
            font=dict(color='white', size=11),
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )
        
        # Update axes
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, gridcolor='rgba(100,100,100,0.2)')
        if include_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor='rgba(100,100,100,0.2)')
        fig.update_yaxes(title_text="Change (%)", row=analytics_row, col=1, gridcolor='rgba(100,100,100,0.2)')
        fig.update_xaxes(title_text="Time (CST)", row=analytics_row, col=1, gridcolor='rgba(100,100,100,0.2)')
        
        # Add range selector buttons
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1D", step="day", stepmode="backward"),
                        dict(count=3, label="3D", step="day", stepmode="backward"),
                        dict(count=7, label="1W", step="day", stepmode="backward"),
                        dict(count=30, label="1M", step="day", stepmode="backward"),
                        dict(count=90, label="3M", step="day", stepmode="backward"),
                        dict(step="all", label="ALL")
                    ]),
                    bgcolor='rgba(50,50,50,0.8)',
                    bordercolor='rgba(100,100,100,0.5)',
                    font=dict(color='white')
                ),
                rangeslider=dict(visible=False),
                type="date"
            )
        )
        
        return fig, chart_data
    
    def generate_explorer_suite(self):
        """Generate a suite of charts for different timeframes and ranges"""
        
        print("🚀 Generating USO Interactive Explorer Suite...")
        print("=" * 60)
        
        # Chart configurations
        configurations = [
            # Short-term trading views
            {'interval': '15min', 'range_days': 3, 'name': 'short_term_15min'},
            {'interval': '30min', 'range_days': 7, 'name': 'swing_30min'},
            {'interval': '60min', 'range_days': 14, 'name': 'swing_hourly'},
            
            # Medium-term views
            {'interval': 'daily', 'range_days': 30, 'name': 'monthly_daily'},
            {'interval': 'daily', 'range_days': 90, 'name': 'quarterly_daily'},
            
            # Long-term views
            {'interval': 'weekly', 'range_days': 365, 'name': 'yearly_weekly'},
        ]
        
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'uso_chart',
                'height': 900,
                'width': 1400,
                'scale': 2
            }
        }
        
        for chart_config in configurations:
            try:
                print(f"\n📊 Creating {chart_config['name']} chart...")
                
                fig, data = self.create_interactive_chart(
                    interval=chart_config['interval'],
                    range_days=chart_config['range_days'],
                    include_volume=True,
                    include_indicators=True
                )
                
                filename = f"uso_explorer_{chart_config['name']}.html"
                fig.write_html(filename, config=config)
                
                print(f"💾 Saved: {filename}")
                print(f"📈 Data points: {len(data):,}")
                print(f"📅 Range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
                
            except Exception as e:
                print(f"❌ Error creating {chart_config['name']}: {e}")
        
        print(f"\n🎯 Explorer Suite Complete!")
        print("Features available in each chart:")
        print("  ✅ Interactive candlestick charts")
        print("  ✅ Dynamic zoom and pan")
        print("  ✅ Range selector buttons (1D, 3D, 1W, 1M, 3M, ALL)")
        print("  ✅ Volume analysis")
        print("  ✅ Price change analytics")
        print("  ✅ Moving averages (SMA 20, 50)")
        print("  ✅ Drawing tools (lines, shapes)")
        print("  ✅ Professional dark theme")
        print("  ✅ Export to PNG capability")
        print("  ✅ Hover tooltips with full OHLC data")

def main():
    """Generate the interactive explorer suite"""
    explorer = USOInteractiveExplorer()
    explorer.generate_explorer_suite()

if __name__ == "__main__":
    main()