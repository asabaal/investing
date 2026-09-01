#!/usr/bin/env python3
"""
USO Multi-Timescale QA Charts
Generate charts for different timescales to identify data discrepancies
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import requests
import os
from datetime import datetime

# Dark theme
pio.templates.default = "plotly_dark"

def fetch_uso_data(interval='15min'):
    """Fetch USO data for specified interval from Alpha Vantage API"""
    
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment")
    
    print(f"📡 Fetching {interval} USO data from Alpha Vantage API...")
    
    # Map intervals to API function calls
    interval_mapping = {
        '15min': ('TIME_SERIES_INTRADAY', '15min'),
        '30min': ('TIME_SERIES_INTRADAY', '30min'), 
        '60min': ('TIME_SERIES_INTRADAY', '60min'),
        'daily': ('TIME_SERIES_DAILY', None)
    }
    
    function, api_interval = interval_mapping[interval]
    
    url = "https://www.alphavantage.co/query"
    params = {
        'function': function,
        'symbol': 'USO',
        'apikey': api_key,
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
    
    # Parse time series data
    if interval == 'daily':
        time_series_key = 'Time Series (Daily)'
    else:
        time_series_key = f'Time Series ({api_interval})'
    
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
    
    print(f"✅ Fetched {len(df):,} {interval} candles")
    print(f"📅 Data range: {df.index.min()} to {df.index.max()}")
    
    return df

def create_timescale_chart(interval='15min', num_candles=100):
    """Create chart for specified timescale"""
    
    print(f"\n📊 Creating {interval} USO chart...")
    
    # Fetch data
    uso_data = fetch_uso_data(interval)
    
    if interval != 'daily':
        # Convert to ET timezone for filtering (only for intraday)
        uso_data_et = uso_data.copy()
        uso_data_et.index = uso_data_et.index.tz_localize('UTC').tz_convert('US/Eastern')
        
        # Filter for regular trading hours (9:30 AM - 4:00 PM ET)
        regular_hours = uso_data_et.between_time('09:30', '16:00')
        
        # Convert to CST for display
        regular_hours_cst = regular_hours.copy()
        regular_hours_cst.index = regular_hours_cst.index.tz_convert('US/Central')
        
        print(f"📊 Total {interval} candles: {len(uso_data):,}")
        print(f"📈 Regular hours candles: {len(regular_hours):,}")
        
        # Get last N candles
        chart_data = regular_hours_cst.tail(num_candles)
    else:
        # For daily data, no need to filter trading hours
        chart_data = uso_data.tail(num_candles)
        print(f"📊 Total daily candles: {len(uso_data):,}")
    
    print(f"📅 Chart data range: {chart_data.index.min()} to {chart_data.index.max()}")
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'USO Price ({interval} candles)', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Format time labels based on interval
    if interval == 'daily':
        x_labels = [f"{ts.strftime('%m/%d/%Y')}" for ts in chart_data.index]
        time_format = '%m/%d/%Y'
    else:
        x_labels = [f"{ts.strftime('%m/%d %H:%M CST')}" for ts in chart_data.index]
        time_format = '%m/%d/%Y %H:%M CST'
    
    x_range = list(range(len(chart_data)))
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=x_range,
            open=chart_data['Open'],
            high=chart_data['High'],
            low=chart_data['Low'],
            close=chart_data['Close'],
            name='USO',
            hovertext=[f"{ts.strftime(time_format)}" for ts in chart_data.index]
        ),
        row=1, col=1
    )
    
    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=x_range,
            y=chart_data['Volume'],
            name='Volume',
            marker_color='rgba(158,202,225,0.6)',
            marker_line_color='rgba(8,48,107,1.0)',
            marker_line_width=1,
            hovertemplate='<b>Volume</b><br>' +
                         'Time: %{customdata}<br>' +
                         'Volume: %{y:,.0f}<br>' +
                         '<extra></extra>',
            customdata=[ts.strftime(time_format) for ts in chart_data.index]
        ),
        row=2, col=1
    )
    
    # Update layout
    timezone_label = "" if interval == 'daily' else " (CST)"
    fig.update_layout(
        title=f"📊 USO - Last {num_candles} {interval.upper()} Candles with Volume{timezone_label}",
        height=800,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    # Set tick labels (show every 5th for intraday, every 10th for daily)
    tick_step = 10 if interval == 'daily' else 5
    tick_indices = list(range(0, len(x_labels), tick_step))
    tick_labels = [x_labels[i] for i in tick_indices]
    
    # Update axes
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(
        title_text=f"Time{timezone_label}",
        tickvals=tick_indices,
        ticktext=tick_labels,
        tickangle=45,
        row=2, col=1
    )
    
    # Configure chart
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'uso_{interval}_chart',
            'height': 800,
            'width': 1200,
            'scale': 1
        }
    }
    
    # Save chart
    filename = f"uso_{interval}_qa_chart.html"
    fig.write_html(filename, config=config)
    print(f"💾 Chart saved as: {filename}")
    
    # Print data summary
    print(f"\n📈 Price Summary (Last {num_candles} {interval} candles):")
    print(f"Latest: ${chart_data['Close'].iloc[-1]:.2f}")
    print(f"High: ${chart_data['High'].max():.2f}")
    print(f"Low: ${chart_data['Low'].min():.2f}")
    print(f"Range: ${chart_data['High'].max() - chart_data['Low'].min():.2f}")
    
    print(f"\n📊 Volume Summary (Last {num_candles} {interval} candles):")
    print(f"Total Volume: {chart_data['Volume'].sum():,}")
    print(f"Average Volume: {chart_data['Volume'].mean():.0f}")
    print(f"Max Volume: {chart_data['Volume'].max():,}")
    
    # Show latest candles for QA
    print(f"\n📋 Latest 3 candles for QA:")
    latest_3 = chart_data.tail(3)
    for timestamp, row in latest_3.iterrows():
        if interval == 'daily':
            time_str = timestamp.strftime('%Y-%m-%d')
        else:
            time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')
        print(f"  {time_str} - O:${row['Open']:.2f} H:${row['High']:.2f} L:${row['Low']:.2f} C:${row['Close']:.2f} V:{row['Volume']:,}")
    
    return chart_data

def run_qa_analysis():
    """Run QA analysis across multiple timescales"""
    
    print("🔍 Starting USO Multi-Timescale QA Analysis")
    print("=" * 60)
    
    timescales = [
        ('30min', 50),  # Last 50 30-minute candles
        ('60min', 30),  # Last 30 hourly candles  
        ('daily', 20)   # Last 20 daily candles
    ]
    
    results = {}
    
    for interval, num_candles in timescales:
        try:
            print(f"\n{'='*20} {interval.upper()} DATA {'='*20}")
            chart_data = create_timescale_chart(interval, num_candles)
            results[interval] = chart_data
            
        except Exception as e:
            print(f"❌ Error processing {interval}: {e}")
            continue
    
    # Summary comparison
    print(f"\n{'='*20} QA COMPARISON SUMMARY {'='*20}")
    
    if results:
        print("Latest Close Prices Comparison:")
        for interval, data in results.items():
            latest_close = data['Close'].iloc[-1]
            latest_time = data.index[-1]
            print(f"  {interval:>6}: ${latest_close:.2f} at {latest_time}")
        
        print("\nVolume Comparison (Latest Period):")
        for interval, data in results.items():
            latest_volume = data['Volume'].iloc[-1]
            print(f"  {interval:>6}: {latest_volume:,}")

if __name__ == "__main__":
    run_qa_analysis()