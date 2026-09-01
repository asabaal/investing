#!/usr/bin/env python3
"""
USO 15-minute chart - Last 100 candles from regular trading hours
Direct API pull for maximum reliability
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

def fetch_15min_uso_direct():
    """Fetch 15-minute USO data directly from Alpha Vantage API"""
    
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment")
    
    print("📡 Fetching 15-minute USO data directly from Alpha Vantage API...")
    
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': 'USO',
        'interval': '15min',
        'apikey': api_key,
        'outputsize': 'full',
        'entitlement': 'delayed'  # Get today's data with 15-min delay
    }
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    # Check for errors
    if 'Error Message' in data:
        raise ValueError(f"API Error: {data['Error Message']}")
    
    if 'Note' in data:
        raise ValueError(f"API Rate Limited: {data['Note']}")
    
    # Parse time series data
    time_series = data['Time Series (15min)']
    
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
    
    print(f"✅ Fetched {len(df):,} 15-minute candles")
    print(f"📅 Data range: {df.index.min()} to {df.index.max()}")
    
    return df

def create_15min_uso_chart():
    """Create 15-minute USO chart with last 100 regular trading hours candles"""
    
    print("📊 Creating 15-minute USO chart...")
    
    # Fetch data directly from API
    uso_data = fetch_15min_uso_direct()
    
    # Convert to ET timezone for proper filtering
    uso_data_et = uso_data.copy()
    uso_data_et.index = uso_data_et.index.tz_localize('UTC').tz_convert('US/Eastern')
    
    # Filter for regular trading hours (9:30 AM - 4:00 PM ET)
    regular_hours = uso_data_et.between_time('09:30', '16:00')
    
    # Convert to CST for display purposes
    regular_hours_cst = regular_hours.copy()
    regular_hours_cst.index = regular_hours_cst.index.tz_convert('US/Central')
    
    print(f"📊 Total 15-min candles: {len(uso_data):,}")
    print(f"📈 Regular hours candles: {len(regular_hours):,}")
    
    # Get last 100 candles from regular trading hours (using CST data)
    last_100 = regular_hours_cst.tail(100)
    print(f"📅 Last 100 regular trading 15-min candles: {last_100.index.min()} to {last_100.index.max()}")
    
    # Create subplot with candlestick and volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price (15-minute candles)', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Create continuous x-axis using range indices instead of datetime (CST times)
    x_labels = [f"{ts.strftime('%m/%d %H:%M CST')}" for ts in last_100.index]
    x_range = list(range(len(last_100)))
    
    # Add candlestick chart with continuous x-axis and custom hover info
    fig.add_trace(
        go.Candlestick(
            x=x_range,
            open=last_100['Open'],
            high=last_100['High'],
            low=last_100['Low'],
            close=last_100['Close'],
            name='USO',
            hovertext=[f"{ts.strftime('%m/%d/%Y %H:%M CST')}" for ts in last_100.index]
        ),
        row=1, col=1
    )
    
    # Add volume bars with continuous x-axis and custom hover info
    fig.add_trace(
        go.Bar(
            x=x_range,
            y=last_100['Volume'],
            name='Volume',
            marker_color='rgba(158,202,225,0.6)',
            marker_line_color='rgba(8,48,107,1.0)',
            marker_line_width=1,
            hovertemplate='<b>Volume</b><br>' +
                         'Time: %{customdata}<br>' +
                         'Volume: %{y:,.0f}<br>' +
                         '<extra></extra>',
            customdata=[ts.strftime('%m/%d/%Y %H:%M CST') for ts in last_100.index]
        ),
        row=2, col=1
    )
    
    # Update layout with custom x-axis labels and hover settings
    fig.update_layout(
        title="📊 USO - Last 100 Regular Trading Hours 15-Minute Candles with Volume (CST)",
        height=800,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'  # Show unified hover info across both subplots
    )
    
    # Set custom x-axis labels - show more frequent labels toward the end
    # Show every 5th label for better time resolution
    tick_indices = list(range(0, len(x_labels), 5))  # Every 5th label instead of 10th
    tick_labels = [x_labels[i] for i in tick_indices]
    
    # Update y-axis labels and x-axis formatting
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    # Update shared x-axis (only need to update bottom subplot since shared_xaxes=True)
    fig.update_xaxes(
        title_text="Time (CST)", 
        tickvals=tick_indices,
        ticktext=tick_labels,
        tickangle=45,
        row=2, col=1
    )
    
    # Configure the chart with interactive settings
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'uso_15min_chart',
            'height': 800,
            'width': 1200,
            'scale': 1
        }
    }
    
    # Save as HTML file with interactive config
    fig.write_html("uso_15min_100_candles.html", config=config)
    print("💾 Chart saved as: uso_15min_100_candles.html")
    
    # Print summary
    print(f"\n📈 Price Summary (Last 100 15-min candles):")
    print(f"Latest: ${last_100['Close'].iloc[-1]:.2f}")
    print(f"High: ${last_100['High'].max():.2f}")
    print(f"Low: ${last_100['Low'].min():.2f}")
    print(f"Range: ${last_100['High'].max() - last_100['Low'].min():.2f}")
    
    print(f"\n📊 Volume Summary (Last 100 15-min candles):")
    print(f"Total Volume: {last_100['Volume'].sum():,}")
    print(f"Average Volume: {last_100['Volume'].mean():.0f}")
    print(f"Max Volume: {last_100['Volume'].max():,}")
    print(f"Min Volume: {last_100['Volume'].min():,}")
    
    # Show the latest few candles for verification
    print(f"\n📋 Latest 5 candles for verification:")
    latest_5 = last_100.tail(5)
    for timestamp, row in latest_5.iterrows():
        print(f"  {timestamp} - O:${row['Open']:.2f} H:${row['High']:.2f} L:${row['Low']:.2f} C:${row['Close']:.2f} V:{row['Volume']:,}")

if __name__ == "__main__":
    create_15min_uso_chart()