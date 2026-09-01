#!/usr/bin/env python3
"""
USO Hourly and 4-Hour Charts
Quick reference charts showing last 5 candles for each timeframe
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

def fetch_uso_hourly_data():
    """Fetch hourly USO data from Alpha Vantage API"""
    
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment")
    
    print("📡 Fetching hourly USO data from Alpha Vantage API...")
    
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': 'USO',
        'interval': '60min',
        'apikey': api_key,
        'outputsize': 'full',
        'entitlement': 'delayed'
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
    time_series = data['Time Series (60min)']
    
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
    
    print(f"✅ Fetched {len(df):,} hourly candles")
    print(f"📅 Data range: {df.index.min()} to {df.index.max()}")
    
    return df

def create_4hour_data(hourly_df):
    """Aggregate hourly data into 4-hour candles"""
    
    # Convert to ET timezone for proper market hours aggregation
    hourly_et = hourly_df.copy()
    hourly_et.index = hourly_et.index.tz_localize('UTC').tz_convert('US/Eastern')
    
    # Filter for regular trading hours only (9:30 AM - 4:00 PM ET)
    regular_hours = hourly_et.between_time('09:30', '16:00')
    
    # Create 4-hour periods starting at 9:30 AM ET
    # 9:30-13:30 (4 hours), 13:30-16:00 (2.5 hours - partial period)
    # We'll group by trading day and create morning/afternoon sessions
    
    four_hour_data = []
    
    # Group by trading day
    for date, day_data in regular_hours.groupby(regular_hours.index.date):
        day_data_sorted = day_data.sort_index()
        
        if len(day_data_sorted) >= 4:  # Need at least 4 hours of data
            # Morning session: 9:30-13:30 (first 4 hours)
            morning_data = day_data_sorted.iloc[:4]
            morning_candle = {
                'datetime': pd.Timestamp.combine(date, pd.Timestamp('13:30').time()).tz_localize('US/Eastern'),
                'Open': morning_data['Open'].iloc[0],
                'High': morning_data['High'].max(),
                'Low': morning_data['Low'].min(),
                'Close': morning_data['Close'].iloc[-1],
                'Volume': morning_data['Volume'].sum()
            }
            four_hour_data.append(morning_candle)
            
            # Afternoon session: 13:30-16:00 (remaining hours)
            if len(day_data_sorted) > 4:
                afternoon_data = day_data_sorted.iloc[4:]
                afternoon_candle = {
                    'datetime': pd.Timestamp.combine(date, pd.Timestamp('16:00').time()).tz_localize('US/Eastern'),
                    'Open': afternoon_data['Open'].iloc[0],
                    'High': afternoon_data['High'].max(),
                    'Low': afternoon_data['Low'].min(),
                    'Close': afternoon_data['Close'].iloc[-1],
                    'Volume': afternoon_data['Volume'].sum()
                }
                four_hour_data.append(afternoon_candle)
    
    # Convert to DataFrame
    four_hour_df = pd.DataFrame(four_hour_data)
    four_hour_df.set_index('datetime', inplace=True)
    four_hour_df.sort_index(inplace=True)
    
    print(f"✅ Created {len(four_hour_df):,} 4-hour candles")
    
    return four_hour_df

def create_timeframe_chart(data, timeframe, num_candles=5):
    """Create chart for specified timeframe"""
    
    print(f"\n📊 Creating {timeframe} USO chart...")
    
    # Convert to CST for display
    data_cst = data.copy()
    if data_cst.index.tz is None:
        data_cst.index = data_cst.index.tz_localize('US/Eastern')
    data_cst.index = data_cst.index.tz_convert('US/Central')
    
    # Get last N candles
    chart_data = data_cst.tail(num_candles)
    
    print(f"📅 Chart data range: {chart_data.index.min()} to {chart_data.index.max()}")
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=(f'USO Price ({timeframe} candles)', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Format time labels
    if timeframe == '4-hour':
        time_format = '%m/%d %H:%M CST'
    else:
        time_format = '%m/%d %H:%M CST'
    
    x_labels = [f"{ts.strftime(time_format)}" for ts in chart_data.index]
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
            hovertext=[f"{ts.strftime('%m/%d/%Y %H:%M CST')}" for ts in chart_data.index]
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
            customdata=[ts.strftime('%m/%d/%Y %H:%M CST') for ts in chart_data.index]
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"📊 USO - Last {num_candles} {timeframe.upper()} Candles (CST)",
        height=600,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    # Use all labels since we only have 5 candles
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(
        title_text="Time (CST)",
        tickvals=x_range,
        ticktext=x_labels,
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
            'filename': f'uso_{timeframe}_chart',
            'height': 600,
            'width': 1000,
            'scale': 1
        }
    }
    
    # Save chart
    filename = f"uso_{timeframe.replace('-', '')}_last5.html"
    fig.write_html(filename, config=config)
    print(f"💾 Chart saved as: {filename}")
    
    # Print candle details
    print(f"\n📋 Last {num_candles} {timeframe} candles:")
    for timestamp, row in chart_data.iterrows():
        print(f"  {timestamp.strftime('%m/%d %H:%M CST')} - O:${row['Open']:.2f} H:${row['High']:.2f} L:${row['Low']:.2f} C:${row['Close']:.2f} V:{row['Volume']:,}")
    
    return chart_data

def main():
    print("📊 Creating USO Hourly and 4-Hour Reference Charts")
    print("=" * 60)
    
    try:
        # Fetch hourly data
        hourly_data = fetch_uso_hourly_data()
        
        # Filter for regular trading hours and convert to CST
        hourly_et = hourly_data.copy()
        hourly_et.index = hourly_et.index.tz_localize('UTC').tz_convert('US/Eastern')
        regular_hours_hourly = hourly_et.between_time('09:30', '16:00')
        
        # Create hourly chart
        print(f"\n{'='*30} HOURLY CHART {'='*30}")
        hourly_chart_data = create_timeframe_chart(regular_hours_hourly, 'hourly', 5)
        
        # Create 4-hour data and chart
        print(f"\n{'='*30} 4-HOUR CHART {'='*30}")
        four_hour_data = create_4hour_data(hourly_data)
        four_hour_chart_data = create_timeframe_chart(four_hour_data, '4-hour', 5)
        
        # Summary
        print(f"\n{'='*25} SUMMARY {'='*25}")
        print(f"📊 Charts created:")
        print(f"  • uso_hourly_last5.html - Last 5 hourly candles")
        print(f"  • uso_4hour_last5.html - Last 5 4-hour candles")
        print(f"📈 Latest hourly close: ${hourly_chart_data['Close'].iloc[-1]:.2f}")
        print(f"📈 Latest 4-hour close: ${four_hour_chart_data['Close'].iloc[-1]:.2f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()