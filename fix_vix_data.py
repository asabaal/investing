#!/usr/bin/env python3
"""
Fix VIX data collection - VIX needs special handling as it's an index
"""

import os
import requests
import pandas as pd
from market_data_database import MarketDataDatabase

def fetch_vix_data():
    """Fetch VIX data using the regular time series function"""
    
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return
    
    # VIX needs TIME_SERIES_DAILY, not TIME_SERIES_DAILY_ADJUSTED
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',  # Note: not DAILY_ADJUSTED
        'symbol': 'VIX',
        'outputsize': 'full',
        'apikey': api_key
    }
    
    print("🔄 Fetching VIX data...")
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    if 'Error Message' in data:
        print(f"❌ API Error: {data['Error Message']}")
        return
    
    if 'Note' in data:
        print(f"❌ Rate limited: {data['Note']}")
        return
    
    # Parse time series data
    time_series_key = 'Time Series (Daily)'
    if time_series_key not in data:
        print(f"❌ Time series data not found. Available keys: {list(data.keys())}")
        return
    
    time_series = data[time_series_key]
    
    # Convert to DataFrame (VIX doesn't have adjusted close)
    df_data = []
    for date_str, values in time_series.items():
        df_data.append({
            'date': date_str,
            'open': float(values['1. open']),
            'high': float(values['2. high']),
            'low': float(values['3. low']),
            'close': float(values['4. close']),
            'adj_close': float(values['4. close']),  # Use close as adj_close for VIX
            'volume': int(values['5. volume'])
        })
    
    df = pd.DataFrame(df_data)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    
    # Store in database manually
    db = MarketDataDatabase()
    db._store_daily_data('VIX', df)
    
    print(f"✅ VIX data stored: {len(df)} records from {df['date'].min().date()} to {df['date'].max().date()}")

if __name__ == "__main__":
    fetch_vix_data()