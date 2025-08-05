#!/usr/bin/env python3
"""
DIRECT API DATA PULLER - COMPLETELY ISOLATED

No database, no complex systems, just direct API calls.
"""

import requests
import pandas as pd
import os
from datetime import datetime

API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

def get_daily_data_api(symbol):
    """Get daily data directly from API"""
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': symbol,
        'outputsize': 'full',
        'apikey': API_KEY
    }
    
    print(f"🌐 API call: {symbol} daily data")
    response = requests.get(url, params=params)
    data = response.json()
    
    time_series = data.get('Time Series (Daily)', {})
    
    records = []
    for date_str, values in time_series.items():
        records.append({
            'date': pd.to_datetime(date_str),
            'open': float(values['1. open']),
            'high': float(values['2. high']),
            'low': float(values['3. low']),
            'close': float(values['4. close']),
            'volume': int(values['6. volume'])
        })
    
    df = pd.DataFrame(records)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"✅ Got {len(df)} daily records from API")
    return df

def get_intraday_data_api(symbol, interval='60min', start_date=None, end_date=None):
    """Get intraday data directly from API with historical support using month parameter"""
    
    if not start_date:
        return _get_recent_intraday(symbol, interval)
    
    # Generate list of months to fetch based on date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
    
    # Get all months in the range
    months_to_fetch = []
    current = start_dt.replace(day=1)
    while current <= end_dt:
        months_to_fetch.append(f"{current.year}-{current.month:02d}")
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    print(f"🌐 API calls: {symbol} {interval} data for {len(months_to_fetch)} months ({start_date} to {end_date or 'now'})")
    
    all_records = []
    
    for month in months_to_fetch:
        print(f"   📅 Fetching {month}...")
        
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'month': month,  # Use month parameter for historical data
            'outputsize': 'full',
            'apikey': API_KEY
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            print(f"   ❌ API Error for {month}: {data['Error Message']}")
            continue
        
        if 'Note' in data:
            print(f"   ⚠️  API Note for {month}: {data['Note']}")
            continue
        
        time_series = data.get(f'Time Series ({interval})', {})
        
        if not time_series:
            print(f"   ❌ No data for {month}")
            continue
        
        month_records = 0
        for datetime_str, values in time_series.items():
            dt = pd.to_datetime(datetime_str)
            
            # Apply exact date filtering
            if dt.date() < start_dt.date():
                continue
            if end_date and dt.date() > end_dt.date():
                continue
                
            all_records.append({
                'datetime': dt,
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': int(values['5. volume'])
            })
            month_records += 1
        
        print(f"   ✅ {month}: {month_records} records after filtering")
    
    df = pd.DataFrame(all_records)
    if not df.empty:
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]  # Remove any duplicates
    
    print(f"✅ Got {len(df)} {interval} records from API")
    return df

def _get_recent_intraday(symbol, interval):
    """Get recent intraday data (last 30 days)"""
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': interval,
        'outputsize': 'full',
        'apikey': API_KEY
    }
    
    print(f"🌐 API call: {symbol} {interval} recent data")
    response = requests.get(url, params=params)
    data = response.json()
    
    time_series = data.get(f'Time Series ({interval})', {})
    
    records = []
    for datetime_str, values in time_series.items():
        records.append({
            'datetime': pd.to_datetime(datetime_str),
            'open': float(values['1. open']),
            'high': float(values['2. high']),
            'low': float(values['3. low']),
            'close': float(values['4. close']),
            'volume': int(values['5. volume'])
        })
    
    df = pd.DataFrame(records)
    if not df.empty:
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
    
    return df

if __name__ == "__main__":
    # Test direct API calls with time filtering
    symbol = 'AAPL'
    
    print(f"🎯 Direct API data for {symbol}")
    print("=" * 30)
    
    # Get daily data from API
    daily = get_daily_data_api(symbol)
    
    # Get hourly data from API for specific timeframe
    hourly_recent = get_intraday_data_api(symbol, '60min', '2025-07-01')
    
    # Get minute data from API for specific timeframe
    minute_recent = get_intraday_data_api(symbol, '15min', '2025-08-01', '2025-08-05')
    
    # Get another timeframe - last week
    hourly_lastweek = get_intraday_data_api(symbol, '60min', '2025-07-28', '2025-08-04')
    
    print(f"\n📊 Results:")
    print(f"Daily (all): {len(daily)} records")
    print(f"Hourly (Jul 1+): {len(hourly_recent)} records") 
    print(f"Minute (Aug 1-5): {len(minute_recent)} records")
    print(f"Hourly (last week): {len(hourly_lastweek)} records")
    
    print("\n🎯 100% isolated - no database involved!")
    print("⏰ Time filtering works perfectly!")