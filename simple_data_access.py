#!/usr/bin/env python3
"""
Simple Data Access - Direct database access without update checks

Use this for backtesting when you just want to use existing data
"""

import sqlite3
import pandas as pd
from market_data_database import get_default_database_path

def get_simple_data(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Get data directly from database without any update checks"""
    
    db_path = get_default_database_path()
    
    query = '''
        SELECT date, open, high, low, close, adj_close, volume
        FROM daily_data 
        WHERE symbol = ?
    '''
    params = [symbol]
    
    if start_date:
        query += ' AND DATE(date) >= ?'
        params.append(start_date)
    
    if end_date:
        query += ' AND DATE(date) <= ?'
        params.append(end_date)
    
    query += ' ORDER BY date'
    
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    
    if df.empty:
        return pd.DataFrame()
    
    # Format data
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.rename(columns={
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Unadjusted_Close',
        'adj_close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    return df

if __name__ == "__main__":
    # Test the simple data access
    import sys
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'SPY'
    start_date = sys.argv[2] if len(sys.argv) > 2 else '2024-01-01'
    end_date = sys.argv[3] if len(sys.argv) > 3 else '2024-01-31'
    
    print(f"Testing simple data access for {symbol} from {start_date} to {end_date}")
    
    data = get_simple_data(symbol, start_date, end_date)
    
    if not data.empty:
        print(f"✓ Got {len(data)} records")
        print(f"  Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"  Sample: {data.iloc[0].to_dict()}")
    else:
        print("✗ No data found")