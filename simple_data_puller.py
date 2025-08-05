#!/usr/bin/env python3
"""
SIMPLE DATA PULLER - Just get the damn data!

No complex logic, no database updates, no fancy fallbacks.
Just pull data from the database and return it.
"""

import pandas as pd
import sqlite3
from market_data_database import get_default_database_path, MarketDataDatabase
import logging

logger = logging.getLogger(__name__)

def get_daily_data(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Get daily data. Period."""
    
    db_path = get_default_database_path()
    
    with sqlite3.connect(db_path) as conn:
        query = 'SELECT date, open, high, low, close, adj_close, volume FROM daily_data WHERE symbol = ?'
        params = [symbol]
        
        if start_date:
            query += ' AND date >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND date <= ?'
            params.append(end_date)
            
        query += ' ORDER BY date'
        
        df = pd.read_sql_query(query, conn, params=params)
    
    if df.empty:
        print(f"No daily data for {symbol}")
        return pd.DataFrame()
    
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Unadjusted_Close', 'adj_close': 'Close', 'volume': 'Volume'
    }, inplace=True)
    
    print(f"✅ Got {len(df)} daily records for {symbol}")
    return df

def get_intraday_data(symbol: str, interval: str = '60min', start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Get intraday data. Period."""
    
    db_path = get_default_database_path()
    
    with sqlite3.connect(db_path) as conn:
        query = 'SELECT datetime, open, high, low, close, volume FROM intraday_data WHERE symbol = ? AND interval = ?'
        params = [symbol, interval]
        
        if start_date:
            query += ' AND date(datetime) >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND date(datetime) <= ?'
            params.append(end_date)
            
        query += ' ORDER BY datetime'
        
        df = pd.read_sql_query(query, conn, params=params)
    
    if df.empty:
        print(f"No {interval} data for {symbol}")
        return pd.DataFrame()
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    }, inplace=True)
    
    print(f"✅ Got {len(df)} {interval} records for {symbol}")
    return df

def update_data(symbol: str, interval: str = 'daily'):
    """Force update data from API."""
    
    db = MarketDataDatabase()
    
    if interval == 'daily':
        success = db.update_daily_data(symbol, force_full_update=True)
    else:
        success = db.update_intraday_data(symbol, interval)
    
    print(f"Update {symbol} {interval}: {'✅ Success' if success else '❌ Failed'}")
    return success

class SimpleSession:
    """Dead simple session - just holds data"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.daily = pd.DataFrame()
        self.intraday = {}
        print(f"📊 Simple session for {symbol}")
    
    def get_daily(self, start_date: str = None, end_date: str = None):
        """Get daily data"""
        self.daily = get_daily_data(self.symbol, start_date, end_date)
        return self.daily
    
    def get_intraday(self, interval: str = '60min', start_date: str = None, end_date: str = None):
        """Get intraday data"""
        data = get_intraday_data(self.symbol, interval, start_date, end_date)
        self.intraday[interval] = data
        return data
    
    def update_and_get_daily(self, start_date: str = None, end_date: str = None):
        """Update then get daily data"""
        update_data(self.symbol, 'daily')
        return self.get_daily(start_date, end_date)
    
    def update_and_get_intraday(self, interval: str = '60min', start_date: str = None, end_date: str = None):
        """Update then get intraday data"""
        update_data(self.symbol, interval)
        return self.get_intraday(interval, start_date, end_date)
    
    def add_intraday(self, interval: str = '60min'):
        """Add/update intraday data for this interval"""
        print(f"🔄 Adding {interval} data for {self.symbol}...")
        success = update_data(self.symbol, interval)
        if success:
            data = self.get_intraday(interval)
            print(f"✅ Added {len(data):,} {interval} records" if not data.empty else f"❌ No {interval} data available")
            return data
        else:
            print(f"❌ Failed to add {interval} data")
            return pd.DataFrame()
    
    def add_daily(self):
        """Add/update daily data"""
        print(f"🔄 Adding daily data for {self.symbol}...")
        success = update_data(self.symbol, 'daily')
        if success:
            data = self.get_daily()
            print(f"✅ Added {len(data):,} daily records" if not data.empty else f"❌ No daily data available")
            return data
        else:
            print(f"❌ Failed to add daily data")
            return pd.DataFrame()
    
    def status(self):
        """Show what we have"""
        print(f"\n📊 {self.symbol} Data Status:")
        print(f"Daily: {len(self.daily)} records" if not self.daily.empty else "Daily: No data")
        for interval, data in self.intraday.items():
            print(f"{interval}: {len(data)} records" if not data.empty else f"{interval}: No data")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simple_data_puller.py SYMBOL")
        print("Example: python simple_data_puller.py SPY")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    
    # Simple test
    session = SimpleSession(symbol)
    
    # Get daily data
    daily = session.get_daily()
    
    # Get whatever intraday data exists
    hourly = session.get_intraday('60min')
    minute = session.get_intraday('15min')
    
    session.status()
    
    if not daily.empty:
        print(f"\nDaily data sample:")
        print(daily.tail())
    
    if not hourly.empty:
        print(f"\n60min data sample:")
        print(hourly.tail())