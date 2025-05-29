#!/usr/bin/env python3
"""
Debug database issues
"""

import sqlite3
import pandas as pd
from market_data_database import MarketDataDatabase

def debug_database():
    """Debug what's in the database"""
    
    db = MarketDataDatabase()
    
    print("🔍 Debugging Database...")
    
    # Check raw database contents
    with sqlite3.connect(db.db_path) as conn:
        # Check what symbols we have
        symbols = conn.execute("SELECT DISTINCT symbol FROM daily_data").fetchall()
        print(f"📊 Symbols in daily_data: {[s[0] for s in symbols]}")
        
        # Check SPY specifically
        spy_count = conn.execute("SELECT COUNT(*) FROM daily_data WHERE symbol = 'SPY'").fetchone()[0]
        print(f"📈 SPY records in daily_data: {spy_count}")
        
        if spy_count > 0:
            # Get sample SPY data
            spy_sample = conn.execute("""
                SELECT date, close FROM daily_data 
                WHERE symbol = 'SPY' 
                ORDER BY date 
                LIMIT 5
            """).fetchall()
            print(f"📅 SPY sample data: {spy_sample}")
            
            # Check date range
            spy_range = conn.execute("""
                SELECT MIN(date), MAX(date) FROM daily_data 
                WHERE symbol = 'SPY'
            """).fetchone()
            print(f"📅 SPY date range: {spy_range}")
        
        # Check metadata
        metadata = conn.execute("SELECT * FROM data_metadata WHERE symbol = 'SPY'").fetchall()
        print(f"🗃️ SPY metadata: {metadata}")
    
    # Test direct database access
    print("\n🧪 Testing direct database get_data...")
    spy_data = db.get_data('SPY')
    print(f"📊 Direct get_data result: {len(spy_data)} records")
    if not spy_data.empty:
        print(f"📅 Date range: {spy_data.index.min()} to {spy_data.index.max()}")
        min_price = spy_data['Close'].min()
        max_price = spy_data['Close'].max()
        print(f"💰 Price range: ${min_price:.2f} to ${max_price:.2f}")
    
    # Test with date filter that should return data
    print("\n🧪 Testing with date filter...")
    spy_data_filtered = db.get_data('SPY', '2024-01-01', '2024-12-31')
    print(f"📊 Filtered data (2024): {len(spy_data_filtered)} records")
    
    return spy_data

if __name__ == "__main__":
    debug_database()