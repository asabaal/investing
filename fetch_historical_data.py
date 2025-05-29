#!/usr/bin/env python3
"""
Fetch comprehensive historical data for the symphony system
"""

import os
import sys
from market_data_database import MarketDataDatabase

def fetch_historical_data():
    """Fetch full historical data for key symbols"""
    
    # Initialize database
    db = MarketDataDatabase()
    
    # Key symbols we need for symphonies
    symbols = [
        # Core ETFs
        'SPY', 'QQQ', 'IWM', 'TLT', 'VTI', 'GLD', 'SHY',
        
        # Tech Giants
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
        
        # Sector ETFs
        'XLK', 'XLF', 'XLE', 'XLV', 'XLI',
        
        # Volatility & Leverage
        'VIX', 'UVXY', 'TQQQ', 'TECL', 'PSQ'
    ]
    
    print(f"📊 Fetching full historical data for {len(symbols)} symbols...")
    print("⏱️ This will take some time due to API rate limiting...")
    
    # Force full update for each symbol
    for i, symbol in enumerate(symbols):
        try:
            print(f"\n🔄 [{i+1}/{len(symbols)}] Fetching {symbol} full history...")
            success = db.update_daily_data(symbol, force_full_update=True)
            
            if success:
                # Check how much data we got
                data = db.get_data(symbol)
                if not data.empty:
                    print(f"✅ {symbol}: {len(data)} records from {data.index.min().date()} to {data.index.max().date()}")
                else:
                    print(f"⚠️ {symbol}: No data retrieved")
            else:
                print(f"❌ {symbol}: Failed to fetch data")
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
            
        # Rate limiting message
        if i < len(symbols) - 1:
            print("⏱️ Rate limiting for 12 seconds...")
    
    # Final stats
    print("\n" + "="*60)
    print("📊 FINAL DATABASE STATISTICS")
    print("="*60)
    
    stats = db.get_database_stats()
    print(f"Symbols: {stats['daily_data']['symbols']}")
    print(f"Total Records: {stats['daily_data']['total_records']}")
    print(f"Date Range: {stats['daily_data']['earliest_date']} to {stats['daily_data']['latest_date']}")
    print(f"Database Size: {stats['database_size_mb']:.1f} MB")
    print(f"Symbols: {', '.join(stats['symbols_in_database'])}")

if __name__ == "__main__":
    fetch_historical_data()