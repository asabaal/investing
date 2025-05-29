#!/usr/bin/env python3
"""
Quick fix to get more historical data
"""

from market_data_database import MarketDataDatabase
import sqlite3

def fix_database_data():
    """Force fetch of full historical data"""
    
    db = MarketDataDatabase()
    
    # Key symbols for testing
    symbols = ['SPY', 'QQQ', 'TLT', 'AAPL', 'MSFT', 'GOOGL']
    
    print("🔧 Forcing full historical data fetch...")
    
    for symbol in symbols:
        try:
            print(f"📊 Fetching full history for {symbol}...")
            success = db.update_daily_data(symbol, force_full_update=True)
            
            if success:
                data = db.get_data(symbol)
                if not data.empty:
                    print(f"✅ {symbol}: {len(data)} records from {data.index.min().date()} to {data.index.max().date()}")
                    
                    # Check if we have enough historical data
                    if len(data) >= 500:
                        print(f"✅ {symbol}: Sufficient data for strategies")
                    else:
                        print(f"⚠️ {symbol}: Only {len(data)} records (may need more)")
                else:
                    print(f"❌ {symbol}: No data retrieved")
            else:
                print(f"❌ {symbol}: Update failed")
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
        
        # Small delay to avoid rate limiting 
        print("⏱️ Rate limiting (12s)...")
        import time
        time.sleep(12)
    
    # Check final database stats
    print("\n📊 Final Database Stats:")
    stats = db.get_database_stats()
    print(f"Symbols: {stats['daily_data']['symbols']}")
    print(f"Total Records: {stats['daily_data']['total_records']}")
    print(f"Date Range: {stats['daily_data']['earliest_date']} to {stats['daily_data']['latest_date']}")

if __name__ == "__main__":
    fix_database_data()