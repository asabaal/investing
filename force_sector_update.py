#!/usr/bin/env python3
"""
Force update sector ETFs
"""

from market_data_database import MarketDataDatabase

def force_update_sectors():
    """Force full update for sector ETFs"""
    
    db = MarketDataDatabase()
    symbols = ['XLK', 'XLE', 'XLF']
    
    for symbol in symbols:
        print(f"🔄 Force updating {symbol}...")
        try:
            success = db.update_daily_data(symbol, force_full_update=True)
            if success:
                # Check the data
                data = db.get_data(symbol)
                if not data.empty:
                    print(f"✅ {symbol}: {len(data)} records")
                else:
                    print(f"⚠️ {symbol}: Update reported success but no data found")
            else:
                print(f"❌ {symbol}: Update failed")
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
            import traceback
            traceback.print_exc()
        
        print("⏱️ Rate limiting (12s)...")
        import time
        time.sleep(12)

if __name__ == "__main__":
    force_update_sectors()