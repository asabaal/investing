#!/usr/bin/env python3
"""
Test the newly added sector ETFs
"""

from market_data_database import MarketDataDatabase

def test_sector_etfs():
    """Test that XLK, XLE, XLF are working"""
    
    db = MarketDataDatabase()
    sector_etfs = ['XLK', 'XLE', 'XLF']
    
    for symbol in sector_etfs:
        try:
            data = db.get_data(symbol, '2023-01-01', '2024-12-31')
            if not data.empty:
                print(f"✅ {symbol}: {len(data)} records from {data.index.min().date()} to {data.index.max().date()}")
                print(f"   Price range: ${data['Close'].min():.2f} to ${data['Close'].max():.2f}")
            else:
                print(f"❌ {symbol}: No data found")
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")

if __name__ == "__main__":
    test_sector_etfs()