#!/usr/bin/env python3
"""
Debug the data format coming from database
"""

from market_data_database import MarketDataDatabase
import pandas as pd

def debug_data_format():
    """Check what the database is actually returning"""
    
    db = MarketDataDatabase()
    spy_data = db.get_data('SPY', '2023-01-01', '2023-03-31')
    
    print(f"📊 Data shape: {spy_data.shape}")
    print(f"🔢 Data types: {spy_data.dtypes}")
    print(f"📝 Columns: {list(spy_data.columns)}")
    print(f"📊 Sample data:")
    print(spy_data.head())
    
    print(f"\n🔍 Checking Close column specifically:")
    close_col = spy_data['Close']
    print(f"Type: {type(close_col)}")
    print(f"Values: {close_col.head()}")
    
    # Check if it's a single value or multiple
    try:
        single_value = close_col.iloc[0]
        print(f"Single value: {single_value} (type: {type(single_value)})")
        
        float_value = float(single_value)
        print(f"✅ Can convert to float: {float_value}")
        
    except Exception as e:
        print(f"❌ Cannot convert to float: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_format()