#!/usr/bin/env python3
"""
SIMPLE DEMO - Exactly what you asked for

1. Pull some data for a security
2. Pull more data for the same security at different timescales  
3. Show the iterative process
"""

from direct_api_puller import get_daily_data_api, get_intraday_data_api

def demo():
    symbol = 'AAPL'
    
    print(f"🎯 ITERATIVE DATA COLLECTION DEMO")
    print(f"Security: {symbol}")
    print("=" * 40)
    
    # Step 1: Pull daily data first (the foundation)
    print("📊 Step 1: Getting daily data...")
    daily = get_daily_data_api(symbol)
    print(f"   Result: {len(daily)} daily records")
    if not daily.empty:
        print(f"   Range: {daily.index.min().date()} to {daily.index.max().date()}")
    
    # Step 2: Pull hourly data for last 2 months
    print("\n⏰ Step 2: Adding hourly data (last 2 months)...")
    hourly = get_intraday_data_api(symbol, '60min', '2025-06-01', '2025-08-05')
    print(f"   Result: {len(hourly)} hourly records")
    if not hourly.empty:
        print(f"   Range: {hourly.index.min()} to {hourly.index.max()}")
    
    # Step 3: Pull 15-minute data for last month
    print("\n⚡ Step 3: Adding 15-minute data (last month)...")
    minute = get_intraday_data_api(symbol, '15min', '2025-07-01', '2025-08-05')
    print(f"   Result: {len(minute)} minute records")
    if not minute.empty:
        print(f"   Range: {minute.index.min()} to {minute.index.max()}")
    
    # Step 4: Pull 5-minute data for last week
    print("\n⚡ Step 4: Adding 5-minute data (last week)...")
    five_min = get_intraday_data_api(symbol, '5min', '2025-07-28', '2025-08-05')
    print(f"   Result: {len(five_min)} 5-minute records")
    if not five_min.empty:
        print(f"   Range: {five_min.index.min()} to {five_min.index.max()}")
    
    # Summary
    total = len(daily) + len(hourly) + len(minute) + len(five_min)
    print(f"\n🎯 FINAL DATASET:")
    print(f"   Daily: {len(daily):,} records")
    print(f"   Hourly: {len(hourly):,} records") 
    print(f"   15-min: {len(minute):,} records")
    print(f"   5-min: {len(five_min):,} records")
    print(f"   TOTAL: {total:,} records")
    
    print(f"\n✅ Perfect for trading analysis!")
    print(f"🚀 Multiple timescales, iteratively collected!")

if __name__ == "__main__":
    demo()