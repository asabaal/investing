#!/usr/bin/env python3
"""
Simple Iterative Data Collection Demo

Shows exactly what you asked for:
1. Pull some data for a security
2. Pull more data for the same security  
3. Do it for multiple timescales
4. Show results

NO COMPLEX BULLSHIT - JUST WORKS
"""

from simple_data_puller import SimpleSession

def demo_iterative_collection(symbol):
    """Demo iterative data collection for one symbol"""
    
    print(f"\n🎯 Iterative Data Collection Demo: {symbol}")
    print("=" * 50)
    
    # Create session
    session = SimpleSession(symbol)
    
    # Step 1: Pull daily data first
    print("📊 Step 1: Getting daily data...")
    daily = session.get_daily()
    if not daily.empty:
        print(f"   ✅ Got {len(daily):,} daily records")
        print(f"   📅 Range: {daily.index.min().date()} to {daily.index.max().date()}")
    else:
        print("   ❌ No daily data")
    
    # Step 2: Pull hourly data  
    print("\n⏰ Step 2: Getting hourly data...")
    hourly = session.get_intraday('60min')
    if not hourly.empty:
        print(f"   ✅ Got {len(hourly):,} hourly records")
        print(f"   📅 Range: {hourly.index.min()} to {hourly.index.max()}")
    else:
        print("   ❌ No hourly data")
    
    # Step 3: Pull minute data
    print("\n⚡ Step 3: Getting minute data...")
    minute = session.get_intraday('15min')
    if not minute.empty:
        print(f"   ✅ Got {len(minute):,} minute records")
        print(f"   📅 Range: {minute.index.min()} to {minute.index.max()}")
    else:
        print("   ❌ No minute data")
    
    # Show final status
    print(f"\n📊 Final Data Summary for {symbol}:")
    total_records = 0
    if not daily.empty:
        total_records += len(daily)
        print(f"   Daily: {len(daily):,} records")
    if not hourly.empty:
        total_records += len(hourly)
        print(f"   Hourly: {len(hourly):,} records") 
    if not minute.empty:
        total_records += len(minute)
        print(f"   Minute: {len(minute):,} records")
    
    print(f"   TOTAL: {total_records:,} records ready for analysis")
    
    return session

def demo_multiple_symbols():
    """Demo with multiple symbols"""
    
    print("🚀 ITERATIVE DATA COLLECTION DEMO")
    print("=" * 60)
    print("Pulling data step by step for multiple securities...")
    
    # Test symbols
    symbols = ['SPY', 'AAPL', 'QQQ']
    
    results = {}
    
    for symbol in symbols:
        try:
            session = demo_iterative_collection(symbol)
            results[symbol] = session
        except Exception as e:
            print(f"\n❌ Error with {symbol}: {e}")
            results[symbol] = None
    
    # Summary across all symbols
    print(f"\n🎯 MULTI-SYMBOL SUMMARY")
    print("=" * 40)
    
    total_all = 0
    for symbol, session in results.items():
        if session:
            symbol_total = 0
            if not session.daily.empty:
                symbol_total += len(session.daily)
            for data in session.intraday.values():
                if not data.empty:
                    symbol_total += len(data)
            
            print(f"✅ {symbol}: {symbol_total:,} total records")
            total_all += symbol_total
        else:
            print(f"❌ {symbol}: Failed")
    
    print(f"\n🚀 GRAND TOTAL: {total_all:,} records across all symbols")
    print("💡 Perfect for your 10 trades/day workflow!")
    
    return results

def demo_add_more_data():
    """Demo adding more data to existing session"""
    
    print(f"\n🔄 ADDING MORE DATA DEMO")
    print("=" * 40)
    
    symbol = 'TSLA'
    session = SimpleSession(symbol)
    
    # Start with daily
    print(f"1️⃣ Starting with daily data for {symbol}...")
    daily = session.get_daily()
    print(f"   📊 Daily: {len(daily):,} records" if not daily.empty else "   ❌ No daily data")
    
    # Add hourly
    print(f"\n2️⃣ Adding hourly data...")
    hourly = session.add_intraday('60min')  # This adds new data
    print(f"   📊 Hourly: {len(hourly):,} records" if not hourly.empty else "   ❌ No hourly data")
    
    # Add minute data  
    print(f"\n3️⃣ Adding minute data...")
    minute = session.add_intraday('15min')  # This adds new data
    print(f"   📊 Minute: {len(minute):,} records" if not minute.empty else "   ❌ No minute data")
    
    print(f"\n✅ {symbol} now has data at multiple timescales!")
    session.status()
    
    return session

if __name__ == "__main__":
    
    print("🎯 SIMPLE ITERATIVE DATA COLLECTION")
    print("🎯 Multiple securities, multiple timescales")
    print("🎯 Exactly what you asked for!")
    
    # Demo 1: Multiple symbols with iterative collection
    results = demo_multiple_symbols()
    
    # Demo 2: Adding more data to existing session
    session = demo_add_more_data()
    
    print(f"\n🎉 DEMO COMPLETE!")
    print("📊 You now have data at multiple timescales for multiple securities")
    print("🚀 Ready for your trading workflow!")
    
    # Show what you can do with the data
    print(f"\n💡 Example usage:")
    print("# Analyze any symbol's data:")
    for symbol in ['SPY', 'AAPL', 'QQQ']:
        if symbol in results and results[symbol]:
            session = results[symbol]
            if not session.daily.empty:
                latest_price = session.daily['Close'].iloc[-1]
                print(f"# {symbol} latest price: ${latest_price:.2f}")