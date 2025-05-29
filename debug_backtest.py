#!/usr/bin/env python3
"""
Debug the backtest float conversion issue
"""

from symphony_core import SymphonyService
from market_data_database import MarketDataDatabase

def debug_simple_backtest():
    """Debug a simple backtest with just SPY"""
    
    # Get data
    db = MarketDataDatabase()
    spy_data = db.get_data('SPY', '2023-01-01', '2023-03-31')
    
    print(f"📊 SPY data: {len(spy_data)} records")
    print(f"📅 Date range: {spy_data.index.min()} to {spy_data.index.max()}")
    min_price = spy_data['Close'].min()
    max_price = spy_data['Close'].max()
    print(f"💰 Price range: ${float(min_price):.2f} to ${float(max_price):.2f}")
    print(f"🔢 Data types: {spy_data.dtypes}")
    print(f"📊 Sample data:\n{spy_data.head()}")
    
    # Test simple price extraction
    try:
        current_price = spy_data['Close'].iloc[-1]
        print(f"📈 Current price (raw): {current_price} (type: {type(current_price)})")
        
        current_price_float = float(current_price)
        print(f"📈 Current price (float): {current_price_float}")
        
    except Exception as e:
        print(f"❌ Error converting to float: {e}")
        import traceback
        traceback.print_exc()
    
    # Create simple symphony
    simple_symphony = {
        "name": "Simple SPY Test",
        "universe": ["SPY"],
        "logic": {
            "allocations": {
                "spy_allocation": {
                    "type": "fixed_allocation",
                    "weights": {
                        "SPY": 1.0
                    }
                }
            }
        }
    }
    
    # Test symphony execution
    service = SymphonyService()
    engine = service.factory.get_engine()
    
    try:
        print("\n🧪 Testing symphony execution...")
        market_data = {"SPY": spy_data}
        
        result = engine.execute_symphony(simple_symphony, market_data, '2023-01-31')
        print(f"✅ Symphony execution successful: {result.allocation}")
        
    except Exception as e:
        print(f"❌ Symphony execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test backtest
    try:
        print("\n🧪 Testing simple backtest...")
        backtester = service.factory.get_backtester()
        
        backtest_results = backtester.backtest(
            simple_symphony, market_data, '2023-01-01', '2023-03-31', 'monthly'
        )
        
        print(f"✅ Backtest successful: {len(backtest_results)} periods")
        if not backtest_results.empty:
            print(backtest_results[['date', 'portfolio_return']].head())
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_simple_backtest()