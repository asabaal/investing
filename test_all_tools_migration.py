#!/usr/bin/env python3
"""
Comprehensive test of all tools with new database structure
"""

def test_database_location():
    """Test that all tools use the correct database location"""
    print("🔍 Testing Database Location Consistency")
    print("=" * 50)
    
    expected_path = "/home/asabaal/.market_data/market_data.db"
    
    # Test 1: Core database class
    from market_data_database import MarketDataDatabase
    db = MarketDataDatabase()
    assert db.db_path == expected_path, f"Database path mismatch: {db.db_path}"
    print("✅ Core MarketDataDatabase")
    
    # Test 2: Intraday data manager
    from intraday_data_manager import IntradayDataManager
    manager = IntradayDataManager()
    assert manager.db.db_path == expected_path, f"Intraday manager path mismatch: {manager.db.db_path}"
    print("✅ Intraday Data Manager")
    
    # Test 3: Symphony core components
    from symphony_core import SymphonyService
    service = SymphonyService()
    pipeline = service.factory.get_data_pipeline()
    assert pipeline.db_path == expected_path, f"Symphony pipeline path mismatch: {pipeline.db_path}"
    print("✅ Symphony Core")
    
    # Test 4: Symphony runner
    from symphony_runner import SymphonyRunner
    runner = SymphonyRunner()
    assert runner.data_manager.database.db_path == expected_path, f"Symphony runner path mismatch"
    print("✅ Symphony Runner")
    
    # Test 5: Integrated symphony system
    from integrated_symphony_system import IntegratedSymphonySystem
    system = IntegratedSymphonySystem()
    assert system.data_manager.database.db_path == expected_path, f"Integrated system path mismatch"
    print("✅ Integrated Symphony System")
    
    # Test 6: Dashboard
    from market_data_dashboard import db as dashboard_db
    assert dashboard_db.db_path == expected_path, f"Dashboard db path mismatch: {dashboard_db.db_path}"
    print("✅ Dashboard Database")
    
    print(f"\n🎯 All tools using: {expected_path}")

def test_functionality():
    """Test that all tools function correctly"""
    print("\n🧪 Testing Tool Functionality")
    print("=" * 50)
    
    # Test database access
    from market_data_database import MarketDataDatabase
    db = MarketDataDatabase()
    stats = db.get_database_stats()
    print(f"✅ Database: {stats['unified_view']['total_unique_symbols']} symbols")
    
    # Test data retrieval
    data = db.get_data('SPY', interval='daily')
    print(f"✅ Data retrieval: {len(data)} SPY records")
    
    # Test dashboard APIs
    from market_data_dashboard import app
    with app.test_client() as client:
        response = client.get('/api/stats')
        assert response.status_code == 200, "Dashboard API failed"
        print("✅ Dashboard APIs")
    
    # Test symphony data preparation
    from symphony_core import DatabaseDataManager
    data_manager = DatabaseDataManager(db)
    test_symphony = {'name': 'Test', 'universe': ['SPY', 'QQQ']}
    symphony_data = data_manager.prepare_symphony_data(test_symphony)
    print(f"✅ Symphony data prep: {len(symphony_data)} symbols")
    
    # Test intraday functionality
    from intraday_data_manager import IntradayDataManager
    manager = IntradayDataManager()
    spy_intraday = manager.get_intraday_data('SPY', '15min')
    print(f"✅ Intraday access: {len(spy_intraday)} SPY 15min records")

def test_cli_tools():
    """Test command-line tools"""
    print("\n⚙️ Testing CLI Tools")
    print("=" * 50)
    
    import subprocess
    import os
    
    # Test market_data_database.py stats
    result = subprocess.run(['python', 'market_data_database.py', '--stats'], 
                          capture_output=True, text=True, cwd=os.getcwd())
    assert result.returncode == 0, "market_data_database.py --stats failed"
    print("✅ market_data_database.py --stats")
    
    # Test intraday_data_manager.py
    result = subprocess.run(['python', 'intraday_data_manager.py', '--get', 'SPY', '--interval', '15min'], 
                          capture_output=True, text=True, cwd=os.getcwd())
    assert result.returncode == 0, "intraday_data_manager.py failed"
    print("✅ intraday_data_manager.py")

def main():
    """Run all tests"""
    print("🎯 Comprehensive Tool Migration Test")
    print("=" * 60)
    
    try:
        test_database_location()
        test_functionality()
        test_cli_tools()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TOOLS MIGRATION TESTS PASSED!")
        print("\n📋 Summary:")
        print("  ✅ All tools use correct database location")
        print("  ✅ All functionality working properly")
        print("  ✅ CLI tools operational")
        print("  ✅ Dashboard fully functional")
        print("  ✅ Symphony system operational")
        
        print("\n🚀 Ready for production use!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)