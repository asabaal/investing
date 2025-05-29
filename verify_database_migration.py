#!/usr/bin/env python3
"""
Verification script for database migration
"""

import os
from pathlib import Path
from market_data_database import MarketDataDatabase, get_default_database_path

def verify_migration():
    """Verify the database migration was successful"""
    print("🔍 Database Migration Verification")
    print("=" * 50)
    
    # Check 1: Database path resolution
    expected_path = Path.home() / '.market_data' / 'market_data.db'
    actual_path = get_default_database_path()
    
    print(f"✅ Default path function: {actual_path}")
    assert actual_path == str(expected_path), "Path function incorrect"
    
    # Check 2: Database file exists
    print(f"✅ Database file exists: {os.path.exists(actual_path)}")
    assert os.path.exists(actual_path), "Database file not found"
    
    # Check 3: Database is functional
    db = MarketDataDatabase()
    print(f"✅ Database path used: {db.db_path}")
    assert db.db_path == actual_path, "Database using wrong path"
    
    # Check 4: Database has data
    stats = db.get_database_stats()
    symbol_count = stats['unified_view']['total_unique_symbols']
    db_size = stats['database_size_mb']
    
    print(f"✅ Database symbols: {symbol_count}")
    print(f"✅ Database size: {db_size:.1f} MB")
    assert symbol_count > 0, "Database has no symbols"
    assert db_size > 100, "Database seems too small"
    
    # Check 5: Old database removed from repo
    old_repo_db = Path('./market_data.db')
    print(f"✅ Old repo database removed: {not old_repo_db.exists()}")
    assert not old_repo_db.exists(), "Old database still in repo"
    
    # Check 6: Data retrieval works
    sample_data = db.get_data('AAPL', interval='daily')
    print(f"✅ Sample data retrieval: {len(sample_data)} AAPL records")
    assert len(sample_data) > 0, "Cannot retrieve data"
    
    # Check 7: Multiple components work
    from market_data_dashboard import app
    with app.test_client() as client:
        response = client.get('/api/stats')
        print(f"✅ Dashboard integration: HTTP {response.status_code}")
        assert response.status_code == 200, "Dashboard not working"
    
    print("\n" + "=" * 50)
    print("🎉 MIGRATION VERIFICATION PASSED!")
    print("\n📊 Summary:")
    print(f"  • Database location: {actual_path}")
    print(f"  • Database size: {db_size:.1f} MB")
    print(f"  • Symbols available: {symbol_count}")
    print(f"  • Repository cleaned: ✅")
    print(f"  • All systems operational: ✅")
    
    print("\n🚀 Ready for git operations!")
    print("  git add .")
    print("  git commit -m 'Move database outside repo'")
    print("  git push")

if __name__ == "__main__":
    verify_migration()