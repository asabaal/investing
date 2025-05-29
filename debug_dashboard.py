#!/usr/bin/env python3
"""
Debug script for dashboard issues
"""

import traceback
import sys

def test_imports():
    """Test all imports"""
    print("🧪 Testing imports...")
    try:
        import flask
        print("✅ Flask imported")
        
        import plotly
        print("✅ Plotly imported")
        
        from market_data_database import MarketDataDatabase
        print("✅ MarketDataDatabase imported")
        
        # Test database connection
        db = MarketDataDatabase()
        print("✅ Database connection successful")
        
        # Test stats
        stats = db.get_database_stats()
        print(f"✅ Database stats loaded: {stats['unified_view']['total_unique_symbols']} symbols")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_dashboard_app():
    """Test dashboard application"""
    print("\n🧪 Testing dashboard app...")
    try:
        from market_data_dashboard import app
        print("✅ Dashboard app imported")
        
        # Test main route
        with app.test_client() as client:
            print("🧪 Testing main route...")
            response = client.get('/')
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ Main page loads successfully")
                return True
            else:
                print(f"❌ Error response: {response.status_code}")
                print(f"Response data: {response.data.decode()}")
                return False
                
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        traceback.print_exc()
        return False

def test_template():
    """Test template rendering"""
    print("\n🧪 Testing template...")
    try:
        from market_data_database import MarketDataDatabase
        from flask import Flask, render_template
        
        app = Flask(__name__)
        
        with app.app_context():
            db = MarketDataDatabase()
            stats = db.get_database_stats()
            
            # Try to render template
            html = render_template('dashboard.html', stats=stats)
            print("✅ Template renders successfully")
            print(f"Template length: {len(html)} characters")
            return True
            
    except Exception as e:
        print(f"❌ Template error: {e}")
        traceback.print_exc()
        return False

def main():
    print("🔍 Dashboard Debug Session")
    print("=" * 50)
    
    # Test 1: Imports
    if not test_imports():
        print("❌ Import test failed - stopping")
        return False
    
    # Test 2: Template
    if not test_template():
        print("❌ Template test failed - stopping")
        return False
    
    # Test 3: Dashboard app
    if not test_dashboard_app():
        print("❌ Dashboard app test failed")
        return False
    
    print("\n" + "=" * 50)
    print("✅ All tests passed! Dashboard should work.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)