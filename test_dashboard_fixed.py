#!/usr/bin/env python3
"""
Quick test to verify dashboard is fully functional
"""

from market_data_dashboard import app
import json

def test_dashboard():
    """Test all dashboard functionality"""
    print("🔧 Testing Fixed Dashboard")
    print("=" * 40)
    
    with app.test_client() as client:
        # Test 1: Main page
        print("🧪 Testing main page...")
        response = client.get('/')
        assert response.status_code == 200, f"Main page failed: {response.status_code}"
        print("✅ Main page loads successfully")
        
        # Test 2: Stats API
        print("🧪 Testing stats API...")
        response = client.get('/api/stats')
        assert response.status_code == 200, f"Stats API failed: {response.status_code}"
        stats = response.get_json()
        assert 'unified_view' in stats, "Stats missing unified_view"
        print(f"✅ Stats API: {stats['unified_view']['total_unique_symbols']} symbols")
        
        # Test 3: Symbols API
        print("🧪 Testing symbols API...")
        response = client.get('/api/symbols')
        assert response.status_code == 200, f"Symbols API failed: {response.status_code}"
        symbols = response.get_json()
        assert len(symbols) > 0, "No symbols returned"
        print(f"✅ Symbols API: {len(symbols)} symbols")
        
        # Test 4: Chart API
        print("🧪 Testing chart API...")
        response = client.get('/api/chart/AAPL?interval=daily&days=30')
        assert response.status_code == 200, f"Chart API failed: {response.status_code}"
        chart_data = response.get_json()
        assert 'data' in chart_data, "Chart missing data"
        print("✅ Chart API: AAPL chart generated")
        
        # Test 5: Comparison API  
        print("🧪 Testing comparison API...")
        response = client.get('/api/comparison?symbols=AAPL,MSFT&days=30')
        assert response.status_code == 200, f"Comparison API failed: {response.status_code}"
        comparison_data = response.get_json()
        assert 'data' in comparison_data, "Comparison missing data"
        print("✅ Comparison API: AAPL vs MSFT comparison")
        
        # Test 6: Heatmap API
        print("🧪 Testing heatmap API...")
        response = client.get('/api/heatmap?days=7')
        assert response.status_code == 200, f"Heatmap API failed: {response.status_code}"
        heatmap_data = response.get_json()
        assert 'data' in heatmap_data, "Heatmap missing data"
        print("✅ Heatmap API: Performance heatmap generated")
        
        # Test 7: SQL Query API
        print("🧪 Testing SQL query API...")
        test_query = "SELECT COUNT(*) as total_records FROM daily_data"
        response = client.post('/api/query', 
                             json={'query': test_query},
                             content_type='application/json')
        assert response.status_code == 200, f"SQL API failed: {response.status_code}"
        sql_result = response.get_json()
        assert 'data' in sql_result, "SQL result missing data"
        print(f"✅ SQL API: Query executed, {sql_result['row_count']} rows")
        
    print("\n" + "=" * 40)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Dashboard is fully functional")
    print("🚀 Ready to launch: python launch_dashboard.py")
    print("🌐 URL: http://localhost:5000")

if __name__ == "__main__":
    test_dashboard()