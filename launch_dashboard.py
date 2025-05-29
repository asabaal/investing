#!/usr/bin/env python3
"""
Quick launcher for Market Data Dashboard

This script starts the dashboard and provides helpful information.
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

def main():
    print("🚀 Market Data Dashboard Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("market_data.db").exists():
        print("❌ market_data.db not found!")
        print("💡 Make sure you're running this from the investing directory")
        return False
    
    # Check dependencies
    try:
        import flask
        import plotly
        print("✅ Dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: pip install flask plotly")
        return False
    
    # Get database stats
    try:
        from market_data_database import MarketDataDatabase
        db = MarketDataDatabase()
        stats = db.get_database_stats()
        print(f"✅ Database loaded: {stats['unified_view']['total_unique_symbols']} symbols")
    except Exception as e:
        print(f"⚠️ Database warning: {e}")
    
    print("\n🌐 Starting dashboard server...")
    print("📍 URL: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop\n")
    
    # Start dashboard
    try:
        # Try to open browser after a short delay
        import threading
        def open_browser():
            time.sleep(2)
            try:
                webbrowser.open('http://localhost:5000')
                print("🌍 Opened dashboard in browser")
            except:
                pass
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start Flask app with better error handling
        from market_data_dashboard import app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
        return True
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)