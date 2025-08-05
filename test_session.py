#!/usr/bin/env python3
"""
Interactive test of the Trading Data Session system
"""

from trading_data_session import TradingDataSession
import pandas as pd

def interactive_test():
    """Interactive test of the trading data session"""
    
    print("🚀 Interactive Trading Data Session Test")
    print("=" * 50)
    
    # Get user input for symbol
    symbol = input("Enter symbol to test (default: AAPL): ").strip().upper()
    if not symbol:
        symbol = 'AAPL'
    
    print(f"\n🎯 Creating session for {symbol}...")
    
    # Create session
    session = TradingDataSession(symbol)
    
    while True:
        print(f"\n📋 Available commands:")
        print("1. Add daily data (full history)")
        print("2. Add daily data (date range)")
        print("3. Add intraday data (specify interval)")
        print("4. Show session status")
        print("5. Get daily data sample")
        print("6. Get intraday data sample")
        print("7. Save session")
        print("8. Export to CSV")
        print("q. Quit")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == '1':
            print("\n🔄 Adding full daily data history...")
            success = session.add_daily_data()
            if success:
                print("✅ Daily data added successfully!")
            else:
                print("❌ Failed to add daily data")
                
        elif choice == '2':
            start_date = input("Start date (YYYY-MM-DD, or Enter for none): ").strip()
            end_date = input("End date (YYYY-MM-DD, or Enter for none): ").strip()
            start_date = start_date if start_date else None
            end_date = end_date if end_date else None
            
            print(f"\n🔄 Adding daily data from {start_date or 'beginning'} to {end_date or 'latest'}...")
            success = session.add_daily_data(start_date, end_date)
            if success:
                print("✅ Daily data added successfully!")
            else:
                print("❌ Failed to add daily data")
                
        elif choice == '3':
            print("Available intervals: 1min, 5min, 15min, 30min, 60min")
            interval = input("Enter interval: ").strip()
            start_date = input("Start date (YYYY-MM-DD, or Enter for none): ").strip()
            end_date = input("End date (YYYY-MM-DD, or Enter for none): ").strip()
            start_date = start_date if start_date else None
            end_date = end_date if end_date else None
            
            print(f"\n🔄 Adding {interval} data from {start_date or 'beginning'} to {end_date or 'latest'}...")
            success = session.add_intraday_data(interval, start_date, end_date)
            if success:
                print("✅ Intraday data added successfully!")
            else:
                print("❌ Failed to add intraday data")
                
        elif choice == '4':
            session.print_session_status()
            
        elif choice == '5':
            daily_data = session.get_daily_data()
            if not daily_data.empty:
                print(f"\n📊 Daily Data Sample ({len(daily_data)} total records):")
                print("First 5 records:")
                print(daily_data.head())
                print("\nLast 5 records:")
                print(daily_data.tail())
            else:
                print("❌ No daily data in session")
                
        elif choice == '6':
            intervals = list(session.intraday_data.keys())
            if intervals:
                print(f"Available intervals: {intervals}")
                interval = input("Enter interval to view: ").strip()
                if interval in intervals:
                    intraday_data = session.get_intraday_data(interval)
                    print(f"\n📈 {interval} Data Sample ({len(intraday_data)} total records):")
                    print("First 5 records:")
                    print(intraday_data.head())
                    print("\nLast 5 records:")
                    print(intraday_data.tail())
                else:
                    print(f"❌ Interval {interval} not found in session")
            else:
                print("❌ No intraday data in session")
                
        elif choice == '7':
            filepath = session.save_session()
            print(f"💾 Session saved to: {filepath}")
            
        elif choice == '8':
            output_dir = input("Output directory (or Enter for default): ").strip()
            output_dir = output_dir if output_dir else None
            files = session.export_to_csv(output_dir)
            print(f"💾 Exported {len(files)} files:")
            for file in files:
                print(f"  - {file}")
        else:
            print("❌ Invalid choice")
    
    print(f"\n👋 Session ended. Final status:")
    session.print_session_status()
    
    return session

if __name__ == "__main__":
    session = interactive_test()