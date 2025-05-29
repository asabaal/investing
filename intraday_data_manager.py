#!/usr/bin/env python3
"""
Intraday Data Manager

Utilities for collecting and managing intraday market data.
The database already supports intraday data - this script makes it easier to use.
"""

import argparse
from market_data_database import MarketDataDatabase
from datetime import datetime

class IntradayDataManager:
    """Manager for intraday data collection and analysis"""
    
    def __init__(self):
        self.db = MarketDataDatabase()
        self.supported_intervals = ['1min', '5min', '15min', '30min', '60min']
    
    def collect_intraday_data(self, symbols: list, interval: str = '15min'):
        """Collect intraday data for symbols"""
        
        if interval not in self.supported_intervals:
            raise ValueError(f"Interval must be one of: {self.supported_intervals}")
        
        print(f"📊 Collecting {interval} data for {len(symbols)} symbols...")
        
        results = {}
        for symbol in symbols:
            try:
                print(f"🔄 Updating {symbol} {interval} data...")
                success = self.db.update_intraday_data(symbol, interval)
                results[symbol] = success
                
                if success:
                    # Check how much data we got
                    data = self.db.get_data(symbol, interval=interval)
                    print(f"✅ {symbol}: {len(data)} {interval} records")
                else:
                    print(f"❌ {symbol}: Failed to update")
                    
            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")
                results[symbol] = False
        
        successful = sum(1 for success in results.values() if success)
        print(f"\n📈 Intraday data collection complete: {successful}/{len(symbols)} successful")
        
        return results
    
    def get_intraday_data(self, symbol: str, interval: str = '15min', 
                         start_date: str = None, end_date: str = None):
        """Get intraday data for analysis"""
        
        print(f"📊 Getting {symbol} {interval} data...")
        data = self.db.get_data(symbol, start_date, end_date, interval)
        
        if not data.empty:
            print(f"✅ Retrieved {len(data)} {interval} records for {symbol}")
            print(f"📅 Date range: {data.index.min()} to {data.index.max()}")
        else:
            print(f"⚠️ No {interval} data found for {symbol}")
        
        return data
    
    def analyze_intraday_patterns(self, symbol: str, interval: str = '15min'):
        """Analyze intraday patterns for a symbol"""
        
        data = self.get_intraday_data(symbol, interval)
        
        if data.empty:
            return None
        
        # Calculate basic intraday statistics
        data['hour'] = data.index.hour
        data['minute'] = data.index.minute
        data['time_of_day'] = data['hour'] + data['minute'] / 60
        data['returns'] = data['Close'].pct_change()
        
        # Group by time of day
        hourly_stats = data.groupby('hour')['returns'].agg([
            'mean', 'std', 'count'
        ]).round(4)
        
        print(f"\n📊 Intraday Pattern Analysis for {symbol} ({interval})")
        print("=" * 60)
        print(hourly_stats)
        
        # Find best and worst hours
        best_hour = hourly_stats['mean'].idxmax()
        worst_hour = hourly_stats['mean'].idxmin()
        
        print(f"\n🏆 Best performing hour: {best_hour}:00 ({hourly_stats.loc[best_hour, 'mean']:.4f} avg return)")
        print(f"📉 Worst performing hour: {worst_hour}:00 ({hourly_stats.loc[worst_hour, 'mean']:.4f} avg return)")
        
        return {
            'hourly_stats': hourly_stats,
            'best_hour': best_hour,
            'worst_hour': worst_hour,
            'total_periods': len(data)
        }

def main():
    """Command line interface for intraday data management"""
    
    parser = argparse.ArgumentParser(description='Intraday Data Manager')
    parser.add_argument('--collect', nargs='+', help='Collect intraday data for symbols')
    parser.add_argument('--interval', default='15min', choices=['1min', '5min', '15min', '30min', '60min'],
                       help='Intraday interval (default: 15min)')
    parser.add_argument('--analyze', help='Analyze intraday patterns for symbol')
    parser.add_argument('--get', help='Get intraday data for symbol')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    manager = IntradayDataManager()
    
    if args.collect:
        # Collect intraday data
        manager.collect_intraday_data(args.collect, args.interval)
    
    elif args.analyze:
        # Analyze patterns
        manager.analyze_intraday_patterns(args.analyze, args.interval)
    
    elif args.get:
        # Get data
        data = manager.get_intraday_data(args.get, args.interval, args.start_date, args.end_date)
        if not data.empty:
            print(f"\nFirst 5 records:")
            print(data.head())
            print(f"\nLast 5 records:")
            print(data.tail())
    
    else:
        print("Use --collect, --analyze, or --get. See --help for details.")

if __name__ == "__main__":
    main()