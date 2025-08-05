#!/usr/bin/env python3
"""
Trading Data Session Manager

Iterative data collection interface for trading analysis workflow.
Allows building up data for a security step by step without committing to main database.

Example Usage:
    # Start session for a security
    session = TradingDataSession('AAPL')
    
    # Get full daily history first
    session.add_daily_data()
    
    # Add 60min data for specific periods iteratively
    session.add_intraday_data('60min', '2023-01-01', '2023-12-31')
    session.add_intraday_data('60min', '2024-01-01', '2024-12-31')
    
    # Add 15min data for recent analysis
    session.add_intraday_data('15min', '2024-06-01', '2024-12-31')
    
    # Access collected data
    daily = session.get_daily_data()
    hourly = session.get_intraday_data('60min')
    minute = session.get_intraday_data('15min')
"""

import pandas as pd
import pickle
from datetime import datetime, date
from typing import Dict, Optional, List
import logging
from pathlib import Path
import os

from market_data_database import MarketDataDatabase

logger = logging.getLogger(__name__)

class TradingDataSession:
    """Session-based iterative data collection for trading analysis"""
    
    def __init__(self, symbol: str, session_name: str = None):
        """
        Initialize trading data session
        
        Args:
            symbol: Stock symbol for this session
            session_name: Optional name for the session (defaults to symbol)
        """
        self.symbol = symbol.upper()
        self.session_name = session_name or self.symbol
        self.db = MarketDataDatabase()
        
        # Data storage - separate containers for each interval
        self.daily_data = pd.DataFrame()
        self.intraday_data = {}  # interval -> DataFrame
        
        # Track what we've collected
        self.data_inventory = {
            'daily': {'collected': False, 'date_range': None, 'records': 0},
            'intraday': {}  # interval -> {collected: bool, date_range: tuple, records: int}
        }
        
        # Session metadata
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        
        logger.info(f"🎯 Started trading data session for {self.symbol}")
    
    def add_daily_data(self, start_date: str = None, end_date: str = None, force_update: bool = False) -> bool:
        """
        Add daily data to the session
        
        Args:
            start_date: Start date (YYYY-MM-DD), None for full history
            end_date: End date (YYYY-MM-DD), None for latest
            force_update: Force API update even if data exists in database
            
        Returns:
            True if successful
        """
        logger.info(f"📊 Adding daily data for {self.symbol}")
        
        try:
            # Get data from database (with smart caching/updating)
            if force_update:
                # Force update in database first
                self.db.update_daily_data(self.symbol, force_full_update=True)
            
            data = self.db.get_data(self.symbol, start_date, end_date, interval='daily')
            
            if data.empty:
                logger.warning(f"⚠️ No daily data available for {self.symbol}")
                return False
            
            # Filter date range if specified and not already filtered
            if start_date or end_date:
                if start_date:
                    data = data[data.index >= pd.to_datetime(start_date)]
                if end_date:
                    data = data[data.index <= pd.to_datetime(end_date)]
            
            # Store in session
            self.daily_data = data.copy()
            
            # Update inventory
            date_range = (data.index.min().strftime('%Y-%m-%d'), 
                         data.index.max().strftime('%Y-%m-%d'))
            
            self.data_inventory['daily'] = {
                'collected': True,
                'date_range': date_range,
                'records': len(data)
            }
            
            self.last_updated = datetime.now()
            
            logger.info(f"✅ Added {len(data)} daily records for {self.symbol}")
            logger.info(f"📅 Date range: {date_range[0]} to {date_range[1]}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add daily data for {self.symbol}: {e}")
            return False
    
    def add_intraday_data(self, interval: str, start_date: str = None, end_date: str = None, 
                         force_update: bool = False) -> bool:
        """
        Add intraday data to the session
        
        Args:
            interval: Intraday interval ('1min', '5min', '15min', '30min', '60min')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_update: Force API update even if data exists in database
            
        Returns:
            True if successful
        """
        logger.info(f"📈 Adding {interval} data for {self.symbol}")
        
        if interval not in ['1min', '5min', '15min', '30min', '60min']:
            logger.error(f"❌ Invalid interval: {interval}")
            return False
        
        try:
            # Strategy: Get ALL available data first, then filter in-memory
            # This avoids multiple API calls and database query issues
            
            if force_update:
                # Force update in database first
                self.db.update_intraday_data(self.symbol, interval)
            
            # Get ALL available data without date filters to avoid database query issues
            # Use skip_update=True to prevent redundant API calls after force_update
            logger.info(f"📊 Getting all available {interval} data for {self.symbol}")
            data = self.db.get_data(self.symbol, None, None, interval=interval, skip_update=force_update)
            
            if data.empty:
                logger.warning(f"⚠️ No {interval} data available for {self.symbol}")
                logger.info(f"💡 Note: Intraday data is typically only available for recent periods (~30 days)")
                return False
            
            logger.info(f"📅 Available data range: {data.index.min()} to {data.index.max()}")
            
            # Store original data before filtering
            original_data = data.copy()
            
            # Apply date filters in-memory (more reliable than SQL date filters)
            original_len = len(data)
            if start_date:
                data = data[data.index >= pd.to_datetime(start_date)]
                logger.info(f"🔍 After start_date filter ({start_date}): {len(data)} records (was {original_len})")
                original_len = len(data)
            
            if end_date:
                data = data[data.index <= pd.to_datetime(end_date)]
                logger.info(f"🔍 After end_date filter ({end_date}): {len(data)} records (was {original_len})")
            
            if data.empty and (start_date or end_date):
                logger.warning(f"⚠️ Date range {start_date} to {end_date} filters excluded all data")
                logger.info(f"💡 Available data is from {original_data.index.min()} to {original_data.index.max()}")
                
                # Use the original unfiltered data instead
                logger.info(f"🔄 Using all available {interval} data instead of the requested date range")
                data = original_data
                logger.info(f"✅ Using all available {interval} data: {len(data)} records")
            
            # Store in session (append or replace)
            if interval in self.intraday_data:
                # Combine with existing data, avoiding duplicates
                existing_data = self.intraday_data[interval]
                combined_data = pd.concat([existing_data, data])
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                combined_data.sort_index(inplace=True)
                self.intraday_data[interval] = combined_data
                logger.info(f"🔄 Combined with existing {interval} data")
            else:
                self.intraday_data[interval] = data.copy()
            
            final_data = self.intraday_data[interval]
            
            # Update inventory
            date_range = (final_data.index.min().strftime('%Y-%m-%d %H:%M:%S'), 
                         final_data.index.max().strftime('%Y-%m-%d %H:%M:%S'))
            
            if interval not in self.data_inventory['intraday']:
                self.data_inventory['intraday'][interval] = {}
            
            self.data_inventory['intraday'][interval] = {
                'collected': True,
                'date_range': date_range,
                'records': len(final_data)
            }
            
            self.last_updated = datetime.now()
            
            logger.info(f"✅ Added {len(data)} new {interval} records for {self.symbol}")
            logger.info(f"📊 Total {interval} records: {len(final_data)}")
            logger.info(f"📅 Date range: {date_range[0]} to {date_range[1]}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add {interval} data for {self.symbol}: {e}")
            return False
    
    def get_daily_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get daily data from session
        
        Args:
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            
        Returns:
            Daily data DataFrame
        """
        if self.daily_data.empty:
            logger.warning(f"⚠️ No daily data in session for {self.symbol}")
            return pd.DataFrame()
        
        data = self.daily_data.copy()
        
        # Apply date filters if specified
        if start_date:
            data = data[data.index >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data.index <= pd.to_datetime(end_date)]
        
        logger.info(f"📊 Retrieved {len(data)} daily records for {self.symbol}")
        
        return data
    
    def get_intraday_data(self, interval: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get intraday data from session
        
        Args:
            interval: Intraday interval
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            
        Returns:
            Intraday data DataFrame
        """
        if interval not in self.intraday_data:
            logger.warning(f"⚠️ No {interval} data in session for {self.symbol}")
            return pd.DataFrame()
        
        data = self.intraday_data[interval].copy()
        
        # Apply date filters if specified
        if start_date:
            data = data[data.index >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data.index <= pd.to_datetime(end_date)]
        
        logger.info(f"📈 Retrieved {len(data)} {interval} records for {self.symbol}")
        
        return data
    
    def get_session_summary(self) -> Dict:
        """Get summary of current session data"""
        
        summary = {
            'symbol': self.symbol,
            'session_name': self.session_name,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'data_inventory': self.data_inventory.copy()
        }
        
        # Add total record counts
        total_records = 0
        if self.data_inventory['daily']['collected']:
            total_records += self.data_inventory['daily']['records']
        
        for interval_data in self.data_inventory['intraday'].values():
            if interval_data['collected']:
                total_records += interval_data['records']
        
        summary['total_records'] = total_records
        summary['intervals_collected'] = list(self.intraday_data.keys())
        summary['has_daily'] = self.data_inventory['daily']['collected']
        
        return summary
    
    def print_session_status(self):
        """Print formatted session status"""
        
        summary = self.get_session_summary()
        
        print(f"\n🎯 Trading Data Session: {summary['session_name']}")
        print("=" * 60)
        print(f"Symbol: {summary['symbol']}")
        print(f"Created: {summary['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Updated: {summary['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Records: {summary['total_records']:,}")
        
        print("\n📊 Data Inventory:")
        print("-" * 40)
        
        # Daily data
        if summary['has_daily']:
            daily_info = summary['data_inventory']['daily']
            print(f"Daily Data: ✅ {daily_info['records']:,} records")
            print(f"  Range: {daily_info['date_range'][0]} to {daily_info['date_range'][1]}")
        else:
            print("Daily Data: ❌ Not collected")
        
        # Intraday data
        if summary['intervals_collected']:
            print("\nIntraday Data:")
            for interval in sorted(summary['intervals_collected']):
                interval_info = summary['data_inventory']['intraday'][interval]
                print(f"  {interval:>6}: ✅ {interval_info['records']:,} records")
                # Show shorter date range for intraday
                start_date = interval_info['date_range'][0][:10]
                end_date = interval_info['date_range'][1][:10]
                print(f"          Range: {start_date} to {end_date}")
        else:
            print("Intraday Data: ❌ None collected")
        
        print("=" * 60)
    
    def save_session(self, filepath: str = None) -> str:
        """
        Save session to file
        
        Args:
            filepath: Optional filepath, defaults to session_name.pkl
            
        Returns:
            Path where session was saved
        """
        if filepath is None:
            # Create sessions directory if it doesn't exist
            sessions_dir = Path("trading_sessions")
            sessions_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = sessions_dir / f"{self.session_name}_{timestamp}.pkl"
        
        # Prepare session data for pickle
        session_data = {
            'symbol': self.symbol,
            'session_name': self.session_name,
            'daily_data': self.daily_data,
            'intraday_data': self.intraday_data,
            'data_inventory': self.data_inventory,
            'created_at': self.created_at,
            'last_updated': self.last_updated
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(session_data, f)
        
        logger.info(f"💾 Session saved to: {filepath}")
        return str(filepath)
    
    @classmethod
    def load_session(cls, filepath: str) -> 'TradingDataSession':
        """
        Load session from file
        
        Args:
            filepath: Path to session file
            
        Returns:
            Loaded TradingDataSession
        """
        with open(filepath, 'rb') as f:
            session_data = pickle.load(f)
        
        # Recreate session
        session = cls(session_data['symbol'], session_data['session_name'])
        session.daily_data = session_data['daily_data']
        session.intraday_data = session_data['intraday_data']
        session.data_inventory = session_data['data_inventory']
        session.created_at = session_data['created_at']
        session.last_updated = session_data['last_updated']
        
        logger.info(f"📂 Session loaded from: {filepath}")
        return session
    
    def export_to_csv(self, output_dir: str = None):
        """
        Export session data to CSV files
        
        Args:
            output_dir: Directory to save CSV files, defaults to symbol_exports/
        """
        if output_dir is None:
            output_dir = Path(f"{self.symbol}_exports")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        exported_files = []
        
        # Export daily data
        if not self.daily_data.empty:
            daily_file = output_dir / f"{self.symbol}_daily.csv"
            self.daily_data.to_csv(daily_file)
            exported_files.append(str(daily_file))
            logger.info(f"💾 Exported daily data to: {daily_file}")
        
        # Export intraday data
        for interval, data in self.intraday_data.items():
            if not data.empty:
                intraday_file = output_dir / f"{self.symbol}_{interval}.csv"
                data.to_csv(intraday_file)
                exported_files.append(str(intraday_file))
                logger.info(f"💾 Exported {interval} data to: {intraday_file}")
        
        return exported_files


def demo_session(symbol: str = 'AAPL'):
    """Demo the trading data session functionality"""
    
    print(f"\n🚀 Trading Data Session Demo with {symbol}")
    print("=" * 60)
    
    # Create session
    session = TradingDataSession(symbol)
    
    # Add daily data first
    print("\n1️⃣ Adding daily data (full history)...")
    session.add_daily_data()
    session.print_session_status()
    
    # Add some 60min data for recent period
    print("\n2️⃣ Adding 60min data for last 6 months...")
    session.add_intraday_data('60min', '2024-06-01', '2024-12-31')
    session.print_session_status()
    
    # Add 15min data for last 3 months
    print("\n3️⃣ Adding 15min data for last 3 months...")
    session.add_intraday_data('15min', '2024-10-01', '2024-12-31')
    session.print_session_status()
    
    # Show how to access the data
    daily = session.get_daily_data()
    hourly = session.get_intraday_data('60min')
    minute = session.get_intraday_data('15min')
    
    print(f"\n📊 Data Access Results:")
    print(f"Daily records: {len(daily):,}")
    print(f"60min records: {len(hourly):,}")
    print(f"15min records: {len(minute):,}")
    
    # Save session
    saved_path = session.save_session()
    print(f"\n💾 Session saved to: {saved_path}")
    
    return session


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Data Session Manager')
    parser.add_argument('--demo', help='Run demo with symbol (default: AAPL)')
    parser.add_argument('--symbol', help='Symbol for new session')
    parser.add_argument('--load', help='Load session from file')
    
    args = parser.parse_args()
    
    if args.demo:
        symbol = args.demo if args.demo != 'demo' else 'AAPL'
        demo_session(symbol)
    elif args.symbol:
        session = TradingDataSession(args.symbol)
        print(f"Started session for {args.symbol}. Use the session object to add data.")
    elif args.load:
        session = TradingDataSession.load_session(args.load)
        session.print_session_status()
    else:
        print("Use --demo [SYMBOL], --symbol SYMBOL, or --load PATH")