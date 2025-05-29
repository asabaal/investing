"""
Market Data Database System

A proper database-first approach for market data management:
- SQLite database for local storage
- Daily data updates via cron job
- Intraday data support
- No rate limiting on local data
- Efficient data retrieval and updates
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import os
import time
import requests
from typing import Dict, List, Optional, Tuple, Any
import json
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataRequest:
    """Data request specification"""
    symbol: str
    start_date: str
    end_date: str
    interval: str = 'daily'  # 'daily', '1min', '5min', '15min', '30min', '60min'

class MarketDataDatabase:
    """Centralized market data database management"""
    
    def __init__(self, db_path: str = "./market_data.db", api_key: str = None):
        self.db_path = db_path
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        
        if not self.api_key:
            raise ValueError("API key required. Set ALPHA_VANTAGE_API_KEY environment variable.")
        
        # Symbol mapping for problematic symbols
        self.symbol_mapping = {
            'VIX': 'VXX',  # VIX index is not directly available, use VXX ETF as proxy
            '^VIX': 'VXX',  # Handle different VIX formats
        }
        
        # Symbols that don't support intraday data
        self.intraday_unsupported = {
            'VIX', '^VIX'  # VIX doesn't have intraday data available through Alpha Vantage
        }
        
        self._init_database()
        logger.info(f"📊 Market Data Database initialized: {db_path}")
    
    def _init_database(self):
        """Initialize database schema"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Daily data table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS daily_data (
                    symbol TEXT,
                    date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, date)
                )
            ''')
            
            # Intraday data table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS intraday_data (
                    symbol TEXT,
                    datetime TIMESTAMP,
                    interval TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, datetime, interval)
                )
            ''')
            
            # Metadata table for tracking updates
            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_metadata (
                    symbol TEXT,
                    interval TEXT,
                    last_update TIMESTAMP,
                    last_date DATE,
                    record_count INTEGER,
                    PRIMARY KEY (symbol, interval)
                )
            ''')
            
            # Create indexes for faster queries
            conn.execute('CREATE INDEX IF NOT EXISTS idx_daily_symbol_date ON daily_data (symbol, date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_intraday_symbol_datetime ON intraday_data (symbol, datetime)')
            
            conn.commit()
    
    def _resolve_symbol(self, symbol: str) -> str:
        """
        Resolve symbol mapping for problematic symbols
        
        Args:
            symbol: Original symbol
            
        Returns:
            Mapped symbol for API calls
        """
        mapped = self.symbol_mapping.get(symbol, symbol)
        if mapped != symbol:
            logger.info(f"🔄 Mapping {symbol} -> {mapped}")
        return mapped
    
    def _supports_intraday(self, symbol: str) -> bool:
        """
        Check if symbol supports intraday data
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if intraday data is supported
        """
        return symbol not in self.intraday_unsupported
    
    def get_data(self, symbol: str, start_date: str = None, end_date: str = None, 
                interval: str = 'daily') -> pd.DataFrame:
        """
        Unified data retrieval - smart fallback from intraday to daily
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD) 
            interval: Data interval ('daily', '1min', '5min', '15min', '30min', '60min')
            
        Returns:
            DataFrame with market data indexed by datetime
        """
        
        logger.info(f"📊 Getting {interval} data for {symbol} from database")
        
        if interval == 'daily':
            # For daily data, first try to get from intraday (preferred) then fallback to daily table
            return self._get_daily_data_unified(symbol, start_date, end_date)
        else:
            # For intraday data, get directly from intraday table
            return self._get_intraday_data_direct(symbol, start_date, end_date, interval)
    
    def _get_daily_data_unified(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get daily data with smart fallback strategy:
        1. Try to derive from intraday data (more comprehensive)
        2. Fallback to daily_data table
        3. Update if needed
        """
        
        # First, check if we have intraday data for this symbol
        with sqlite3.connect(self.db_path) as conn:
            intraday_check = conn.execute(
                'SELECT COUNT(*) FROM intraday_data WHERE symbol = ? LIMIT 1',
                [symbol]
            ).fetchone()[0]
            
            if intraday_check > 0:
                # We have intraday data - derive daily from it
                logger.info(f"📈 Deriving daily data from intraday for {symbol}")
                return self._derive_daily_from_intraday(symbol, start_date, end_date)
            else:
                # No intraday data - use daily table or update
                logger.info(f"📊 Using daily table for {symbol}")
                return self._get_daily_data_direct(symbol, start_date, end_date)
    
    def _derive_daily_from_intraday(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Derive daily OHLCV data from intraday data"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Get all intraday data and aggregate to daily
            query = '''
                SELECT 
                    DATE(datetime) as date,
                    MIN(datetime) as first_time,
                    MAX(datetime) as last_time
                FROM intraday_data 
                WHERE symbol = ?
            '''
            params = [symbol]
            
            if start_date:
                query += ' AND DATE(datetime) >= ?'
                params.append(start_date)
            
            if end_date:
                query += ' AND DATE(datetime) <= ?'
                params.append(end_date)
            
            query += ' GROUP BY DATE(datetime) ORDER BY date'
            
            date_ranges = pd.read_sql_query(query, conn, params=params)
            
            if date_ranges.empty:
                logger.warning(f"⚠️ No intraday data found for {symbol}")
                return pd.DataFrame()
            
            # For each date, get OHLCV
            daily_data = []
            for _, row in date_ranges.iterrows():
                date_str = row['date']
                
                # Get OHLCV for this date
                ohlcv_query = '''
                    SELECT 
                        ? as date,
                        (SELECT open FROM intraday_data WHERE symbol = ? AND DATE(datetime) = ? ORDER BY datetime ASC LIMIT 1) as open,
                        MAX(high) as high,
                        MIN(low) as low,
                        (SELECT close FROM intraday_data WHERE symbol = ? AND DATE(datetime) = ? ORDER BY datetime DESC LIMIT 1) as close,
                        SUM(volume) as volume
                    FROM intraday_data 
                    WHERE symbol = ? AND DATE(datetime) = ?
                '''
                
                result = conn.execute(ohlcv_query, [date_str, symbol, date_str, symbol, date_str, symbol, date_str]).fetchone()
                
                if result and result[1] is not None:  # Ensure we have data
                    daily_data.append({
                        'date': result[0],
                        'open': result[1],
                        'high': result[2], 
                        'low': result[3],
                        'close': result[4],
                        'volume': result[5] or 0
                    })
            
            if daily_data:
                df = pd.DataFrame(daily_data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df['adj_close'] = df['close']  # For compatibility
                
                # Format columns to match the expected format (title case)
                df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Unadjusted_Close',
                    'adj_close': 'Close',
                    'volume': 'Volume'
                }, inplace=True)
                
                logger.info(f"✅ Derived {len(df)} daily records from intraday for {symbol}")
                return df
            else:
                return pd.DataFrame()
    
    def _get_daily_data_direct(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get data directly from daily_data table"""
        
        # Check if we need to update data first
        self._ensure_data_current(symbol, 'daily')
        
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT date, open, high, low, close, adj_close, volume
                FROM daily_data 
                WHERE symbol = ?
            '''
            params = [symbol]
            
            if start_date:
                query += ' AND date >= ?'
                params.append(start_date)
            
            if end_date:
                query += ' AND date <= ?'
                params.append(end_date)
            
            query += ' ORDER BY date'
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                logger.warning(f"⚠️ No daily data found for {symbol} in database")
                return pd.DataFrame()
            
            # Format daily data
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Unadjusted_Close',
                'adj_close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            logger.info(f"✅ Retrieved {len(df)} daily records for {symbol}")
            return df
    
    def _get_intraday_data_direct(self, symbol: str, start_date: str = None, end_date: str = None, interval: str = '15min') -> pd.DataFrame:
        """Get data directly from intraday_data table"""
        
        # Check if we need to update data first
        self._ensure_data_current(symbol, interval)
        
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT datetime, open, high, low, close, volume
                FROM intraday_data 
                WHERE symbol = ? AND interval = ?
            '''
            params = [symbol, interval]
            
            if start_date:
                query += ' AND date(datetime) >= ?'
                params.append(start_date)
            
            if end_date:
                query += ' AND date(datetime) <= ?'
                params.append(end_date)
            
            query += ' ORDER BY datetime'
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                logger.warning(f"⚠️ No {interval} data found for {symbol} in database")
                return pd.DataFrame()
            
            # Format intraday data
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low', 
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            logger.info(f"✅ Retrieved {len(df)} {interval} records for {symbol}")
            return df
    
    def update_daily_data(self, symbol: str, force_full_update: bool = False) -> bool:
        """
        Update daily data for a symbol
        
        Args:
            symbol: Stock symbol to update
            force_full_update: Whether to fetch all historical data
            
        Returns:
            True if update successful
        """
        
        logger.info(f"🔄 Updating daily data for {symbol}")
        
        try:
            # Resolve symbol mapping (e.g., VIX -> VXX)
            api_symbol = self._resolve_symbol(symbol)
            
            # Determine date range to fetch
            if force_full_update:
                # Fetch all available data
                fetch_size = 'full'
                logger.info(f"📅 Full update requested for {symbol}")
            else:
                # Check last update date
                last_date = self._get_last_update_date(symbol, 'daily')
                if last_date:
                    days_behind = (date.today() - last_date).days
                    if days_behind <= 1:
                        logger.info(f"✅ {symbol} daily data is current")
                        return True
                    else:
                        logger.info(f"📅 {symbol} is {days_behind} days behind, updating...")
                
                fetch_size = 'compact'  # Last 100 days
            
            # Fetch from API with rate limiting (only when actually calling API)
            data = self._fetch_daily_from_api(api_symbol, fetch_size)
            
            if data.empty:
                logger.error(f"❌ No data received for {symbol}")
                return False
            
            # Store in database
            self._store_daily_data(symbol, data)
            
            logger.info(f"✅ Updated {len(data)} daily records for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update {symbol}: {e}")
            return False
    
    def update_intraday_data(self, symbol: str, interval: str = '15min') -> bool:
        """
        Update intraday data for a symbol
        
        Args:
            symbol: Stock symbol
            interval: Intraday interval
            
        Returns:
            True if update successful
        """
        
        logger.info(f"🔄 Updating {interval} data for {symbol}")
        
        try:
            # Check if symbol supports intraday data
            if not self._supports_intraday(symbol):
                logger.warning(f"⚠️ {symbol} does not support intraday data")
                return False
            
            # Resolve symbol mapping (e.g., VIX -> VXX)
            api_symbol = self._resolve_symbol(symbol)
            
            # Fetch from API
            data = self._fetch_intraday_from_api(api_symbol, interval)
            
            if data.empty:
                logger.error(f"❌ No intraday data received for {symbol}")
                return False
            
            # Store in database
            self._store_intraday_data(symbol, data, interval)
            
            logger.info(f"✅ Updated {len(data)} intraday records for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update intraday {symbol}: {e}")
            return False
    
    def bulk_update_symbols(self, symbols: List[str], interval: str = 'daily') -> Dict[str, bool]:
        """
        Update multiple symbols with proper rate limiting
        
        Args:
            symbols: List of symbols to update
            interval: Data interval to update
            
        Returns:
            Dictionary of symbol -> success status
        """
        
        logger.info(f"🔄 Bulk updating {len(symbols)} symbols ({interval})")
        
        results = {}
        
        for i, symbol in enumerate(symbols):
            try:
                if interval == 'daily':
                    success = self.update_daily_data(symbol)
                else:
                    success = self.update_intraday_data(symbol, interval)
                
                results[symbol] = success
                
                # Rate limiting only when calling APIs
                if i < len(symbols) - 1:
                    logger.info(f"⏱️ Rate limiting (12s)...")
                    time.sleep(12)
                
            except Exception as e:
                logger.error(f"❌ Error updating {symbol}: {e}")
                results[symbol] = False
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"✅ Bulk update complete: {successful}/{len(symbols)} successful")
        
        return results
    
    def _ensure_data_current(self, symbol: str, interval: str):
        """Ensure data is current, update if needed"""
        
        last_date = self._get_last_update_date(symbol, interval)
        
        if not last_date:
            logger.info(f"📊 No existing data for {symbol} ({interval}), fetching now...")
            if interval == 'daily':
                self.update_daily_data(symbol, force_full_update=True)
            else:
                self.update_intraday_data(symbol, interval)
            return
        
        # Check if data is stale
        days_behind = (date.today() - last_date).days
        
        if interval == 'daily' and days_behind > 1:
            logger.info(f"🔄 {symbol} daily data is {days_behind} days stale, updating...")
            self.update_daily_data(symbol)
        elif interval != 'daily' and days_behind > 0:
            logger.info(f"🔄 {symbol} intraday data is stale, updating...")
            self.update_intraday_data(symbol, interval)
    
    def _get_last_update_date(self, symbol: str, interval: str) -> Optional[date]:
        """Get the last update date for a symbol"""
        
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute('''
                SELECT last_date FROM data_metadata 
                WHERE symbol = ? AND interval = ?
            ''', (symbol, interval)).fetchone()
            
            if result:
                return datetime.strptime(result[0], '%Y-%m-%d').date()
        
        return None
    
    def _fetch_daily_from_api(self, symbol: str, outputsize: str = 'compact') -> pd.DataFrame:
        """Fetch daily data from Alpha Vantage API"""
        
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': outputsize,
            'apikey': self.api_key
        }
        
        logger.info(f"🌐 API call: {symbol} daily data ({outputsize})")
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        
        if 'Note' in data:
            raise ValueError(f"API Rate Limited: {data['Note']}")
        
        # Parse time series data
        time_series_key = 'Time Series (Daily)'
        if time_series_key not in data:
            raise ValueError(f"Time series data not found. Available keys: {list(data.keys())}")
        
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        df_data = []
        for date_str, values in time_series.items():
            df_data.append({
                'date': date_str,
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'adj_close': float(values['5. adjusted close']),
                'volume': int(values['6. volume'])
            })
        
        df = pd.DataFrame(df_data)
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        
        return df
    
    def _fetch_intraday_from_api(self, symbol: str, interval: str) -> pd.DataFrame:
        """Fetch intraday data from Alpha Vantage API"""
        
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'apikey': self.api_key,
            'outputsize': 'full'
        }
        
        logger.info(f"🌐 API call: {symbol} {interval} intraday data")
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Check for errors
        if 'Error Message' in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        
        if 'Note' in data:
            raise ValueError(f"API Rate Limited: {data['Note']}")
        
        # Parse time series data
        time_series_key = f'Time Series ({interval})'
        if time_series_key not in data:
            raise ValueError(f"Time series data not found. Available keys: {list(data.keys())}")
        
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        df_data = []
        for datetime_str, values in time_series.items():
            df_data.append({
                'datetime': datetime_str,
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': int(values['5. volume'])
            })
        
        df = pd.DataFrame(df_data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.sort_values('datetime', inplace=True)
        
        return df
    
    def _store_daily_data(self, symbol: str, data: pd.DataFrame):
        """Store daily data in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Prepare data for insertion
            data_to_insert = data[['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']].copy()
            data_to_insert['symbol'] = symbol
            
            # Use INSERT OR REPLACE to handle duplicates
            data_to_insert.to_sql('daily_data_temp', conn, if_exists='replace', index=False)
            
            # Update main table
            conn.execute('''
                INSERT OR REPLACE INTO daily_data 
                (symbol, date, open, high, low, close, adj_close, volume)
                SELECT symbol, date, open, high, low, close, adj_close, volume
                FROM daily_data_temp
            ''')
            
            # Update metadata
            conn.execute('''
                INSERT OR REPLACE INTO data_metadata 
                (symbol, interval, last_update, last_date, record_count)
                VALUES (?, 'daily', CURRENT_TIMESTAMP, ?, ?)
            ''', (symbol, data['date'].max().strftime('%Y-%m-%d'), len(data)))
            
            # Clean up temp table
            conn.execute('DROP TABLE daily_data_temp')
            
            conn.commit()
    
    def _store_intraday_data(self, symbol: str, data: pd.DataFrame, interval: str):
        """Store intraday data in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Prepare data for insertion
            data_to_insert = data[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
            data_to_insert['symbol'] = symbol
            data_to_insert['interval'] = interval
            
            # Use INSERT OR REPLACE to handle duplicates  
            data_to_insert.to_sql('intraday_data_temp', conn, if_exists='replace', index=False)
            
            # Update main table
            conn.execute('''
                INSERT OR REPLACE INTO intraday_data 
                (symbol, datetime, interval, open, high, low, close, volume)
                SELECT symbol, datetime, interval, open, high, low, close, volume
                FROM intraday_data_temp
            ''')
            
            # Update metadata
            last_date = data['datetime'].max().date().strftime('%Y-%m-%d')
            conn.execute('''
                INSERT OR REPLACE INTO data_metadata 
                (symbol, interval, last_update, last_date, record_count)
                VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?)
            ''', (symbol, interval, last_date, len(data)))
            
            # Clean up temp table
            conn.execute('DROP TABLE intraday_data_temp')
            
            conn.commit()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Daily data stats
            daily_stats = conn.execute('''
                SELECT 
                    COUNT(DISTINCT symbol) as symbols,
                    COUNT(*) as total_records,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date
                FROM daily_data
            ''').fetchone()
            
            # Intraday data stats
            intraday_stats = conn.execute('''
                SELECT 
                    COUNT(DISTINCT symbol) as symbols,
                    COUNT(DISTINCT interval) as intervals,
                    COUNT(*) as total_records
                FROM intraday_data
            ''').fetchone()
            
            # All symbols from both tables (unified view)
            all_symbols = set()
            daily_symbols = {row[0] for row in conn.execute('SELECT DISTINCT symbol FROM daily_data')}
            intraday_symbols = {row[0] for row in conn.execute('SELECT DISTINCT symbol FROM intraday_data')}
            all_symbols = daily_symbols.union(intraday_symbols)
            
            # Get date range from both tables
            date_ranges = []
            if daily_stats[2]:  # If we have daily data
                date_ranges.append(daily_stats[2])
                date_ranges.append(daily_stats[3])
            
            intraday_dates = conn.execute('SELECT MIN(datetime), MAX(datetime) FROM intraday_data').fetchone()
            if intraday_dates[0]:  # If we have intraday data
                date_ranges.extend(intraday_dates)
            
            earliest_date = min(date_ranges) if date_ranges else None
            latest_date = max(date_ranges) if date_ranges else None
            
        return {
            'daily_data': {
                'symbols': daily_stats[0] or 0,
                'total_records': daily_stats[1] or 0,
                'earliest_date': daily_stats[2],
                'latest_date': daily_stats[3]
            },
            'intraday_data': {
                'symbols': intraday_stats[0] or 0,
                'intervals': intraday_stats[1] or 0,
                'total_records': intraday_stats[2] or 0
            },
            'unified_view': {
                'total_unique_symbols': len(all_symbols),
                'daily_only_symbols': len(daily_symbols - intraday_symbols),
                'intraday_only_symbols': len(intraday_symbols - daily_symbols),
                'both_tables_symbols': len(daily_symbols.intersection(intraday_symbols)),
                'earliest_date': earliest_date,
                'latest_date': latest_date
            },
            'symbols_in_database': sorted(list(all_symbols)),
            'symbols_breakdown': {
                'daily_only': sorted(list(daily_symbols - intraday_symbols)),
                'intraday_only': sorted(list(intraday_symbols - daily_symbols)),
                'both_tables': sorted(list(daily_symbols.intersection(intraday_symbols)))
            },
            'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
        }


# Daily update script for cron job
def daily_update_script():
    """Script to run daily for updating market data"""
    
    # Load configuration
    config_file = 'data_update_config.json'
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        symbols = config.get('symbols', ['SPY', 'QQQ', 'IWM'])
    else:
        # Default symbols
        symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'VTI', 'AAPL', 'MSFT', 'GOOGL']
        
        # Create config file
        with open(config_file, 'w') as f:
            json.dump({'symbols': symbols}, f, indent=2)
        
        logger.info(f"📄 Created config file: {config_file}")
    
    # Initialize database
    db = MarketDataDatabase()
    
    # Update all symbols
    logger.info(f"🔄 Daily update starting for {len(symbols)} symbols")
    results = db.bulk_update_symbols(symbols, 'daily')
    
    successful = sum(1 for success in results.values() if success)
    logger.info(f"✅ Daily update completed: {successful}/{len(symbols)} successful")
    
    # Print database stats
    stats = db.get_database_stats()
    logger.info(f"📊 Database stats: {stats}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Market Data Database Management")
    parser.add_argument('--daily-update', action='store_true', help='Run daily update')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--init-symbols', nargs='+', help='Initialize database with symbols')
    
    args = parser.parse_args()
    
    if args.daily_update:
        daily_update_script()
    elif args.stats:
        db = MarketDataDatabase()
        stats = db.get_database_stats()
        print(json.dumps(stats, indent=2, default=str))
    elif args.init_symbols:
        db = MarketDataDatabase()
        results = db.bulk_update_symbols(args.init_symbols, 'daily')
        print(f"Initialized {sum(results.values())}/{len(args.init_symbols)} symbols")
    else:
        print("Use --daily-update, --stats, or --init-symbols")
