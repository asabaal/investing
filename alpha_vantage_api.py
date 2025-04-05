"""
Alpha Vantage API Integration Module

This module provides an interface to the Alpha Vantage API for retrieving
financial market data needed for evaluating Composer symphonies.
"""

import os
import time
import json
import logging
import requests
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import sqlite3

# Configure logging
logger = logging.getLogger(__name__)

class AlphaVantageClient:
    """
    Client for interacting with Alpha Vantage API.
    
    This class provides methods to fetch various types of market data from 
    Alpha Vantage, with built-in rate limiting and caching.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: str = 'https://www.alphavantage.co/query',
        cache_dir: str = './cache',
        db_path: str = 'trading_data.db',
        max_requests_per_minute: int = 5
    ):
        """
        Initialize AlphaVantageClient.
        
        Args:
            api_key: Alpha Vantage API key (can also be set via ALPHA_VANTAGE_API_KEY env var)
            base_url: Base URL for Alpha Vantage API
            cache_dir: Directory to store cached data
            db_path: Path to SQLite database for caching
            max_requests_per_minute: Rate limit for API requests
        """
        self.api_key = api_key or os.environ.get('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided or set as ALPHA_VANTAGE_API_KEY environment variable")
            
        self.base_url = base_url
        self.cache_dir = cache_dir
        self.db_path = db_path
        self.max_requests_per_minute = max_requests_per_minute
        self.last_request_time = 0
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize database for more efficient caching
        self._init_db()
        
    def _init_db(self):
        """Initialize database tables for caching."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create alpha_vantage_cache table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alpha_vantage_cache (
                request_key TEXT PRIMARY KEY,
                data TEXT,
                timestamp TEXT
            )
        ''')
        
        # Create market_data_sources table to track data provenance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data_sources (
                symbol TEXT,
                data_type TEXT,
                source TEXT,
                last_updated TEXT,
                PRIMARY KEY (symbol, data_type)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _rate_limit(self):
        """
        Implement rate limiting to avoid API request limits.
        """
        elapsed = time.time() - self.last_request_time
        min_interval = 60.0 / self.max_requests_per_minute
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
            
        self.last_request_time = time.time()
        
    def _get_cache_key(self, function: str, symbol: str, **params) -> str:
        """
        Generate a unique cache key for the request.
        """
        param_str = json.dumps(params, sort_keys=True)
        return f"{function}_{symbol}_{param_str}"
        
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """
        Retrieve data from cache if available and not expired.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT data, timestamp FROM alpha_vantage_cache WHERE request_key = ?",
            (cache_key,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            data_str, timestamp_str = result
            timestamp = datetime.fromisoformat(timestamp_str)
            
            # Check if cache is expired (24 hours for daily data)
            if datetime.now() - timestamp < timedelta(hours=24):
                return json.loads(data_str)
                
        return None
        
    def _save_to_cache(self, cache_key: str, data: Dict):
        """
        Save API response to cache.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT OR REPLACE INTO alpha_vantage_cache (request_key, data, timestamp) VALUES (?, ?, ?)",
            (cache_key, json.dumps(data), datetime.now().isoformat())
        )
        
        conn.commit()
        conn.close()
        
    def _update_data_source(self, symbol: str, data_type: str):
        """
        Update market data source tracking.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT OR REPLACE INTO market_data_sources 
            (symbol, data_type, source, last_updated)
            VALUES (?, ?, 'alpha_vantage', ?)
            """,
            (symbol, data_type, datetime.now().isoformat())
        )
        
        conn.commit()
        conn.close()
        
    def _make_request(self, params: Dict) -> Dict:
        """
        Make an API request to Alpha Vantage.
        """
        # Add API key to parameters
        params['apikey'] = self.api_key
        
        # Check cache first
        function = params.get('function', '')
        symbol = params.get('symbol', '')
        cache_params = {k: v for k, v in params.items() if k not in ['function', 'symbol', 'apikey']}
        cache_key = self._get_cache_key(function, symbol, **cache_params)
        
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            logger.info(f"Retrieved cached data for {symbol}, function {function}")
            return cached_data
        
        # Apply rate limiting
        self._rate_limit()
        
        # Make the request
        logger.info(f"Making API request for {symbol}, function {function}")
        response = requests.get(self.base_url, params=params)
        
        if response.status_code != 200:
            raise RuntimeError(f"API request failed with status code {response.status_code}: {response.text}")
        
        # Parse the response
        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(f"Failed to parse JSON response: {response.text}")
        
        # Check for API error messages
        if "Error Message" in data:
            raise RuntimeError(f"API returned error: {data['Error Message']}")
        
        # Cache the response
        self._save_to_cache(cache_key, data)
        self._update_data_source(symbol, function)
        
        return data
    
    def get_daily(self, symbol: str, outputsize: str = 'compact', datatype: str = 'json') -> pd.DataFrame:
        """
        Get daily time series data for a symbol.
        
        Args:
            symbol: Stock symbol
            outputsize: 'compact' (100 days) or 'full' (20+ years)
            datatype: 'json' or 'csv'
            
        Returns:
            DataFrame with daily time series data
        """
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': outputsize,
            'datatype': datatype
        }
        
        data = self._make_request(params)
        
        # Convert to DataFrame
        if "Time Series (Daily)" in data:
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index(ascending=True)
            
            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            # Rename columns
            df = df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. adjusted close': 'adjusted_close',
                '6. volume': 'volume',
                '7. dividend amount': 'dividend_amount',
                '8. split coefficient': 'split_coefficient'
            })
            
            return df
        else:
            logger.error(f"Unexpected response format: {data}")
            raise RuntimeError(f"Unexpected response format: {data}")
    
    def get_intraday(self, symbol: str, interval: str = '5min', outputsize: str = 'compact') -> pd.DataFrame:
        """
        Get intraday time series data for a symbol.
        
        Args:
            symbol: Stock symbol
            interval: '1min', '5min', '15min', '30min', or '60min'
            outputsize: 'compact' or 'full'
            
        Returns:
            DataFrame with intraday time series data
        """
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'outputsize': outputsize
        }
        
        data = self._make_request(params)
        
        # Convert to DataFrame
        key = f"Time Series ({interval})"
        if key in data:
            df = pd.DataFrame.from_dict(data[key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index(ascending=True)
            
            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            # Rename columns
            df = df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            })
            
            return df
        else:
            logger.error(f"Unexpected response format: {data}")
            raise RuntimeError(f"Unexpected response format: {data}")
    
    def get_weekly(self, symbol: str) -> pd.DataFrame:
        """
        Get weekly time series data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with weekly time series data
        """
        params = {
            'function': 'TIME_SERIES_WEEKLY_ADJUSTED',
            'symbol': symbol
        }
        
        data = self._make_request(params)
        
        # Convert to DataFrame
        if "Weekly Adjusted Time Series" in data:
            df = pd.DataFrame.from_dict(data["Weekly Adjusted Time Series"], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index(ascending=True)
            
            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            # Rename columns
            df = df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. adjusted close': 'adjusted_close',
                '6. volume': 'volume',
                '7. dividend amount': 'dividend_amount'
            })
            
            return df
        else:
            logger.error(f"Unexpected response format: {data}")
            raise RuntimeError(f"Unexpected response format: {data}")
    
    def get_monthly(self, symbol: str) -> pd.DataFrame:
        """
        Get monthly time series data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with monthly time series data
        """
        params = {
            'function': 'TIME_SERIES_MONTHLY_ADJUSTED',
            'symbol': symbol
        }
        
        data = self._make_request(params)
        
        # Convert to DataFrame
        if "Monthly Adjusted Time Series" in data:
            df = pd.DataFrame.from_dict(data["Monthly Adjusted Time Series"], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index(ascending=True)
            
            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            # Rename columns
            df = df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. adjusted close': 'adjusted_close',
                '6. volume': 'volume',
                '7. dividend amount': 'dividend_amount'
            })
            
            return df
        else:
            logger.error(f"Unexpected response format: {data}")
            raise RuntimeError(f"Unexpected response format: {data}")
    
    def get_quote(self, symbol: str) -> Dict:
        """
        Get current quote for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with quote data
        """
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol
        }
        
        data = self._make_request(params)
        
        if "Global Quote" in data:
            return data["Global Quote"]
        else:
            logger.error(f"Unexpected response format: {data}")
            raise RuntimeError(f"Unexpected response format: {data}")
    
    def get_symbols_batch(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Get daily data for multiple symbols efficiently.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        result = {}
        
        for symbol in symbols:
            try:
                result[symbol] = self.get_daily(symbol)
            except Exception as e:
                logger.error(f"Failed to get data for {symbol}: {str(e)}")
        
        return result
    
    def get_sma(self, symbol: str, interval: str = 'daily', time_period: int = 20, series_type: str = 'close') -> pd.DataFrame:
        """
        Get Simple Moving Average (SMA) values for a symbol.
        
        Args:
            symbol: Stock symbol
            interval: 'daily', 'weekly', 'monthly'
            time_period: Number of periods
            series_type: 'close', 'open', 'high', 'low'
            
        Returns:
            DataFrame with SMA values
        """
        params = {
            'function': 'SMA',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            'series_type': series_type
        }
        
        data = self._make_request(params)
        
        # Convert to DataFrame
        if "Technical Analysis: SMA" in data:
            df = pd.DataFrame.from_dict(data["Technical Analysis: SMA"], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index(ascending=True)
            
            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            return df
        else:
            logger.error(f"Unexpected response format: {data}")
            raise RuntimeError(f"Unexpected response format: {data}")
    
    def get_rsi(self, symbol: str, interval: str = 'daily', time_period: int = 14, series_type: str = 'close') -> pd.DataFrame:
        """
        Get Relative Strength Index (RSI) values for a symbol.
        
        Args:
            symbol: Stock symbol
            interval: 'daily', 'weekly', 'monthly'
            time_period: Number of periods
            series_type: 'close', 'open', 'high', 'low'
            
        Returns:
            DataFrame with RSI values
        """
        params = {
            'function': 'RSI',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            'series_type': series_type
        }
        
        data = self._make_request(params)
        
        # Convert to DataFrame
        if "Technical Analysis: RSI" in data:
            df = pd.DataFrame.from_dict(data["Technical Analysis: RSI"], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index(ascending=True)
            
            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            return df
        else:
            logger.error(f"Unexpected response format: {data}")
            raise RuntimeError(f"Unexpected response format: {data}")
    
    def get_macd(self, symbol: str, interval: str = 'daily', series_type: str = 'close', 
               fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Get MACD values for a symbol.
        
        Args:
            symbol: Stock symbol
            interval: 'daily', 'weekly', 'monthly'
            series_type: 'close', 'open', 'high', 'low'
            fast_period: Fast period
            slow_period: Slow period
            signal_period: Signal period
            
        Returns:
            DataFrame with MACD values
        """
        params = {
            'function': 'MACD',
            'symbol': symbol,
            'interval': interval,
            'series_type': series_type,
            'fastperiod': fast_period,
            'slowperiod': slow_period,
            'signalperiod': signal_period
        }
        
        data = self._make_request(params)
        
        # Convert to DataFrame
        if "Technical Analysis: MACD" in data:
            df = pd.DataFrame.from_dict(data["Technical Analysis: MACD"], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index(ascending=True)
            
            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            return df
        else:
            logger.error(f"Unexpected response format: {data}")
            raise RuntimeError(f"Unexpected response format: {data}")
    
    def get_bbands(self, symbol: str, interval: str = 'daily', time_period: int = 20, 
                 series_type: str = 'close', nbdevup: int = 2, nbdevdn: int = 2) -> pd.DataFrame:
        """
        Get Bollinger Bands values for a symbol.
        
        Args:
            symbol: Stock symbol
            interval: 'daily', 'weekly', 'monthly'
            time_period: Number of periods
            series_type: 'close', 'open', 'high', 'low'
            nbdevup: Standard deviations above the mean
            nbdevdn: Standard deviations below the mean
            
        Returns:
            DataFrame with Bollinger Bands values
        """
        params = {
            'function': 'BBANDS',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            'series_type': series_type,
            'nbdevup': nbdevup,
            'nbdevdn': nbdevdn
        }
        
        data = self._make_request(params)
        
        # Convert to DataFrame
        if "Technical Analysis: BBANDS" in data:
            df = pd.DataFrame.from_dict(data["Technical Analysis: BBANDS"], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index(ascending=True)
            
            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            return df
        else:
            logger.error(f"Unexpected response format: {data}")
            raise RuntimeError(f"Unexpected response format: {data}")
    
    def get_atr(self, symbol: str, interval: str = 'daily', time_period: int = 14) -> pd.DataFrame:
        """
        Get Average True Range (ATR) values for a symbol.
        
        Args:
            symbol: Stock symbol
            interval: 'daily', 'weekly', 'monthly'
            time_period: Number of periods
            
        Returns:
            DataFrame with ATR values
        """
        params = {
            'function': 'ATR',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period
        }
        
        data = self._make_request(params)
        
        # Convert to DataFrame
        if "Technical Analysis: ATR" in data:
            df = pd.DataFrame.from_dict(data["Technical Analysis: ATR"], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index(ascending=True)
            
            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            return df
        else:
            logger.error(f"Unexpected response format: {data}")
            raise RuntimeError(f"Unexpected response format: {data}")

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create client instance
    client = AlphaVantageClient()
    
    # Get daily data for a symbol
    try:
        df = client.get_daily('AAPL')
        print(f"Got {len(df)} days of data for AAPL")
        print(df.head())
    except Exception as e:
        logger.error(f"Error getting daily data: {str(e)}")
    
    # Get current quote
    try:
        quote = client.get_quote('MSFT')
        print(f"MSFT quote: {quote}")
    except Exception as e:
        logger.error(f"Error getting quote: {str(e)}")
