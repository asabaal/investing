"""
Alpha Vantage API Client

This module provides a client for accessing Alpha Vantage APIs
for financial market data.
"""

import json
import logging
import os
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List

import pandas as pd

# Try to import fallback functions
try:
    from alpha_vantage_api_fallback import fallback_daily_adjusted
    HAS_FALLBACKS = True
except ImportError:
    HAS_FALLBACKS = False

# Configure logging
logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Manages API rate limits with persistent logging.
    
    Tracks API calls in a log file to ensure limits aren't exceeded
    across multiple program executions.
    """
    
    def __init__(self, log_dir='./logs', calls_per_minute=75, calls_per_day=None):
        """
        Initialize rate limiter.
        
        Args:
            log_dir: Directory for log files
            calls_per_minute: Maximum API calls per minute
            calls_per_day: Maximum API calls per day (None for unlimited)
        """
        self.log_dir = log_dir
        self.calls_per_minute = calls_per_minute
        self.calls_per_day = calls_per_day
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create or update .gitignore to exclude logs
        self._update_gitignore()
        
        # Log file path uses current date
        self.log_file = os.path.join(log_dir, f"api_calls_{datetime.now().strftime('%Y-%m-%d')}.json")
        
        # Initialize or load call history
        self._initialize_log()
    
    def _update_gitignore(self):
        """Ensure logs are excluded from git tracking."""
        gitignore_path = '.gitignore'
        log_pattern = f"{self.log_dir}/*.json"
        
        # Check if .gitignore exists and has our pattern
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r') as f:
                content = f.read()
                
            if log_pattern not in content:
                with open(gitignore_path, 'a') as f:
                    f.write(f"\n# API call logs\n{log_pattern}\n")
        else:
            # Create new .gitignore
            with open(gitignore_path, 'w') as f:
                f.write(f"# API call logs\n{log_pattern}\n")
    
    def _initialize_log(self):
        """Initialize or load call history."""
        today = datetime.now().strftime('%Y-%m-%d')
        self.log_file = os.path.join(self.log_dir, f"api_calls_{today}.json")
        
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                self.call_history = json.load(f)
        else:
            # Check if there's a log from a previous day and remove it
            for old_file in os.listdir(self.log_dir):
                if old_file.startswith('api_calls_') and old_file.endswith('.json'):
                    if old_file != os.path.basename(self.log_file):
                        try:
                            os.remove(os.path.join(self.log_dir, old_file))
                        except:
                            pass
            
            # Initialize new log
            self.call_history = []
            self._save_log()
    
    def _save_log(self):
        """Save call history to log file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.call_history, f)
    
    def check_rate_limit(self):
        """
        Check if we're within rate limits and wait if necessary.
        
        Returns:
            float: Time waited in seconds
        """
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        
        # Check if we need to create a new log for a new day
        if not os.path.exists(self.log_file) or not self.log_file.endswith(f"{today}.json"):
            self._initialize_log()
        
        # Get recent calls within the last minute
        one_minute_ago = (now - timedelta(minutes=1)).timestamp()
        recent_calls = [call for call in self.call_history if call > one_minute_ago]
        
        # Check daily limit if applicable
        if self.calls_per_day is not None:
            if len(self.call_history) >= self.calls_per_day:
                # We've hit the daily limit - calculate time until tomorrow
                tomorrow = datetime.combine(now.date() + timedelta(days=1), datetime.min.time())
                wait_time = (tomorrow - now).total_seconds()
                time.sleep(wait_time)
                
                # Reset for new day
                self._initialize_log()
                return wait_time
        
        # Check minute limit
        if len(recent_calls) >= self.calls_per_minute:
            # Calculate oldest timestamp in the last 75 calls
            if len(recent_calls) > 0:
                oldest = min(recent_calls)
                # Wait until we're below the limit
                wait_time = max(0, oldest + 60 - now.timestamp())
                
                if wait_time > 0:
                    time.sleep(wait_time)
                    return wait_time
        
        return 0
    
    def log_api_call(self):
        """Log an API call and update the history."""
        now = datetime.now().timestamp()
        self.call_history.append(now)
        self._save_log()
    
    def wait_if_needed(self):
        """Check rate limit and log an API call."""
        wait_time = self.check_rate_limit()
        self.log_api_call()
        return wait_time

class AlphaVantageClient:
    """
    Client for accessing Alpha Vantage API endpoints.
    
    This client provides methods for accessing various Alpha Vantage
    endpoints and handles data formatting, rate limiting, and caching.
    
    Attributes:
        api_key: Alpha Vantage API key
        base_url: Base URL for Alpha Vantage API
        last_request_time: Timestamp of last API request
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = 'https://www.alphavantage.co/query',
        premium: bool = True,
        use_fallbacks: bool = True
    ):
        """
        Initialize Alpha Vantage client.
        
        Args:
            api_key: Your Alpha Vantage API key
            base_url: Base URL for Alpha Vantage API
            premium: Whether you have a premium subscription
            use_fallbacks: Use fallback methods for premium endpoints
        """
        self.api_key = api_key
        self.base_url = base_url
        self.use_fallbacks = use_fallbacks and HAS_FALLBACKS
        
        # Set appropriate limits based on subscription
        if premium:
            # Premium tier with 75 calls per minute
            self.rate_limiter = RateLimiter(calls_per_minute=75)
        else:
            # Free tier with 5 calls per minute, 500 per day
            self.rate_limiter = RateLimiter(calls_per_minute=5, calls_per_day=500)
        
        # Keep track of last request time
        self.last_request_time = 0
        
        # Initialize cache
        self.cache = {}
    
    def _rate_limit(self):
        """Apply rate limiting before making a request."""
        wait_time = self.rate_limiter.wait_if_needed()
        if wait_time > 0:
            logger.info(f"Rate limit applied: waited {wait_time:.2f} seconds")
    
    def clean_data(self, data):
        """
        Clean data frame by handling missing values.
        
        Parameters:
        data (pandas.DataFrame): DataFrame to clean
        
        Returns:
        pandas.DataFrame: Cleaned DataFrame
        """
        if data is None or data.empty:
            logger.warning("Empty data provided for cleaning")
            return data
        
        cleaned = data.copy()
        
        # Handle missing volume data
        if 'volume' in cleaned.columns and cleaned['volume'].isna().any():
            # Log the issue
            logger.info(f"Fixing {cleaned['volume'].isna().sum()} missing volume values")
            
            # First try forward/backward fill
            cleaned['volume'] = cleaned['volume'].fillna(method='ffill').fillna(method='bfill')
            
            # If still have NaNs, use median of non-NaN values
            if cleaned['volume'].isna().any():
                median_volume = cleaned['volume'].median()
                cleaned['volume'] = cleaned['volume'].fillna(median_volume)
                logger.info(f"Used median volume {median_volume} to fill remaining NaNs")
        
        return cleaned
    
    def _make_request(self, params: Dict[str, str]) -> Dict:
        """
        Make a request to the Alpha Vantage API.
        
        Args:
            params: Dictionary of query parameters
        
        Returns:
            JSON response as dictionary
        
        Raises:
            Exception: If the API returns an error
        """
        # Add API key to params
        params['apikey'] = self.api_key
        
        # Debug log to verify API key usage
        logger.debug(f"Making request with API key: {params['apikey'][:4]}...")
        
        # Rate limit API calls
        self._rate_limit()
        
        # Make the request
        try:
            response = requests.get(self.base_url, params=params)
            
            # Handle errors
            if response.status_code != 200:
                error_msg = f"API request failed with status code {response.status_code}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Parse response
            data = response.json()
            
            # Check for error messages
            if 'Error Message' in data:
                error_msg = f"API returned error: {data['Error Message']}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Check for premium endpoint error
            if isinstance(data, dict) and 'Information' in data and 'premium endpoint' in data['Information']:
                error_msg = f"Premium endpoint required: {data['Information']}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Check for other information messages
            if 'Information' in data:
                error_msg = f"Unexpected response format: {data}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Check for empty response
            if not data:
                error_msg = "API returned empty response"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            return data
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            raise
    
    def _check_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Check if data is in cache.
        
        Args:
            cache_key: Key to check in cache
            
        Returns:
            Cached DataFrame or None if not in cache
        """
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        return None
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """
        Cache data for future use.
        
        Args:
            cache_key: Key to store in cache
            data: DataFrame to cache
        """
        self.cache[cache_key] = data.copy()
    
    def _parse_time_series(self, data: Dict, time_series_key: str) -> pd.DataFrame:
        """
        Parse time series data from Alpha Vantage response.
        
        Args:
            data: Response data from Alpha Vantage
            time_series_key: Key for time series data in response
            
        Returns:
            DataFrame with parsed time series data
        """
        if time_series_key not in data:
            error_msg = f"Expected {time_series_key} in response"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Convert column names (remove number prefixes)
        df.columns = [col.split('. ')[1] for col in df.columns]
        
        # Convert to numeric
        df = df.apply(pd.to_numeric)
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df = df.sort_index()
        
        # Rename columns to match our expected format
        column_map = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'adjusted close': 'adjusted_close',
            'volume': 'volume',
            'dividend amount': 'dividend_amount',
            'split coefficient': 'split_coefficient'
        }
        
        # Only rename columns that exist
        rename_map = {col: column_map[col] for col in df.columns if col in column_map}
        df = df.rename(columns=rename_map)
        
        return df
    
    def get_daily(
        self,
        symbol: str,
        outputsize: str = 'compact',
        datatype: str = 'json'
    ) -> pd.DataFrame:
        """
        Get daily time series for a symbol.
        
        Args:
            symbol: The stock symbol
            outputsize: 'compact' (100 data points) or 'full' (20+ years)
            datatype: 'json' or 'csv'
        
        Returns:
            DataFrame with daily OHLCV data
        """
        # Check cache first
        cache_key = f"{symbol}_TIME_SERIES_DAILY_ADJUSTED"
        cached_data = self._check_cache(cache_key)
        if cached_data is not None:
            logger.info(f"Retrieved cached data for {symbol}, function TIME_SERIES_DAILY_ADJUSTED")
            # Even if we use cached data, still clean it
            return self.clean_data(cached_data)
        
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': outputsize,
            'datatype': datatype
        }
        
        try:
            logger.info(f"Making API request for {symbol}, function TIME_SERIES_DAILY_ADJUSTED")
            data = self._make_request(params)
            df = self._parse_time_series(data, 'Time Series (Daily)')
            
            # Cache the data
            self._cache_data(cache_key, df)
            
            # Clean the data before returning
            return self.clean_data(df)
        except ValueError as e:
            # Check if this is a premium endpoint error and fallbacks are enabled
            if "premium endpoint" in str(e) and self.use_fallbacks:
                logger.warning(f"Using fallback for {symbol} due to premium endpoint restriction")
                try:
                    # Try fallback method
                    fallback_data = fallback_daily_adjusted(self, symbol, outputsize)
                    
                    # Cache the fallback data
                    self._cache_data(cache_key, fallback_data)
                    
                    # Clean and return the data
                    return self.clean_data(fallback_data)
                except Exception as fallback_error:
                    logger.error(f"Fallback failed for {symbol}: {str(fallback_error)}")
                    raise e
            else:
                raise
        except Exception as e:
            logger.error(f"Failed to get daily data for {symbol}: {str(e)}")
            raise
    
    def get_quote(self, symbol: str) -> Dict:
        """
        Get current quote for a symbol.
        
        Args:
            symbol: The stock symbol
        
        Returns:
            Dictionary with quote data
        """
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol
        }
        
        data = self._make_request(params)
        
        # Extract quote data
        if 'Global Quote' not in data:
            error_msg = f"Unexpected response format: {data}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        quote = data['Global Quote']
        
        # Create a more user-friendly dictionary
        return {
            'symbol': quote.get('01. symbol', ''),
            'open': float(quote.get('02. open', 0)),
            'high': float(quote.get('03. high', 0)),
            'low': float(quote.get('04. low', 0)),
            'price': float(quote.get('05. price', 0)),
            'volume': int(quote.get('06. volume', 0)),
            'latest trading day': quote.get('07. latest trading day', ''),
            'previous_close': float(quote.get('08. previous close', 0)),
            'change': float(quote.get('09. change', 0)),
            'change_percent': float(quote.get('10. change percent', '0%').strip('%'))
        }
    
    def get_rsi(
        self,
        symbol: str,
        interval: str = 'daily',
        time_period: int = 14,
        series_type: str = 'close'
    ) -> pd.DataFrame:
        """
        Get RSI technical indicator for a symbol.
        
        Args:
            symbol: The stock symbol
            interval: Time interval ('daily', 'weekly', 'monthly')
            time_period: Number of periods for calculation
            series_type: Price type ('close', 'open', 'high', 'low')
        
        Returns:
            DataFrame with RSI values
        """
        params = {
            'function': 'RSI',
            'symbol': symbol,
            'interval': interval,
            'time_period': str(time_period),
            'series_type': series_type
        }
        
        data = self._make_request(params)
        
        # Extract indicator data
        if 'Technical Analysis: RSI' not in data:
            error_msg = f"Unexpected response format: {data}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        indicator_data = data['Technical Analysis: RSI']
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(indicator_data, orient='index')
        
        # Rename column
        df.columns = ['RSI']
        
        # Convert to numeric
        df = df.apply(pd.to_numeric)
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df = df.sort_index()
        
        # Clean data before returning
        return self.clean_data(df)
    
    def get_macd(
        self,
        symbol: str,
        interval: str = 'daily',
        series_type: str = 'close',
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9
    ) -> pd.DataFrame:
        """
        Get MACD technical indicator for a symbol.
        
        Args:
            symbol: The stock symbol
            interval: Time interval ('daily', 'weekly', 'monthly')
            series_type: Price type ('close', 'open', 'high', 'low')
            fastperiod: Fast period for calculation
            slowperiod: Slow period for calculation
            signalperiod: Signal period for calculation
        
        Returns:
            DataFrame with MACD values
        """
        params = {
            'function': 'MACD',
            'symbol': symbol,
            'interval': interval,
            'series_type': series_type,
            'fastperiod': str(fastperiod),
            'slowperiod': str(slowperiod),
            'signalperiod': str(signalperiod)
        }
        
        data = self._make_request(params)
        
        # Extract indicator data
        if 'Technical Analysis: MACD' not in data:
            error_msg = f"Unexpected response format: {data}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        indicator_data = data['Technical Analysis: MACD']
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(indicator_data, orient='index')
        
        # Rename columns
        df.columns = ['MACD', 'MACD_Hist', 'MACD_Signal']
        
        # Convert to numeric
        df = df.apply(pd.to_numeric)
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df = df.sort_index()
        
        # Clean data before returning
        return self.clean_data(df)
    
    def get_sma(
        self,
        symbol: str,
        interval: str = 'daily',
        time_period: int = 50,
        series_type: str = 'close'
    ) -> pd.DataFrame:
        """
        Get Simple Moving Average for a symbol.
        
        Args:
            symbol: The stock symbol
            interval: Time interval ('daily', 'weekly', 'monthly')
            time_period: Number of periods for calculation
            series_type: Price type ('close', 'open', 'high', 'low')
        
        Returns:
            DataFrame with SMA values
        """
        params = {
            'function': 'SMA',
            'symbol': symbol,
            'interval': interval,
            'time_period': str(time_period),
            'series_type': series_type
        }
        
        data = self._make_request(params)
        
        # Extract indicator data
        if 'Technical Analysis: SMA' not in data:
            error_msg = f"Unexpected response format: {data}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        indicator_data = data['Technical Analysis: SMA']
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(indicator_data, orient='index')
        
        # Rename column
        df.columns = ['SMA']
        
        # Convert to numeric
        df = df.apply(pd.to_numeric)
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df = df.sort_index()
        
        # Clean data before returning
        return self.clean_data(df)
    
    def get_sector_performance(self) -> Dict:
        """
        Get sector performance data.
        
        Returns:
            Dictionary with sector performance data
        """
        params = {
            'function': 'SECTOR'
        }
        
        data = self._make_request(params)
        
        # Clean up metadata keys
        metadata_keys = [key for key in data.keys() if key.startswith('Meta')]
        for key in metadata_keys:
            del data[key]
        
        return data

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create client
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("Please set ALPHA_VANTAGE_API_KEY environment variable")
    else:
        client = AlphaVantageClient(api_key=api_key)
        
        # Get daily data
        df = client.get_daily('SPY')
        print(df.head())
        
        # Get quote
        quote = client.get_quote('SPY')
        print(quote)
