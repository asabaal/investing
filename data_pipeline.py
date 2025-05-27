"""
Market Data Pipeline

Clean interface for fetching market data from various sources.
Supports both daily and intraday data for symphony backtesting and execution.
"""

import os
import pandas as pd
import numpy as np
import requests
import time
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import json
from dataclasses import dataclass

@dataclass
class DataConfig:
    """Configuration for data fetching"""
    source: str = 'alpha_vantage'
    api_key: str = None
    cache_dir: str = './data_cache'
    rate_limit_delay: float = 12.0  # seconds between API calls
    
class MarketDataPipeline:
    """Unified interface for fetching market data"""
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        
        # Set API key from environment if not provided
        if not self.config.api_key:
            self.config.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            
        if not self.config.api_key:
            raise ValueError("API key required. Set ALPHA_VANTAGE_API_KEY environment variable.")
        
        # Create cache directory
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        self.last_api_call = 0
    
    def get_daily_data(self, symbol: str, start_date: str = None, end_date: str = None, 
                      use_cache: bool = True) -> pd.DataFrame:
        """
        Get daily OHLCV data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date (YYYY-MM-DD) or None for all available
            end_date: End date (YYYY-MM-DD) or None for latest
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data indexed by date
        """
        
        cache_file = os.path.join(self.config.cache_dir, f"{symbol}_daily.csv")
        
        # Check cache first
        if use_cache and os.path.exists(cache_file):
            try:
                cached_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                
                # Check if cache is recent (less than 1 day old)
                cache_age = time.time() - os.path.getmtime(cache_file)
                if cache_age < 86400:  # 24 hours
                    print(f"Using cached data for {symbol}")
                    return self._filter_date_range(cached_data, start_date, end_date)
            except Exception as e:
                print(f"Error reading cache for {symbol}: {e}")
        
        # Fetch fresh data
        print(f"Fetching daily data for {symbol}...")
        data = self._fetch_alpha_vantage_daily(symbol)
        
        if data is not None and not data.empty:
            # Cache the data
            try:
                data.to_csv(cache_file)
                print(f"Cached data for {symbol}")
            except Exception as e:
                print(f"Error caching data for {symbol}: {e}")
            
            return self._filter_date_range(data, start_date, end_date)
        
        return pd.DataFrame()
    
    def get_intraday_data(self, symbol: str, interval: str = '15min', 
                         extended_hours: bool = False) -> pd.DataFrame:
        """
        Get intraday data for a symbol
        
        Args:
            symbol: Stock symbol
            interval: '1min', '5min', '15min', '30min', '60min'
            extended_hours: Include extended hours trading
            
        Returns:
            DataFrame with intraday OHLCV data
        """
        
        print(f"Fetching intraday data for {symbol} ({interval})...")
        return self._fetch_alpha_vantage_intraday(symbol, interval, extended_hours)
    
    def get_multiple_symbols(self, symbols: List[str], data_type: str = 'daily', 
                           **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple symbols with rate limiting
        
        Args:
            symbols: List of stock symbols
            data_type: 'daily' or 'intraday'
            **kwargs: Additional arguments for data fetching
            
        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        
        data_dict = {}
        
        for i, symbol in enumerate(symbols):
            try:
                if data_type == 'daily':
                    data = self.get_daily_data(symbol, **kwargs)
                elif data_type == 'intraday':
                    data = self.get_intraday_data(symbol, **kwargs)
                else:
                    raise ValueError(f"Unknown data_type: {data_type}")
                
                if not data.empty:
                    data_dict[symbol] = data
                    print(f"✓ {symbol}: {len(data)} records")
                else:
                    print(f"✗ {symbol}: No data received")
                
                # Rate limiting (except for last symbol)
                if i < len(symbols) - 1:
                    self._rate_limit()
                    
            except Exception as e:
                print(f"✗ {symbol}: Error - {e}")
                continue
        
        return data_dict
    
    def _fetch_alpha_vantage_daily(self, symbol: str) -> pd.DataFrame:
        """Fetch daily data from Alpha Vantage API"""
        
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': 'full',  # Get full history
            'apikey': self.config.api_key
        }
        
        self._rate_limit()
        
        try:
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
                available_keys = list(data.keys())
                raise ValueError(f"Time series data not found. Available keys: {available_keys}")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for date_str, values in time_series.items():
                df_data.append({
                    'Date': pd.to_datetime(date_str),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Adj_Close': float(values['5. adjusted close']),
                    'Volume': int(values['6. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Use adjusted close as the main close price
            df['Close'] = df['Adj_Close']
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching {symbol}: {e}")
            return pd.DataFrame()
        except (KeyError, ValueError, TypeError) as e:
            print(f"Data parsing error for {symbol}: {e}")
            return pd.DataFrame()
    
    def _fetch_alpha_vantage_intraday(self, symbol: str, interval: str, 
                                    extended_hours: bool) -> pd.DataFrame:
        """Fetch intraday data from Alpha Vantage API"""
        
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'apikey': self.config.api_key,
            'extended_hours': 'true' if extended_hours else 'false',
            'outputsize': 'full'
        }
        
        self._rate_limit()
        
        try:
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
                available_keys = list(data.keys())
                raise ValueError(f"Time series data not found. Available keys: {available_keys}")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for datetime_str, values in time_series.items():
                df_data.append({
                    'Datetime': pd.to_datetime(datetime_str),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': int(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Datetime', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching intraday {symbol}: {e}")
            return pd.DataFrame()
        except (KeyError, ValueError, TypeError) as e:
            print(f"Data parsing error for intraday {symbol}: {e}")
            return pd.DataFrame()
    
    def _filter_date_range(self, data: pd.DataFrame, start_date: str = None, 
                          end_date: str = None) -> pd.DataFrame:
        """Filter DataFrame by date range"""
        
        if data.empty:
            return data
        
        filtered_data = data.copy()
        
        if start_date:
            filtered_data = filtered_data[filtered_data.index >= start_date]
        
        if end_date:
            filtered_data = filtered_data[filtered_data.index <= end_date]
        
        return filtered_data
    
    def _rate_limit(self):
        """Implement rate limiting for API calls"""
        
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.config.rate_limit_delay:
            sleep_time = self.config.rate_limit_delay - time_since_last_call
            print(f"Rate limiting: waiting {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()


class SymphonyDataManager:
    """High-level data management for symphonies"""
    
    def __init__(self, data_pipeline: MarketDataPipeline = None):
        self.pipeline = data_pipeline or MarketDataPipeline()
    
    def prepare_symphony_data(self, symphony_config: dict, start_date: str = None, 
                            end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Prepare all data needed for a symphony
        
        Args:
            symphony_config: Symphony configuration dictionary
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping symbol -> DataFrame with market data
        """
        
        universe = symphony_config.get('universe', [])
        
        if not universe:
            raise ValueError("No universe defined in symphony configuration")
        
        print(f"Preparing data for symphony: {symphony_config.get('name', 'Unknown')}")
        print(f"Universe: {universe}")
        
        # Fetch data for all symbols
        market_data = self.pipeline.get_multiple_symbols(
            symbols=universe,
            data_type='daily',
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        
        # Validate we have sufficient data
        min_required_days = 100  # Minimum for most metrics
        valid_symbols = []
        
        for symbol, data in market_data.items():
            if len(data) >= min_required_days:
                valid_symbols.append(symbol)
            else:
                print(f"Warning: {symbol} has only {len(data)} days of data (min {min_required_days})")
        
        print(f"Valid symbols with sufficient data: {valid_symbols}")
        
        return {symbol: data for symbol, data in market_data.items() if symbol in valid_symbols}


# Utility functions for data analysis
def analyze_data_quality(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Analyze quality of fetched data"""
    
    analysis = []
    
    for symbol, data in data_dict.items():
        if data.empty:
            analysis.append({
                'Symbol': symbol,
                'Records': 0,
                'Date_Range': 'No Data',
                'Missing_Values': 'N/A',
                'Data_Quality': 'FAIL'
            })
            continue
        
        missing_values = data.isnull().sum().sum()
        date_range = f"{data.index.min().date()} to {data.index.max().date()}"
        
        # Quality score based on completeness and recency
        days_old = (pd.Timestamp.now() - data.index.max()).days
        completeness = 1 - (missing_values / (len(data) * len(data.columns)))
        recency_score = max(0, 1 - days_old / 30)  # Penalize data older than 30 days
        
        quality_score = (completeness * 0.7 + recency_score * 0.3)
        
        if quality_score > 0.8:
            quality = 'GOOD'
        elif quality_score > 0.6:
            quality = 'OK'
        else:
            quality = 'POOR'
        
        analysis.append({
            'Symbol': symbol,
            'Records': len(data),
            'Date_Range': date_range,
            'Missing_Values': missing_values,
            'Days_Old': days_old,
            'Quality_Score': f"{quality_score:.2f}",
            'Data_Quality': quality
        })
    
    return pd.DataFrame(analysis)


# Example usage and testing
if __name__ == "__main__":
    
    print("Market Data Pipeline Test")
    print("=" * 50)
    
    # Test configuration
    config = DataConfig(
        rate_limit_delay=1.0  # Faster for testing
    )
    
    pipeline = MarketDataPipeline(config)
    
    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'SPY']
    
    print(f"Testing with symbols: {test_symbols}")
    print("-" * 30)
    
    # Test daily data fetching
    try:
        data_dict = pipeline.get_multiple_symbols(
            symbols=test_symbols,
            data_type='daily',
            start_date='2024-01-01',
            use_cache=True
        )
        
        print("\nData Quality Analysis:")
        quality_df = analyze_data_quality(data_dict)
        print(quality_df.to_string(index=False))
        
        # Test symphony data manager
        print("\n" + "=" * 50)
        print("Testing Symphony Data Manager")
        
        sample_symphony = {
            'name': 'Test Symphony',
            'universe': test_symbols
        }
        
        data_manager = SymphonyDataManager(pipeline)
        symphony_data = data_manager.prepare_symphony_data(
            sample_symphony,
            start_date='2024-01-01'
        )
        
        print(f"\nSymphony data prepared for {len(symphony_data)} symbols")
        
        for symbol, data in symphony_data.items():
            print(f"{symbol}: {len(data)} records from {data.index.min().date()} to {data.index.max().date()}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
