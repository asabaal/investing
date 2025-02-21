import pandas as pd
import sqlite3
import yfinance as yf

import logging
import requests
import time

from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, Tuple, Union, List

class StockDataSource(ABC):
    """Abstract base class for stock data sources"""
    
    @abstractmethod
    def fetch_data(self, symbol: str, start_date: Optional[datetime] = None, 
                  end_date: Optional[datetime] = None, period: Optional[str] = None) -> pd.DataFrame:
        """Fetch data for a given symbol"""
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Get the name of this data source"""
        pass
    
    @abstractmethod
    def can_fetch_data(self) -> bool:
        """Check if this source is currently able to fetch data"""
        pass

class YFinanceSource(StockDataSource):
    """YFinance implementation of stock data source"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_source_name(self) -> str:
        return "yfinance"
    
    def can_fetch_data(self) -> bool:
        return True  # YFinance doesn't have API limits
    
    def fetch_data(self, symbols: Union[str, List[str]], start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None, period: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data for one or more symbols
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        all_data = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data, used_period = self._get_history_with_fallback(ticker, period, start_date, end_date)
                
                if not data.empty:
                    # Standardize the data format
                    standardized_data = self._standardize_data(data, symbol)
                    all_data.append(standardized_data)
                    
            except Exception as e:
                self.logger.error(f"Error fetching YFinance data for {symbol}: {str(e)}")
                continue
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        
        return pd.DataFrame()
    
    def _get_history_with_fallback(self, ticker: yf.Ticker, period: Optional[str] = None,
                                 start: Optional[datetime] = None, 
                                 end: Optional[datetime] = None) -> Tuple[pd.DataFrame, Optional[str]]:
        """Implementation of YFinance's fallback strategy"""
        if period:
            data = ticker.history(period=period)
            if not data.empty:
                return data, period

        if start:
            data = ticker.history(start=start, end=end)
            if not data.empty:
                return data, None

        fallback_periods = ['5Y', '1Y', '6M', '1M', '5D', '1D']
        for fallback_period in fallback_periods:
            try:
                data = ticker.history(period=fallback_period)
                if not data.empty:
                    self.logger.info(f"Used fallback period {fallback_period} for {ticker.ticker}")
                    return data, fallback_period
            except Exception as e:
                self.logger.debug(f"Fallback {fallback_period} failed for {ticker.ticker}: {e}")
                continue

        return pd.DataFrame(), None
    
    def _standardize_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Standardize YFinance data format"""
        df = data.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df['Symbol'] = symbol
        
        # Ensure all required columns exist
        if 'Dividends' not in df.columns:
            df['Dividends'] = 0
        if 'Stock Splits' not in df.columns:
            df['Stock_Splits'] = 0
        
        # Rename columns to standard format
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Dividends': 'dividends',
            'Stock Splits': 'stock_splits'
        })
        
        df['data_source'] = self.get_source_name()
        df['last_updated'] = datetime.now()
        
        return df

class AlphaVantageSource(StockDataSource):
    """Alpha Vantage implementation of stock data source"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.daily_limit = 25
        self.calls_made_today = 0
        self.last_call_date = None
        self.logger = logging.getLogger(__name__)

    def get_source_name(self) -> str:
        return "alpha_vantage"
    
    def can_fetch_data(self) -> bool:
        today = datetime.now().date()
        
        # Reset counter if it's a new day
        if self.last_call_date != today:
            self.calls_made_today = 0
            self.last_call_date = today
        
        return self.calls_made_today < self.daily_limit
    
    def fetch_data(self, symbol: str, start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None, period: Optional[str] = None) -> pd.DataFrame:
        if not self.can_fetch_data():
            self.logger.warning("Alpha Vantage API limit reached for today")
            return pd.DataFrame()
        
        try:
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "full"
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if "Time Series (Daily)" not in data:
                self.logger.error(f"Error getting Alpha Vantage data for {symbol}: {data.get('Note', 'Unknown error')}")
                return pd.DataFrame()
            
            # Update API call counter
            self.calls_made_today += 1
            
            # Convert to DataFrame and standardize
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            df = self._standardize_data(df, symbol)
            
            # Filter by date range if specified
            if start_date:
                df = df[df['date'] >= start_date]
            if end_date:
                df = df[df['date'] <= end_date]
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching Alpha Vantage data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _standardize_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Standardize Alpha Vantage data format"""
        df = data.copy()
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. adjusted close': 'adjusted_close',
            '6. volume': 'volume',
            '7. dividend amount': 'dividends'
        })
        
        df = df.reset_index()
        df = df.rename(columns={'index': 'date'})
        df['date'] = df['date'].dt.date
        df['symbol'] = symbol
        df['stock_splits'] = 0  # Alpha Vantage doesn't provide split data
        df['data_source'] = self.get_source_name()
        df['last_updated'] = datetime.now()
        
        return df

class StockDataFetcher:
    """Enhanced version of StockDataFetcher that supports multiple data sources"""
    
    def __init__(self, db_path: str = 'stock_data.db', av_api_key: Optional[str] = None):
        self.db_path = db_path
        self.timeout = 30.0
        self.logger = logging.getLogger(__name__)
        
        # Initialize data sources
        self.sources: List[StockDataSource] = [
            YFinanceSource()
        ]
        
        if av_api_key:
            self.sources.append(AlphaVantageSource(av_api_key))
        
        self.setup_database()
    
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        try:
            yield conn
        finally:
            conn.close()
    
    def setup_database(self):
        """Create database tables if they don't exist"""
        try:
            with self.get_db_connection() as conn:
                c = conn.cursor()
                
                c.execute('''
                    CREATE TABLE IF NOT EXISTS daily_prices (
                        date DATE,
                        symbol TEXT,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER,
                        dividends REAL,
                        stock_splits REAL,
                        data_sources TEXT,  -- Changed from data_source to data_sources
                        last_updated TIMESTAMP,
                        PRIMARY KEY (date, symbol)
                    )
                ''')
                
                # Add indices for better performance
                c.execute('CREATE INDEX IF NOT EXISTS idx_symbol_date ON daily_prices(symbol, date)')
                c.execute('CREATE INDEX IF NOT EXISTS idx_symbol_sources ON daily_prices(symbol, data_sources)')
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error setting up database: {str(e)}")
            raise

    def fetch_data(self, symbols: Union[str, List[str]], start_date: Optional[str] = None,
                end_date: Optional[str] = None, period: Optional[str] = None,
                verify_across_sources: bool = False, replace: bool = False) -> None:
        """
        Fetch data using all available sources
        
        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date for data fetch
            end_date: End date for data fetch
            period: Time period for data fetch
            verify_across_sources: Whether to fetch from all sources for verification
            replace: If True, removes existing data before inserting new data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        try:
            if not replace and not start_date:
                # Get latest date in database for these symbols
                with self.get_db_connection() as conn:
                    placeholders = ','.join(['?' for _ in symbols])
                    query = f"""
                        SELECT symbol, MAX(date) as last_date, GROUP_CONCAT(data_sources) as sources
                        FROM daily_prices 
                        WHERE symbol IN ({placeholders})
                        GROUP BY symbol
                    """
                    last_dates = pd.read_sql_query(query, conn, params=symbols)
                    
                    if not last_dates.empty:
                        min_date = pd.to_datetime(last_dates['last_date'].min()) + timedelta(days=1)
                        if min_date.date() >= datetime.now().date():
                            self.logger.info(f"Data already up to date for batch: {symbols}")
                            return
                        start_date = min_date
            
            dfs = []
            used_sources = set()
            
            # Try primary source first
            primary_source = self.sources[0]
            batch_data = primary_source.fetch_data(symbols, start_date, end_date, period)
            if not batch_data.empty:
                dfs.append(batch_data)
                used_sources.add(primary_source.get_source_name())
            
            # If verification requested and we have other sources, use them
            if verify_across_sources:
                for source in self.sources[1:]:
                    if source.can_fetch_data():
                        try:
                            secondary_data = source.fetch_data(symbols, start_date, end_date, period)
                            if not secondary_data.empty:
                                dfs.append(secondary_data)
                                used_sources.add(source.get_source_name())
                        except Exception as e:
                            self.logger.error(f"Error fetching from {source.get_source_name()}: {e}")
            
            if dfs:
                final_df = self._merge_data_sources(dfs)
                
                # Add source tracking
                final_df['data_sources'] = '|'.join(sorted(used_sources))
                
                # Save batch data
                symbol_str = ','.join(symbols)  # For logging purposes
                self._save_to_db(symbol_str, final_df, replace)
                
                self.logger.info(f"Data fetched from sources: {', '.join(used_sources)}")
                
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbols}: {str(e)}")
            raise
    
    def update_symbols(self, symbols: Union[str, List[str]], batch_size: int = 50):
        """Update data for given symbols with latest prices"""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        self.logger.info(f"Starting update for {len(symbols)} symbols with batch size {batch_size}")
                
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}: {batch}")
            
            try:
                # Get latest dates for symbols in batch
                with self.get_db_connection() as conn:
                    placeholders = ','.join(['?' for _ in batch])
                    query = f"""
                        SELECT symbol, MAX(date) as last_date 
                        FROM daily_prices 
                        WHERE symbol IN ({placeholders})
                        GROUP BY symbol
                    """
                    last_dates = pd.read_sql_query(query, conn, params=batch)
                    self.logger.debug(f"Found last dates for batch: {last_dates.to_dict() if not last_dates.empty else 'No dates'}")
                
                if not last_dates.empty:
                    min_date = pd.to_datetime(last_dates['last_date'].min()) + timedelta(days=1)
                    self.logger.info(f"Fetching data from {min_date} for batch")
                    
                    if min_date.date() < datetime.now().date():
                        # Process entire batch at once
                        for source in self.sources:
                            self.logger.info(f"Trying source: {source.get_source_name()}")
                            if source.can_fetch_data():
                                data = source.fetch_data(batch, start_date=min_date)
                                if not data.empty:
                                    self.logger.info(f"Got data from {source.get_source_name()}, saving batch")
                                    self._save_to_db(batch[0], data)  # First symbol represents the batch
                                    break
                            else:
                                self.logger.warning(f"Source {source.get_source_name()} cannot fetch data")
                else:
                    self.logger.info(f"No existing data found, fetching full history for batch: {batch}")
                    self.fetch_data(batch, period='max')
                
                # Sleep between batches
                if i + batch_size < len(symbols):
                    self.logger.debug("Sleeping between batches")
                    time.sleep(5)
                            
            except Exception as e:
                self.logger.error(f"Error updating batch starting with {batch[0]}: {str(e)}")
                if "429" in str(e) or "rate limit" in str(e).lower():
                    self.logger.warning("Rate limit hit, sleeping for 60 seconds")
                    time.sleep(60)
                    i -= batch_size  # Retry this batch
                        
    def _merge_data_sources(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge data from multiple sources with conflict resolution"""
        if len(dfs) == 1:
            return dfs[0]
        
        # Convert dates to datetime for merging
        for df in dfs:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
        
        # Start with the primary source
        final_df = dfs[0].copy()
        
        # Keep track of all sources used
        all_sources = [final_df['data_source'].iloc[0]]
        
        for df in dfs[1:]:
            current_source = df['data_source'].iloc[0]
            all_sources.append(current_source)
            
            # Get numeric columns to average
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            
            # Merge on date and symbol
            merged = pd.merge(final_df, df, on=['date', 'symbol'], how='outer', suffixes=(None, '_y'))
            
            # Average the numeric columns
            for col in numeric_cols:
                if f'{col}_y' in merged.columns:
                    merged[col] = merged[[col, f'{col}_y']].mean(axis=1)
                    merged.drop(f'{col}_y', axis=1, inplace=True)
            
            # Combine data sources
            merged['data_sources'] = '|'.join(sorted(set(all_sources)))
            
            # Keep most recent last_updated
            merged['last_updated'] = merged.apply(
                lambda x: max(pd.to_datetime(x['last_updated']), 
                            pd.to_datetime(x['last_updated_y']))
                if 'last_updated_y' in merged.columns else x['last_updated'],
                axis=1
            )
            
            # Clean up remaining _y columns
            y_columns = [col for col in merged.columns if col.endswith('_y')]
            merged.drop(columns=y_columns, inplace=True)
            
            final_df = merged
        
        return final_df
    
    def _save_to_db(self, symbol: str, data: pd.DataFrame, replace: bool = False):
        """Save data to database with duplicate handling"""
        if data.empty:
            return
            
        try:
            with self.get_db_connection() as conn:
                if replace:
                    conn.execute('DELETE FROM daily_prices WHERE symbol = ?', (symbol,))
                    conn.commit()
                
                # Standardize column names
                data.columns = [col.lower() for col in data.columns]
                
                # Ensure all required columns exist
                required_cols = {
                    'dividends': 0,
                    'stock_splits': 0,
                    'data_source': 'unknown',
                    'last_updated': datetime.now()
                }
                
                for col, default in required_cols.items():
                    if col not in data.columns:
                        data[col] = default
                
                # Convert dates to proper format
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date']).dt.date
                
                # Get existing dates to avoid duplicates
                if not replace:
                    existing_dates = pd.read_sql_query(
                        'SELECT date FROM daily_prices WHERE symbol = ?',
                        conn,
                        params=(symbol,)
                    )
                    
                    if not existing_dates.empty:
                        existing_dates['date'] = pd.to_datetime(existing_dates['date']).dt.date
                        data = data[~data['date'].isin(existing_dates['date'])]
                
                if not data.empty:
                    data.to_sql('daily_prices', conn, if_exists='append', index=False,
                              method='multi', chunksize=500)
                    self.logger.info(f"Added {len(data)} new records for {symbol}")
                else:
                    self.logger.info(f"No new data to save for {symbol}")
                    
        except Exception as e:
            self.logger.error(f"Error saving data for {symbol}: {str(e)}")
            raise