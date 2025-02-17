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
    
    def fetch_data(self, symbol: str, start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None, period: Optional[str] = None) -> pd.DataFrame:
        try:
            ticker = yf.Ticker(symbol)
            data, used_period = self._get_history_with_fallback(ticker, period, start_date, end_date)
            
            if data.empty:
                self.logger.warning(f"No data found for {symbol} using YFinance")
                return pd.DataFrame()
            
            # Standardize the data format
            data = self._standardize_data(data, symbol)
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching YFinance data for {symbol}: {str(e)}")
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
    def __init__(self, db_path='stock_data.db'):
        """Initialize with path to SQLite database"""
        self.db_path = db_path
        self.timeout = 30.0  # 30 seconds timeout        
        self.setup_database()

    @contextmanager
    def get_db_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        try:
            yield conn
        finally:
            conn.close()

    def get_history_with_fallback(self, ticker, period='max', start=None, end=None):
        """
        Attempt to get history with fallback periods if max doesn't work.
        Returns DataFrame and the period that worked.
        """
        #breakpoint()
        # First try the requested period/dates
        if period:
            data = ticker.history(period=period)
            if not data.empty:
                return data, period
                
        # If start date is specified, try that
        if start:
            data = ticker.history(start=start, end=end)
            if not data.empty:
                return data, None
        
        # Fallback periods from longest to shortest
        fallback_periods = ['5Y', '1Y', '6M', '1M', '5D', '1D']
        
        for fallback_period in fallback_periods:
            try:
                data = ticker.history(period=fallback_period)
                if not data.empty:
                    logging.info(f"Used fallback period {fallback_period} for {ticker.ticker}")
                    return data, fallback_period
            except Exception as e:
                logging.debug(f"Fallback {fallback_period} failed for {ticker.ticker}: {e}")
                continue
                
        # If all fallbacks fail, return empty DataFrame
        return pd.DataFrame(), None

    def fetch_data(self, symbols, start_date=None, end_date=None, period=None, replace=False):
        """
        Fetch historical data for given symbols and date range or period
        
        Parameters:
        symbols (str or list): Single symbol or list of symbols
        start_date (str): Start date in YYYY-MM-DD format (optional)
        end_date (str): End date in YYYY-MM-DD format (optional)
        period (str): Time period - e.g., '1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'
                    If specified, overrides start_date and end_date
        replace (bool): If True, removes existing data for the symbol before inserting
                    If False, will only fetch and append newer data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        #breakpoint()
        conn = sqlite3.connect(self.db_path)

        for symbol in symbols:
            try:
                if not replace:
                    # Get latest date in database for this symbol
                    query = "SELECT MAX(date) FROM daily_prices WHERE symbol = ?"
                    last_date = pd.read_sql_query(query, conn, params=(symbol,)).iloc[0, 0]

                    if last_date:
                        # Convert to datetime and add one day
                        start_date = datetime.strptime(last_date, '%Y-%m-%d').date() + timedelta(days=1)
                        if start_date >= datetime.now().date():
                            logging.info(f"Data already up to date for {symbol}")
                            continue
                
                # Get data from Yahoo Finance
                ticker = yf.Ticker(symbol)

                # If no existing data found or replace is True, try to get maximum history
                if (not last_date and not start_date) or (period and replace and period == 'max'):
                    data, used_period = self.get_history_with_fallback(ticker, period='max')
                else:
                    # Otherwise use the start_date
                    data, used_period = self.get_history_with_fallback(ticker, start=start_date, end=end_date)
                if data.empty:
                    logging.warning(f"No data found for {symbol}")
                    continue

                # Prepare data for database
                data = data.reset_index()
                data['Date'] = data['Date'].dt.date
                # Include Dividends and Stock Splits if available
                if 'Dividends' in data.columns:
                    dividends = data['Dividends']
                else:
                    dividends = pd.Series([0] * len(data))
                if 'Stock Splits' in data.columns:
                    splits = data['Stock Splits']
                else:
                    splits = pd.Series([0] * len(data))
                
                df = pd.DataFrame({
                    'Date': data['Date'],
                    'Symbol': symbol,
                    'Open': data['Open'],
                    'High': data['High'],
                    'Low': data['Low'],
                    'Close': data['Close'],
                    'Volume': data['Volume'],
                    'Dividends': dividends,
                    'Stock_Splits': splits,
                    'last_updated': datetime.now()
                })

                # If replace is True, delete existing data for this symbol
                if replace:
                    conn.execute('DELETE FROM daily_prices WHERE symbol = ?', (symbol,))
                    conn.commit()
                    
                df.to_sql('daily_prices', conn, if_exists='append', index=False,
                         method='multi', chunksize=500)

                logging.info(f"Successfully saved data for {symbol}")

                # Sleep briefly to avoid hitting rate limits
                time.sleep(1)

            except Exception as e:
                logging.error(f"Error fetching data for {symbol}: {str(e)}")

        conn.close()

    def update_symbols(self, symbols, batch_size=50):
        """Update data for given symbols with latest prices"""
        #breakpoint()
        if isinstance(symbols, str):
            symbols = [symbols]

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            try:
                # Get latest dates for all symbols in batch
                conn = sqlite3.connect(self.db_path)
                query = "SELECT symbol, MAX(date) as last_date FROM daily_prices WHERE symbol IN ({}) GROUP BY symbol".format(
                    ','.join(['?'] * len(batch))
                )
                last_dates = pd.read_sql_query(query, conn, params=batch)
                conn.close()
                
                # Find earliest last date in batch
                min_date = None
                if not last_dates.empty:
                    min_date = pd.to_datetime(last_dates['last_date'].min()) + timedelta(days=1)
                
                # If we have a valid start date and it's not in the future
                if min_date and min_date.date() < datetime.now().date():
                    # Fetch data for entire batch
                    tickers = [yf.Ticker(s) for s in batch]
                    all_data = {}
                    for ticker in tickers:
                        try:
                            data = ticker.history(start=min_date)
                            if not data.empty:
                                all_data[ticker.ticker] = data
                        except Exception as e:
                            logging.error(f"Error fetching {ticker.ticker}: {e}")
                    
                    # Process each symbol's data
                    for symbol in batch:
                        if symbol in all_data and not all_data[symbol].empty:
                            self.save_to_db(symbol, all_data[symbol])
                else:
                    # No existing data, fetch max history
                    self.fetch_data(batch, period='max')
                
                # Sleep between batches
                if i + batch_size < len(symbols):
                    time.sleep(5)
                    
            except Exception as e:
                logging.error(f"Error updating batch starting with {batch[0]}: {str(e)}")
                # If rate limit hit, increase sleep time and retry
                if "429" in str(e) or "rate limit" in str(e).lower():
                    time.sleep(60)
                    i -= batch_size  # Retry this batch


class EnhancedStockDataFetcher:
    """Enhanced version of your StockDataFetcher that supports multiple data sources"""
    
    def __init__(self, db_path: str = 'stock_data.db', av_api_key: Optional[str] = None):
        self.db_path = db_path
        self.timeout = 30.0
        
        # Initialize data sources
        self.sources: List[StockDataSource] = [
            YFinanceSource()
        ]
        
        if av_api_key:
            self.sources.append(AlphaVantageSource(av_api_key))
        
        self.setup_database()
    
    def setup_database(self):
        """Create database tables if they don't exist with improved error handling"""
        try:
            with self.get_db_connection() as conn:
                c = conn.cursor()
                
                # Create tables with improved schema
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
                        last_updated TIMESTAMP,
                        PRIMARY KEY (date, symbol)
                    )
                ''')
                
                # Create index for faster queries
                c.execute('CREATE INDEX IF NOT EXISTS idx_symbol_date ON daily_prices(symbol, date)')
                
                conn.commit()

        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                print(f"Database is locked. Waiting up to {self.timeout} seconds for access...")
            raise
        except Exception as e:
            print(f"Error setting up database: {str(e)}")
            raise
    
    def fetch_data(self, symbols: Union[str, List[str]], start_date: Optional[str] = None,
                  end_date: Optional[str] = None, period: Optional[str] = None,
                  verify_across_sources: bool = False) -> None:
        """
        Fetch data using all available sources
        
        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date for data fetch
            end_date: End date for data fetch
            period: Time period for data fetch
            verify_across_sources: Whether to fetch from all sources for verification
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        for symbol in symbols:
            dfs = []
            
            # Try primary source first (YFinance)
            primary_df = self.sources[0].fetch_data(symbol, start_date, end_date, period)
            if not primary_df.empty:
                dfs.append(primary_df)
            
            # If verification requested and we have other sources, use them
            if verify_across_sources:
                for source in self.sources[1:]:
                    if source.can_fetch_data():
                        secondary_df = source.fetch_data(symbol, start_date, end_date, period)
                        if not secondary_df.empty:
                            dfs.append(secondary_df)
            
            if dfs:
                # Merge data from all sources
                final_df = self._merge_data_sources(dfs)
                # Save to database
                self._save_to_db(symbol, final_df)
    
    def _merge_data_sources(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge data from multiple sources, handling conflicts intelligently
        This is where you'd implement your logic for resolving conflicts between sources
        """
        if len(dfs) == 1:
            return dfs[0]
        
        # Start with the primary source
        final_df = dfs[0]
        
        # Merge additional sources
        for df in dfs[1:]:
            # Implement your merging logic here
            # For example, you might want to:
            # - Keep the source with the most recent timestamp
            # - Average values across sources
            # - Flag discrepancies for review
            pass
        
        return final_df
    
    def _save_to_db(self, symbol, data):
        """Save data for a symbol to the database without duplicating or removing existing records"""
        #breakpoint()
        if data.empty:
            return
            
        conn = sqlite3.connect(self.db_path)
        try:
            # Prepare the new data
            df = data.reset_index()
            df['Date'] = df['Date'].dt.date
            df['Symbol'] = symbol
            df['last_updated'] = datetime.now()
            
            # Ensure all required columns exist
            if 'Dividends' not in df.columns:
                df['Dividends'] = 0
            if 'Stock Splits' not in df.columns:
                df['Stock_Splits'] = 0
            else:
                df = df.rename(columns = {"Stock Splits": "Stock_Splits"})
            
            df.columns = [col.lower() for col in df.columns]
            
            # Get existing dates for this symbol
            existing_dates = pd.read_sql_query(
                'SELECT date FROM daily_prices WHERE symbol = ?',
                conn,
                params=(symbol,)
            )
            
            if not existing_dates.empty:
                existing_dates['date'] = pd.to_datetime(existing_dates['date']).dt.date
                # Filter out dates we already have
                df = df[~df['date'].isin(existing_dates['date'])]
            
            if not df.empty:
                # Save only the new dates to database
                df.to_sql('daily_prices', conn, if_exists='append', index=False,
                         method='multi', chunksize=500)
                logging.info(f"Added {len(df)} new records for {symbol}")
            else:
                logging.info(f"No new data to save for {symbol}")
                
        except Exception as e:
            logging.error(f"Error saving data for {symbol}: {e}")
            conn.rollback()
        finally:
            conn.close()