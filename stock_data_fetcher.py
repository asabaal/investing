import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import time
import logging

class StockDataFetcher:
    def __init__(self, db_path='stock_data.db'):
        """Initialize with path to SQLite database"""
        self.db_path = db_path
        self.setup_database()

    def setup_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
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
        conn.close()

    def get_history_with_fallback(self, ticker, period='max', start=None, end=None):
        """
        Attempt to get history with fallback periods if max doesn't work.
        Returns DataFrame and the period that worked.
        """
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

    def save_to_db(self, symbol, data):
        """Save data for a symbol to the database without duplicating or removing existing records"""
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
            
            # Get existing dates for this symbol
            existing_dates = pd.read_sql_query(
                'SELECT date FROM daily_prices WHERE symbol = ?',
                conn,
                params=(symbol,)
            )
            
            if not existing_dates.empty:
                existing_dates['date'] = pd.to_datetime(existing_dates['date']).dt.date
                # Filter out dates we already have
                df = df[~df['Date'].isin(existing_dates['date'])]
            
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

    def update_symbols(self, symbols, batch_size=50):
        """Update data for given symbols with latest prices"""
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