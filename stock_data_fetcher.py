import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import time


class StockDataFetcher:
    def __init__(self, db_path='stock_data.db'):
        """Initialize with path to SQLite database"""
        self.db_path = db_path
        self.setup_database()

    def setup_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create tables
        c.execute('''
            CREATE TABLE IF NOT EXISTS daily_prices (
                date DATE,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (date, symbol)
            )
        ''')
        conn.commit()
        conn.close()

    def fetch_data(self, symbols, start_date=None, end_date=None):
        """
        Fetch historical data for given symbols and date range
        If no dates provided, fetches max history
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        for symbol in symbols:
            try:
                # Get data from Yahoo Finance
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)

                if df.empty:
                    print(f"No data found for {symbol}")
                    continue

                # Prepare data for database
                df = df.reset_index()
                df['Date'] = df['Date'].dt.date
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

                # Add symbol column
                df.insert(1, 'Symbol', symbol)

                # Save to database
                conn = sqlite3.connect(self.db_path)
                df.to_sql('daily_prices', conn, if_exists='append', index=False,
                         method='multi', chunksize=500)
                conn.close()

                print(f"Successfully saved data for {symbol}")

                # Sleep briefly to avoid hitting rate limits
                time.sleep(1)

            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")

    def update_symbols(self, symbols):
        """Update data for given symbols with latest prices"""
        conn = sqlite3.connect(self.db_path)

        for symbol in symbols:
            try:
                # Get latest date in database for this symbol
                query = f"""
                    SELECT MAX(date)
                    FROM daily_prices
                    WHERE symbol = '{symbol}'
                """
                last_date = pd.read_sql_query(query, conn).iloc[0, 0]

                if last_date:
                    # Convert to datetime and add one day
                    start_date = datetime.strptime(last_date, '%Y-%m-%d').date() + timedelta(days=1)
                    if start_date >= datetime.now().date():
                        print(f"Data already up to date for {symbol}")
                        continue
                else:
                    start_date = None

                # Fetch new data
                self.fetch_data(symbol, start_date=start_date)

            except Exception as e:
                print(f"Error updating {symbol}: {str(e)}")

        conn.close()