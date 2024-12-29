import yfinance as yf
import pandas as pd
import sqlite3
from stock_data_fetcher import StockDataFetcher
import time
from datetime import datetime
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_fetcher.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_sp500_symbols():
    """Get current S&P 500 symbols"""
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        return table['Symbol'].tolist()
    except Exception as e:
        logging.error(f"Error fetching S&P 500 symbols: {e}")
        return []

def get_nasdaq_symbols():
    """Get NASDAQ-listed symbols"""
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[1]
        return table['Ticker'].tolist()
    except Exception as e:
        logging.error(f"Error fetching NASDAQ symbols: {e}")
        return []

def fetch_all_historical_data(batch_size=10, sleep_time=30):
    """
    Fetch historical data for all available symbols
    
    Args:
        batch_size (int): Number of symbols to process before sleeping
        sleep_time (int): Seconds to sleep between batches
    """
    # Initialize fetcher
    fetcher = StockDataFetcher(db_path='full_stock_history.db')
    
    # Get symbols from multiple sources
    sp500_symbols = get_sp500_symbols()
    nasdaq_symbols = get_nasdaq_symbols()
    
    # Combine and deduplicate symbols
    all_symbols = list(set(sp500_symbols + nasdaq_symbols))
    logging.info(f"Total unique symbols to process: {len(all_symbols)}")
    
    # Process symbols in batches
    for i in range(0, len(all_symbols), batch_size):
        batch = all_symbols[i:i + batch_size]
        logging.info(f"Processing batch {i//batch_size + 1} of {len(all_symbols)//batch_size + 1}")
        
        # Fetch data for batch
        try:
            fetcher.fetch_data(batch)
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
        
        # Sleep between batches to avoid rate limiting
        if i + batch_size < len(all_symbols):
            logging.info(f"Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
    
    logging.info("Completed fetching historical data")

def check_database_status():
    """Print summary statistics about the database"""
    try:
        fetcher = StockDataFetcher(db_path='full_stock_history.db')
        conn = sqlite3.connect(fetcher.db_path)
        
        # Get total number of symbols
        symbols_query = "SELECT COUNT(DISTINCT symbol) FROM daily_prices"
        total_symbols = pd.read_sql_query(symbols_query, conn).iloc[0, 0]
        
        # Get date range
        date_query = """
            SELECT 
                MIN(date) as earliest_date,
                MAX(date) as latest_date
            FROM daily_prices
        """
        date_range = pd.read_sql_query(date_query, conn)
        
        # Get total number of records
        count_query = "SELECT COUNT(*) FROM daily_prices"
        total_records = pd.read_sql_query(count_query, conn).iloc[0, 0]
        
        logging.info(f"""
        Database Status:
        ----------------
        Total Symbols: {total_symbols}
        Date Range: {date_range['earliest_date'].iloc[0]} to {date_range['latest_date'].iloc[0]}
        Total Records: {total_records:,}
        """)
        
        conn.close()
    except Exception as e:
        logging.error(f"Error checking database status: {e}")

if __name__ == "__main__":
    logging.info("Starting historical data fetch process")
    
    # Fetch all historical data
    fetch_all_historical_data()
    
    # Check final database status
    check_database_status()