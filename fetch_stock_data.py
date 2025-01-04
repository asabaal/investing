import pandas as pd
import logging
import sys
import sqlite3
import requests
from stock_data_fetcher import StockDataFetcher

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
    """Get NASDAQ symbols using yfinance"""
    try:
        # Try to get ^IXIC (NASDAQ Composite) constituents
        import yfinance as yf
        nasdaq = yf.Ticker("^IXIC")
        symbols = [holding['symbol'] for holding in nasdaq.holdings]
        if symbols:
            return list(set(symbols))
    except:
        pass
    
    try:
        # Fallback to NASDAQ API
        url = "https://api.nasdaq.com/api/screener/stocks"
        params = {
            "exchange": "NASDAQ",
            "download": "true"
        }
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        return [stock['symbol'] for stock in data['data']['rows']]
    except Exception as e:
        logging.error(f"Error fetching NASDAQ symbols: {e}")
        # Fallback to major NASDAQ stocks if all else fails
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA']

def fetch_all_historical_data(batch_size=50, sleep_time=5):
    """Fetch historical data with optimized batching"""
    # Initialize fetcher
    fetcher = StockDataFetcher(db_path='full_stock_history.db')
    # Get all symbols
    sp500_symbols = get_sp500_symbols()
    nasdaq_symbols = get_nasdaq_symbols()
    indices = ['^IXIC', '^NDX', '^GSPC', '^DJI', 'QQQ', 'SPY']
    
    all_symbols = list(set(sp500_symbols + nasdaq_symbols + indices))
    logging.info(f"Total unique symbols: {len(all_symbols)}")

    # Update all symbols
    fetcher.update_symbols(all_symbols, batch_size=batch_size)
    #fetcher.update_symbols(["EDBLW"])
    
    # Check database status
    #check_database_status(fetcher)

def check_database_status(fetcher):
    """Print summary statistics about the database"""
    try:
        conn = sqlite3.connect(fetcher.db_path)
        
        stats = {
            'total_symbols': pd.read_sql_query(
                "SELECT COUNT(DISTINCT symbol) FROM daily_prices", conn).iloc[0, 0],
            'total_records': pd.read_sql_query(
                "SELECT COUNT(*) FROM daily_prices", conn).iloc[0, 0],
            'date_range': pd.read_sql_query("""
                SELECT MIN(date) as earliest, MAX(date) as latest 
                FROM daily_prices""", conn).iloc[0].to_dict(),
            'last_update': pd.read_sql_query(
                "SELECT MAX(last_updated) FROM daily_prices", conn).iloc[0, 0]
        }
        
        logging.info("Database Status:")
        logging.info(f"Total Symbols: {stats['total_symbols']}")
        logging.info(f"Total Records: {stats['total_records']:,}")
        logging.info(f"Date Range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
        logging.info(f"Last Updated: {stats['last_update']}")
        
        conn.close()
    except Exception as e:
        logging.error(f"Error checking database status: {e}")

if __name__ == "__main__":
    logging.info("Starting historical data fetch process")
    fetch_all_historical_data()