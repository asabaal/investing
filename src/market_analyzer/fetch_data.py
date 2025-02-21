import pandas as pd
import logging
import sys
import sqlite3
import requests
import time
from typing import List, Optional, Union
from market_analyzer.api import APIConfigHandler
from market_analyzer.data_fetcher import StockDataFetcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_fetcher.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class MarketDataFetcher:
    def __init__(self, db_path: str = 'full_stock_history.db', config_dir: str = '.config'):
        self.logger = logging.getLogger(__name__)
        self.config_handler = APIConfigHandler(config_dir)
        
        # Get API keys
        av_api_key = self.config_handler.get_api_key('alpha_vantage')
        if not av_api_key:
            self.logger.warning("No Alpha Vantage API key found. Only YFinance will be used.")
        
        # Initialize fetcher with both sources
        self.fetcher = StockDataFetcher(db_path=db_path, av_api_key=av_api_key)
    
    def get_sp500_symbols(self) -> List[str]:
        """Get current S&P 500 symbols"""
        try:
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            return table['Symbol'].tolist()
        except Exception as e:
            self.logger.error(f"Error fetching S&P 500 symbols: {e}")
            return []

    def get_nasdaq_symbols(self) -> List[str]:
        """Get NASDAQ symbols using multiple fallback methods"""
        try:
            # Try to get ^IXIC (NASDAQ Composite) constituents
            import yfinance as yf
            nasdaq = yf.Ticker("^IXIC")
            symbols = [holding['symbol'] for holding in nasdaq.holdings]
            if symbols:
                return list(set(symbols))
        except Exception as e:
            self.logger.debug(f"Failed to get NASDAQ symbols from yfinance: {e}")
        
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
            self.logger.error(f"Error fetching NASDAQ symbols: {e}")
            # Fallback to major NASDAQ stocks
            return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA']

    def fetch_all_historical_data(self, batch_size: int = 50, verify_sources: bool = True) -> None:
        """
        Fetch historical data with optimized batching and source verification
        
        Args:
            batch_size: Number of symbols to process in each batch
            verify_sources: Whether to verify data across multiple sources when available
        """
        # Get all symbols
        sp500_symbols = self.get_sp500_symbols()
        nasdaq_symbols = self.get_nasdaq_symbols()
        indices = ['^IXIC', '^NDX', '^GSPC', '^DJI', 'QQQ', 'SPY']
        
        all_symbols = list(set(sp500_symbols + nasdaq_symbols + indices))
        self.logger.info(f"Total unique symbols to process: {len(all_symbols)}")
        
        # Process in batches
        for i in range(0, len(all_symbols), batch_size):
            batch = all_symbols[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1} of {len(all_symbols)//batch_size + 1}")
            
            # Check API limits before processing batch
            if not self.config_handler.can_make_api_call('alpha_vantage'):
                remaining_calls = self.config_handler.get_remaining_calls('alpha_vantage')
                self.logger.warning(f"Alpha Vantage API limit reached. Remaining calls: {remaining_calls}")
                # Continue with YFinance only
                verify_sources = False
            
            try:
                # Update symbols with verification if possible
                self.fetcher.update_symbols(batch, batch_size=batch_size)
                
                # Record API call if successful and using Alpha Vantage
                if verify_sources:
                    self.config_handler.record_api_call('alpha_vantage')
                
                # Status update
                self.check_database_status()
                
                # Sleep between batches
                if i + batch_size < len(all_symbols):
                    time.sleep(5)  # Avoid rate limits
                    
            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
                continue
    
    def check_database_status(self) -> None:
        """Print summary statistics about the database"""
        try:
            conn = sqlite3.connect(self.fetcher.db_path)
            
            stats = {
                'total_symbols': pd.read_sql_query(
                    "SELECT COUNT(DISTINCT symbol) FROM daily_prices", conn).iloc[0, 0],
                'total_records': pd.read_sql_query(
                    "SELECT COUNT(*) FROM daily_prices", conn).iloc[0, 0],
                'date_range': pd.read_sql_query("""
                    SELECT MIN(date) as earliest, MAX(date) as latest 
                    FROM daily_prices""", conn).iloc[0].to_dict(),
                'last_update': pd.read_sql_query(
                    "SELECT MAX(last_updated) FROM daily_prices", conn).iloc[0, 0],
                'sources': pd.read_sql_query(
                    "SELECT DISTINCT data_source FROM daily_prices", conn)['data_source'].tolist()
            }
            
            self.logger.info("Database Status:")
            self.logger.info(f"Total Symbols: {stats['total_symbols']}")
            self.logger.info(f"Total Records: {stats['total_records']:,}")
            self.logger.info(f"Date Range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
            self.logger.info(f"Last Updated: {stats['last_update']}")
            self.logger.info(f"Data Sources: {', '.join(stats['sources'])}")
            
            conn.close()
        except Exception as e:
            self.logger.error(f"Error checking database status: {e}")

    def fetch_symbols(self, symbols: Union[str, List[str]], start_date: Optional[str] = None,
                    end_date: Optional[str] = None, verify_sources: bool = True) -> None:
        """
        Fetch data for specific symbols
        
        Args:
            symbols: Single symbol or list of symbols to fetch
            start_date: Optional start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format
            verify_sources: Whether to verify data across multiple sources
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        symbols = list(set(symbols))  # Remove duplicates
        self.logger.info(f"Fetching data for {len(symbols)} symbols")
        
        # Check API availability if verification is requested
        if verify_sources and not self.config_handler.can_make_api_call('alpha_vantage'):
            remaining_calls = self.config_handler.get_remaining_calls('alpha_vantage')
            self.logger.warning(f"Alpha Vantage API limit reached. Remaining calls: {remaining_calls}")
            verify_sources = False
        
        try:
            self.fetcher.fetch_data(
                symbols,
                start_date=start_date,
                end_date=end_date,
                verify_across_sources=verify_sources
            )
            
            if verify_sources:
                self.config_handler.record_api_call('alpha_vantage')
            
            # Provide status update
            self.check_database_status()
            
        except Exception as e:
            self.logger.error(f"Error fetching data for symbols {symbols}: {e}")
            raise

if __name__ == "__main__":
    fetcher = MarketDataFetcher()
    logging.info("Starting historical data fetch process")
    fetcher.fetch_all_historical_data(verify_sources=True)