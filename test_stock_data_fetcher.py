import pytest
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
from unittest.mock import patch, MagicMock
from stock_data_fetcher import StockDataFetcher

class TestStockDataFetcher:
    @pytest.fixture
    def test_db_path(self):
        """Fixture to provide test database path"""
        path = 'test_stock_data.db'
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    @pytest.fixture
    def fetcher(self, test_db_path):
        """Fixture to provide StockDataFetcher instance"""
        return StockDataFetcher(db_path=test_db_path)

    def test_database_initialization(self, fetcher, test_db_path):
        """Test if database and tables are created properly"""
        assert os.path.exists(test_db_path)
        
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='daily_prices'")
        table_schema = cursor.fetchone()[0].lower()
        conn.close()
        
        required_columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        for column in required_columns:
            assert column in table_schema

    def create_mock_data(self, periods=2):
        """Helper to create mock data in yfinance format"""
        dates = pd.date_range(start='2024-01-01', periods=periods)
        return pd.DataFrame({
            'Open': [100.0 + i for i in range(periods)],
            'High': [102.0 + i for i in range(periods)],
            'Low': [98.0 + i for i in range(periods)],
            'Close': [101.0 + i for i in range(periods)],
            'Volume': [1000000 + i*100000 for i in range(periods)]
        }, index=dates)

    @patch('yfinance.Ticker')
    def test_fetch_data_single_symbol(self, mock_ticker, fetcher):
        """Test fetching data for a single symbol"""
        # Create mock data
        mock_data = self.create_mock_data(periods=2)
        
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Fetch data
        fetcher.fetch_data('AAPL')
        
        # Verify data in database
        conn = sqlite3.connect(fetcher.db_path)
        df = pd.read_sql('SELECT * FROM daily_prices WHERE symbol = "AAPL" ORDER BY date', conn)
        conn.close()
        
        assert len(df) == 2
        assert all(df['symbol'] == 'AAPL')
        assert df['open'].iloc[0] == 100.0
        assert df['open'].iloc[1] == 101.0

    @patch('yfinance.Ticker')
    def test_fetch_data_multiple_symbols(self, mock_ticker, fetcher):
        """Test fetching data for multiple symbols"""
        # Create mock data
        mock_data = self.create_mock_data(periods=1)
        
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Fetch data for multiple symbols
        symbols = ['AAPL', 'GOOGL']
        fetcher.fetch_data(symbols)
        
        # Verify data in database
        conn = sqlite3.connect(fetcher.db_path)
        df = pd.read_sql('SELECT DISTINCT symbol FROM daily_prices', conn)
        conn.close()
        
        assert set(df['symbol']) == set(symbols)

    def test_fetch_data_empty_response(self, fetcher):
        """Test handling of empty data response"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.history.return_value = pd.DataFrame()
            mock_ticker.return_value = mock_ticker_instance
            
            fetcher.fetch_data('INVALID')
            
            conn = sqlite3.connect(fetcher.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM daily_prices WHERE symbol = "INVALID"')
            count = cursor.fetchone()[0]
            conn.close()
            
            assert count == 0

    @patch('yfinance.Ticker')
    def test_update_symbols(self, mock_ticker, fetcher):
        """Test updating symbols with new data"""
        # First, insert some initial data
        initial_data = pd.DataFrame({
            'date': ['2024-01-01'],
            'symbol': ['AAPL'],
            'open': [100.0],
            'high': [102.0],
            'low': [98.0],
            'close': [101.0],
            'volume': [1000000]
        })
        
        conn = sqlite3.connect(fetcher.db_path)
        initial_data.to_sql('daily_prices', conn, if_exists='append', index=False)
        conn.close()
        
        # Create mock data for update
        mock_data = self.create_mock_data(periods=1)
        mock_data.index = pd.date_range(start='2024-01-02', periods=1)
        
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Update data
        fetcher.update_symbols(['AAPL'])
        
        # Verify data in database
        conn = sqlite3.connect(fetcher.db_path)
        df = pd.read_sql('SELECT * FROM daily_prices WHERE symbol = "AAPL" ORDER BY date', conn)
        conn.close()
        
        assert len(df) == 2  # Should have both initial and new data
        assert df['date'].iloc[0] == '2024-01-01'
        assert df['date'].iloc[1] == '2024-01-02'

    def test_error_handling(self, fetcher):
        """Test error handling when fetching data"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.history.side_effect = Exception("API Error")
            mock_ticker.return_value = mock_ticker_instance
            
            fetcher.fetch_data('AAPL')
            
            conn = sqlite3.connect(fetcher.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM daily_prices WHERE symbol = "AAPL"')
            count = cursor.fetchone()[0]
            conn.close()
            
            assert count == 0

if __name__ == '__main__':
    pytest.main([__file__])
