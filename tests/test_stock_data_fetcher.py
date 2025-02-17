import pytest
import pandas as pd
import sqlite3
from datetime import datetime
import os
from stock_data_fetcher import StockDataFetcher

@pytest.fixture
def test_db():
    """Fixture to create test database path"""
    db_path = 'test_stock_data.db'
    yield db_path
    # Cleanup after test
    if os.path.exists(db_path):
        os.remove(db_path)

@pytest.fixture
def fetcher(test_db):
    """Fixture to create StockDataFetcher instance"""
    return StockDataFetcher(db_path=test_db)

@pytest.fixture
def mock_stock_data():
    """Fixture for sample stock data"""
    return pd.DataFrame({
        'Date': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        'Open': [100.0, 101.0],
        'High': [102.0, 103.0],
        'Low': [98.0, 99.0],
        'Close': [101.0, 102.0],
        'Volume': [1000000, 1100000],
        'Dividends': [0, 0],
        'Stock Splits': [0, 0]
    }).set_index('Date')

def test_database_setup(fetcher, test_db):
    """Test if database and table are created with correct schema"""
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='daily_prices'
    """)
    table_exists = cursor.fetchone() is not None
    
    # Get table schema
    cursor.execute("PRAGMA table_info(daily_prices)")
    columns = {col[1].lower(): col[2].upper() for col in cursor.fetchall()}
    
    conn.close()
    
    assert table_exists
    
    # Verify schema
    expected_columns = {
        'date': 'DATE',
        'symbol': 'TEXT',
        'open': 'REAL',
        'high': 'REAL',
        'low': 'REAL',
        'close': 'REAL',
        'volume': 'INTEGER',
        'dividends': 'REAL',
        'stock_splits': 'REAL',
        'last_updated': 'TIMESTAMP'
    }
    
    assert columns == expected_columns

def test_fetch_single_symbol(fetcher, mock_stock_data, mocker):
    """Test fetching data for a single symbol"""
    # Mock yfinance Ticker
    mock_ticker = mocker.patch('yfinance.Ticker')
    mock_ticker_instance = mocker.MagicMock()
    mock_ticker_instance.history.return_value = mock_stock_data
    mock_ticker.return_value = mock_ticker_instance
    
    # Mock get_history_with_fallback
    mocker.patch.object(
        fetcher, 
        'get_history_with_fallback',
        return_value=(mock_stock_data, 'max')
    )
    
    # Fetch data
    fetcher.fetch_data('AAPL')
    
    # Verify saved data
    conn = sqlite3.connect(fetcher.db_path)
    saved_data = pd.read_sql('SELECT * FROM daily_prices', conn)
    conn.close()
    
    assert len(saved_data) == 2
    assert saved_data['symbol'].unique()[0] == 'AAPL'
    assert not saved_data.empty

def test_fetch_multiple_symbols(fetcher, mock_stock_data, mocker):
    """Test fetching data for multiple symbols"""
    # Mock yfinance Ticker
    mock_ticker = mocker.patch('yfinance.Ticker')
    mock_ticker_instance = mocker.MagicMock()
    mock_ticker_instance.history.return_value = mock_stock_data
    mock_ticker.return_value = mock_ticker_instance
    
    # Mock get_history_with_fallback
    mocker.patch.object(
        fetcher, 
        'get_history_with_fallback',
        return_value=(mock_stock_data, 'max')
    )
    
    # Fetch data for multiple symbols
    symbols = ['AAPL', 'GOOGL']
    fetcher.fetch_data(symbols)
    
    # Verify saved data
    conn = sqlite3.connect(fetcher.db_path)
    saved_data = pd.read_sql('SELECT * FROM daily_prices', conn)
    conn.close()
    
    assert len(saved_data) == 4  # 2 days * 2 symbols
    assert set(saved_data['symbol'].unique()) == set(symbols)

def test_update_symbols(fetcher, mocker):
    """Test updating symbols with new data"""
    # Insert initial data
    conn = sqlite3.connect(fetcher.db_path)
    initial_data = pd.DataFrame({
        'Date': [datetime(2024, 1, 1)],
        'Symbol': ['AAPL'],
        'Open': [100.0],
        'High': [102.0],
        'Low': [98.0],
        'Close': [101.0],
        'Volume': [1000000],
        'Dividends': [0],
        'Stock_Splits': [0],
        'last_updated': [datetime(2024, 1, 1)]
    })
    initial_data.to_sql('daily_prices', conn, if_exists='append', index=False)
    conn.close()
    
    # Mock new data
    new_data = pd.DataFrame({
        'Date': [datetime(2024, 1, 2)],
        'Symbol': ['AAPL'],
        'Open': [101.0],
        'High': [103.0],
        'Low': [99.0],
        'Close': [102.0],
        'Volume': [1100000],
        'Dividends': [0],
        'Stock_Splits': [0],
        'last_updated': [datetime(2024, 1, 2)],
    })
    
    # Mock yfinance
    mock_ticker = mocker.patch('yfinance.Ticker')
    mock_ticker_instance = mocker.MagicMock()
    mock_ticker_instance.history.return_value = new_data
    mock_ticker.return_value = mock_ticker_instance
    
    # Patch save_to_db instead of get_history_with_fallback
    mocker.patch.object(
        fetcher,
        'save_to_db',
        side_effect=lambda symbol, data: data.reset_index().to_sql('daily_prices', 
                                                                  sqlite3.connect(fetcher.db_path), 
                                                                  if_exists='append', 
                                                                  index=False)
    )
    
    # Update symbols
    fetcher.update_symbols(['AAPL'])
    
    # Verify updated data
    conn = sqlite3.connect(fetcher.db_path)
    saved_data = pd.read_sql('SELECT * FROM daily_prices ORDER BY date', conn)
    conn.close()
    breakpoint()
    assert len(saved_data) == 2
    assert len(saved_data['date'].unique()) == 2

def test_empty_response_handling(fetcher, mocker):
    """Test handling of empty response from yfinance"""
    # Mock empty response
    mock_ticker = mocker.patch('yfinance.Ticker')
    mock_ticker_instance = mocker.MagicMock()
    mock_ticker_instance.history.return_value = pd.DataFrame()
    mock_ticker.return_value = mock_ticker_instance
    
    # Mock get_history_with_fallback
    mocker.patch.object(
        fetcher, 
        'get_history_with_fallback',
        return_value=(pd.DataFrame(), None)
    )
    
    # Attempt to fetch data
    fetcher.fetch_data('INVALID')
    
    # Verify no data was saved
    conn = sqlite3.connect(fetcher.db_path)
    saved_data = pd.read_sql('SELECT * FROM daily_prices', conn)
    conn.close()
    
    assert len(saved_data) == 0

def test_error_handling(fetcher, mocker):
    """Test error handling during data fetching"""
    # Mock API error
    mock_ticker = mocker.patch('yfinance.Ticker')
    mock_ticker_instance = mocker.MagicMock()
    mock_ticker_instance.history.side_effect = Exception("API Error")
    mock_ticker.return_value = mock_ticker_instance
    
    # Mock get_history_with_fallback to raise exception
    mocker.patch.object(
        fetcher, 
        'get_history_with_fallback',
        side_effect=Exception("API Error")
    )
    
    # Attempt to fetch data
    fetcher.fetch_data('AAPL')
    
    # Verify no data was saved
    conn = sqlite3.connect(fetcher.db_path)
    saved_data = pd.read_sql('SELECT * FROM daily_prices', conn)
    conn.close()
    
    assert len(saved_data) == 0