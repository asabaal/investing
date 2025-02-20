import pytest
import pandas as pd
import sqlite3
import os
from datetime import datetime

from stock_data_fetcher import StockDataFetcher, YFinanceSource, AlphaVantageSource

@pytest.fixture
def test_db():
    """Fixture to create test database path"""
    db_path = 'test_stock_data.db'
    yield db_path
    # Cleanup after test
    if os.path.exists(db_path):
        os.remove(db_path)

@pytest.fixture
def mock_stock_data():
    """Fixture for sample stock data"""
    return pd.DataFrame({
        'date': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        'open': [100.0, 101.0],
        'high': [102.0, 103.0],
        'low': [98.0, 99.0],
        'close': [101.0, 102.0],
        'volume': [1000000, 1100000],
        'dividends': [0, 0],
        'stock_splits': [0, 0],
        'symbol': ['AAPL', 'AAPL'],
        'data_source': ['yfinance', 'yfinance'],
        'last_updated': [datetime.now(), datetime.now()]
    })

@pytest.fixture
def mock_av_data():
    """Fixture for sample Alpha Vantage data"""
    return pd.DataFrame({
        'date': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        'open': [100.2, 101.2],
        'high': [102.2, 103.2],
        'low': [98.2, 99.2],
        'close': [101.2, 102.2],
        'volume': [1000200, 1100200],
        'dividends': [0, 0],
        'stock_splits': [0, 0],
        'symbol': ['AAPL', 'AAPL'],
        'data_source': ['alpha_vantage', 'alpha_vantage'],
        'last_updated': [datetime.now(), datetime.now()]
    })

@pytest.fixture
def fetcher(test_db):
    """Fixture to create StockDataFetcher instance"""
    return StockDataFetcher(db_path=test_db, av_api_key='test_key')

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
    
    # Verify enhanced schema with new columns
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
        'data_source': 'TEXT',
        'last_updated': 'TIMESTAMP'
    }
    
    assert columns == expected_columns

def test_data_source_initialization(fetcher):
    """Test proper initialization of data sources"""
    assert len(fetcher.sources) == 2
    assert isinstance(fetcher.sources[0], YFinanceSource)
    assert isinstance(fetcher.sources[1], AlphaVantageSource)

@pytest.mark.parametrize("source_class", [YFinanceSource, AlphaVantageSource])
def test_data_source_interface(source_class):
    """Test that data sources implement required interface"""
    source = source_class() if source_class == YFinanceSource else source_class("test_key")
    assert hasattr(source, 'fetch_data')
    assert hasattr(source, 'get_source_name')
    assert hasattr(source, 'can_fetch_data')

def test_fetch_single_symbol(fetcher, mock_stock_data, mocker):
    """Test fetching data for a single symbol from primary source"""
    # Mock YFinance source
    mock_yf_source = mocker.MagicMock()
    mock_yf_source.fetch_data.return_value = mock_stock_data
    mock_yf_source.get_source_name.return_value = "yfinance"
    fetcher.sources[0] = mock_yf_source
    
    # Fetch data
    fetcher.fetch_data('AAPL')
    
    # Verify saved data
    with fetcher.get_db_connection() as conn:
        saved_data = pd.read_sql('SELECT * FROM daily_prices', conn)
    
    assert len(saved_data) == 2
    assert saved_data['symbol'].unique()[0] == 'AAPL'
    assert saved_data['data_source'].unique()[0] == 'yfinance'

def test_fetch_with_verification(fetcher, mock_stock_data, mock_av_data, mocker):
    """Test fetching data with verification across sources"""
    # Mock both sources
    mock_yf_source = mocker.MagicMock()
    mock_yf_source.fetch_data.return_value = mock_stock_data
    mock_yf_source.get_source_name.return_value = "yfinance"
    
    mock_av_source = mocker.MagicMock()
    mock_av_source.fetch_data.return_value = mock_av_data
    mock_av_source.get_source_name.return_value = "alpha_vantage"
    mock_av_source.can_fetch_data.return_value = True
    
    fetcher.sources = [mock_yf_source, mock_av_source]
    
    # Fetch data with verification
    fetcher.fetch_data('AAPL', verify_across_sources=True)
    
    # Verify both sources were called
    mock_yf_source.fetch_data.assert_called_once()
    mock_av_source.fetch_data.assert_called_once()
    
    # Verify merged data was saved
    with fetcher.get_db_connection() as conn:
        saved_data = pd.read_sql('SELECT * FROM daily_prices', conn)
    
    assert len(saved_data) == 2  # Should have merged duplicate dates
    assert set(saved_data['data_source'].unique()) == {'yfinance'}  # Primary source should be used

def test_update_symbols_batch_processing(fetcher, mock_stock_data, mocker):
    """Test updating multiple symbols in batches"""
    # Mock YFinance source
    mock_yf_source = mocker.MagicMock()
    mock_yf_source.fetch_data.return_value = mock_stock_data
    fetcher.sources[0] = mock_yf_source
    
    # Update symbols in batches
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    fetcher.update_symbols(symbols, batch_size=2)
    
    # Verify number of calls to fetch_data
    assert mock_yf_source.fetch_data.call_count == 2  # Should be called once per batch

def test_data_source_fallback(fetcher, mock_stock_data, mocker):
    """Test fallback to secondary source when primary fails"""
    # Mock YFinance source to fail
    mock_yf_source = mocker.MagicMock()
    mock_yf_source.fetch_data.return_value = pd.DataFrame()
    
    # Mock Alpha Vantage source to succeed
    mock_av_source = mocker.MagicMock()
    mock_av_source.fetch_data.return_value = mock_stock_data
    mock_av_source.can_fetch_data.return_value = True
    
    fetcher.sources = [mock_yf_source, mock_av_source]
    
    # Fetch data with verification
    fetcher.fetch_data('AAPL', verify_across_sources=True)
    
    # Verify fallback worked
    with fetcher.get_db_connection() as conn:
        saved_data = pd.read_sql('SELECT * FROM daily_prices', conn)
    
    assert len(saved_data) == 2
    assert not saved_data.empty

def test_merge_data_sources(fetcher, mock_stock_data, mock_av_data):
    """Test merging data from multiple sources"""
    # Modify second source data slightly
    mock_av_data.loc[0, 'close'] = 101.5  # Create a conflict
    
    merged_data = fetcher._merge_data_sources([mock_stock_data, mock_av_data])
    
    assert len(merged_data) == 2  # Should maintain unique dates
    assert merged_data.iloc[0]['close'] == 101.35  # Should average conflicting values

def test_error_handling(fetcher, mocker):
    """Test error handling during data fetching"""
    # Mock source to raise exception
    mock_source = mocker.MagicMock()
    mock_source.fetch_data.side_effect = Exception("API Error")
    fetcher.sources = [mock_source]
    
    # Attempt to fetch data
    fetcher.fetch_data('AAPL')
    
    # Verify no data was saved
    with fetcher.get_db_connection() as conn:
        saved_data = pd.read_sql('SELECT * FROM daily_prices', conn)
    
    assert len(saved_data) == 0

def test_replace_existing_data(fetcher, mock_stock_data, mocker):
    """Test replacing existing data for a symbol"""
    # Mock YFinance source
    mock_source = mocker.MagicMock()
    mock_source.fetch_data.return_value = mock_stock_data
    fetcher.sources = [mock_source]
    
    # Insert initial data
    fetcher.fetch_data('AAPL')
    
    # Modify mock data and replace
    mock_stock_data.loc[0, 'close'] = 150.0
    mock_source.fetch_data.return_value = mock_stock_data
    
    fetcher.fetch_data('AAPL', replace=True)
    
    # Verify data was replaced
    with fetcher.get_db_connection() as conn:
        saved_data = pd.read_sql('SELECT * FROM daily_prices WHERE symbol = "AAPL"', conn)
    
    assert len(saved_data) == 2
    assert saved_data.iloc[0]['close'] == 150.0

def test_database_connection_context_manager(fetcher):
    """Test database connection context manager"""
    with fetcher.get_db_connection() as conn:
        assert isinstance(conn, sqlite3.Connection)
        # Connection should be open
        assert not conn.closed
    
    # Connection should be closed after context
    assert conn.closed