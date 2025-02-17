import pytest
import pandas as pd
import sqlite3
from unittest.mock import patch, MagicMock
import logging
from fetch_stock_data import (
    get_sp500_symbols,
    get_nasdaq_symbols,
    fetch_all_historical_data,
    check_database_status
)

@pytest.fixture
def mock_sp500_data():
    """Mock S&P 500 data from Wikipedia"""
    return pd.DataFrame({
        'Symbol': ['AAPL', 'GOOGL', 'MSFT'],
        'Security': ['Apple Inc.', 'Alphabet Inc.', 'Microsoft Corporation'],
    })

@pytest.fixture
def mock_nasdaq_api_response():
    """Mock NASDAQ API response"""
    return {
        'data': {
            'rows': [
                {'symbol': 'AAPL'},
                {'symbol': 'AMZN'},
                {'symbol': 'NFLX'}
            ]
        }
    }

@pytest.fixture
def mock_db_path(tmp_path):
    """Create a temporary database path"""
    return tmp_path / "test_stock_history.db"

def test_get_sp500_symbols(mock_sp500_data, mocker):
    """Test fetching S&P 500 symbols"""
    mocker.patch('pandas.read_html', return_value=[mock_sp500_data])
    
    symbols = get_sp500_symbols()
    
    assert isinstance(symbols, list)
    assert len(symbols) == 3
    assert 'AAPL' in symbols
    assert 'GOOGL' in symbols

def test_get_nasdaq_symbols(mock_nasdaq_api_response, mocker):
    """Test fetching NASDAQ symbols"""
    # Mock requests.get
    mock_response = MagicMock()
    mock_response.json.return_value = mock_nasdaq_api_response
    mocker.patch('requests.get', return_value=mock_response)
    
    # Mock yfinance fallback just in case
    mock_ticker = mocker.patch('yfinance.Ticker')
    mock_ticker.return_value.holdings = []
    
    symbols = get_nasdaq_symbols()
    
    assert isinstance(symbols, list)
    assert len(symbols) == 3
    assert all(s in ['AAPL', 'AMZN', 'NFLX'] for s in symbols)

def test_get_nasdaq_symbols_error(mocker):
    """Test error handling when fetching NASDAQ symbols"""
    # Mock both primary and fallback methods to fail
    mocker.patch('requests.get', side_effect=Exception("API Error"))
    mock_ticker = mocker.patch('yfinance.Ticker')
    mock_ticker.return_value.holdings = []
    
    symbols = get_nasdaq_symbols()
    
    # Should return fallback major NASDAQ stocks
    assert isinstance(symbols, list)
    assert len(symbols) == 7  # Major NASDAQ stocks fallback
    assert 'AAPL' in symbols
    assert 'MSFT' in symbols

def test_fetch_all_historical_data(mock_sp500_data, mock_nasdaq_api_response, mock_db_path, mocker):
    """Test the full data fetching process"""
    # Mock symbol fetching functions
    mocker.patch('fetch_stock_data.get_sp500_symbols', return_value=['AAPL', 'GOOGL'])
    mocker.patch('fetch_stock_data.get_nasdaq_symbols', return_value=['AAPL', 'AMZN'])
    
    # Mock StockDataFetcher
    mock_fetcher_class = mocker.patch('fetch_stock_data.StockDataFetcher')
    mock_fetcher_instance = MagicMock()
    mock_fetcher_class.return_value = mock_fetcher_instance
    
    # Mock time.sleep to speed up tests
    mocker.patch('time.sleep')
    
    # Run the fetch process
    fetch_all_historical_data(batch_size=2, sleep_time=1)
    
    # Verify StockDataFetcher was initialized
    mock_fetcher_class.assert_called_once()
    
    # Verify update_symbols was called
    assert mock_fetcher_instance.update_symbols.called
    update_calls = mock_fetcher_instance.update_symbols.call_args_list
    assert len(update_calls) == 1
    
    # Get the symbols that were passed to update_symbols
    symbols_updated = update_calls[0].args[0]
    deduped_symbols = set(s for s in symbols_updated if not s.startswith('^'))
    
    # Verify we have the expected stock symbols (ignoring indices)
    assert len(deduped_symbols) == 3
    assert all(s in deduped_symbols for s in ['AAPL', 'GOOGL', 'AMZN'])

def test_check_database_status(mock_db_path, mocker):
    """Test database status checking"""
    # Create mock fetcher
    mock_fetcher = MagicMock()
    mock_fetcher.db_path = str(mock_db_path)
    
    # Mock sqlite3.connect
    mock_conn = MagicMock()
    mocker.patch('sqlite3.connect', return_value=mock_conn)
    
    # Mock pd.read_sql_query results
    mock_read_sql = mocker.patch('pandas.read_sql_query')
    mock_read_sql.side_effect = [
        pd.DataFrame({'count': [100]}),
        pd.DataFrame({
            'min': ['2020-01-01'],
            'max': ['2024-01-01']
        }, columns=['min', 'max']),
        pd.DataFrame({'count': [1000000]})
    ]
    
    # Run status check
    check_database_status(mock_fetcher)
    
    # Verify expected number of database queries
    expected_queries = [
        "SELECT COUNT(DISTINCT symbol) FROM daily_prices",
        "SELECT MIN(date) as min, MAX(date) as max FROM daily_prices",
        "SELECT COUNT(*) FROM daily_prices"
    ]
    
    actual_queries = [call.args[0] for call in mock_read_sql.call_args_list]
    assert len(actual_queries) == len(expected_queries)
    
    # Close should be called once
    mock_conn.close.assert_called_once()

def test_fetch_all_historical_data_with_errors(mock_db_path, mocker):
    """Test error handling during the fetch process"""
    # Mock symbol fetching
    mocker.patch('fetch_stock_data.get_sp500_symbols', return_value=['AAPL', 'GOOGL'])
    mocker.patch('fetch_stock_data.get_nasdaq_symbols', return_value=['AAPL', 'AMZN'])
    
    # Mock StockDataFetcher to raise an exception
    mock_fetcher = mocker.patch('fetch_stock_data.StockDataFetcher')
    mock_fetcher_instance = MagicMock()
    mock_fetcher_instance.update_symbols.side_effect = Exception("API Error")
    mock_fetcher.return_value = mock_fetcher_instance
    
    # Mock time.sleep
    mocker.patch('time.sleep')
    
    # Mock logger
    mock_logger = mocker.patch('logging.error')
    
    # Run the fetch process (should not raise exception)
    fetch_all_historical_data(batch_size=2, sleep_time=1)
    
    # Verify error was logged
    assert mock_logger.called
    error_messages = [str(call.args[0]) for call in mock_logger.call_args_list]
    assert any("Error" in msg for msg in error_messages)