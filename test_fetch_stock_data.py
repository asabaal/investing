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
def mock_nasdaq_data():
    """Mock NASDAQ-100 data from Wikipedia"""
    return pd.DataFrame({
        'Ticker': ['AAPL', 'AMZN', 'NFLX'],
        'Company': ['Apple Inc.', 'Amazon.com', 'Netflix, Inc.'],
    })

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

def test_get_sp500_symbols_error(mocker):
    """Test error handling when fetching S&P 500 symbols"""
    mocker.patch('pandas.read_html', side_effect=Exception("Connection error"))
    
    symbols = get_sp500_symbols()
    
    assert isinstance(symbols, list)
    assert len(symbols) == 0

def test_get_nasdaq_symbols(mock_nasdaq_data, mocker):
    """Test fetching NASDAQ symbols"""
    mocker.patch('pandas.read_html', return_value=[pd.DataFrame(), mock_nasdaq_data])
    
    symbols = get_nasdaq_symbols()
    
    assert isinstance(symbols, list)
    assert len(symbols) == 3
    assert 'AAPL' in symbols
    assert 'AMZN' in symbols

def test_get_nasdaq_symbols_error(mocker):
    """Test error handling when fetching NASDAQ symbols"""
    mocker.patch('pandas.read_html', side_effect=Exception("Connection error"))
    
    symbols = get_nasdaq_symbols()
    
    assert isinstance(symbols, list)
    assert len(symbols) == 0

def test_fetch_all_historical_data(mock_sp500_data, mock_nasdaq_data, mock_db_path, mocker):
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
    
    # Verify fetch_data was called with correct batches
    calls = mock_fetcher_instance.fetch_data.call_args_list
    assert len(calls) > 0
    
    # Verify batch size
    for call in calls[:-1]:  # All but last batch
        assert len(call.args[0]) <= 2

def test_check_database_status(mock_db_path, mocker):
    """Test database status checking"""
    # Mock sqlite3.connect
    mock_conn = MagicMock()
    mocker.patch('sqlite3.connect', return_value=mock_conn)
    
    # Mock pd.read_sql_query results
    mock_symbols_result = pd.DataFrame({'count': [100]})
    mock_date_result = pd.DataFrame({
        'earliest_date': ['2020-01-01'],
        'latest_date': ['2024-01-01']
    })
    mock_records_result = pd.DataFrame({'count': [1000000]})
    
    # Setup mock for pd.read_sql_query
    mock_read_sql = mocker.patch('pandas.read_sql_query')
    mock_read_sql.side_effect = [
        mock_symbols_result,
        mock_date_result,
        mock_records_result
    ]
    
    # Run status check
    check_database_status()
    
    # Verify database connection was made
    assert mock_conn.close.called
    
    # Verify queries were executed
    assert mock_read_sql.call_count == 3

def test_fetch_all_historical_data_with_errors(mock_sp500_data, mock_nasdaq_data, mock_db_path, mocker):
    """Test error handling during the fetch process"""
    # Mock symbol fetching
    mocker.patch('fetch_stock_data.get_sp500_symbols', return_value=['AAPL', 'GOOGL'])
    mocker.patch('fetch_stock_data.get_nasdaq_symbols', return_value=['AAPL', 'AMZN'])
    
    # Mock StockDataFetcher to raise an exception
    mock_fetcher = mocker.patch('fetch_stock_data.StockDataFetcher')
    mock_fetcher_instance = MagicMock()
    mock_fetcher_instance.fetch_data.side_effect = Exception("API Error")
    mock_fetcher.return_value = mock_fetcher_instance
    
    # Mock time.sleep
    mocker.patch('time.sleep')
    
    # Mock logger instead of using LogCaptureFixture
    mock_logger = mocker.patch('logging.error')
    
    # Run the fetch process
    fetch_all_historical_data(batch_size=2, sleep_time=1)
    
    # Verify error was logged
    assert mock_logger.called
    assert any("Error processing batch" in str(call) for call in mock_logger.call_args_list)

if __name__ == '__main__':
    pytest.main(['-v'])