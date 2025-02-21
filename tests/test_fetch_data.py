import pytest
import pandas as pd
import requests
from unittest.mock import patch, MagicMock
from datetime import datetime
from market_analyzer.data_fetcher import MarketDataFetcher

@pytest.fixture
def test_config_dir(tmp_path):
    """Create a temporary directory for test config files"""
    return str(tmp_path / "test_config")

@pytest.fixture
def mock_stock_data():
    """Fixture for sample stock data"""
    return pd.DataFrame({
        'date': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        'symbol': ['AAPL', 'AAPL'],
        'open': [100.0, 101.0],
        'high': [102.0, 103.0],
        'low': [98.0, 99.0],
        'close': [101.0, 102.0],
        'volume': [1000000, 1100000],
        'dividends': [0, 0],
        'stock_splits': [0, 0],
        'data_source': ['yfinance', 'yfinance'],
        'last_updated': [datetime.now(), datetime.now()]
    })

@pytest.fixture
def fetcher(test_config_dir):
    """Create MarketDataFetcher instance with test configuration"""
    with patch('pathlib.Path.write_text'):  # Mock config file writing
        return MarketDataFetcher(db_path=':memory:', config_dir=test_config_dir)

def test_get_sp500_symbols(fetcher, mocker):
    """Test S&P 500 symbols retrieval"""
    mock_data = pd.DataFrame({'Symbol': ['AAPL', 'GOOGL']})
    mocker.patch('pandas.read_html', return_value=[mock_data])
    
    symbols = fetcher.get_sp500_symbols()
    assert len(symbols) == 2
    assert 'AAPL' in symbols

def test_get_nasdaq_symbols_primary(fetcher, mocker):
    """Test NASDAQ symbols retrieval using primary method"""
    mock_ticker = mocker.MagicMock()
    mock_ticker.holdings = [{'symbol': 'AAPL'}, {'symbol': 'MSFT'}]
    mocker.patch('yfinance.Ticker', return_value=mock_ticker)
    
    symbols = fetcher.get_nasdaq_symbols()
    assert len(symbols) == 2
    assert 'AAPL' in symbols

def test_get_nasdaq_symbols_fallback(fetcher, mocker):
    """Test NASDAQ symbols retrieval fallback"""
    # Mock yfinance to fail
    mocker.patch('yfinance.Ticker', side_effect=Exception("API Error"))
    
    # Mock NASDAQ API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        'data': {'rows': [{'symbol': 'AAPL'}, {'symbol': 'MSFT'}]}
    }
    mocker.patch('requests.get', return_value=mock_response)
    
    symbols = fetcher.get_nasdaq_symbols()
    assert len(symbols) == 2
    assert 'AAPL' in symbols

def test_fetch_symbols(fetcher, mock_stock_data, mocker):
    """Test fetching specific symbols"""
    # Mock the underlying StockDataFetcher
    mock_stock_fetcher = mocker.MagicMock()
    fetcher.fetcher = mock_stock_fetcher
    
    # Mock config handler
    mock_config = mocker.MagicMock()
    mock_config.can_make_api_call.return_value = True
    fetcher.config_handler = mock_config
    
    # Test fetching single symbol
    fetcher.fetch_symbols('AAPL')
    mock_stock_fetcher.fetch_data.assert_called_once()
    
    # Test fetching multiple symbols
    mock_stock_fetcher.fetch_data.reset_mock()
    fetcher.fetch_symbols(['AAPL', 'GOOGL'])
    mock_stock_fetcher.fetch_data.assert_called_once()

def test_fetch_symbols_with_dates(fetcher, mocker):
    """Test fetching symbols with date range"""
    mock_stock_fetcher = mocker.MagicMock()
    fetcher.fetcher = mock_stock_fetcher
    
    start_date = '2024-01-01'
    end_date = '2024-01-31'
    
    fetcher.fetch_symbols('AAPL', start_date=start_date, end_date=end_date)
    
    mock_stock_fetcher.fetch_data.assert_called_once_with(
        ['AAPL'],
        start_date=start_date,
        end_date=end_date,
        verify_across_sources=True
    )

def test_fetch_symbols_api_limit(fetcher, mocker):
    """Test handling API limits when fetching symbols"""
    mock_config = mocker.MagicMock()
    mock_config.can_make_api_call.return_value = False
    mock_config.get_remaining_calls.return_value = 0
    fetcher.config_handler = mock_config
    
    mock_stock_fetcher = mocker.MagicMock()
    fetcher.fetcher = mock_stock_fetcher
    
    fetcher.fetch_symbols('AAPL', verify_sources=True)
    
    # Should call fetch_data with verify_across_sources=False when API limit reached
    mock_stock_fetcher.fetch_data.assert_called_once_with(
        ['AAPL'],
        start_date=None,
        end_date=None,
        verify_across_sources=False
    )

def test_fetch_all_historical_data(fetcher, mocker):
    """Test fetching all historical data"""
    # Mock symbol retrieval
    mocker.patch.object(fetcher, 'get_sp500_symbols', return_value=['AAPL', 'GOOGL'])
    mocker.patch.object(fetcher, 'get_nasdaq_symbols', return_value=['MSFT', 'AMZN'])
    
    # Mock the fetcher
    mock_stock_fetcher = mocker.MagicMock()
    fetcher.fetcher = mock_stock_fetcher
    
    fetcher.fetch_all_historical_data(batch_size=2)
    
    # Should process unique symbols (4 symbols + 6 indices = 10 symbols)
    expected_calls = 5  # 10 symbols / batch_size of 2
    assert mock_stock_fetcher.update_symbols.call_count == expected_calls

def test_check_database_status(fetcher, mocker):
    """Test database status checking"""
    mock_conn = MagicMock()
    mock_execute = MagicMock()
    
    # Mock database queries
    def mock_read_sql(*args, **kwargs):
        if "COUNT(DISTINCT symbol)" in args[0]:
            return pd.DataFrame({'count': [100]})
        elif "COUNT(*)" in args[0]:
            return pd.DataFrame({'count': [1000]})
        elif "MIN(date)" in args[0]:
            return pd.DataFrame({'earliest': ['2024-01-01'], 'latest': ['2024-01-31']})
        elif "MAX(last_updated)" in args[0]:
            return pd.DataFrame({'max': ['2024-01-31']})
        else:
            return pd.DataFrame({'data_source': ['yfinance', 'alpha_vantage']})
    
    mocker.patch('sqlite3.connect', return_value=mock_conn)
    mocker.patch('pandas.read_sql_query', side_effect=mock_read_sql)
    
    # Should not raise any exceptions
    fetcher.check_database_status()

def test_fetch_symbols_single(fetcher, mock_stock_data, mocker):
    """Test fetching a single symbol"""
    # Mock the underlying fetcher
    mock_fetcher = mocker.MagicMock()
    fetcher.fetcher = mock_fetcher
    mock_fetcher.fetch_data.return_value = mock_stock_data
    
    # Mock config handler
    mock_config = mocker.MagicMock()
    mock_config.can_make_api_call.return_value = True
    fetcher.config_handler = mock_config
    
    # Test with single symbol as string
    fetcher.fetch_symbols('AAPL')
    mock_fetcher.fetch_data.assert_called_once_with(
        ['AAPL'],
        start_date=None,
        end_date=None,
        verify_across_sources=True
    )

def test_fetch_symbols_multiple(fetcher, mock_stock_data, mocker):
    """Test fetching multiple symbols"""
    mock_fetcher = mocker.MagicMock()
    fetcher.fetcher = mock_fetcher
    mock_fetcher.fetch_data.return_value = mock_stock_data
    
    mock_config = mocker.MagicMock()
    mock_config.can_make_api_call.return_value = True
    fetcher.config_handler = mock_config
    
    # Test with list of symbols
    symbols = ['AAPL', 'GOOGL', 'AAPL']  # Intentional duplicate
    fetcher.fetch_symbols(symbols)
    
    # Should remove duplicates
    mock_fetcher.fetch_data.assert_called_once_with(
        ['AAPL', 'GOOGL'],  # Duplicates removed
        start_date=None,
        end_date=None,
        verify_across_sources=True
    )

def test_fetch_symbols_with_date_range(fetcher, mock_stock_data, mocker):
    """Test fetching symbols with specific date range"""
    mock_fetcher = mocker.MagicMock()
    fetcher.fetcher = mock_fetcher
    mock_fetcher.fetch_data.return_value = mock_stock_data
    
    mock_config = mocker.MagicMock()
    mock_config.can_make_api_call.return_value = True
    fetcher.config_handler = mock_config
    
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    
    fetcher.fetch_symbols(['AAPL', 'GOOGL'], start_date=start_date, end_date=end_date)
    mock_fetcher.fetch_data.assert_called_once_with(
        ['AAPL', 'GOOGL'],
        start_date=start_date,
        end_date=end_date,
        verify_across_sources=True
    )

def test_fetch_symbols_api_limit_handling(fetcher, mock_stock_data, mocker):
    """Test handling of API limits during symbol fetch"""
    mock_fetcher = mocker.MagicMock()
    fetcher.fetcher = mock_fetcher
    mock_fetcher.fetch_data.return_value = mock_stock_data
    
    # Mock config handler to simulate API limit reached
    mock_config = mocker.MagicMock()
    mock_config.can_make_api_call.return_value = False
    mock_config.get_remaining_calls.return_value = 0
    fetcher.config_handler = mock_config
    
    fetcher.fetch_symbols(['AAPL', 'GOOGL'], verify_sources=True)
    
    # Should fall back to single source when API limit reached
    mock_fetcher.fetch_data.assert_called_once_with(
        ['AAPL', 'GOOGL'],
        start_date=None,
        end_date=None,
        verify_across_sources=False  # Should be False when API limit reached
    )

def test_fetch_symbols_error_handling(fetcher, mocker):
    """Test error handling during symbol fetch"""
    mock_fetcher = mocker.MagicMock()
    fetcher.fetcher = mock_fetcher
    mock_fetcher.fetch_data.side_effect = Exception("API Error")
    
    mock_config = mocker.MagicMock()
    mock_config.can_make_api_call.return_value = True
    fetcher.config_handler = mock_config
    
    # Should raise the exception
    with pytest.raises(Exception) as exc_info:
        fetcher.fetch_symbols(['AAPL'])
    assert "API Error" in str(exc_info.value)

def test_fetch_symbols_api_call_recording(fetcher, mock_stock_data, mocker):
    """Test that API calls are properly recorded"""
    mock_fetcher = mocker.MagicMock()
    fetcher.fetcher = mock_fetcher
    mock_fetcher.fetch_data.return_value = mock_stock_data
    
    mock_config = mocker.MagicMock()
    mock_config.can_make_api_call.return_value = True
    fetcher.config_handler = mock_config
    
    fetcher.fetch_symbols(['AAPL'], verify_sources=True)
    
    # Should record API call when verification is used
    mock_config.record_api_call.assert_called_once_with('alpha_vantage')
    
    # Reset mocks
    mock_config.record_api_call.reset_mock()
    
    # Shouldn't record API call when verification is disabled
    fetcher.fetch_symbols(['AAPL'], verify_sources=False)
    mock_config.record_api_call.assert_not_called()

def test_fetch_symbols_status_update(fetcher, mock_stock_data, mocker):
    """Test that database status is checked after fetch"""
    mock_fetcher = mocker.MagicMock()
    fetcher.fetcher = mock_fetcher
    mock_fetcher.fetch_data.return_value = mock_stock_data
    
    mock_config = mocker.MagicMock()
    mock_config.can_make_api_call.return_value = True
    fetcher.config_handler = mock_config
    
    # Mock check_database_status
    mock_check_status = mocker.patch.object(fetcher, 'check_database_status')
    
    fetcher.fetch_symbols(['AAPL'])
    
    # Should call check_database_status after fetch
    mock_check_status.assert_called_once()    