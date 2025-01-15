import pytest
import pandas as pd
import numpy as np
from market_analyzer import MarketAnalyzer  # Assume this is the file name

@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    # Create date range
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    
    # Generate sample data for two indices and one stock
    data = {}
    
    # S&P 500 synthetic data
    sp500_data = pd.DataFrame({
        'open': np.linspace(4000, 4500, len(dates)) + np.random.normal(0, 50, len(dates)),
        'high': np.linspace(4050, 4550, len(dates)) + np.random.normal(0, 50, len(dates)),
        'low': np.linspace(3950, 4450, len(dates)) + np.random.normal(0, 50, len(dates)),
        'close': np.linspace(4000, 4500, len(dates)) + np.random.normal(0, 50, len(dates)),
        'volume': np.random.uniform(1e6, 5e6, len(dates))
    }, index=dates)
    data['^GSPC'] = sp500_data

    # Dow Jones synthetic data
    dow_data = pd.DataFrame({
        'open': np.linspace(33000, 35000, len(dates)) + np.random.normal(0, 300, len(dates)),
        'high': np.linspace(33050, 35050, len(dates)) + np.random.normal(0, 300, len(dates)),
        'low': np.linspace(32950, 34950, len(dates)) + np.random.normal(0, 300, len(dates)),
        'close': np.linspace(33000, 35000, len(dates)) + np.random.normal(0, 300, len(dates)),
        'volume': np.random.uniform(5e5, 2e6, len(dates))
    }, index=dates)
    data['^DJI'] = dow_data

    # Sample stock data
    stock_data = pd.DataFrame({
        'open': np.linspace(100, 150, len(dates)) + np.random.normal(0, 5, len(dates)),
        'high': np.linspace(102, 152, len(dates)) + np.random.normal(0, 5, len(dates)),
        'low': np.linspace(98, 148, len(dates)) + np.random.normal(0, 5, len(dates)),
        'close': np.linspace(100, 150, len(dates)) + np.random.normal(0, 5, len(dates)),
        'volume': np.random.uniform(1e5, 1e6, len(dates))
    }, index=dates)
    data['AAPL'] = stock_data

    return data

@pytest.fixture
def market_analyzer(sample_data):
    """Create a MarketAnalyzer instance with sample data."""
    return MarketAnalyzer(
        data=sample_data,
        market_indices=['^GSPC', '^DJI'],
        benchmark_index='^GSPC'
    )

class TestMarketAnalyzerInitialization:
    """Test the initialization and basic setup of MarketAnalyzer."""

    def test_initialization(self, market_analyzer, sample_data):
        """Test if MarketAnalyzer initializes correctly with sample data."""
        assert isinstance(market_analyzer, MarketAnalyzer)
        assert market_analyzer.data == sample_data
        assert market_analyzer.market_indices == ['^GSPC', '^DJI']
        assert market_analyzer.benchmark_index == '^GSPC'

    def test_returns_data_preparation(self, market_analyzer):
        """Test if returns data is correctly calculated and stored."""
        for symbol in market_analyzer.data.keys():
            assert symbol in market_analyzer.returns_data
            returns_df = market_analyzer.returns_data[symbol]
            
            # Check if all required columns are present
            assert all(col in returns_df.columns 
                      for col in ['daily_return', 'log_return', 'volume'])
            
            # Check if returns are properly calculated
            price_data = market_analyzer.data[symbol]['close']
            expected_daily_return = price_data.pct_change()
            pd.testing.assert_series_equal(
                returns_df['daily_return'],
                expected_daily_return,
                check_names=False
            )

class TestMarketAnalyzerFeatures:
    """Test feature calculation methods of MarketAnalyzer."""

    def test_rolling_features(self, market_analyzer):
        """Test calculation of rolling window features."""
        symbol = '^GSPC'
        windows = [5, 21]
        features = market_analyzer.calculate_rolling_features(symbol, windows=windows)

        # Check if all expected features are present
        expected_columns = []
        for window in windows:
            expected_columns.extend([
                f'return_mean_{window}d',
                f'return_std_{window}d',
                f'return_skew_{window}d',
                f'return_kurt_{window}d',
                f'price_mean_{window}d',
                f'price_std_{window}d',
                f'volume_mean_{window}d',
                f'volume_std_{window}d',
                f'RSI_{window}d'
            ])
        
        assert all(col in features.columns for col in expected_columns)
        
        # Check if calculations are correct for a specific window
        window = 5
        symbol_data = market_analyzer.data[symbol]['close']
        returns = market_analyzer.returns_data[symbol]['daily_return']
        
        # Test mean calculation
        expected_mean = returns.rolling(window=window).mean()
        pd.testing.assert_series_equal(
            features[f'return_mean_{window}d'],
            expected_mean,
            check_names=False
        )

    def test_rsi_calculation(self, market_analyzer):
        """Test RSI calculation."""
        # Test with a controlled sequence of prices
        prices = pd.Series([
            10, 10.5, 10.3, 10.2, 10.4, 10.3, 10.7,
            10.5, 10.2, 10.3, 10.4, 10.8, 10.9, 11.0,
            11.1, 11.2, 11.3, 11.4, 11.5, 11.6
        ])
        window = 14
        rsi = market_analyzer._calculate_rsi(prices, window)
        
        # Basic checks
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(prices)
        
        # Check if RSI values are within valid range (excluding NaN values)
        valid_rsi = rsi.dropna()
        assert valid_rsi.between(0, 100).all(), f"RSI values out of range: {valid_rsi}"
        
        # First window-1 values should be NaN due to rolling window
        assert rsi[:window-1].isna().all()
        
        # Test with purely increasing prices
        rising_prices = pd.Series(np.linspace(100, 200, 50))
        rising_rsi = market_analyzer._calculate_rsi(rising_prices, window)
        # RSI should be high (>70) for consistently rising prices
        assert rising_rsi.iloc[-1] > 70, f"RSI for rising prices: {rising_rsi.iloc[-1]}"
        
        # Test with purely decreasing prices
        falling_prices = pd.Series(np.linspace(200, 100, 50))
        falling_rsi = market_analyzer._calculate_rsi(falling_prices, window)
        # RSI should be low (<30) for consistently falling prices
        assert falling_rsi.iloc[-1] < 30, f"RSI for falling prices: {falling_rsi.iloc[-1]}"

class TestMarketAnalyzerRegimes:
    """Test regime detection functionality."""

    def test_volatility_regime_detection(self, market_analyzer):
        """Test GARCH-based volatility regime detection."""
        symbol = '^GSPC'
        lookback = 50  # Shorter lookback for testing
        regime_data = market_analyzer.detect_volatility_regime(symbol, lookback)
        
        # Check if regime data has expected columns
        assert all(col in regime_data.columns for col in ['volatility', 'regime'])
        
        # Check if regimes are properly classified
        assert regime_data['regime'].dropna().isin([0, 1, 2]).all()
        
        # Check if volatility values are non-negative
        assert (regime_data['volatility'].dropna() >= 0).all()

class TestMarketAnalyzerState:
    """Test market state analysis functionality."""

    def test_market_state_analysis(self, market_analyzer):
        """Test overall market state analysis."""
        start_date = '2023-06-01'
        end_date = '2023-12-31'
        market_state = market_analyzer.analyze_market_state(start_date, end_date)
        
        # Check if analysis is performed for all indices
        assert all(index in market_state for index in market_analyzer.market_indices)
        
        # Check if all required features are calculated
        for index, features in market_state.items():
            assert all(col in features.columns for col in [
                'trend_20d',
                'volatility_20d',
                'momentum_20d'
            ])
            
            # Check date range
            assert features.index[0].strftime('%Y-%m-%d') >= start_date
            assert features.index[-1].strftime('%Y-%m-%d') <= end_date
            
            # Check if trend indicators are binary
            assert features['trend_20d'].isin([0, 1]).all()
            
            # Check if volatility is non-negative
            assert (features['volatility_20d'].dropna() >= 0).all()