import pytest
import pandas as pd
import numpy as np
from market_analyzer import MarketAnalyzer, PatternRecognition, TechnicalPattern

@pytest.fixture
def pattern_data():
    """Create sample data with clear technical patterns for testing.
    
    This fixture creates three distinct patterns:
    
    1. Head and Shoulders Pattern:
       - Left shoulder:  Peak at 110
       - Trough after:   100
       - Head:           120 (clearly higher than shoulders)
       - Trough after:   100
       - Right shoulder: 111 (within 2% of left shoulder)
       - Final trough:   100
       
       This matches our detection criteria:
       - Head is >2% higher than shoulders (120 vs 110)
       - Shoulders within 2% of each other (110 vs 111)
       - Neckline is horizontal at 100
       
    2. Double Bottom Pattern:
       - First bottom:  80
       - Middle peak:   95
       - Second bottom: 81 (within 2% of first bottom)
       - Final rise:    100
       
       This matches our detection criteria:
       - Bottoms within 2% of each other (80 vs 81)
       - Clear peak between bottoms
       - Significant bounce from bottoms
       
    3. Volume-Price Divergence:
       We create opposing trends in price and volume:
       - Rising prices with declining volume
       - Clear directional difference for detection
    """
    # Create date range
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='B')
    
    # Head and Shoulders pattern (90 days total)
    h_and_s = np.concatenate([
        np.linspace(100, 110, 15),  # Left shoulder up
        np.linspace(110, 100, 15),  # Left shoulder down
        np.linspace(100, 120, 15),  # Head up
        np.linspace(120, 100, 15),  # Head down
        np.linspace(100, 111, 15),  # Right shoulder up (within 2% of left)
        np.linspace(111, 100, 15)   # Right shoulder down
    ])
    
    # Double Bottom pattern (60 days total)
    double_bottom = np.concatenate([
        np.linspace(100, 80, 15),   # First bottom down
        np.linspace(80, 95, 15),    # Rise between bottoms
        np.linspace(95, 81, 15),    # Second bottom (within 2% of first)
        np.linspace(81, 100, 15)    # Final rise
    ])
    
    # Create base price data
    prices = np.concatenate([
        h_and_s,                    # First 90 days: H&S pattern
        double_bottom,              # Next 60 days: Double bottom
        np.full(len(dates)-150, 100)  # Fill remaining days with stable price
    ])
    
    # Create volume data with clear divergence
    volumes = np.concatenate([
        np.linspace(1e6, 1e6, 90),      # Stable volume for H&S
        np.linspace(1e6, 1e6, 60),      # Stable volume for double bottom
        np.linspace(2e6, 1e6, 50),      # Declining volume despite stable price
        np.full(len(dates)-200, 1e6)    # Fill remaining days
    ])
    
    return pd.Series(prices, index=dates), pd.Series(volumes, index=dates)

@pytest.fixture
def pattern_recognition(pattern_data):
    """Create PatternRecognition instance with sample data."""
    prices, volumes = pattern_data
    return PatternRecognition(prices, volumes)

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

def test_pattern_data_validation(pattern_data):
    """Verify that our test data contains the patterns we expect."""
    prices, volumes = pattern_data
    
    # Get the specific segments we want to verify
    h_and_s = prices[:90]  # First 90 days contain H&S pattern
    double_bottom = prices[90:150]  # Next 60 days contain double bottom
    
    # Head and Shoulders validation
    left_shoulder = h_and_s[14]   # Peak of left shoulder
    head = h_and_s[44]            # Peak of head
    right_shoulder = h_and_s[74]  # Peak of right shoulder
    neckline_left = h_and_s[29]   # Trough after left shoulder
    neckline_right = h_and_s[59]  # Trough after head
    
    # Verify H&S properties
    shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
    head_height = (head - max(left_shoulder, right_shoulder)) / max(left_shoulder, right_shoulder)
    neckline_slope = abs(neckline_left - neckline_right) / neckline_left
    
    assert shoulder_diff < 0.02, f"Shoulders differ by {shoulder_diff:.1%}, should be <2%"
    assert head_height > 0.02, f"Head only {head_height:.1%} above shoulders, should be >2%"
    assert neckline_slope < 0.02, f"Neckline slope {neckline_slope:.1%}, should be <2%"
    
    # Double Bottom validation
    bottom1 = double_bottom[14]    # First bottom
    bottom2 = double_bottom[44]    # Second bottom
    middle_peak = double_bottom[29]  # Peak between bottoms
    
    # Verify Double Bottom properties
    bottom_diff = abs(bottom1 - bottom2) / bottom1
    bounce_height = (middle_peak - min(bottom1, bottom2)) / min(bottom1, bottom2)
    
    assert bottom_diff < 0.02, f"Bottoms differ by {bottom_diff:.1%}, should be <2%"
    assert bounce_height > 0.05, f"Bounce only {bounce_height:.1%}, should be >5%"
    
    # Volume-Price divergence validation
    divergence_prices = prices[150:200]  # 50 days of divergence
    divergence_volumes = volumes[150:200]
    
    price_direction = np.sign(divergence_prices.iloc[-1] - divergence_prices.iloc[0])
    volume_direction = np.sign(divergence_volumes.iloc[-1] - divergence_volumes.iloc[0])
    
    assert price_direction != volume_direction, "Price and volume should move in opposite directions"

class TestTechnicalPattern:
    """Test the TechnicalPattern data class."""
    
    def test_pattern_creation(self):
        """Test creation of TechnicalPattern instances."""
        pattern = TechnicalPattern(
            pattern_type="HEAD_AND_SHOULDERS",
            start_idx=0,
            end_idx=10,
            confidence=0.8,
            price_range=(100.0, 120.0)
        )
        
        assert pattern.pattern_type == "HEAD_AND_SHOULDERS"
        assert pattern.start_idx == 0
        assert pattern.end_idx == 10
        assert pattern.confidence == 0.8
        assert pattern.price_range == (100.0, 120.0)

class TestPatternRecognition:
    """Test pattern detection and analysis functionality."""
    
    def test_swing_points_detection(self, pattern_recognition):
        """Test identification of swing highs and lows."""
        highs, lows = pattern_recognition.find_swing_points()
        
        assert isinstance(highs, np.ndarray)
        assert isinstance(lows, np.ndarray)
        assert len(highs) > 0
        assert len(lows) > 0
        
        # Verify highs are local maxima
        for idx in highs:
            window_slice = slice(max(0, idx-5), min(len(pattern_recognition.prices), idx+6))
            local_max = pattern_recognition.prices.iloc[window_slice].max()
            assert pattern_recognition.prices.iloc[idx] == local_max
    
    def test_head_and_shoulders_detection(self, pattern_recognition):
        """Test head and shoulders pattern detection."""
        patterns = pattern_recognition.detect_head_and_shoulders()
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        
        for pattern in patterns:
            assert isinstance(pattern, TechnicalPattern)
            assert pattern.pattern_type == "HEAD_AND_SHOULDERS"
            assert 0 <= pattern.confidence <= 1
            
            # Check pattern structure
            prices = pattern_recognition.prices
            left_shoulder = prices.iloc[pattern.start_idx]
            right_shoulder_idx = pattern.end_idx
            head_idx = pattern.start_idx + (right_shoulder_idx - pattern.start_idx) // 2
            
            assert prices.iloc[head_idx] > left_shoulder
            assert prices.iloc[head_idx] > prices.iloc[right_shoulder_idx]

    def test_head_and_shoulders_validation(self, pattern_recognition):
        """Test specific validation scenarios for head and shoulders patterns."""
        patterns = pattern_recognition.detect_head_and_shoulders()
        
        for pattern in patterns:
            if pattern.specific_points:
                head_idx = pattern.specific_points['head'][0]
                
                # Head should be global maximum in pattern
                pattern_slice = pattern_recognition.prices.iloc[pattern.start_idx:pattern.end_idx]
                assert pattern_recognition.prices.iloc[head_idx] == pattern_slice.max()
                
                # Verify troughs
                left_trough_idx = pattern.specific_points['left_trough'][0]
                right_trough_idx = pattern.specific_points['right_trough'][0]
                
                assert left_trough_idx > pattern.start_idx
                assert left_trough_idx < head_idx
                assert right_trough_idx > head_idx
                assert right_trough_idx < pattern.end_idx   

    def test_double_bottom_detection(self, pattern_recognition):
        """Test double bottom pattern detection."""
        patterns = pattern_recognition.detect_double_bottom()
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        
        for pattern in patterns:
            assert isinstance(pattern, TechnicalPattern)
            assert pattern.pattern_type == "DOUBLE_BOTTOM"
            assert 0 <= pattern.confidence <= 1
            
            # Check pattern structure
            prices = pattern_recognition.prices
            bottom1 = prices.iloc[pattern.start_idx]
            bottom2 = prices.iloc[pattern.end_idx]
            
            # Bottoms should be at similar levels
            assert abs(bottom1 - bottom2) / bottom1 < 0.02
            
            # Middle point should be higher
            middle_idx = (pattern.start_idx + pattern.end_idx) // 2
            middle_price = prices.iloc[middle_idx]
            assert middle_price > min(bottom1, bottom2)
    
    def test_volume_price_divergence(self, pattern_recognition):
        """Test volume-price divergence detection."""
        patterns = pattern_recognition.detect_volume_price_divergence()
        
        assert isinstance(patterns, list)
        
        for pattern in patterns:
            assert isinstance(pattern, TechnicalPattern)
            assert pattern.pattern_type == "VOLUME_PRICE_DIVERGENCE"
            assert 0 <= pattern.confidence <= 1
            
            # Check divergence
            price_change = (pattern_recognition.prices.iloc[pattern.end_idx] - 
                          pattern_recognition.prices.iloc[pattern.start_idx])
            volume_change = (pattern_recognition.volumes.iloc[pattern.end_idx] - 
                           pattern_recognition.volumes.iloc[pattern.start_idx])
            
            # Price and volume should move in opposite directions
            assert np.sign(price_change) != np.sign(volume_change)

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

class TestMarketAnalyzerPatterns:
    """Test pattern analysis integration with MarketAnalyzer."""
    
    @pytest.fixture
    def market_analyzer_with_patterns(self, pattern_data):
        """Create MarketAnalyzer instance with pattern test data."""
        prices, volumes = pattern_data
        data = {
            'TEST': pd.DataFrame({
                'open': prices,
                'high': prices * 1.01,
                'low': prices * 0.99,
                'close': prices,
                'volume': volumes
            })
        }
        return MarketAnalyzer(data=data, market_indices=[])
    
    def test_analyze_patterns(self, market_analyzer_with_patterns):
        """Test pattern analysis through MarketAnalyzer interface."""
        patterns = market_analyzer_with_patterns.analyze_patterns('TEST')
        
        # Check all pattern types are present
        assert all(key in patterns for key in [
            'head_and_shoulders',
            'double_bottom',
            'volume_price_divergence'
        ])
        
        # Check each pattern type
        for pattern_list in patterns.values():
            assert isinstance(pattern_list, list)
            for pattern in pattern_list:
                assert isinstance(pattern, TechnicalPattern)
                assert 0 <= pattern.confidence <= 1
                assert pattern.start_idx < pattern.end_idx
                assert pattern.price_range[0] < pattern.price_range[1]
    
    def test_analyze_patterns_date_range(self, market_analyzer_with_patterns):
        """Test pattern analysis with date filtering."""
        # Use dates that match our synthetic data patterns
        start_date = '2023-02-01'
        end_date = '2023-11-30'
        
        patterns = market_analyzer_with_patterns.analyze_patterns(
            'TEST',
            start_date=start_date,
            end_date=end_date
        )
        
        # Check that we got some patterns
        assert any(len(pattern_list) > 0 for pattern_list in patterns.values())
        
        # Verify patterns are within date range
        for pattern_list in patterns.values():
            for pattern in pattern_list:
                pattern_dates = market_analyzer_with_patterns.data['TEST'].index
                pattern_start = pattern_dates[pattern.start_idx]
                pattern_end = pattern_dates[pattern.end_idx]
                
                # Add debugging output
                if not (pattern_start >= pd.Timestamp(start_date) and 
                       pattern_end <= pd.Timestamp(end_date)):
                    print(f"Pattern outside range: {pattern_start} to {pattern_end}")
                
                assert pattern_start >= pd.Timestamp(start_date)
                assert pattern_end <= pd.Timestamp(end_date)          