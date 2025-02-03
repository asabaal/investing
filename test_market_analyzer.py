import pytest
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from market_analyzer import MarketAnalyzer, PatternRecognition, TechnicalPattern, LeadLagAnalyzer, MarketVisualizer, RiskAnalyzer, VolumePatternType, VolumePattern
from typing import Dict, Any

@pytest.fixture
def volume_price_test_1():
    """
    Create test data with clear examples of each pattern type.
    Returns both the data and expected point-by-point classifications.
    """
    prices = []
    volumes = []
    expected_points = []
    
    # DIVERGENCE: 5 days of rising price, falling volume
    prices.extend(np.linspace(100, 110, 5))
    volumes.extend(np.linspace(2e6, 1e6, 5))
    expected_points.extend([VolumePatternType.DIVERGENCE] * 5)
    
    # NEUTRAL: 3 days flat
    prices.extend([110] * 3)
    volumes.extend([1e6] * 3)
    expected_points.extend([VolumePatternType.NEUTRAL] * 3)
    
    # VOLUME_FORCE: 4 days flat price, rising volume
    prices.extend([110] * 4)
    volumes.extend(np.linspace(1.03e6, 2e6, 4))
    expected_points.extend([VolumePatternType.VOLUME_FORCE] * 4)
    
    # CONCORDANT: 3 days both rising (should be ignored in pattern detection)
    prices.extend(np.linspace(115, 130, 3))
    volumes.extend(np.linspace(2e6, 3e6, 3))
    expected_points.extend([VolumePatternType.CONCORDANT] * 3)
    
    # NON_CONFIRMATION: 4 days falling price, flat volume
    prices.extend(np.linspace(125, 110, 4))
    volumes.extend([3e6] * 4)
    expected_points.extend([VolumePatternType.NON_CONFIRMATION] * 4)
    
    dates = pd.date_range(start='2023-01-01', periods=len(prices), freq='B')
    df = pd.DataFrame({
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    # Expected consolidated patterns (4+ points)
    expected_patterns = [
        VolumePattern(
            pattern_type=VolumePatternType.DIVERGENCE,
            start_idx=1,
            end_idx=4,
            price_range=(102.5, 110),
            volume_range=(1e6, 1.75e6)
        ),
        VolumePattern(
            pattern_type=VolumePatternType.VOLUME_FORCE,
            start_idx=8,
            end_idx=11,
            price_range=(110, 110),
            volume_range=(1.03e6, 2e6)
        ),
        VolumePattern(
            pattern_type=VolumePatternType.NON_CONFIRMATION,
            start_idx=15,
            end_idx=18,
            price_range=(110, 125),
            volume_range=(3e6, 3e6)
        )
    ]
    
    return df, expected_points, expected_patterns

@pytest.fixture
def volume_price_test_2():
    """
    Create test data with edge cases and transitions.
    """
    prices = []
    volumes = []
    expected_points = [VolumePatternType.NEUTRAL]
    
    # Start with 3 days DIVERGENCE (not enough for pattern)
    prices.extend(np.linspace(100, 105, 4))
    volumes.extend(np.linspace(2e6, 1.8e6, 4))
    expected_points.extend([VolumePatternType.DIVERGENCE] * 3)
    
    # Follow with 5 days DIVERGENCE (forms pattern with previous points)
    prices.extend(np.linspace(110, 120, 5))
    volumes.extend(np.linspace(1.7e6, 1.4e6, 5))
    expected_points.extend([VolumePatternType.DIVERGENCE] * 5)
    
    # One day NEUTRAL
    prices.append(120)
    volumes.append(1.4e6)
    expected_points.append(VolumePatternType.NEUTRAL)
    
    # 4 days slight movement (should be NEUTRAL due to small changes)
    prices.extend(np.linspace(120, 120.1, 4))  # Very small price change
    volumes.extend(np.linspace(1.4e6, 1.41e6, 4))  # Very small volume change
    expected_points.extend([VolumePatternType.NEUTRAL] * 4)
    
    # 6 days alternating (should not form any pattern)
    for _ in range(3):
        prices.extend([125, 126])
        volumes.extend([1.4e6, 1.6e6])
    expected_points.extend([VolumePatternType.NEUTRAL, VolumePatternType.CONCORDANT, VolumePatternType.CONCORDANT, VolumePatternType.CONCORDANT, VolumePatternType.CONCORDANT, VolumePatternType.CONCORDANT])
    
    dates = pd.date_range(start='2023-01-01', periods=len(prices), freq='B')
    df = pd.DataFrame({
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    # Expected consolidated patterns (4+ points)
    expected_patterns = [
        VolumePattern(
            pattern_type=VolumePatternType.DIVERGENCE,
            start_idx=1,
            end_idx=8,
            price_range=(101.66666666666667, 120),
            volume_range=(1.4e6, 1933333.3333333333)
        ),
        VolumePattern(
            pattern_type=VolumePatternType.NEUTRAL,
            start_idx=9,
            end_idx=13,
            price_range=(120, 120.1),
            volume_range=(1.4e6, 1.41e6)
        )
    ]
    
    return df, expected_points, expected_patterns

@pytest.fixture
def lead_lag_sample_returns_data():
    """Create sample return data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
    np.random.seed(42)
    
    # Create correlated returns for testing
    returns_A = np.random.normal(0, 0.01, len(dates))
    # B follows A with lag
    returns_B = np.roll(returns_A, 2) + np.random.normal(0, 0.005, len(dates))
    # C is independent
    returns_C = np.random.normal(0, 0.01, len(dates))
    
    data = {
        'A': pd.DataFrame({
            'daily_return': pd.Series(returns_A, index=dates)
        }),
        'B': pd.DataFrame({
            'daily_return': pd.Series(returns_B, index=dates)
        }),
        'C': pd.DataFrame({
            'daily_return': pd.Series(returns_C, index=dates)
        })
    }
    return data

@pytest.fixture
def lead_lag_analyzer(lead_lag_sample_returns_data):
    """Create LeadLagAnalyzer instance with sample data."""
    return LeadLagAnalyzer(lead_lag_sample_returns_data)

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
        market_indices=['^GSPC', '^DJI']
    )

class TestRelationshipAnalysis:
    """Tests for the analyze_relationships method."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing relationships."""
        dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
        np.random.seed(42)
        
        # Create base returns with known relationships
        base_returns = np.random.normal(0, 0.01, len(dates))
        
        # SPY leads QQQ
        spy_returns = base_returns
        qqq_returns = np.roll(base_returns, 2) + np.random.normal(0, 0.003, len(dates))
        
        # AAPL independent
        aapl_returns = np.random.normal(0, 0.015, len(dates))
        
        data = {
            'SPY': pd.DataFrame({
                'daily_return': pd.Series(spy_returns, index=dates),
                'close': pd.Series(1000 * (1 + spy_returns).cumprod(), index=dates),
                'volume': pd.Series(np.random.randint(1000000, 2000000, len(dates)), index=dates)
            }),
            'QQQ': pd.DataFrame({
                'daily_return': pd.Series(qqq_returns, index=dates),
                'close': pd.Series(900 * (1 + qqq_returns).cumprod(), index=dates),
                'volume': pd.Series(np.random.randint(800000, 1600000, len(dates)), index=dates)
            }),
            'AAPL': pd.DataFrame({
                'daily_return': pd.Series(aapl_returns, index=dates),
                'close': pd.Series(150 * (1 + aapl_returns).cumprod(), index=dates),
                'volume': pd.Series(np.random.randint(500000, 1000000, len(dates)), index=dates)
            })
        }
        return data

    @pytest.fixture
    def market_analyzer(self, sample_market_data):
        """Create MarketAnalyzer instance with sample data."""
        return MarketAnalyzer(data=sample_market_data)

    def test_analyze_relationships_basic(self, market_analyzer):
        """Test basic functionality of analyze_relationships."""
        symbols = ['SPY', 'QQQ', 'AAPL']
        results = market_analyzer.analyze_relationships(symbols)
        
        # Check all expected keys are present
        expected_keys = {'cross_correlations', 'relationship_network', 
                        'market_leaders', 'granger_causality'}
        assert set(results.keys()) == expected_keys
        
        # Check cross_correlations structure
        assert isinstance(results['cross_correlations'], pd.DataFrame)
        assert set(results['cross_correlations'].columns) == {'symbol1', 'symbol2', 'lag', 'correlation'}
        
        # Check relationship_network structure
        assert isinstance(results['relationship_network'], nx.Graph)
        assert set(results['relationship_network'].nodes()) <= set(symbols)
        
        # Check market_leaders structure
        assert isinstance(results['market_leaders'], dict)
        assert set(results['market_leaders'].keys()) <= set(symbols)
        
        # Check granger_causality structure
        assert isinstance(results['granger_causality'], dict)
        for key in results['granger_causality']:
            assert '->' in key
            assert isinstance(results['granger_causality'][key], dict)

    def test_relationship_thresholds(self, market_analyzer):
        """Test behavior with different correlation thresholds."""
        symbols = ['SPY', 'QQQ', 'AAPL']
        
        # Test with high threshold
        high_thresh_results = market_analyzer.analyze_relationships(
            symbols, correlation_threshold=0.8
        )
        
        # Test with low threshold
        low_thresh_results = market_analyzer.analyze_relationships(
            symbols, correlation_threshold=0.2
        )
        
        # High threshold should have fewer edges
        assert (len(high_thresh_results['relationship_network'].edges()) <=
                len(low_thresh_results['relationship_network'].edges()))

    def test_max_lags_impact(self, market_analyzer):
        """Test impact of different max_lags values."""
        symbols = ['SPY', 'QQQ']
        
        # Test with different lag values
        short_lag_results = market_analyzer.analyze_relationships(symbols, max_lags=2)
        long_lag_results = market_analyzer.analyze_relationships(symbols, max_lags=10)
        
        # More lags should mean more correlation rows
        assert (len(short_lag_results['cross_correlations']) <
                len(long_lag_results['cross_correlations']))
        
        # More lags should mean more Granger test results
        assert all(len(v) <= 2 for v in short_lag_results['granger_causality'].values())
        assert all(len(v) <= 10 for v in long_lag_results['granger_causality'].values())


class TestRegimeAnalysis:
    """Tests for the analyze_regimes method."""
    
    @pytest.fixture
    def sample_regime_data(self):
        """Create sample data with known regime changes."""
        dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
        n_points = len(dates)
        np.random.seed(42)
        
        # Calculate segment sizes to ensure we use all points
        segment_size = n_points // 3
        remainder = n_points % 3
        segment_sizes = [segment_size] * 3
        # Distribute remainder across segments
        for i in range(remainder):
            segment_sizes[i] += 1
            
        # Create returns with regime shifts
        returns = []
        volumes = []
        
        # Normal regime
        returns.extend(np.random.normal(0.001, 0.01, segment_sizes[0]))
        volumes.extend(np.random.normal(1000000, 100000, segment_sizes[0]))
        
        # High volatility regime
        returns.extend(np.random.normal(-0.002, 0.03, segment_sizes[1]))
        volumes.extend(np.random.normal(2000000, 300000, segment_sizes[1]))
        
        # Low volatility regime
        returns.extend(np.random.normal(0.0005, 0.005, segment_sizes[2]))
        volumes.extend(np.random.normal(800000, 50000, segment_sizes[2]))
        
        returns = np.array(returns)
        volumes = np.array(volumes)
        assert len(returns) == n_points  # Verify length matches
        prices = 100 * (1 + returns).cumprod()
        
        data = {
            'TEST': pd.DataFrame({
                'daily_return': pd.Series(returns, index=dates),
                'close': pd.Series(prices, index=dates),
                'volume': pd.Series(volumes, index=dates)
            })
        }
        return data

    @pytest.fixture
    def regime_analyzer(self, sample_regime_data):
        """Create MarketAnalyzer instance with regime test data."""
        return MarketAnalyzer(data=sample_regime_data)

    def test_analyze_regimes_basic(self, regime_analyzer):
        """Test basic functionality of analyze_regimes."""
        results = regime_analyzer.analyze_regimes('TEST')
        
        # Check all expected keys are present
        expected_keys = {'hmm_regimes', 'structural_breaks', 
                        'combined_regimes', 'volatility_regimes'}
        assert set(results.keys()) == expected_keys
        
        # Check each result type
        assert isinstance(results['hmm_regimes'], pd.DataFrame)
        assert isinstance(results['structural_breaks'], pd.DataFrame)
        assert isinstance(results['combined_regimes'], pd.DataFrame)
        assert isinstance(results['volatility_regimes'], pd.DataFrame)
        
        # Check hmm_regimes structure
        assert 'regime' in results['hmm_regimes'].columns
        assert 'regime_type' in results['hmm_regimes'].columns
        
        # Check structural_breaks structure
        assert 'significant_break' in results['structural_breaks'].columns
        
        # Check combined_regimes structure
        assert 'composite_regime' in results['combined_regimes'].columns

    def test_date_filtering(self, regime_analyzer):
        """Test date range filtering in analyze_regimes."""
        start_date = '2020-06-01'
        end_date = '2021-06-01'
        
        results = regime_analyzer.analyze_regimes(
            'TEST', start_date=start_date, end_date=end_date
        )
        
        # Check date ranges
        for df in results.values():
            assert df.index[0].strftime('%Y-%m-%d') >= start_date
            assert df.index[-1].strftime('%Y-%m-%d') <= end_date

    def test_hmm_states_parameter(self, regime_analyzer):
        """Test impact of different HMM state numbers."""
        results_2_states = regime_analyzer.analyze_regimes('TEST', hmm_states=2)
        results_4_states = regime_analyzer.analyze_regimes('TEST', hmm_states=4)
        
        # Check number of unique regimes
        assert results_2_states['hmm_regimes']['regime'].nunique() <= 2
        assert results_4_states['hmm_regimes']['regime'].nunique() <= 4

    def test_window_parameter(self, regime_analyzer):
        """Test impact of different window sizes."""
        # Use larger windows to ensure HMM convergence
        results_small_window = regime_analyzer.analyze_regimes('TEST', window=180)
        results_large_window = regime_analyzer.analyze_regimes('TEST', window=360)
        
        # Test structural breaks (independent of HMM convergence)
        small_breaks = results_small_window['structural_breaks']['significant_break'].sum()
        large_breaks = results_large_window['structural_breaks']['significant_break'].sum()
        
        # Verify break detection and basic properties
        assert isinstance(small_breaks, (int, np.int64))
        assert isinstance(large_breaks, (int, np.int64))
        
        # Check that we have some breaks detected
        assert small_breaks > 0
        assert large_breaks > 0
        
        # Test that smaller windows are generally more sensitive
        # but don't strictly require more breaks
        assert small_breaks >= large_breaks * 0.5  # Allow some flexibility

    def test_regime_characteristics(self, regime_analyzer):
        """Test characteristics of detected regimes."""
        results = regime_analyzer.analyze_regimes('TEST')
        
        # Check regime types
        regime_types = results['hmm_regimes']['regime_type'].unique()
        assert all(rt.split('_')[0] in ['bullish', 'bearish'] for rt in regime_types)
        assert all('vol' in rt for rt in regime_types)
        
        # Check trend identification
        trends = results['combined_regimes']['trend'].unique()
        assert set(trends) <= {'uptrend', 'downtrend'}
        
        # Check composite regime format
        composite_regimes = results['combined_regimes']['composite_regime'].unique()
        assert all('_' in cr for cr in composite_regimes)
        assert all(cr.endswith(('uptrend', 'downtrend')) for cr in composite_regimes)

class TestLeadLagBasics:
    """Basic functionality tests for LeadLagAnalyzer."""
    
    def test_init(self, lead_lag_sample_returns_data):
        """Test initialization of LeadLagAnalyzer."""
        analyzer = LeadLagAnalyzer(lead_lag_sample_returns_data)
        assert analyzer.returns_data == lead_lag_sample_returns_data
        assert analyzer.relationships == {}

    def test_calculate_cross_correlations(self, lead_lag_analyzer):
        """Test cross-correlation calculation."""
        symbols = ['A', 'B', 'C']
        results = lead_lag_analyzer.calculate_cross_correlations(symbols, max_lags=3)
        
        # Check basic properties of the results
        assert isinstance(results, pd.DataFrame)
        assert set(results.columns) == {'symbol1', 'symbol2', 'lag', 'correlation'}
        assert all(s in symbols for s in results['symbol1'].unique())
        assert all(s in symbols for s in results['symbol2'].unique())
        assert all(-3 <= lag <= 3 for lag in results['lag'])
        assert all(-1 <= corr <= 1 for corr in results['correlation'])

    def test_granger_causality(self, lead_lag_analyzer):
        """Test Granger causality testing."""
        results = lead_lag_analyzer.test_granger_causality('A', 'B', max_lag=3)
        
        # Check basic properties of results
        assert isinstance(results, dict)
        assert all(f'lag_{i}' in results for i in range(1, 4))
        assert all(isinstance(v, float) for v in results.values())
        assert all(0 <= v <= 1 for v in results.values() if not np.isnan(v))


class TestNetworkAnalysis:
    """Tests for network-related functionality."""
    
    def test_build_relationship_network(self, lead_lag_analyzer):
        """Test network building functionality."""
        symbols = ['A', 'B', 'C']
        G = lead_lag_analyzer.build_relationship_network(symbols, threshold=0.1)
        
        # Check basic properties of the network
        assert isinstance(G, nx.Graph)
        assert set(G.nodes()) == set(symbols)
        assert all(isinstance(d['weight'], float) for _, _, d in G.edges(data=True))
        assert all(-1 <= d['weight'] <= 1 for _, _, d in G.edges(data=True))

    def test_find_market_leaders(self, lead_lag_analyzer):
        """Test market leader identification."""
        symbols = ['A', 'B', 'C']
        scores = lead_lag_analyzer.find_market_leaders(symbols, max_lag=3)
        
        # Check basic properties of the results
        assert isinstance(scores, dict)
        assert set(scores.keys()) == set(symbols)
        assert all(isinstance(v, float) for v in scores.values())
        assert all(0 <= v <= 1 for v in scores.values())
        assert any(v == 1 for v in scores.values()) or all(v == 0 for v in scores.values())


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_and_single_inputs(self, lead_lag_analyzer):
        """Test handling of empty and single-symbol inputs."""
        # Test with empty symbol list
        empty_corr = lead_lag_analyzer.calculate_cross_correlations([])
        assert len(empty_corr) == 0
        
        # Test with single symbol
        single_corr = lead_lag_analyzer.calculate_cross_correlations(['A'])
        assert len(single_corr) == 0

    def test_missing_data(self, lead_lag_sample_returns_data):
        """Test handling of missing data."""
        # Add empty series
        lead_lag_sample_returns_data['D'] = pd.DataFrame({
            'daily_return': pd.Series([], dtype=float)
        })
        analyzer_with_missing = LeadLagAnalyzer(lead_lag_sample_returns_data)
        results = analyzer_with_missing.test_granger_causality('A', 'D')
        assert all(np.isnan(v) for v in results.values())

    def test_data_alignment(self, lead_lag_sample_returns_data):
        """Test handling of differently aligned time series."""
        # Create data with different date ranges
        dates1 = pd.date_range('2020-01-01', '2021-12-31')
        dates2 = pd.date_range('2020-06-01', '2021-06-30')
        
        lead_lag_sample_returns_data['E'] = pd.DataFrame({
            'daily_return': pd.Series(np.random.normal(0, 0.01, len(dates1)), index=dates1)
        })
        lead_lag_sample_returns_data['F'] = pd.DataFrame({
            'daily_return': pd.Series(np.random.normal(0, 0.01, len(dates2)), index=dates2)
        })
        
        analyzer = LeadLagAnalyzer(lead_lag_sample_returns_data)
        results = analyzer.calculate_cross_correlations(['E', 'F'])
        
        # Check that results are based on properly aligned data
        assert not results['correlation'].isnull().any()
        assert len(results) > 0

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
    
    def test_volume_price_detection_1(self, volume_price_test_1):
        """Test detection of consolidated patterns."""
        
        df, _, expected_patterns = volume_price_test_1
        pattern_recognition = PatternRecognition(df["close"], df["volume"])
        patterns = pattern_recognition.detect_volume_price_patterns()

        assert len(patterns) == len(expected_patterns)
        for detected, expected in zip(patterns, expected_patterns):
            assert detected.pattern_type == expected.pattern_type
            assert detected.start_idx == expected.start_idx
            assert detected.end_idx == expected.end_idx
            assert np.allclose(detected.price_range, expected.price_range)
            assert np.allclose(detected.volume_range, expected.volume_range)
        
        pass

    def test_volume_price_detection_2(self, volume_price_test_2):
        """Test detection of consolidated patterns."""
        
        df, _, expected_patterns = volume_price_test_2
        pattern_recognition = PatternRecognition(df["close"], df["volume"])
        patterns = pattern_recognition.detect_volume_price_patterns()
        assert len(patterns) == len(expected_patterns)
        for detected, expected in zip(patterns, expected_patterns):
            assert detected.pattern_type == expected.pattern_type
            assert detected.start_idx == expected.start_idx
            assert detected.end_idx == expected.end_idx
            assert np.allclose(detected.price_range, expected.price_range)
            assert np.allclose(detected.volume_range, expected.volume_range)
        
        pass    

class TestMarketAnalyzerInitialization:
    """Test the initialization and basic setup of MarketAnalyzer."""

    def test_initialization(self, market_analyzer, sample_data):
        """Test if MarketAnalyzer initializes correctly with sample data."""
        assert isinstance(market_analyzer, MarketAnalyzer)
        assert market_analyzer.data == sample_data
        assert market_analyzer.market_indices == ['^GSPC', '^DJI']

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
            }, index = prices.index)
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

        data = market_analyzer_with_patterns.data['TEST'].copy()
        data = data.loc[data.index >= start_date]
        data = data.loc[data.index <= end_date]

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
                pattern_dates = data.index
                pattern_start = pattern_dates[pattern.start_idx]
                pattern_end = pattern_dates[pattern.end_idx]
                
                # Add debugging output
                if not (pattern_start >= pd.Timestamp(start_date) and 
                       pattern_end <= pd.Timestamp(end_date)):
                    print(f"Pattern outside range: {pattern_start} to {pattern_end}")
                
                assert pattern_start >= pd.Timestamp(start_date)
                assert pattern_end <= pd.Timestamp(end_date)          


class TestMarketVisualizer:
    @pytest.fixture
    def sample_stock_data(self) -> Dict[str, pd.DataFrame]:
        """Create sample stock price data for testing."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        symbols = ['AAPL', 'GOOGL']
        
        data = {}
        for symbol in symbols:
            # Create a double bottom pattern in the close prices
            base_price = 100
            prices = [
                base_price + 10,  # Start higher
                base_price,       # First bottom
                base_price + 5,   # Middle peak
                base_price,       # Second bottom
                base_price + 15   # End higher
            ]
            # Pad with additional random prices
            prices = prices + list(np.random.uniform(base_price, base_price + 20, len(dates) - len(prices)))
            
            df = pd.DataFrame(
                {
                    'close': prices,
                    'volume': np.random.uniform(1000000, 5000000, len(dates))
                },
                index=dates
            )
            
            # Generate OHLC data around close prices
            df['open'] = df['close'] + np.random.uniform(-5, 5, len(df))
            df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 5, len(df))
            df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 5, len(df))
            
            data[symbol] = df
            
        return data

    @pytest.fixture
    def sample_results(self) -> Dict[str, Any]:
        """Create sample analysis results for testing."""
        class Pattern:
            def __init__(self, start_idx, end_idx, price_range):
                self.start_idx = start_idx
                self.end_idx = end_idx
                self.price_range = price_range

        return {
            'patterns': {
                'double_bottom': [
                    Pattern(1, 3, (95, 110))  # Indices correspond to actual pattern in data
                ]
            },
            'regimes': {
                    'combined': pd.DataFrame({
                        'regime': ['bull', 'bear', 'bull'],
                        'bull_prob': [0.8, 0.3, 0.7],
                        'bear_prob': [0.2, 0.7, 0.3]
                    }, index=pd.date_range('2023-01-01', '2023-01-03'))
            },
            'relationship_network': nx.Graph([
                ('AAPL', 'GOOGL', {'weight': 0.75})
            ]),
            'cross_correlations': pd.DataFrame({
                'symbol1': ['AAPL', 'AAPL', 'GOOGL'],
                'lag': [-1, 0, 1],
                'correlation': [0.5, 0.75, 0.6]
            })
        }

    @pytest.fixture
    def visualizer(self, sample_stock_data, sample_results):
        """Create MarketVisualizer instance with sample data."""
        return MarketVisualizer(sample_stock_data, sample_results)

    @pytest.fixture
    def sample_volatility_surface(self) -> pd.DataFrame:
        """Create sample volatility surface data."""
        windows = [5, 21, 63, 252]
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        surface_data = []
        for window in windows:
            for quantile in quantiles:
                surface_data.append({
                    'window': window,
                    'quantile': quantile,
                    'volatility': np.random.uniform(0.1, 0.4)
                })
        return pd.DataFrame(surface_data)

    @pytest.fixture
    def sample_risk_metrics(self) -> Dict[str, Any]:
        """Create sample risk metrics data."""
        return {
            'var': {
                'historical': 0.025,
                'parametric': 0.028,
                'monte_carlo': 0.027
            },
            'expected_shortfall': {
                0.90: 0.032,
                0.95: 0.038,
                0.99: 0.045
            }
        }

    @pytest.fixture
    def sample_stress_test(self) -> pd.DataFrame:
        """Create sample stress test results."""
        scenarios = ['Market Crash', 'Rate Hike', 'Tech Bubble', 'Recovery']
        return pd.DataFrame({
            'scenario': scenarios,
            'price_change': [-0.15, -0.05, -0.20, 0.10],
            'stressed_var': [0.03, 0.02, 0.04, 0.015]
        })

    @pytest.fixture
    def extended_visualizer(self, sample_stock_data, sample_results, sample_volatility_surface, sample_risk_metrics, sample_stress_test):
        """Create MarketVisualizer instance with additional risk data."""
        results = sample_results.copy()
        results.update({
            'volatility_surface': sample_volatility_surface,
            'risk_metrics': sample_risk_metrics,
            'stress_test': sample_stress_test
        })
        return MarketVisualizer(sample_stock_data, results)

    def test_plot_price_patterns(self, visualizer):
        """Test price pattern visualization."""
        fig = visualizer.plot_price_patterns('AAPL')
        
        # Verify figure structure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Should have at least candlestick and volume traces
        assert isinstance(fig.data[0], go.Candlestick)  # First trace should be candlestick
        assert isinstance(fig.data[1], go.Bar)  # Second trace should be volume
        
        # Verify pattern annotations
        shapes = fig.layout.shapes
        assert len(shapes) == 1  # Should have one pattern rectangle
        
        annotations = fig.layout.annotations
        assert len(annotations) == 1  # Should have one pattern label

    def test_plot_price_patterns_date_filtering(self, visualizer):
        """Test price pattern visualization with date filtering."""
        start_date = '2023-01-02'  # Include the pattern period
        end_date = '2023-01-04'
        
        fig = visualizer.plot_price_patterns('AAPL', start_date, end_date)
        
        # Verify date range in plot
        candlestick_trace = fig.data[0]
        dates = pd.to_datetime(candlestick_trace.x)
        assert min(dates) >= pd.to_datetime(start_date)
        assert max(dates) <= pd.to_datetime(end_date)

    def test_plot_regimes(self, visualizer):
        """Test regime visualization."""
        fig = visualizer.plot_regimes('AAPL', regime_type='combined')
        
        # Verify figure structure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # Price line, regime indicator, and probabilities
        
        # Verify traces
        trace_names = [trace.name for trace in fig.data]
        assert 'Price' in trace_names
        assert any('prob' in name.lower() for name in trace_names)

    def test_plot_regimes_invalid_type(self, visualizer):
        """Test regime visualization with invalid regime type."""
        with pytest.raises(ValueError):
            visualizer.plot_regimes('AAPL', regime_type='invalid')

    def test_plot_network(self, visualizer):
        """Test network visualization."""
        fig = visualizer.plot_network(min_correlation=0.5)
        
        # Verify figure structure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Should have edge and node traces
        
        # Verify node and edge data
        edge_trace, node_trace = fig.data
        assert isinstance(edge_trace, go.Scatter)
        assert isinstance(node_trace, go.Scatter)
        
        # Verify node count
        unique_nodes = len(set(node_trace.text))
        assert unique_nodes == 2  # Should have AAPL and GOOGL

    def test_plot_lead_lag_heatmap(self, visualizer):
        """Test lead-lag heatmap visualization."""
        fig = visualizer.plot_lead_lag_heatmap()
        
        # Verify figure structure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # Should have one heatmap trace
        assert isinstance(fig.data[0], go.Heatmap)
        
        # Verify heatmap dimensions
        heatmap = fig.data[0]
        assert len(heatmap.x) == 3  # Three lag values
        assert len(heatmap.y) == 2  # Two symbols

    def test_invalid_symbol(self, visualizer):
        """Test handling of invalid symbol."""
        with pytest.raises(KeyError):
            visualizer.plot_price_patterns('INVALID')

    def test_missing_results(self, sample_stock_data):
        """Test handling of missing results."""
        visualizer = MarketVisualizer(sample_stock_data, {})
        
        with pytest.raises(ValueError):
            visualizer.plot_network()
            
        with pytest.raises(ValueError):
            visualizer.plot_lead_lag_heatmap()

    def test_plot_volatility_surface(self, extended_visualizer):
        """Test volatility surface visualization."""
        fig = extended_visualizer.plot_volatility_surface()
        
        # Verify figure structure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Surface)
        
        # Verify axes labels and title
        assert fig.layout.scene.xaxis.title.text == 'Time Window (days)'
        assert fig.layout.scene.yaxis.title.text == 'Quantile'
        assert fig.layout.scene.zaxis.title.text == 'Volatility'

    def test_plot_risk_metrics(self, extended_visualizer):
        """Test risk metrics dashboard visualization."""
        fig = extended_visualizer.plot_risk_metrics()
        
        # Verify figure structure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 4  # Should have VaR, ES, stress test, and returns plots
        
        # Verify subplot titles
        subplot_titles = fig.layout.annotations
        expected_titles = ['VaR Comparison', 'Expected Shortfall', 
                         'Stress Test Scenarios', 'Return Distribution']
        for title, expected in zip(subplot_titles, expected_titles):
            assert title.text == expected

        # Verify data presence
        var_trace = fig.data[0]
        assert isinstance(var_trace, go.Bar)
        assert len(var_trace.x) == 3  # Three VaR methods

    def test_plot_efficient_frontier(self, extended_visualizer):
        """Test efficient frontier visualization."""
        # Create sample efficient frontier data
        ef_data = pd.DataFrame({
            'volatility': np.linspace(0.1, 0.4, 20),
            'return': np.linspace(0.05, 0.15, 20),
            'sharpe_ratio': np.linspace(0.5, 2.0, 20)
        })
        
        fig = extended_visualizer.plot_efficient_frontier(ef_data)
        
        # Verify figure structure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1  # Efficient frontier line plus individual assets
        
        # Verify efficient frontier trace
        ef_trace = fig.data[0]
        assert isinstance(ef_trace, go.Scatter)
        assert len(ef_trace.x) == len(ef_data)
        
        # Verify layout
        assert fig.layout.xaxis.title.text == 'Portfolio Volatility'
        assert fig.layout.yaxis.title.text == 'Portfolio Return'

    def test_missing_risk_metrics(self, visualizer):
        """Test visualization with missing risk metrics."""
        fig = visualizer.plot_risk_metrics()
        
        # Should still create figure but with fewer traces
        assert isinstance(fig, go.Figure)
        assert len(fig.data) <= 4  # Some traces might be missing

    def test_invalid_efficient_frontier_data(self, visualizer):
        """Test efficient frontier plot with invalid data."""
        invalid_ef_data = pd.DataFrame()
        
        with pytest.raises(Exception):
            visualizer.plot_efficient_frontier(invalid_ef_data)            

class TestRiskAnalyzer:
    @pytest.fixture
    def sample_returns(self) -> pd.Series:
        """Create sample return series with known properties."""
        np.random.seed(42)
        # Create returns with known mean and volatility
        returns = pd.Series(
            np.random.normal(loc=0.03, scale=0.02, size=252),
            index=pd.date_range('2023-01-01', periods=252, freq='B')
        )
        return returns

    @pytest.fixture
    def sample_prices(self, sample_returns) -> pd.Series:
        """Create sample price series from returns."""
        initial_price = 100
        prices = initial_price * (1 + sample_returns).cumprod()
        prices.index = sample_returns.index
        return prices

    @pytest.fixture
    def risk_analyzer(self, sample_returns, sample_prices) -> RiskAnalyzer:
        """Create RiskAnalyzer instance with sample data."""
        return RiskAnalyzer(sample_returns, sample_prices)

    def test_calculate_var_historical(self, risk_analyzer):
        """Test historical VaR calculation."""
        var_results = risk_analyzer.calculate_var(
            confidence_level=0.95,
            time_horizon=1,
            method='historical'
        )
        
        assert 'historical_var' in var_results
        assert isinstance(var_results['historical_var'], float)
        assert var_results['historical_var'] > 0
 
        # Test different confidence levels
        var_90 = risk_analyzer.calculate_var(confidence_level=0.90)['historical_var']
        var_95 = risk_analyzer.calculate_var(confidence_level=0.95)['historical_var']
        var_99 = risk_analyzer.calculate_var(confidence_level=0.99)['historical_var']
        
        assert var_90 > var_95 > var_99  # Since these are negative numbers

    def test_calculate_var_parametric(self, risk_analyzer):
        """Test parametric VaR calculation."""
        var_results = risk_analyzer.calculate_var(
            confidence_level=0.95,
            time_horizon=1,
            method='parametric'
        )
        
        assert 'parametric_var' in var_results
        assert isinstance(var_results['parametric_var'], float)
        
        # Verify scaling with time horizon
        var_1d = var_results['parametric_var']
        var_10d = risk_analyzer.calculate_var(
            confidence_level=0.95,
            time_horizon=10,
            method='parametric'
        )['parametric_var']
        
        # Should approximately scale with square root of time
        assert np.isclose(var_10d, var_1d * np.sqrt(10), rtol=0.1)

    def test_calculate_var_monte_carlo(self, risk_analyzer):
        """Test Monte Carlo VaR calculation."""
        var_results = risk_analyzer.calculate_var(
            confidence_level=0.95,
            time_horizon=1,
            method='monte_carlo'
        )
        
        assert 'monte_carlo_var' in var_results
        assert isinstance(var_results['monte_carlo_var'], float)
        
        # Run multiple times to check stability
        results = [
            risk_analyzer.calculate_var(method='monte_carlo')['monte_carlo_var']
            for _ in range(5)
        ]
        
        # Results should be similar but not identical
        assert len(set(results)) > 1  # Should be random
        assert np.std(results) < 0.01  # But not too random

    def test_calculate_expected_shortfall(self, risk_analyzer):
        """Test Expected Shortfall calculation."""
        es = risk_analyzer.calculate_expected_shortfall(
            confidence_level=0.95,
            time_horizon=1
        )
        
        assert isinstance(es, float)
        assert es < 0  # ES should be negative for losses
        
        # ES should be more extreme than VaR
        var = risk_analyzer.calculate_var(
            confidence_level=0.95,
            method='historical'
        )['historical_var']
        assert abs(es) > abs(var)

    def test_stress_test(self, risk_analyzer):
        """Test stress testing functionality."""
        scenarios = {
            'Market Crash': -0.15,
            'Rate Hike': -0.05,
            'Recovery': 0.10
        }
        
        results = risk_analyzer.stress_test(scenarios)
        
        # Verify results structure
        assert isinstance(results, pd.DataFrame)
        assert set(results.columns) == {
            'scenario', 'price_shock', 'stressed_price',
            'price_change', 'normal_var', 'stressed_var'
        }
        
        # Verify calculations
        assert len(results) == len(scenarios)
        assert all(results['stressed_var'] >= results['normal_var'])  # Stress increases VaR

    def test_calculate_volatility_surface(self, risk_analyzer):
        """Test volatility surface calculation."""
        windows = [5, 21, 63]
        quantiles = [0.1, 0.5, 0.9]

        surface = risk_analyzer.calculate_volatility_surface(
            windows=windows,
            quantiles=quantiles
        )

        # Verify surface structure
        assert isinstance(surface, pd.DataFrame)
        assert set(surface.columns) == {'window', 'quantile', 'volatility', 'return'}
        assert len(surface) == len(windows) * len(quantiles)
        
        # Verify volatility properties
        assert all(surface['volatility'] > 0)  # Volatility should be positive
        assert all(surface['volatility'] < 1)  # Annualized vol should be reasonable

    def test_invalid_inputs(self, risk_analyzer):
        """Test handling of invalid inputs."""
        # Invalid confidence level

        with pytest.raises(ValueError):
            risk_analyzer.calculate_var(confidence_level=1.5)
        
        # Invalid VaR method
        with pytest.raises(ValueError):
            risk_analyzer.calculate_var(method='invalid_method')
        
        # Invalid time horizon
        with pytest.raises(ValueError):
            risk_analyzer.calculate_var(time_horizon=-1)        
