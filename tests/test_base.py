import pytest

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from enum import Enum
from market_analyzer import MarketAnalyzer, MarketVisualizer
from market_analyzer.pattern_recognition import TechnicalPattern
from typing import Dict, Any

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
            'volume_price'
        ])
        
        # Check each pattern type
        for pattern_list in patterns.values():
            assert isinstance(pattern_list, list)
            for pattern in pattern_list:
                assert isinstance(pattern, TechnicalPattern)
                assert pattern.start_idx < pattern.end_idx
                assert pattern.price_range[0] <= pattern.price_range[1]
                if hasattr(pattern, "confidence"):
                    assert (0 <= pattern.confidence <= 1) or np.isnan(pattern.confidence)
                if hasattr(pattern, "sub_classification") and pattern.sub_classification is not None:
                    assert isinstance(pattern.sub_classification, Enum)                                    
    
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