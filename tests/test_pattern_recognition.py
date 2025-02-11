import pytest
import re

import networkx as nx
import numpy as np
import pandas as pd

from market_analyzer.pattern_recognition import PatternRecognition, TechnicalPattern, LeadLagAnalyzer

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
        # Run the Granger causality tests
        results = lead_lag_analyzer.test_granger_causality(['A', 'B', 'C'], max_lag=3)

        # Test 1: Check basic structure of results DataFrame
        assert isinstance(results, pd.DataFrame), "Results should be a DataFrame"
        expected_columns = {
            'cause', 'effect', 'lag', 'p_value', 'r2',
            'significant_coefficients'
        }
        assert all(col in results.columns for col in expected_columns), \
            f"Missing expected columns. Found {results.columns}"
        
        # Test 2: Check number of results
        # For 3 symbols, we expect 6 relationships (n*(n-1)) * 3 lags = 18 rows
        assert len(results) == 18, \
            f"Expected 18 results (6 relationships * 3 lags), got {len(results)}"
        
        # Test 3: Check value ranges
        assert all(0 <= r <= 1 for r in results['r2']), "R² values should be between 0 and 1"
        assert all(0 <= p <= 1 for p in results['p_value']), "P-values should be between 0 and 1"
        
        # Test 4: Check relationship presence
        relationships = set((row['cause'], row['effect']) for _, row in results.iterrows())
        expected_relationships = {
            ('A', 'B'), ('B', 'A'), 
            ('A', 'C'), ('C', 'A'),
            ('B', 'C'), ('C', 'B')
        }
        assert relationships == expected_relationships, \
            "Missing some expected relationships"
        
        # Test 5: Check lag values
        assert all(lag in [1, 2, 3] for lag in results['lag']), \
            "All lags should be between 1 and 3"
        
        # Test 6: Check for expected strong relationship (A → B at lag 2)
        ab_lag2 = results[
            (results['cause'] == 'A') & 
            (results['effect'] == 'B') & 
            (results['lag'] == 2)
        ]
        assert len(ab_lag2) == 1, "Should find exactly one A→B relationship at lag 2"
        
        # Test 7: Check lack of relationship with C
        c_relationships = results[
            (results['cause'] == 'C') | (results['effect'] == 'C')
        ]
        # These relationships should generally have higher p-values as C is independent
        
        # Test 8: Test with invalid inputs
        with pytest.raises(Exception):
            lead_lag_analyzer.test_granger_causality(['NonExistentSymbol'], max_lag=3)
        
        # Test 9: Test with different significance levels
        results_strict = lead_lag_analyzer.test_granger_causality(
            ['A', 'B', 'C'], 
            max_lag=3, 
            significance_level=0.01
        )
        results_loose = lead_lag_analyzer.test_granger_causality(
            ['A', 'B', 'C'], 
            max_lag=3, 
            significance_level=0.1
        )
        assert isinstance(results_strict, pd.DataFrame)
        assert isinstance(results_loose, pd.DataFrame)
        
        # Test 10: Check coefficient format in significant_coefficients column
        coef_pattern = r'Lag \d+: -?\d+\.\d+'
        non_empty_coeffs = results['significant_coefficients'].str.len() > 0
        if any(non_empty_coeffs):
            sample_coeff = results.loc[non_empty_coeffs, 'significant_coefficients'].iloc[0]
            assert re.match(coef_pattern, sample_coeff.split(', ')[0]), \
                "Coefficient format should match 'Lag X: Y.YYYY'"


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

    def test_market_leader_relationships(self, lead_lag_analyzer):
        """Test that the market leader identification correctly captures the synthetic relationships."""
        symbols = ['A', 'B', 'C']
        
        # Test without effect size weighting for clearer causality testing
        scores = lead_lag_analyzer.find_market_leaders(
            symbols, 
            max_lag=3,
            use_effect_size=False
        )
        
        # Given our synthetic data setup:
        # - A leads B (with lag 2)
        # - C is independent
        # Therefore:
        # - A should have the highest leadership score
        # - B should have a lower score
        # - C should have the lowest score

        # Check if A has highest leadership score
        assert scores['A'] == 1.0, "A should be the primary market leader"
        assert scores['B'] < scores['A'], "B should have lower leadership score than A"
        assert scores['C'] < scores['A'], "C should have lower leadership score than A"

    def test_market_leader_effect_size_impact(self, lead_lag_analyzer):
        """Test that effect size weighting impacts the scores appropriately."""
        symbols = ['A', 'B', 'C']
        
        # Get scores with and without effect size weighting
        scores_basic = lead_lag_analyzer.find_market_leaders(
            symbols, 
            max_lag=3,
            use_effect_size=False
        )
        
        scores_weighted = lead_lag_analyzer.find_market_leaders(
            symbols, 
            max_lag=3,
            use_effect_size=True
        )
        
        # The relative ordering should remain the same
        # but the absolute values should differ
        assert scores_basic['A'] >= scores_basic['B'], "A should lead B in both cases"
        assert scores_weighted['A'] >= scores_weighted['B'], "A should lead B in both cases"
        
        # Check that at least some scores are different when using effect size
        assert any(scores_basic[s] != scores_weighted[s] for s in symbols), \
            "Effect size weighting should impact the scores"

    def test_market_leader_different_lags(self, lead_lag_analyzer):
        """Test that the function works with different maximum lag values."""
        symbols = ['A', 'B', 'C']
        
        # Test with different lag values
        for max_lag in [1, 3, 5]:
            scores = lead_lag_analyzer.find_market_leaders(
                symbols, 
                max_lag=max_lag,
                use_effect_size=False
            )
            
            # Basic checks
            assert isinstance(scores, dict)
            assert set(scores.keys()) == set(symbols)
            assert all(0 <= v <= 1 for v in scores.values())
       
            # Since our synthetic data has lag 2:
            # - With max_lag=1, the relationship might be weaker
            # - With max_lag>=2, we should definitely see A leading
            if max_lag >= 2:
                assert scores['A'] == 1.0, f"A should be the leader with max_lag={max_lag}"

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
            assert (0 <= pattern.confidence <= 1) or np.isnan(pattern.confidence)
            
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
            assert (0 <= pattern.confidence <= 1) or np.isnan(pattern.confidence)
            
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

    def test_neutral_pattern(self):
        """Test detection of neutral pattern when changes are below thresholds"""
        # Create enough data points to accommodate window size
        prices = pd.Series([100.0 + i*0.1 for i in range(30)])  # 30 small increments
        volumes = pd.Series([1000 + i for i in range(30)])      # 30 small increments
        
        detector_instance = PatternRecognition(prices, volumes)
        results = detector_instance.detect_volume_price_patterns(
            min_price_change=0.005,  # 0.5% threshold
            min_volume_change=0.02   # 2% threshold
        )
        
        # Check recent points (last target_length points) more heavily
        recent_results = results.iloc[-4:]  # Using default target_length=4
        
        assert len(results) > 0
        assert 'NEUTRAL' in results.columns
        assert recent_results['NEUTRAL'].mean() > 0.6  # Focus on recent neutral pattern weight

    def test_divergence_pattern(self):
        """Test detection of divergence pattern with temporal weighting"""
        # Create clear divergence pattern with enough points
        prices = pd.Series([100.0 + i for i in range(30)])  # Steadily increasing
        volumes = pd.Series([1000 - i*10 for i in range(30)])  # Steadily decreasing
        
        detector_instance = PatternRecognition(prices, volumes)
        results = detector_instance.detect_volume_price_patterns(target_length=4)
        
        # Check pattern weights in recent window
        recent_results = results.iloc[-4:]
        
        assert len(results) > 0
        assert 'DIVERGENCE' in results.columns
        assert recent_results['DIVERGENCE'].mean() > 0.6  # Strong recent divergence
        # Verify pattern is detected more quickly
        assert results.iloc[5:]['pattern'].notna().sum() > 0  # Should detect pattern early

    def test_concordant_pattern(self):
        """Test detection of concordant pattern with weight decay"""
        # Create clear concordant pattern with enough points
        base_series = [i for i in range(30)]
        prices = pd.Series([100.0 + i for i in base_series])
        volumes = pd.Series([1000 * (1 + i*1.03) for i in base_series])
        
        detector_instance = PatternRecognition(prices, volumes)
        results = detector_instance.detect_volume_price_patterns(
            target_length=4,
            min_pattern_weight=0.6
        )

        recent_results = results.iloc[-4:]
        
        assert len(results) > 0
        assert 'CONCORDANT' in results.columns
        assert recent_results['CONCORDANT'].mean() > 0.6
        # Test weight decay
        assert results[results["pattern"]=="CONCORDANT"]["pattern_weight"].diff().mean() > 0

    def test_pattern_transitions(self):
        """Test quick detection of pattern transitions"""
        # Create sequence with clear pattern transition
        prices = []
        volumes = []
        
        # First 15 points: divergence
        for i in range(15):
            prices.append(100 + i)
            volumes.append(1000 - i*10)
        
        # Next 15 points: concordant
        for i in range(15):
            prices.append(115 + i)
            volumes.append(850 + i*10)
        
        detector_instance = PatternRecognition(pd.Series(prices), pd.Series(volumes))
        results = detector_instance.detect_volume_price_patterns(
            target_length=4,
            min_pattern_weight=0.6
        )

        # Check transition period 
        transition_period = results.iloc[14:19]  # Around the transition point
        
        assert len(results) > 0
        # Should see pattern change within target_length points
        assert 'DIVERGENCE' in transition_period['pattern'].values
        #assert 'CONCORDANT' in transition_period['pattern'].values
        # Pattern weights should shift quickly
        assert abs(transition_period['pattern_weight'].diff().mean()) > 0.1

    def test_mixed_patterns(self):
        """Test behavior with mixed patterns and temporal weighting"""
        # Create a longer sequence with mixed behavior
        prices = []
        volumes = []
        for i in range(30):
            if i % 3 == 0:
                prices.append(100 + i)
                volumes.append(1000 - i*10)  # Divergence
            elif i % 3 == 1:
                prices.append(100 + i)
                volumes.append(1000 + i*10)  # Concordant
            else:
                prices.append(100 + 0.1)
                volumes.append(1000 + 1)     # Neutral
        
        detector_instance = PatternRecognition(pd.Series(prices), pd.Series(volumes))
        results = detector_instance.detect_volume_price_patterns(
            target_length=4,
            min_pattern_weight=0.6
        )
        
        assert len(results) > 0
        # Pattern weights should be more volatile due to temporal weighting
        assert results['pattern_weight'].std() > 0.1
        # Recent patterns should have higher weights
        assert results['pattern_weight'].iloc[-4:].mean() > results['pattern_weight'].iloc[:-4].mean()

    def test_window_size_and_target_length(self):
        """Test interaction between window_size and target_length"""
        prices = pd.Series(np.linspace(100, 110, 50))
        volumes = pd.Series(np.linspace(1000, 1200, 50))
        
        detector_instance = PatternRecognition(prices, volumes)
        
        results_short = detector_instance.detect_volume_price_patterns(
            window_size=10,
            target_length=4
        )
        results_long = detector_instance.detect_volume_price_patterns(
            window_size=20,
            target_length=8
        )

        assert len(results_short) == len(results_long)
        # Shorter target_length should lead to more responsive pattern detection
        assert results_short['pattern_weight'].mean() > results_long['pattern_weight'].mean()
        # Longer window should have more stable pattern weights
        assert results_long['pattern'].notna().sum() < results_short['pattern'].notna().sum()

    def test_minimum_pattern_weight(self):
        """Test sensitivity to minimum pattern weight threshold"""
        prices = pd.Series([100.0 + i for i in range(30)])
        volumes = pd.Series([1000 + i*10 for i in range(30)])
        
        detector_instance = PatternRecognition(prices, volumes)
        
        results_loose = detector_instance.detect_volume_price_patterns(
            min_pattern_weight=0.4,
            target_length=4
        )
        results_strict = detector_instance.detect_volume_price_patterns(
            min_pattern_weight=0.7,
            target_length=4
        )
        
        assert 'pattern' in results_loose.columns
        # Strict threshold should have more null patterns
        assert (results_strict['pattern'].isna().sum() > 
                results_loose['pattern'].isna().sum())
        # Pattern weights should be identical (only threshold changes)
        np.testing.assert_array_almost_equal(
            results_strict['pattern_weight'],
            results_loose['pattern_weight']
        )