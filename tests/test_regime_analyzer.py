import pytest

import numpy as np
import pandas as pd

from market_analyzer.regime_analyzer import RegimeAnalyzer

@pytest.fixture
def sample_regime_data():
    """Fixture providing sample market data with distinct regimes"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Create DataFrame with all required columns
    data = pd.DataFrame(index=dates)
    
    # Price data showing clear trend
    data['price'] = np.concatenate([
        np.linspace(100, 150, 50),  # Uptrend
        np.linspace(150, 120, 50)   # Downtrend
    ])
    
    # Returns with distinct volatility regimes
    data['returns'] = np.concatenate([
        np.random.normal(0.02, 0.01, 50),  # Low volatility, positive returns
        np.random.normal(-0.01, 0.03, 50)  # High volatility, negative returns
    ])
    
    # Volume with regime shifts
    data['volume'] = np.concatenate([
        np.random.normal(1000, 100, 50),   # Normal volume
        np.random.normal(2000, 200, 50)    # High volume
    ])
    
    # Additional features for HMM
    data['volatility'] = data['returns'].rolling(20).std()
    data['momentum'] = data['returns'].rolling(10).mean()
    
    return data.fillna(method='bfill')

class TestRegimeAnalyzer:
    def test_basic_functionality(self, sample_regime_data):
        """Test basic functionality with all detection methods"""
        detector = RegimeAnalyzer(n_states=2, window=20)
        results = detector.fit_transform(sample_regime_data)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(sample_regime_data)
        assert 'regime' in results.columns
        assert 'regime_type' in results.columns
        assert 'significant_break' in results.columns
        assert 'trend' in results.columns
        assert 'composite_regime' in results.columns

    def test_hmm_only(self, sample_regime_data):
        """Test using only HMM detection"""
        detector = RegimeAnalyzer(
            n_states=2,
            detection_methods=['hmm']
        )
        results = detector.fit_transform(sample_regime_data)
        
        assert 'regime' in results.columns
        assert 'composite_regime' not in results.columns
        assert 'significant_break' not in results.columns

    def test_breaks_only(self, sample_regime_data):
        """Test using only structural break detection"""
        detector = RegimeAnalyzer(
            detection_methods=['breaks']
        )
        results = detector.fit_transform(sample_regime_data)
        
        assert 'significant_break' in results.columns
        assert 'regime' not in results.columns
        assert 'trend' not in results.columns

    def test_custom_regime_labels(self, sample_regime_data):
        """Test with custom regime labels"""
        custom_labels = {0: 'calm', 1: 'volatile'}
        detector = RegimeAnalyzer(
            n_states=2,
            regime_labels=custom_labels,
            detection_methods=['hmm']
        )
        results = detector.fit_transform(sample_regime_data)
        
        assert set(results['regime_type'].unique()).issubset({'calm', 'volatile'})

    def test_input_validation(self, sample_regime_data):
        """Test various input validation scenarios"""
        detector = RegimeAnalyzer()
        
        # Test empty DataFrame
        with pytest.raises(ValueError, match="X cannot be empty"):
            detector.fit(pd.DataFrame())
        
        # Test missing required columns for breaks
        detector_breaks = RegimeAnalyzer(detection_methods=['breaks'])
        invalid_data = sample_regime_data.drop(columns=['volume'])
        with pytest.raises(ValueError, match="must contain 'returns' and 'volume' columns"):
            detector_breaks.fit(invalid_data)
        
        # Test missing required columns for trend
        detector_trend = RegimeAnalyzer(detection_methods=['trend'])
        invalid_data = sample_regime_data.drop(columns=['price'])
        with pytest.raises(ValueError, match="must contain 'price' column"):
            detector_trend.fit(invalid_data)
        
        # Test non-DataFrame input
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            detector.fit(sample_regime_data.values)

    def test_transform_before_fit(self, sample_regime_data):
        """Test that transform raises error if called before fit"""
        detector = RegimeAnalyzer(detection_methods=['hmm'])
        with pytest.raises(ValueError, match="Must fit HMM before transform"):
            detector.transform(sample_regime_data)

    def test_nan_handling(self, sample_regime_data):
        """Test handling of NaN values in features"""
        data_with_nans = sample_regime_data.copy()
        data_with_nans.iloc[10:15] = np.nan
        
        detector = RegimeAnalyzer()
        results = detector.fit_transform(data_with_nans)
        
        assert isinstance(results, pd.DataFrame)
        assert not results['regime'].isna().any()
        assert not results['significant_break'].isna().any()
        assert not results['trend'].isna().any()

    @pytest.mark.parametrize("n_states", [2, 3, 4])
    def test_different_state_numbers(self, sample_regime_data, n_states):
        """Test with different numbers of states"""
        detector = RegimeAnalyzer(
            n_states=n_states,
            detection_methods=['hmm']
        )
        results = detector.fit_transform(sample_regime_data)
        
        assert len(results['regime'].unique()) == n_states
        for i in range(n_states):
            assert f'regime_{i}_prob' in results.columns

    def test_standardization(self, sample_regime_data):
        """Test standardization option"""
        # Test with standardization
        detector_std = RegimeAnalyzer(
            standardize=True,
            detection_methods=['hmm']
        )
        results_std = detector_std.fit_transform(sample_regime_data)
        
        # Test without standardization
        detector_no_std = RegimeAnalyzer(
            standardize=False,
            detection_methods=['hmm']
        )
        results_no_std = detector_no_std.fit_transform(sample_regime_data)
        
        # Results should be different due to scaling
        assert not np.allclose(
            results_std['regime'].values,
            results_no_std['regime'].values
        )

    def test_structural_breaks(self, sample_regime_data):
        """Test structural break detection specifics"""
        detector = RegimeAnalyzer(
            window=20,
            detection_methods=['breaks']
        )
        results = detector.fit_transform(sample_regime_data)
        
        # Check break indicators
        assert 'break_chow' in results.columns
        assert 'break_volatility' in results.columns
        assert 'break_volume' in results.columns
        assert 'significant_break' in results.columns

        # First few periods should be NaN (minimum required for break detection)
        assert results['break_chow'].iloc[:5].isna().all()  # Need at least 3 points each for pre/post
        
        # After minimum required points, should start getting values
        assert not results['break_chow'].iloc[6:].isna().all()
        
        # Should have some breaks detected
        assert results['significant_break'].sum() > 0

    def test_trend_detection(self, sample_regime_data):
        """Test trend detection specifics"""
        # Create more pronounced trend data
        sample_data = sample_regime_data.copy()
        sample_data['price'] = np.concatenate([
            np.linspace(100, 200, 50),  # Steeper uptrend
            np.linspace(200, 100, 50)   # Steeper downtrend
        ])
        
        detector = RegimeAnalyzer(
            window=20,
            detection_methods=['trend']
        )
        results = detector.fit_transform(sample_data)
        
        assert 'trend' in results.columns
        assert set(results['trend'].unique()) == {'uptrend', 'downtrend'}

        # Allow for moving average lag at the transition point
        # Check middle portions of each half where trend should be clear
        first_quarter = results['trend'][20:40]  # Check middle of first half
        assert (first_quarter == 'uptrend').mean() > 0.7
        
        last_quarter = results['trend'][60:80]   # Check middle of second half
        assert (last_quarter == 'downtrend').mean() > 0.7


    def test_composite_regime_format(self, sample_regime_data):
        """Test the format of composite regime strings"""
        detector = RegimeAnalyzer(
            n_states=2,
            detection_methods=['hmm', 'breaks', 'trend']
        )
        results = detector.fit_transform(sample_regime_data)
        
        # Check that composite regimes combine information correctly
        for regime in results['composite_regime']:
            components = regime.split('_')
            
            # Should contain regime type
            assert any(c in ['bullish', 'bearish', 'neutral'] for c in components)
            
            # Should contain trend
            assert any(c in ['uptrend', 'downtrend'] for c in components)
            
            # Might contain transition
            if 'transition' in components:
                assert results.loc[results['composite_regime'] == regime, 'significant_break'].iloc[0]

    def test_hmm_convergence(self, sample_regime_data):
        """Test handling of HMM convergence scenarios"""
        # Create very noisy data that might cause convergence issues
        noisy_data = sample_regime_data.copy()
        noisy_data = noisy_data + np.random.normal(0, 10, size=noisy_data.shape)
        
        detector = RegimeAnalyzer(
            n_states=10,  # Excessive number of states to potentially cause non-convergence
            detection_methods=['hmm']
        )
        results = detector.fit_transform(noisy_data)
        
        # Should still return valid results
        assert isinstance(results, pd.DataFrame)
        assert 'regime' in results.columns
        assert 'regime_type' in results.columns

    def test_break_detection_initialization(self, sample_regime_data):
        """Test that break detection handles initialization period properly"""
        detector = RegimeAnalyzer(
            window=20,
            detection_methods=['breaks']
        )
        results = detector.fit_transform(sample_regime_data)
        
        # Check early periods
        early_results = results.iloc[:detector.window]
        
        # Should have some valid calculations even in early periods
        assert not early_results['break_chow'].isna().all()
        assert not early_results['break_volatility'].isna().all()
        assert not early_results['break_volume'].isna().all()
        
        # Values should be continuous (no sudden jumps)
        for col in ['break_chow', 'break_volatility', 'break_volume']:
            values = results[col].dropna()
            if len(values) > 1:
                differences = np.abs(values.diff().dropna())
                assert differences.max() < 10  # No extreme jumps
        
        # Significant breaks should be properly initialized
        assert not results['significant_break'].isna().any()

    def test_break_detection_edge_cases(self, sample_regime_data):
        """Test break detection with various edge cases"""
        # Create test data with some edge cases
        edge_data = sample_regime_data.copy()
        
        # Add some constant periods (zero variance)
        edge_data.iloc[10:15, edge_data.columns.get_loc('returns')] = 0.0
        edge_data.iloc[30:35, edge_data.columns.get_loc('volume')] = 1000.0
        
        # Add some extreme jumps
        edge_data.iloc[50, edge_data.columns.get_loc('returns')] = edge_data['returns'].mean() + 10 * edge_data['returns'].std()
        
        detector = RegimeAnalyzer(
            window=20,
            detection_methods=['breaks']
        )
        results = detector.fit_transform(edge_data)
        
        # Should handle constant periods without errors
        assert not results['break_chow'].isna().all()
        assert not results['break_volatility'].isna().all()
        assert not results['break_volume'].isna().all()
        
        # Should detect the extreme jump
        jump_idx = 50
        assert results.iloc[jump_idx:jump_idx+5]['significant_break'].any()