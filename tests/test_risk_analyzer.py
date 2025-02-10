import pytest

import numpy as np
import pandas as pd

from market_analyzer.risk_analyzer import RiskAnalyzer

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
