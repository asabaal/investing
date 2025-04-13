"""
Tests for technical indicators module.

This file contains unit tests for the technical indicator calculation functions
to ensure they provide correct results for known inputs.
"""

import unittest
import numpy as np
import pandas as pd
from technical_indicators import (
    calculate_rsi,
    calculate_sma,
    calculate_ema,
    calculate_momentum,
    calculate_volatility,
    calculate_drawdown,
    calculate_returns
)


class TestTechnicalIndicators(unittest.TestCase):
    """Test cases for technical indicator calculations."""
    
    def setUp(self):
        """Set up test data."""
        # Create flat price series
        self.flat_prices = np.array([100.0] * 50)
        
        # Create ascending price series
        self.ascending_prices = np.array([100.0 + i for i in range(50)])
        
        # Create descending price series
        self.descending_prices = np.array([150.0 - i for i in range(50)])
        
        # Create oscillating price series
        self.oscillating_prices = np.array([
            100.0 + 10 * np.sin(i * np.pi / 10) for i in range(50)
        ])
        
        # Create series with gaps
        self.prices_with_gaps = np.array([100.0] * 30)
        self.prices_with_gaps[10:20] = np.nan
        
        # Create real-world-like price series
        np.random.seed(42)  # For reproducibility
        self.realistic_prices = np.array([100.0])
        for i in range(49):
            change = np.random.normal(0.0005, 0.01)  # Small upward drift with noise
            self.realistic_prices = np.append(
                self.realistic_prices, 
                self.realistic_prices[-1] * (1 + change)
            )
        
        # Convert to pandas Series for some tests
        self.realistic_prices_series = pd.Series(self.realistic_prices)
    
    def test_calculate_rsi_flat(self):
        """Test RSI calculation with flat prices."""
        rsi = calculate_rsi(self.flat_prices, period=14)
        
        # RSI should be 50 for flat prices (after the initial period)
        self.assertTrue(np.isnan(rsi[0:14]).all(), "First 14 values should be NaN")
        self.assertTrue(np.allclose(rsi[14:], 50.0, rtol=1e-10, atol=1e-10, equal_nan=True),
                       "RSI should be 50 for flat prices")
    
    def test_calculate_rsi_ascending(self):
        """Test RSI calculation with consistently rising prices."""
        rsi = calculate_rsi(self.ascending_prices, period=14)
        
        # RSI should approach 100 for consistently rising prices
        self.assertTrue(np.isnan(rsi[0:14]).all(), "First 14 values should be NaN")
        self.assertTrue(np.all(rsi[14:] > 50), "RSI should be above 50 for rising prices")
        self.assertTrue(rsi[-1] > 90, "RSI should approach 100 for consistently rising prices")
    
    def test_calculate_rsi_descending(self):
        """Test RSI calculation with consistently falling prices."""
        rsi = calculate_rsi(self.descending_prices, period=14)
        
        # RSI should approach 0 for consistently falling prices
        self.assertTrue(np.isnan(rsi[0:14]).all(), "First 14 values should be NaN")
        self.assertTrue(np.all(rsi[14:] < 50), "RSI should be below 50 for falling prices")
        self.assertTrue(rsi[-1] < 10, "RSI should approach 0 for consistently falling prices")
    
    def test_calculate_rsi_realistic(self):
        """Test RSI calculation with realistic price movements."""
        # Test with numpy array
        rsi_np = calculate_rsi(self.realistic_prices, period=14)
        
        # Test with pandas Series
        rsi_pd = calculate_rsi(self.realistic_prices_series, period=14)
        
        # Results should be between 0 and 100
        self.assertTrue(np.all((rsi_np[14:] >= 0) & (rsi_np[14:] <= 100)),
                       "RSI values should be between 0 and 100")
        
        # Results should be the same for numpy array and pandas Series
        self.assertTrue(np.allclose(rsi_np, rsi_pd, rtol=1e-10, atol=1e-10, equal_nan=True),
                       "RSI calculation should give same results for numpy array and pandas Series")
    
    def test_calculate_sma(self):
        """Test SMA calculation."""
        period = 10
        
        # Test flat prices
        sma_flat = calculate_sma(self.flat_prices, period=period)
        self.assertTrue(np.isnan(sma_flat[0:period-1]).all(), f"First {period-1} values should be NaN")
        self.assertTrue(np.allclose(sma_flat[period-1:], 100.0, rtol=1e-10, atol=1e-10, equal_nan=True),
                       "SMA should equal price for flat prices")
        
        # Test ascending prices
        sma_ascending = calculate_sma(self.ascending_prices, period=period)
        self.assertTrue(np.isnan(sma_ascending[0:period-1]).all(), f"First {period-1} values should be NaN")
        
        # For ascending prices, SMA should lag behind the price by (period-1)/2 days
        # At index i, SMA should equal the price at index i-(period-1)/2
        lag = (period - 1) / 2
        for i in range(period - 1, len(self.ascending_prices)):
            expected_price = self.ascending_prices[int(i - lag)]
            self.assertAlmostEqual(sma_ascending[i], expected_price, delta=0.001,
                                  msg=f"SMA at index {i} should equal price at index {int(i - lag)}")
    
    def test_calculate_ema(self):
        """Test EMA calculation."""
        period = 10
        
        # Test flat prices
        ema_flat = calculate_ema(self.flat_prices, period=period)
        self.assertTrue(np.isnan(ema_flat[0:period-1]).all(), f"First {period-1} values should be NaN")
        self.assertTrue(np.allclose(ema_flat[period-1:], 100.0, rtol=1e-10, atol=1e-10, equal_nan=True),
                       "EMA should equal price for flat prices")
        
        # Test with pandas Series
        ema_pd = calculate_ema(pd.Series(self.realistic_prices), period=period)
        self.assertTrue(np.isnan(ema_pd[0:period-1]).all(), f"First {period-1} values should be NaN")
        
        # For realistic prices, EMA should be less responsive to recent price changes than price itself
        # but more responsive than SMA
        sma = calculate_sma(self.realistic_prices, period=period)
        
        # Calculate absolute differences from price
        ema_diff = np.abs(ema_pd[period:] - self.realistic_prices[period:])
        sma_diff = np.abs(sma[period:] - self.realistic_prices[period:])
        
        # EMA should track price more closely than SMA on average
        self.assertLess(np.nanmean(ema_diff), np.nanmean(sma_diff),
                        "EMA should track price more closely than SMA on average")
    
    def test_calculate_momentum(self):
        """Test momentum calculation."""
        period = 10
        
        # Test flat prices
        momentum_flat = calculate_momentum(self.flat_prices, period=period)
        self.assertTrue(np.isnan(momentum_flat[0:period]).all(), f"First {period} values should be NaN")
        self.assertTrue(np.allclose(momentum_flat[period:], 0.0, rtol=1e-10, atol=1e-10, equal_nan=True),
                       "Momentum should be 0 for flat prices")
        
        # Test ascending prices
        momentum_ascending = calculate_momentum(self.ascending_prices, period=period)
        self.assertTrue(np.isnan(momentum_ascending[0:period]).all(), f"First {period} values should be NaN")
        
        # For ascending prices, momentum should be positive and decreasing percentage-wise
        self.assertTrue(np.all(momentum_ascending[period:] > 0),
                       "Momentum should be positive for ascending prices")
        
        # Calculate expected momentum values
        expected_momentum = np.zeros_like(self.ascending_prices, dtype=float)
        for i in range(period, len(self.ascending_prices)):
            expected_momentum[i] = (self.ascending_prices[i] - self.ascending_prices[i - period]) / self.ascending_prices[i - period]
        
        self.assertTrue(np.allclose(momentum_ascending, expected_momentum, rtol=1e-10, atol=1e-10, equal_nan=True),
                        "Momentum calculation should match expected values")
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        period = 20
        
        # Test flat prices
        volatility_flat = calculate_volatility(self.flat_prices, period=period)
        self.assertTrue(np.isnan(volatility_flat[0:period]).all(), f"First {period} values should be NaN")
        self.assertTrue(np.allclose(volatility_flat[period:], 0.0, rtol=1e-10, atol=1e-10, equal_nan=True),
                       "Volatility should be 0 for flat prices")
        
        # Test realistic prices
        volatility_realistic = calculate_volatility(self.realistic_prices, period=period)
        self.assertTrue(np.isnan(volatility_realistic[0:period]).all(), f"First {period} values should be NaN")
        self.assertTrue(np.all(volatility_realistic[period:] >= 0),
                        "Volatility should be non-negative")
    
    def test_calculate_drawdown(self):
        """Test drawdown calculation."""
        # Test flat prices
        dd_flat, max_dd_flat = calculate_drawdown(self.flat_prices)
        self.assertTrue(np.allclose(dd_flat, 0.0, rtol=1e-10, atol=1e-10),
                       "Drawdown should be 0 for flat prices")
        self.assertTrue(np.allclose(max_dd_flat, 0.0, rtol=1e-10, atol=1e-10),
                       "Max drawdown should be 0 for flat prices")
        
        # Test ascending prices
        dd_ascending, max_dd_ascending = calculate_drawdown(self.ascending_prices)
        self.assertTrue(np.allclose(dd_ascending, 0.0, rtol=1e-10, atol=1e-10),
                       "Drawdown should be 0 for ascending prices")
        self.assertTrue(np.allclose(max_dd_ascending, 0.0, rtol=1e-10, atol=1e-10),
                       "Max drawdown should be 0 for ascending prices")
        
        # Test descending prices
        dd_descending, max_dd_descending = calculate_drawdown(self.descending_prices)
        
        # Calculate expected drawdown for descending prices
        expected_dd = np.zeros_like(self.descending_prices, dtype=float)
        for i in range(len(self.descending_prices)):
            expected_dd[i] = (self.descending_prices[0] - self.descending_prices[i]) / self.descending_prices[0]
        
        self.assertTrue(np.allclose(dd_descending, expected_dd, rtol=1e-10, atol=1e-10),
                        "Drawdown calculation should match expected values for descending prices")
        self.assertTrue(np.allclose(max_dd_descending, expected_dd, rtol=1e-10, atol=1e-10),
                        "Max drawdown should equal current drawdown for consistently falling prices")
    
    def test_calculate_returns(self):
        """Test returns calculation."""
        period = 10
        
        # Test flat prices
        returns_flat = calculate_returns(self.flat_prices, period=period)
        self.assertTrue(np.isnan(returns_flat[0:period]).all(), f"First {period} values should be NaN")
        self.assertTrue(np.allclose(returns_flat[period:], 0.0, rtol=1e-10, atol=1e-10, equal_nan=True),
                        "Returns should be 0 for flat prices")
        
        # Test ascending prices
        returns_ascending = calculate_returns(self.ascending_prices, period=period)
        self.assertTrue(np.isnan(returns_ascending[0:period]).all(), f"First {period} values should be NaN")
        
        # Calculate expected returns for ascending prices
        expected_returns = np.zeros_like(self.ascending_prices, dtype=float)
        for i in range(period, len(self.ascending_prices)):
            expected_returns[i] = (self.ascending_prices[i] - self.ascending_prices[i - period]) / self.ascending_prices[i - period]
        
        self.assertTrue(np.allclose(returns_ascending, expected_returns, rtol=1e-10, atol=1e-10, equal_nan=True),
                        "Returns calculation should match expected values for ascending prices")


if __name__ == "__main__":
    unittest.main()
